"""Golden-capture + verify harness for figna_utils optimizations.

Usage:
  python test_figna_opt.py capture   # run CURRENT impl, save inputs/outputs/timing to golden file
  python test_figna_opt.py verify    # run (now-optimized) impl on saved inputs, compare bit-exact + timing
"""
import sys, time, torch
import utils.figna_utils as F

GOLDEN = "/tmp/figna_golden.pt"
DEV = "cuda"

def make_dup_kn(x, gs):  # duplicate along K (qcol)
    K, N = x.shape; x2 = x.clone()
    for k0 in range(0, K, gs): x2[k0:k0+gs, :] = x2[k0:k0+1, :].expand(gs, N)
    return x2

def make_dup_nk(x, gs):  # duplicate along N (qrow)
    K, N = x.shape; x2 = x.clone()
    for n0 in range(0, N, gs): x2[:, n0:n0+gs] = x2[:, n0:n0+1].expand(K, gs)
    return x2

# (name, kind, M, K, N, groupsize, zero_nonzero)
CONFIGS = [
    ("qcol_main_z0",   "qcol", 256, 256, 256, 32,  False),
    ("qcol_z_nonzero", "qcol", 128, 256, 256, 32,  True),
    ("qcol_linear_z0", "qcol", 64,  512, 512, 32,  False),   # linear-like, bigger K
    ("qcol_attn_z0",   "qcol", 100, 128, 100, 128, False),   # attn Q@K^T-like (K=head_dim)
    ("qcol_perf_z0",   "qcol", 128, 2048,2048,32,  False),   # perf: big K linear
    ("qrow_main",      "qrow", 256, 256, 256, 32,  False),
    ("qrow_attn",      "qrow", 100, 112, 128, 128, False),   # attn P@V-like, unaligned K=112
    ("qrow_perf",      "qrow", 128, 2048,256, 32,  False),
]

def build_inputs(name, kind, M, K, N, gs, znz, seed):
    torch.manual_seed(seed)
    inp = torch.tanh(torch.randn(M, K, device=DEV, dtype=torch.float16))
    wt  = torch.randint(-8, 7, (K, N), device=DEV, dtype=torch.int8)
    sc_base = torch.rand(K, N, device=DEV, dtype=torch.float16) * 0.1
    ze_base = torch.zeros(K, N, device=DEV, dtype=torch.int16)
    if znz:
        ze_base = torch.randint(-2, 3, (K, N), device=DEV, dtype=torch.int16)
    if kind == "qcol":
        sc = make_dup_kn(sc_base, gs); ze = make_dup_kn(ze_base, gs)
    else:
        sc = make_dup_nk(sc_base, gs); ze = make_dup_nk(ze_base, gs)
    return inp, wt, sc, ze

def run_one(kind, inp, wt, sc, ze, gs):
    fn = F.fpint_gemm_qcol_real_2scomp_torch if kind == "qcol" else F.fpint_gemm_qrow_real_2scomp_torch
    torch.cuda.synchronize(); t0 = time.time()
    out = fn(inp, wt, sc, ze, groupsize=gs, out_dtype=torch.float16)
    torch.cuda.synchronize(); dt = time.time() - t0
    return out, dt

def capture():
    data = {}
    for i, (name, kind, M, K, N, gs, znz) in enumerate(CONFIGS):
        inp, wt, sc, ze = build_inputs(name, kind, M, K, N, gs, znz, seed=i)
        # warmup + timed
        run_one(kind, inp, wt, sc, ze, gs)
        out, dt = run_one(kind, inp, wt, sc, ze, gs)
        data[name] = dict(kind=kind, gs=gs, inp=inp.cpu(), wt=wt.cpu(), sc=sc.cpu(),
                          ze=ze.cpu(), out=out.cpu(), dt=dt)
        print(f"[capture] {name:16s} {kind} M={M} K={K} N={N} gs={gs} znz={znz}  t={dt*1000:.1f}ms")
    torch.save(data, GOLDEN)
    print(f"saved golden -> {GOLDEN}")

def verify():
    data = torch.load(GOLDEN)
    print(f"{'config':16s} {'exact':9s} {'close':9s} {'maxdiff':11s} {'old(ms)':9s} {'new(ms)':9s} {'speedup':7s}")
    allok = True
    for name, d in data.items():
        inp = d["inp"].to(DEV); wt = d["wt"].to(DEV); sc = d["sc"].to(DEV); ze = d["ze"].to(DEV)
        gold = d["out"].to(DEV)
        run_one(d["kind"], inp, wt, sc, ze, d["gs"])  # warmup
        new, dt = run_one(d["kind"], inp, wt, sc, ze, d["gs"])
        exact = torch.equal(new, gold)
        maxdiff = (new.float() - gold.float()).abs().max().item()
        speedup = d["dt"] / dt if dt > 0 else 0
        # acceptance = the design's own golden test tolerance (allclose, same as __main__:
        # qcol atol/rtol 1e-3, qrow 1e-2). fp16-ULP-level reordering diffs are OK.
        atol = 1e-3 if d["kind"] == "qcol" else 1e-2
        close = torch.allclose(new.float(), gold.float(), atol=atol, rtol=atol)
        allok = allok and close
        flag = "" if close else "  <-- FAIL"
        print(f"{name:16s} {str(exact):9s} {str(close):9s} {maxdiff:<11.2e} {d['dt']*1000:<9.1f} {dt*1000:<9.1f} {speedup:<7.1f}{flag}")
    print("ALL PASS (allclose)" if allok else "SOME FAILED")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "capture"
    (capture if mode == "capture" else verify)()
