"""Golden-capture + verify for custom_fp_int_attention (#4 head-batching).

  python test_attn_opt.py capture   # current per-head-loop impl -> golden
  python test_attn_opt.py verify    # optimized impl vs golden (allclose + speedup)
"""
import sys, time, torch
import utils.figna_utils as F

GOLDEN = "/tmp/attn_golden.pt"
DEV = "cuda"

# (name, batch, Hq, Hkv, seq, head_dim, causal)
CONFIGS = [
    ("mha_s64",   1, 32, 32, 64,  128, True),
    ("gqa_s64",   1, 32, 8,  64,  128, True),
    ("gqa_s128",  1, 32, 8,  128, 128, True),
    ("gqa_s256",  1, 32, 8,  256, 128, True),
    ("gqa_s512",  1, 32, 8,  512, 128, True),
]

def build(name, b, Hq, Hkv, S, D, seed):
    torch.manual_seed(seed)
    q = torch.tanh(torch.randn(b, Hq, S, D, dtype=torch.float16, device=DEV))
    k_int = torch.randint(-8, 7, (b, Hkv, S, D), device=DEV, dtype=torch.int8)
    v_int = torch.randint(-8, 7, (b, Hkv, S, D), device=DEV, dtype=torch.int8)
    ks = torch.rand(b, Hkv, S, D, device=DEV, dtype=torch.float16) * 0.05
    vs = torch.rand(b, Hkv, S, D, device=DEV, dtype=torch.float16) * 0.05
    kz = torch.zeros(b, Hkv, S, D, device=DEV, dtype=torch.int32)
    vz = torch.zeros(b, Hkv, S, D, device=DEV, dtype=torch.int32)
    return q, k_int, v_int, ks, kz, vs, vz

def run(args, causal):
    q, k_int, v_int, ks, kz, vs, vz = args
    torch.cuda.synchronize(); t0 = time.time()
    out = F.custom_fp_int_attention(q, k_int, v_int, ks, kz, vs, vz, is_causal=causal)
    torch.cuda.synchronize(); return out, time.time() - t0

def capture():
    data = {}
    for i, (name, b, Hq, Hkv, S, D, causal) in enumerate(CONFIGS):
        args = build(name, b, Hq, Hkv, S, D, seed=i)
        run(args, causal); out, dt = run(args, causal)
        data[name] = dict(args=[a.cpu() for a in args], causal=causal, out=out.cpu(), dt=dt)
        print(f"[capture] {name:10s} b={b} Hq={Hq} Hkv={Hkv} S={S} D={D}  t={dt*1000:.1f}ms")
    torch.save(data, GOLDEN); print(f"saved -> {GOLDEN}")

def verify():
    data = torch.load(GOLDEN)
    print(f"{'config':10s} {'exact':7s} {'close':7s} {'maxdiff':11s} {'old(ms)':9s} {'new(ms)':9s} {'speedup':7s}")
    allok = True
    for name, d in data.items():
        args = [a.to(DEV) for a in d["args"]]; gold = d["out"].to(DEV)
        run(args, d["causal"]); new, dt = run(args, d["causal"])
        exact = torch.equal(new, gold)
        maxdiff = (new.float() - gold.float()).abs().max().item()
        close = torch.allclose(new.float(), gold.float(), atol=1e-2, rtol=1e-2)
        sp = d["dt"] / dt if dt > 0 else 0
        allok = allok and close
        flag = "" if close else "  <-- FAIL"
        print(f"{name:10s} {str(exact):7s} {str(close):7s} {maxdiff:<11.2e} {d['dt']*1000:<9.1f} {dt*1000:<9.1f} {sp:<7.1f}{flag}")
    print("ALL PASS (allclose)" if allok else "SOME FAILED")

if __name__ == "__main__":
    (capture if (len(sys.argv) > 1 and sys.argv[1] == "capture") else verify)()
