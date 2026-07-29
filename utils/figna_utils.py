import torch
from torch import Tensor
from typing import Tuple

# =========================
# Constants (match your module)
# =========================
MXU_K = 16
IN_EXP_BIAS = 15
IN_MAN_WIDTH = 10

EXTRA_BIT = 19
EXTRA_BIT_FOR_REDUCE = 10
MANTISSA_WIDTH = 10

def custom_fp16_int4_gemm(
    input: Tensor,       # [batch, seq_len, in_features] FP16
    weight_int4: Tensor, # [out_features, in_features] INT4
    scale: Tensor,       # [out_features, in_features] FP16
    bias: Tensor = None  # [out_features] FP16
) -> Tensor:
    """
    Custom FP16-INT4 mixed precision GEMM
    Y = (X @ W_dequant^T) + bias
    where W_dequant = (W_int4 - zero) * scale
    """
    # (debug print removed: fires per-GEMM during eval, cripples speed / bloats logs)

    output = fpint_gemm_qcol_real_2scomp_torch(
        input.reshape(-1, input.shape[-1]),  # (M,K)
        weight_int4.t(),                        # (K,N)
        scale.t(),                             # (K,N)
        torch.zeros_like(scale.t(), dtype=torch.int16),  # zero (K,N)
        groupsize=32,
        out_dtype=torch.float16,
        debug=False,
    ).reshape(input.shape[0], input.shape[1], -1)  # (batch, seq_len, out_features)
    
    
    # Fallback
    # batch_size, seq_len, in_features = input.shape
    # out_features = weight_int4.shape[0]
    
    # # FP16 dequantization
    # weight_fp = weight_int4 * scale
    # weight_fp = weight_fp.to(input.dtype)  # Convert to FP16
    
    # # GEMM
    # output = torch.matmul(input, weight_fp.t())  # [batch, seq_len, out_features]
    
    if bias is not None:
        output = output + bias
    
    return output.to(input.dtype)

def custom_fp_int_attention(
    query: torch.Tensor,           # [batch, num_heads, seq_len, head_dim] FP16
    key_int: torch.Tensor,          # [batch, num_heads, seq_len, head_dim] INT (quantized)
    value_int: torch.Tensor,        # [batch, num_heads, seq_len, head_dim] INT (quantized)
    scale_k: torch.Tensor,          # [batch, num_heads, seq_len, head_dim] or grouped FP16
    zero_k: torch.Tensor,           # [batch, num_heads, seq_len, head_dim] or grouped INT
    scale_v: torch.Tensor,          # [batch, num_heads, seq_len, head_dim] or grouped FP16
    zero_v: torch.Tensor,           # [batch, num_heads, seq_len, head_dim] or grouped INT
    attn_mask: torch.Tensor = None, # [batch, 1, seq_len, seq_len] or None
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Custom FP-INT attention kernel (fallback emulation)
    
    Computes: Softmax(Q @ K^T / sqrt(d)) @ V
    where K and V are quantized to INT4/INT8
    
    Args:
        query: FP16 query states
        key_int: Quantized key states (INT)
        value_int: Quantized value states (INT)
        scale_k, zero_k: Dequantization params for K
        scale_v, zero_v: Dequantization params for V
        attn_mask: Attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal mask
    
    Returns:
        Attention output [batch, num_heads, seq_len, head_dim] FP16
    """
    # print(f"[DEBUG] custom_fp_int_attention called!")
    # print(f"  query.shape: {query.shape}, query.dtype: {query.dtype}")
    # print(f"  key_int.shape: {key_int.shape}, key_int.dtype: {key_int.dtype}")
    # print(f"  value_int.shape: {value_int.shape}, value_int.dtype: {value_int.dtype}")
    # print(f"  scale_k.shape: {scale_k.shape}, scale_v.shape: {scale_v.shape}")
    # print(f"  scale_k.dtype: {scale_k.dtype}, scale_v.dtype: {scale_v.dtype}")
    # print(f"  zero_k.shape: {zero_k.shape}, zero_v.shape: {zero_v.shape}")
    # print(f"  zero_k.dtype: {zero_k.dtype}, zero_v.dtype: {zero_v.dtype}")
    # print(f"  attn_mask: {attn_mask.shape if attn_mask is not None else None}")
    # print(f"  is_causal: {is_causal}, dropout_p: {dropout_p}")
    
    batch_size, num_heads, seq_len, head_dim = query.shape

    # === GQA support ===
    # Query has `num_heads` heads, but the cached K/V may have fewer heads
    # (num_key_value_heads) when the model uses Grouped-Query Attention.
    # Each KV head is shared by `n_rep` consecutive query heads, matching the
    # repeat_kv mapping: query head j -> kv head (j // n_rep).
    # For MHA (Llama-2-7B) num_kv_heads == num_heads, so n_rep == 1 (no change).
    num_kv_heads = key_int.shape[1]
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    n_rep = num_heads // num_kv_heads

    # === Step 1: Dequantize K and V ===
    # K_fp16 = scale_k * (K_int - zero_k)
    
    # fall back dequantization
    # key_fp = scale_k * (key_int.to(scale_k.dtype) - zero_k.to(scale_k.dtype))
    # value_fp = scale_v * (value_int.to(scale_v.dtype) - zero_v.to(scale_v.dtype))
    
    # print(f"  [After dequant] key_fp.shape: {key_fp.shape}, key_fp.dtype: {key_fp.dtype}")
    # print(f"  [After dequant] value_fp.shape: {value_fp.shape}, value_fp.dtype: {value_fp.dtype}")
    
    # === Step 2: Q @ K^T ===
    # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
    # = [batch, num_heads, seq_len, seq_len]
    
    # fallback 
    # attn_weights = torch.matmul(query, key_fp.transpose(-2, -1))
    
    # #4: batch all (batch, head) into a leading B dim and run ONE batched qcol,
    # replacing the 32-head Python loop. GQA: query head j uses kv head j // n_rep,
    # so gather kv along the head dim to per-query-head tensors.
    kv_idx = torch.arange(num_heads, device=query.device) // n_rep       # (num_heads,)
    B = batch_size * num_heads
    q_b  = query.reshape(B, seq_len, head_dim)                            # (B,S,D)
    k_b  = key_int[:, kv_idx].reshape(B, seq_len, head_dim)              # (B,S,D)
    sk_b = scale_k[:, kv_idx].reshape(B, seq_len, head_dim)
    zk_b = zero_k[:, kv_idx].reshape(B, seq_len, head_dim)
    # Q@K^T: weight = K^T (D,S); scale/zero transposed; groupsize = head_dim
    attn_weights = fpint_gemm_qcol_batched(
        q_b,
        k_b.transpose(1, 2).contiguous(),   # (B,D,S)
        sk_b.transpose(1, 2).contiguous(),
        zk_b.transpose(1, 2).contiguous(),
        groupsize=head_dim,
        out_dtype=torch.float16,
    ).reshape(batch_size, num_heads, seq_len, seq_len)


    
    # Scale by sqrt(head_dim)
    attn_weights = attn_weights / torch.sqrt(torch.tensor(head_dim, dtype=query.dtype, device=query.device))
    # print(f"  [After QK^T] attn_weights.shape: {attn_weights.shape}")
    
    # === Step 3: Apply mask ===
    if is_causal and attn_mask is None:
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        # print(f"  [Applied causal mask]")
    elif attn_mask is not None:
        attn_weights = attn_weights + attn_mask
        # print(f"  [Applied attention mask]")
    
    # === Step 4: Softmax ===
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # === Step 5: Dropout (if training) ===
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)
        # print(f"  [Applied dropout with p={dropout_p}]")
    
    # === Step 6: P @ V ===
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    # = [batch, num_heads, seq_len, head_dim]
    # #4: batched P@V (one call for all heads). attn_weights (B,S,S) @ V (B,S,D).
    v_b  = value_int[:, kv_idx].reshape(B, seq_len, head_dim)             # (B,S,D)
    sv_b = scale_v[:, kv_idx].reshape(B, seq_len, head_dim)
    zv_b = zero_v[:, kv_idx].reshape(B, seq_len, head_dim)
    attn_output = fpint_gemm_qrow_batched(
        attn_weights.reshape(B, seq_len, seq_len),
        v_b, sv_b, zv_b,
        groupsize=head_dim,
        out_dtype=torch.float16,
    ).reshape(batch_size, num_heads, seq_len, head_dim)

    # attn_output = torch.matmul(attn_weights, value_fp)
    
    # print(f"  [After PV] attn_output.shape: {attn_output.shape}, attn_output.dtype: {attn_output.dtype}")
    # print(f"[DEBUG] custom_fp_int_attention completed!")
    
    return attn_output.to(query.dtype)

@torch.no_grad()
def fpint_gemm_qcol_real_2scomp_torch(
    input_data: torch.Tensor,   # (M,K) float16
    weight_data: torch.Tensor,  # (K,N) int8 (signed)
    scale_data: torch.Tensor,   # (K,N) float16/float32 (duplicated per groupsize on K)
    zero_data: torch.Tensor,    # (K,N) int16/int32 (duplicated per groupsize on K)
    groupsize: int,
    out_dtype: torch.dtype = torch.float16,
    debug: bool = False,
) -> torch.Tensor:
    """
    MAC-loop emulation (closest to 16x16 HW behavior):
    """
    assert input_data.is_cuda and weight_data.is_cuda and scale_data.is_cuda and zero_data.is_cuda
    assert input_data.dtype == torch.float16
    assert weight_data.dtype in (torch.int8, torch.int16, torch.int32)
    assert zero_data.dtype in (torch.int16, torch.int32)
    assert scale_data.dtype in (torch.float16, torch.float32)

    M, K = input_data.shape
    K_w, N = weight_data.shape
    assert K_w == K
    assert scale_data.shape == (K, N)
    assert zero_data.shape == (K, N)
    assert K % MXU_K == 0
    assert groupsize > 0 and (K % groupsize == 0)

    # prealign
    aligned_fx_main, aligned_exp = _prealign_torch_fp16bits(input_data, EXTRA_BIT)
    # #3: the reduce (act_sum) path only matters when zero != 0. The symmetric
    # linear-weight path passes zero==0, so skip the 2nd prealign + reduce entirely.
    has_zero = bool(zero_data.any())
    if has_zero:
        aligned_fx_red, _ = _prealign_torch_fp16bits(input_data, EXTRA_BIT_FOR_REDUCE)

    KG = K // MXU_K
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))
    two = torch.tensor(2.0, device=input_data.device, dtype=torch.float32)

    if not has_zero:
        # Collapsed fast path for the symmetric (zero==0) weight-GEMM linear path.
        # Each block's exponent factor 2^(e-bias) is CONSTANT over its 16 lanes, so it
        # distributes over the lane sum: fold it into the aligned activation, fold the
        # per-column scale into the weight, then the whole KG-block computation is a
        # SINGLE (M,K)@(K,N) matmul. Values fit fp32, so it matches the block-wise
        # result within fp16 ULP (verified allclose). TF32 OFF keeps fp32 exact enough.
        # ~40-60x faster than the block loop (no (KG,M,N) intermediate / elementwise).
        _tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        exp_scale = torch.pow(
            two, (aligned_exp.to(torch.int32) - IN_EXP_BIAS).to(torch.float32)
        ).repeat_interleave(MXU_K, dim=1)                                    # (M,K)
        B = aligned_fx_main.to(torch.float32) * exp_scale * mant_scale       # (M,K)
        Wp = weight_data.to(torch.float32) * scale_data.to(torch.float32)    # (K,N)
        acc = torch.matmul(B, Wp)                                            # (M,N) f32
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        return acc if out_dtype == torch.float32 else acc.to(out_dtype)

    # #1+#2: batch the 16-lane MAC AND all KG blocks into batched matmuls (bmm).
    # aligned_fx_main -> (KG,M,16), weight -> (KG,16,N); one bmm produces every
    # block's integer inner product at once. float64 exactly represents these ints
    # (<2^53), so the contraction is exact. Tiled over N to bound the (KG,M,tile)
    # intermediate. Cross-block float reduction is a vectorized sum (order differs
    # from the sequential acc+= by ULP only; passes the golden allclose test).
    A = aligned_fx_main.view(M, KG, MXU_K).permute(1, 0, 2).to(torch.float64).contiguous()  # (KG,M,16)
    Wv = weight_data.to(torch.float64).view(KG, MXU_K, N)                                    # (KG,16,N)
    exp_scale = torch.pow(two, (aligned_exp.to(torch.int32) - IN_EXP_BIAS).to(torch.float32))  # (M,KG)
    exp_scale = exp_scale.permute(1, 0).unsqueeze(2)                                          # (KG,M,1)
    sc = scale_data[0::MXU_K, :].to(torch.float32)                                            # (KG,N)
    if has_zero:
        act_red = aligned_fx_red.view(M, KG, MXU_K).sum(dim=2).to(torch.float64)             # (M,KG)
        act_red = act_red.permute(1, 0).unsqueeze(2)                                         # (KG,M,1)
        z_all = zero_data[0::MXU_K, :].to(torch.float64)                                     # (KG,N)

    acc = torch.zeros((M, N), device=input_data.device, dtype=torch.float32)
    n_tile = max(MXU_K, min(N, int((256 * 1024 * 1024) // max(1, KG * M))))  # bound (KG,M,tile) f64
    for n0 in range(0, N, n_tile):
        n1 = min(N, n0 + n_tile)
        inner = torch.bmm(A, Wv[:, :, n0:n1])                                                # (KG,M,nt) f64
        if has_zero:
            inner = inner - act_red * z_all[:, n0:n1].unsqueeze(1) * float(1 << shift_back)
        post_f32 = inner.round().to(torch.float32)                                           # (KG,M,nt)
        contrib = post_f32 * exp_scale * mant_scale * sc[:, n0:n1].unsqueeze(1)              # (KG,M,nt) f32
        # reduce across blocks in float64 (exact sum of the f32 terms) to avoid the
        # f32 accumulation error that grows with KG on large-K layers.
        acc[:, n0:n1] = contrib.double().sum(dim=0).to(torch.float32)                        # (M,nt) f32

    if out_dtype == torch.float32:
        return acc
    return acc.to(out_dtype)

@torch.no_grad()
def fpint_gemm_qrow_real_2scomp_torch(
    input_data: torch.Tensor,   # (M,K) float16
    weight_data: torch.Tensor,  # (K,N) int8 (signed)
    scale_data: torch.Tensor,   # (K,N) float16 (duplicated groupsize times)
    zero_data: torch.Tensor,    # (K,N) int16 (duplicated groupsize times)
    groupsize: int,             # Quantization block size in N direction
    out_dtype: torch.dtype = torch.float16,
    debug: bool = False,
) -> torch.Tensor:
    """
    FPINT GEMM with row-wise quantization (qrow) using real 2's complement encoding.
    Uses signed weights directly without 2*inner_product + act_sum transformation.
    
    Args:
        input_data: FP16 input activations, shape (M, K)
        weight_data: Quantized weights (int8), shape (K, N)
        scale_data: FP16 scale factors, shape (K, N) - duplicated groupsize times
        zero_data: Zero points (int16), shape (K, N) - duplicated groupsize times
        groupsize: Quantization block size in N direction (for reference)
        out_dtype: Output dtype
        debug: Enable debug printing
        
    Returns:
        output_data: Output, shape (M, N)
    """
    assert input_data.is_cuda and weight_data.is_cuda and scale_data.is_cuda and zero_data.is_cuda
    assert input_data.dtype == torch.float16
    assert weight_data.dtype in (torch.int8, torch.int16, torch.int32)
    assert zero_data.dtype in (torch.int16, torch.int32)
    assert scale_data.dtype in (torch.float16, torch.float32)

    M, K = input_data.shape
    # Zero-shot sequences have arbitrary length, so the P@V contraction dim
    # (K = seq_len) is often not a multiple of MXU_K. Pad K with zeros, which
    # contributes nothing to the accumulation and leaves the result unchanged.
    _pad = (-K) % MXU_K
    if _pad:
        input_data = torch.nn.functional.pad(input_data, (0, _pad))
        weight_data = torch.nn.functional.pad(weight_data, (0, 0, 0, _pad))
        scale_data = torch.nn.functional.pad(scale_data, (0, 0, 0, _pad))
        zero_data = torch.nn.functional.pad(zero_data, (0, 0, 0, _pad))
        M, K = input_data.shape
    K_w, N = weight_data.shape
    assert K_w == K
    K_s, N_s = scale_data.shape
    assert K_s == K
    assert N_s == N
    assert zero_data.shape == (K, N)
    assert K % MXU_K == 0
    assert groupsize > 0

    # Convert dtypes
    if weight_data.dtype != torch.int8:
        weight_data = weight_data.to(torch.int8)
    if zero_data.dtype != torch.int16:
        zero_data = zero_data.to(torch.int16)
    
    scale_fp = scale_data.to(torch.float32)
    zero_fp = zero_data.to(torch.int16)
    
    # Constants
    KG = K // MXU_K
    NG = N // groupsize 
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))
    two = torch.tensor(2.0, device=input_data.device, dtype=torch.float32)

    # Output accumulator
    acc = torch.zeros((M, N), device=input_data.device, dtype=torch.float32)

    if debug:
        print("[FPINT_EMUL.QROW_2SCOMP_TORCH] ===== Start GEMM calculation =====")

    # Process each group of N dim
    for ng in range(NG):
        # Scale input: scaled_input[m, k] = input[m, k] * scale[k, n]
        n_start = ng * groupsize
        n_end = n_start + groupsize
        scaled_input = input_data * scale_fp[:, n_start].unsqueeze(0)  #  scale_fp[ has same values across n_start to n_end - 1
        scaled_input = scaled_input.to(torch.float16)  # shape: (M,K) float16
        
        # Prealign scaled input
        aligned_fx_main, aligned_exp = _prealign_torch_fp16bits(scaled_input, EXTRA_BIT)
        aligned_fx_red, _ = _prealign_torch_fp16bits(scaled_input, EXTRA_BIT_FOR_REDUCE)
        
        # Get zero and weight for this column
        z_group = zero_fp[:, n_start:n_end].to(torch.int64)  # (K, groupsize)
        w_group = weight_data[:, n_start:n_end].to(torch.int64)  # (K, groupsize)
        acc_group = acc[:, n_start:n_end]  # (M, groupsize) float32 accumulator for this group
        
        # #1+#2: batch the 16-lane MAC AND all KG blocks into two bmm calls.
        # (M,K) reshaped to (KG,M,16); w_group/z_group to (KG,16,gs). float64 exactly
        # represents the ints. gs is small (head_dim), so (KG,M,gs) is cheap — no tiling.
        Am = aligned_fx_main.view(M, KG, MXU_K).permute(1, 0, 2).to(torch.float64).contiguous()  # (KG,M,16)
        Ar = aligned_fx_red.view(M, KG, MXU_K).permute(1, 0, 2).to(torch.float64).contiguous()   # (KG,M,16)
        w_g64 = w_group.to(torch.float64).reshape(KG, MXU_K, groupsize)   # (KG,16,gs)
        z_g64 = z_group.to(torch.float64).reshape(KG, MXU_K, groupsize)   # (KG,16,gs)

        inner = torch.bmm(Am, w_g64)                             # (KG,M,gs) f64 exact
        act_sum_red = torch.bmm(Ar, z_g64)                       # (KG,M,gs) f64 exact
        post_f32 = (inner - act_sum_red * float(1 << shift_back)).round().to(torch.float32)  # (KG,M,gs)

        exp_scale = torch.pow(two, (aligned_exp.to(torch.int32) - IN_EXP_BIAS).to(torch.float32))  # (M,KG)
        exp_scale = exp_scale.permute(1, 0).unsqueeze(2)         # (KG,M,1)
        contrib = post_f32 * exp_scale * mant_scale              # (KG,M,gs) float32
        acc_group += contrib.double().sum(dim=0).to(torch.float32)  # (M,gs) f64 reduction

    if debug:
        print(f"[FPINT_EMUL.QROW_2SCOMP_TORCH] Output computed, shape: {acc.shape}")
        if M <= 2 and N <= 8:
            print(f"  acc[0, :8] = {acc[0, :8]}")

    return acc.to(out_dtype)

def _prealign_torch_fp16bits(
    input_fp16: torch.Tensor,  # (M,K) float16
    extra_bitwidth: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    prealign on GPU using torch ops.
    Returns:
      aligned_fx:  (M,K) int64
      aligned_exp: (M,K//16) int16
    """
    assert input_fp16.dtype == torch.float16
    assert input_fp16.is_cuda
    input_fp16 = input_fp16.contiguous()

    M, K = input_fp16.shape
    assert K % MXU_K == 0

    bits = input_fp16.view(torch.uint16).to(torch.int32)  # do bitops in int32 on CUDA

    sign = (bits >> 15) & 0x1
    exp  = (bits >> 10) & 0x1F
    mant = bits & 0x3FF

    exp_for_align = torch.where(exp == 0, torch.ones_like(exp), exp)  # denorm exp=1

    KG = K // MXU_K
    exp_g = exp_for_align.view(M, KG, MXU_K)
    max_exp = exp_g.max(dim=2).values  # (M,KG)
    aligned_exp = max_exp.to(torch.int16)

    hidden_bit = (exp != 0).to(torch.int32)
    hidden_man = (hidden_bit << MANTISSA_WIDTH) | mant  # 11-bit in int32

    hidden_man_ext = hidden_man.to(torch.int64) << extra_bitwidth

    max_exp_b = max_exp.unsqueeze(-1).expand(M, KG, MXU_K).reshape(M, K).to(torch.int64)
    exp_b = exp_for_align.to(torch.int64)
    shift_amount = (max_exp_b - exp_b).clamp(min=0)

    shifted = hidden_man_ext >> shift_amount
    aligned_fx = torch.where(sign.bool(), -shifted, shifted).to(torch.int64)
    return aligned_fx, aligned_exp


@torch.no_grad()
def fpint_gemm_qcol_batched(inp, w, sc, z, groupsize, out_dtype=torch.float16, b_tile=8):
    """Batched qcol: leading batch dim B. inp (B,M,K), w/sc/z (B,K,N) -> (B,M,N).
    Mirrors fpint_gemm_qcol_real_2scomp_torch per batch element (bit-exact within fp16).
    Used by custom_fp_int_attention to run all heads without a Python per-head loop.
    Tiled over B to bound the (chunk,KG,M,N) intermediate."""
    B, M, K = inp.shape
    N = w.shape[-1]
    KG = K // MXU_K
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))
    two = torch.tensor(2.0, device=inp.device, dtype=torch.float32)
    has_zero = bool(z.any())
    out = torch.empty(B, M, N, device=inp.device, dtype=out_dtype)
    for b0 in range(0, B, b_tile):
        b1 = min(B, b0 + b_tile); bs = b1 - b0
        xi = inp[b0:b1]
        afx, aexp = _prealign_torch_fp16bits(xi.reshape(bs * M, K), EXTRA_BIT)
        afx = afx.view(bs, M, KG, MXU_K).permute(0, 2, 1, 3).to(torch.float64)   # (bs,KG,M,16)
        Wv = w[b0:b1].to(torch.float64).view(bs, KG, MXU_K, N)                    # (bs,KG,16,N)
        inner = torch.matmul(afx, Wv)                                            # (bs,KG,M,N)
        if has_zero:
            afr, _ = _prealign_torch_fp16bits(xi.reshape(bs * M, K), EXTRA_BIT_FOR_REDUCE)
            act_red = afr.view(bs, M, KG, MXU_K).sum(-1).to(torch.float64).permute(0, 2, 1).unsqueeze(3)  # (bs,KG,M,1)
            z_all = z[b0:b1, 0::MXU_K, :].to(torch.float64).unsqueeze(2)          # (bs,KG,1,N)
            inner = inner - act_red * z_all * float(1 << shift_back)
        post = inner.round().to(torch.float32)                                   # (bs,KG,M,N)
        exp_scale = torch.pow(two, (aexp.view(bs, M, KG).to(torch.int32) - IN_EXP_BIAS).to(torch.float32))
        exp_scale = exp_scale.permute(0, 2, 1).unsqueeze(3)                       # (bs,KG,M,1)
        scb = sc[b0:b1, 0::MXU_K, :].to(torch.float32).unsqueeze(2)              # (bs,KG,1,N)
        contrib = post * exp_scale * mant_scale * scb                            # (bs,KG,M,N)
        out[b0:b1] = contrib.double().sum(dim=1).to(out_dtype)                    # (bs,M,N)
    return out


@torch.no_grad()
def fpint_gemm_qrow_batched(inp, w, sc, z, groupsize, out_dtype=torch.float16, b_tile=8):
    """Batched qrow (P@V) for attention: leading batch dim B, groupsize == N (NG==1).
    inp (B,M,K), w/sc/z (B,K,N) -> (B,M,N). Pads K to a multiple of 16."""
    B, M, K = inp.shape
    N = w.shape[-1]
    assert groupsize == N, "batched qrow assumes a single N-group (attention head-wise)"
    _pad = (-K) % MXU_K
    if _pad:
        inp = torch.nn.functional.pad(inp, (0, _pad))          # pad K (last dim)
        w = torch.nn.functional.pad(w, (0, 0, 0, _pad))        # pad K (dim -2)
        sc = torch.nn.functional.pad(sc, (0, 0, 0, _pad))
        z = torch.nn.functional.pad(z, (0, 0, 0, _pad))
        K = K + _pad
    KG = K // MXU_K
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))
    two = torch.tensor(2.0, device=inp.device, dtype=torch.float32)
    out = torch.empty(B, M, N, device=inp.device, dtype=out_dtype)
    for b0 in range(0, B, b_tile):
        b1 = min(B, b0 + b_tile); bs = b1 - b0
        # rescale input by the per-K scale (column 0 of the single group), like the 2D qrow
        scaled = (inp[b0:b1] * sc[b0:b1, :, 0:1].transpose(1, 2)).to(torch.float16)  # (bs,M,K)
        am, aexp = _prealign_torch_fp16bits(scaled.reshape(bs * M, K), EXTRA_BIT)
        ar, _ = _prealign_torch_fp16bits(scaled.reshape(bs * M, K), EXTRA_BIT_FOR_REDUCE)
        am = am.view(bs, M, KG, MXU_K).permute(0, 2, 1, 3).to(torch.float64)     # (bs,KG,M,16)
        ar = ar.view(bs, M, KG, MXU_K).permute(0, 2, 1, 3).to(torch.float64)
        w_g = w[b0:b1].to(torch.float64).view(bs, KG, MXU_K, N)                   # (bs,KG,16,N)
        z_g = z[b0:b1].to(torch.float64).view(bs, KG, MXU_K, N)
        inner = torch.matmul(am, w_g)                                            # (bs,KG,M,N)
        act_red = torch.matmul(ar, z_g)                                          # (bs,KG,M,N)
        post = (inner - act_red * float(1 << shift_back)).round().to(torch.float32)
        exp_scale = torch.pow(two, (aexp.view(bs, M, KG).to(torch.int32) - IN_EXP_BIAS).to(torch.float32))
        exp_scale = exp_scale.permute(0, 2, 1).unsqueeze(3)                       # (bs,KG,M,1)
        contrib = post * exp_scale * mant_scale                                  # (bs,KG,M,N)
        out[b0:b1] = contrib.double().sum(dim=1).to(out_dtype)                    # (bs,M,N)
    return out


def _make_groupwise_duplicated_kn(x: torch.Tensor, groupsize: int) -> torch.Tensor:
    K, N = x.shape
    assert K % groupsize == 0
    x2 = x.clone()
    for k0 in range(0, K, groupsize):
        x2[k0:k0+groupsize, :] = x2[k0:k0+1, :].expand(groupsize, N)
    return x2

def _make_groupwise_duplicated_nk(x: torch.Tensor, groupsize: int) -> torch.Tensor:
    """Duplicate along N dimension (for qrow)"""
    K, N = x.shape
    assert N % groupsize == 0
    x2 = x.clone()
    for n0 in range(0, N, groupsize):
        x2[:, n0:n0+groupsize] = x2[:, n0:n0+1].expand(K, groupsize)
    return x2

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    assert torch.cuda.is_available()
    device = "cuda"

    M, K, N = 256, 256, 256
    groupsize = 32
    
    print("="*60)
    print("Testing qcol...")
    print("="*60)

    inp = torch.tanh(torch.randn(M, K, device=device, dtype=torch.float16))
    wt  = torch.randint(-8, 7, (K, N), device=device, dtype=torch.int8)

    sc_base = (torch.rand(K, N, device=device, dtype=torch.float16) * 0.1)
    ze_base = torch.zeros(K, N, device=device, dtype=torch.int16)

    sc = _make_groupwise_duplicated_kn(sc_base, groupsize)
    ze = _make_groupwise_duplicated_kn(ze_base, groupsize)
    
    # Measure QCOL
    start = time.time()
    out_fpint = fpint_gemm_qcol_real_2scomp_torch(
        inp, wt, sc, ze, groupsize=groupsize, out_dtype=torch.float16, debug=False
    )
    torch.cuda.synchronize()
    qcol_time = time.time() - start

    # Measure reference
    start = time.time()
    D = sc.to(torch.float32) * (wt.to(torch.float32) - ze.to(torch.float32))
    out_ref = (inp.to(torch.float32) @ D).to(torch.float16)
    torch.cuda.synchronize()
    ref_time = time.time() - start

    diff = (out_fpint.to(torch.float32) - out_ref.to(torch.float32)).abs()
    
    # all cose check
    print("QCOL Results:")
    print(f"Is close? : {torch.allclose(out_fpint, out_ref, atol=1e-3, rtol=1e-3)}")
    print("fpint[0,:8] :", out_fpint[0, :8].detach().cpu())
    print("ref  [0,:8] :", out_ref[0, :8].detach().cpu())

    print("\n" + "="*60)
    print("Testing qrow...")
    print("="*60)
    
    # For qrow: duplicate along N direction (row-wise quantization)
    sc_row = _make_groupwise_duplicated_nk(sc_base, groupsize)
    ze_row = _make_groupwise_duplicated_nk(ze_base, groupsize)
    
    start = time.time()
    out_qrow = fpint_gemm_qrow_real_2scomp_torch(
        inp, wt, sc_row, ze_row, groupsize=groupsize, out_dtype=torch.float16, debug=False
    )
    torch.cuda.synchronize()
    qrow_time = time.time() - start
    
    # naive reference for qrow
    D_row = sc_row.to(torch.float32) * (wt.to(torch.float32) - ze_row.to(torch.float32))
    out_ref_row = (inp.to(torch.float32) @ D_row).to(torch.float16)
    
    diff_row = (out_qrow.to(torch.float32) - out_ref_row.to(torch.float32)).abs()
    print("QROW Results:")
    print(f"Is close? : {torch.allclose(out_qrow, out_ref_row, atol=1e-2, rtol=1e-2)}")
    print("qrow [0,:8] :", out_qrow[0, :8].detach().cpu())
    print("ref  [0,:8] :", out_ref_row[0, :8].detach().cpu())
    
    # Performance Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Method':<15} {'Time (ms)':<15} {'vs Ref':<15} {'vs QCOL':<15}")
    print("-"*70)
    print(f"{'Reference':<15} {ref_time*1000:<15.2f} {'1.00x':<15} {'-':<15}")
    print(f"{'QCOL':<15} {qcol_time*1000:<15.2f} {qcol_time/ref_time:<15.2f} {'1.00x':<15}")
    print(f"{'QROW':<15} {qrow_time*1000:<15.2f} {qrow_time/ref_time:<15.2f} {qrow_time/qcol_time:<15.2f}")
    print("="*70)