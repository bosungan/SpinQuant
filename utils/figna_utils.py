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
    print("Custom FP16-INT4 GEMM called!")
    
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
    
    # custom gemm
    attn_weights = torch.zeros((batch_size, num_heads, seq_len, seq_len), device=query.device, dtype=torch.float16)
    for i in range(batch_size):
        for j in range(num_heads):
            attn_weights[i, j] = fpint_gemm_qcol_real_2scomp_torch(
                query[i, j].contiguous(), # (seq_len, head_dim)
                key_int[i, j].t().contiguous(), # (head_dim, seq_len)
                scale_k[i, j].t().contiguous(), # (head_dim, seq_len)
                zero_k[i, j].t().contiguous(), # (head_dim, seq_len)
                groupsize=head_dim, # headwise quantization
                out_dtype=torch.float16
            )
            
    
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
    attn_output = torch.zeros((batch_size, num_heads, seq_len, head_dim), device=query.device, dtype=torch.float16)
    for i in range(batch_size):
        for j in range(num_heads):
            attn_output[i, j] = fp_int_gemm_qrow_real_2scomp_torch(
                attn_weights[i, j].contiguous(), # (seq_len, seq_len)
                value_int[i, j].contiguous(), # (seq_len, head_dim)
                scale_v[i, j].contiguous(), # (seq_len, head_dim)
                zero_v[i, j].contiguous(), # (seq_len, head_dim)
                groupsize=head_dim, # headwise quantization
                out_dtype=torch.float16
            )
            
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
    aligned_fx_red, _ = _prealign_torch_fp16bits(input_data, EXTRA_BIT_FOR_REDUCE)

    KG = K // MXU_K
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))

    # float accumulator (matches your python model style)
    acc = torch.zeros((M, N), device=input_data.device, dtype=torch.float32)
    two = torch.tensor(2.0, device=input_data.device, dtype=torch.float32)

    for g in range(KG):
        k0 = g * MXU_K
        k1 = k0 + MXU_K

        inner = torch.zeros((M, N), device=input_data.device, dtype=torch.int64)

        # 16-lane MAC: inner += a[:,lane] * w[lane,:]
        for lane in range(MXU_K):
            a = aligned_fx_main[:, k0 + lane].view(M, 1)                 # (M,1) int64
            w = weight_data[k0 + lane, :].to(torch.int64).view(1, N)     # (1,N) int64
            inner += a * w                                               # broadcast → (M,N)

        # === act_sum_for_reduce: int64 sum of 16 lanes (reduce precision path) ===
        act_sum_red = torch.zeros((M, 1), device=input_data.device, dtype=torch.int64)
        for lane in range(MXU_K):
            act_sum_red += aligned_fx_red[:, k0 + lane].view(M, 1)

        # qcol constants (duplicated per groupsize, but we just sample k0)
        z  = zero_data[k0, :].to(torch.int64).view(1, N)                 # (1,N)
        sc = scale_data[k0, :].to(torch.float32).view(1, N)              # (1,N)

        # post_inner_product (int64 domain)
        post = inner - ((z * act_sum_red) << shift_back)                 # (M,N) int64

        # exponent restore (per m, per block g)
        e = aligned_exp[:, g].to(torch.int32)                            # (M,)
        exp_scale = torch.pow(two, (e - IN_EXP_BIAS).to(torch.float32)).view(M, 1)
        post_fp = post.to(torch.float32) * exp_scale * mant_scale        # (M,N) float32
        acc += post_fp * sc                                              # (M,N) float32

        if debug and g == 0:
            print("[DEBUG block0]")
            print(" inner[0,0] =", int(inner[0,0].item()))
            print(" act_sum_red[0] =", int(act_sum_red[0,0].item()))
            print(" z[0] =", int(z[0,0].item()))
            print(" exp(m=0) =", int(e[0].item()))
            print(" sc[0] =", float(sc[0,0].item()))
            print(" acc[0,0] =", float(acc[0,0].item()))

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
    K_w, N = weight_data.shape
    assert K_w == K
    K_s, N_s = scale_data.shape
    assert K_s == K
    assert N_s == N
    assert zero_data.shape == (K, N)
    assert K % MXU_K == 0
    assert groupsize > 0

    if debug:
        print(f"[FPINT_EMUL.QROW_2SCOMP_TORCH] M={M}, N={N}, K={K}, groupsize={groupsize}")
        print(f"  input: {input_data.shape} {input_data.dtype}")
        print(f"  weight: {weight_data.shape} {weight_data.dtype}")
        print(f"  scale: {scale_data.shape} {scale_data.dtype}")
        print(f"  zero: {zero_data.shape} {zero_data.dtype}")

    # Convert dtypes
    if weight_data.dtype != torch.int8:
        weight_data = weight_data.to(torch.int8)
    if zero_data.dtype != torch.int16:
        zero_data = zero_data.to(torch.int16)
    
    scale_fp = scale_data.to(torch.float32)
    zero_fp = zero_data.to(torch.int16)
    
    # Constants
    KG = K // MXU_K
    shift_back = EXTRA_BIT - EXTRA_BIT_FOR_REDUCE
    mant_scale = 2.0 ** (-(IN_MAN_WIDTH + EXTRA_BIT))
    two = torch.tensor(2.0, device=input_data.device, dtype=torch.float32)

    # Output accumulator
    acc = torch.zeros((M, N), device=input_data.device, dtype=torch.float32)

    if debug:
        print("[FPINT_EMUL.QROW_2SCOMP_TORCH] ===== Start GEMM calculation =====")

    # Process each output column n
    for n in range(N):
        # Scale input: scaled_input[m, k] = input[m, k] * scale[k, n]
        scaled_input = input_data * scale_fp[:, n].unsqueeze(0)  # (M, K)
        scaled_input = scaled_input.to(torch.float16)  # Ensure it's FP16 for prealignment
        
        # Prealign scaled input
        aligned_fx_main, aligned_exp = _prealign_torch_fp16bits(scaled_input, EXTRA_BIT)
        aligned_fx_red, _ = _prealign_torch_fp16bits(scaled_input, EXTRA_BIT_FOR_REDUCE)
        
        # Get zero and weight for this column
        z_col = zero_fp[:, n].to(torch.int64)  # (K,)
        w_col = weight_data[:, n].to(torch.int64)  # (K,)
        
        # Process K dimension in blocks of MXU_K
        for g in range(KG):
            k0 = g * MXU_K
            k1 = k0 + MXU_K
            
            # Initialize accumulators (vectorized over M)
            inner = torch.zeros(M, device=input_data.device, dtype=torch.int64)
            act_sum_red = torch.zeros(M, device=input_data.device, dtype=torch.int64)
            
            # MAC over MXU_K lanes (vectorized)
            for lane in range(MXU_K):
                k = k0 + lane
                a_main = aligned_fx_main[:, k]  # (M,)
                a_red = aligned_fx_red[:, k]    # (M,)
                w = w_col[k]                    # scalar
                z = z_col[k]                    # scalar
                
                inner += a_main * w
                act_sum_red += a_red * z
            
            # Post-processing (real 2's complement: direct signed weight)
            post = inner - (act_sum_red << shift_back)  # (M,)
            
            # Convert to float with exponent restoration
            e = aligned_exp[:, g].to(torch.int32)  # (M,)
            exp_scale = torch.pow(two, (e - IN_EXP_BIAS).to(torch.float32))  # (M,)
            post_fp = post.to(torch.float32) * exp_scale * mant_scale  # (M,)
            
            # No scaling factor for real 2's complement
            scaled_post = post_fp  # (M,)
            
            acc[:, n] += scaled_post
            
            if debug and n < 2 and g == 0:
                print(f"[QROW_REAL_2SCOMP] n={n}, g={g}")
                print(f"  z[{k0}]={int(z_col[k0].item())}, w[{k0}]={int(w_col[k0].item())}")
                print(f"  inner[0]={int(inner[0].item())}, act_sum_red[0]={int(act_sum_red[0].item())}")
                print(f"  post[0]={int(post[0].item())}, exp[0]={int(e[0].item())}")
                print(f"  post_fp[0]={float(post_fp[0].item()):.6f}, scaled[0]={float(scaled_post[0].item()):.6f}")

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

    out_fpint = fpint_gemm_qcol_real_2scomp_torch(
        inp, wt, sc, ze, groupsize=groupsize, out_dtype=torch.float16, debug=False
    )

    # naive reference: inp @ (scale*(w-zero))  (this is "math reference", not HW-prealign reference)
    D = sc.to(torch.float32) * (wt.to(torch.float32) - ze.to(torch.float32))
    out_ref = (inp.to(torch.float32) @ D).to(torch.float16)

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
    
    out_qrow = fpint_gemm_qrow_real_2scomp_torch(
        inp, wt, sc_row, ze_row, groupsize=groupsize, out_dtype=torch.float16, debug=False
    )
    
    # naive reference for qrow
    D_row = sc_row.to(torch.float32) * (wt.to(torch.float32) - ze_row.to(torch.float32))
    out_ref_row = (inp.to(torch.float32) @ D_row).to(torch.float16)
    
    diff_row = (out_qrow.to(torch.float32) - out_ref_row.to(torch.float32)).abs()
    print("QROW Results:")
    print(f"Is close? : {torch.allclose(out_qrow, out_ref_row, atol=1e-2, rtol=1e-2)}")
    print("qrow [0,:8] :", out_qrow[0, :8].detach().cpu())
    print("ref  [0,:8] :", out_ref_row[0, :8].detach().cpu())