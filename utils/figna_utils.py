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
    print(f"[DEBUG] custom_fp_int_attention called!")
    print(f"  query.shape: {query.shape}, query.dtype: {query.dtype}")
    print(f"  key_int.shape: {key_int.shape}, key_int.dtype: {key_int.dtype}")
    print(f"  value_int.shape: {value_int.shape}, value_int.dtype: {value_int.dtype}")
    print(f"  scale_k.shape: {scale_k.shape}, scale_v.shape: {scale_v.shape}")
    print(f"  scale_k.dtype: {scale_k.dtype}, scale_v.dtype: {scale_v.dtype}")
    print(f"  zero_k.shape: {zero_k.shape}, zero_v.shape: {zero_v.shape}")
    print(f"  zero_k.dtype: {zero_k.dtype}, zero_v.dtype: {zero_v.dtype}")
    print(f"  attn_mask: {attn_mask.shape if attn_mask is not None else None}")
    print(f"  is_causal: {is_causal}, dropout_p: {dropout_p}")
    
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    # === Step 1: Dequantize K and V ===
    # K_fp16 = scale_k * (K_int - zero_k)
    
    # fall back dequantization
    # key_fp = scale_k * (key_int.to(scale_k.dtype) - zero_k.to(scale_k.dtype))
    value_fp = scale_v * (value_int.to(scale_v.dtype) - zero_v.to(scale_v.dtype))
    
    # print(f"  [After dequant] key_fp.shape: {key_fp.shape}, key_fp.dtype: {key_fp.dtype}")
    print(f"  [After dequant] value_fp.shape: {value_fp.shape}, value_fp.dtype: {value_fp.dtype}")
    
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
    print(f"  [After QK^T] attn_weights.shape: {attn_weights.shape}")
    
    # === Step 3: Apply mask ===
    if is_causal and attn_mask is None:
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=query.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        print(f"  [Applied causal mask]")
    elif attn_mask is not None:
        attn_weights = attn_weights + attn_mask
        print(f"  [Applied attention mask]")
    
    # === Step 4: Softmax ===
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    
    # === Step 5: Dropout (if training) ===
    if dropout_p > 0.0:
        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=True)
        print(f"  [Applied dropout with p={dropout_p}]")
    
    # === Step 6: P @ V ===
    # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
    # = [batch, num_heads, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, value_fp)
    
    print(f"  [After PV] attn_output.shape: {attn_output.shape}, attn_output.dtype: {attn_output.dtype}")
    print(f"[DEBUG] custom_fp_int_attention completed!")
    
    return attn_output.to(query.dtype)



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
def fpint_gemm_qcol_real_2scomp_torch(
    input_data: torch.Tensor,   # Shape: (M, K), dtype: float16
    weight_data: torch.Tensor,  # Shape: (K, N), dtype: int8
    scale_data: torch.Tensor,   # Shape: (KG, N), dtype: float16
    zero_data: torch.Tensor,    # Shape: (KG, N), dtype: int16 or float16
    groupsize: int,             # Quantization block size in K direction
    debug: bool = False
) -> torch.Tensor:
    """
    FPINT GEMM with column-wise quantization (qcol) using 2's complement encoding - PyTorch version
    
    Args:
        input_data: FP16 input activations, shape (M, K)
        weight_data: Quantized weights (int8), shape (K, N)
        scale_data: FP16 scale factors, shape (K//groupsize, N)
        zero_data: Zero points (int16 or fp16), shape (K//groupsize, N)
        groupsize: Quantization block size in K direction
        debug: Enable debug printing
        
    Returns:
        output_data: FP16 output, shape (M, N)
    """
    M, K = input_data.shape
    K_w, N = weight_data.shape
    assert K == K_w, f"K mismatch: input {K} vs weight {K_w}"
    
    device = input_data.device
    dtype = input_data.dtype
    
    # KG = number of quantization blocks in K direction
    KG = scale_data.shape[0]
    
    if debug:
        print(f"[FPINT_EMUL.QCOL_REAL_2SCOMP_TORCH] M={M}, N={N}, K={K}, KG={KG}, groupsize={groupsize}")
        print(f"  input: {input_data.shape} {input_data.dtype}")
        print(f"  weight: {weight_data.shape} {weight_data.dtype}")
        print(f"  scale: {scale_data.shape} {scale_data.dtype}")
        print(f"  zero: {zero_data.shape} {zero_data.dtype}")
    
    # Ensure weight_data is int8
    if weight_data.dtype != torch.int8:
        weight_data = weight_data.to(torch.int8)
    
    # Ensure zero_data is int16 for proper handling
    if zero_data.dtype == torch.float16:
        zero_data = zero_data.to(torch.int16)
    
    # Convert to float for computation
    weight_fp = weight_data.float()
    scale_fp = scale_data.float()
    zero_fp = zero_data.float()
    
    # Expand scale and zero to match K dimension
    # scale: (KG, N) -> (K, N) by repeating each block groupsize times
    scale_expanded = scale_fp.repeat_interleave(groupsize, dim=0)[:K, :]  # (K, N)
    zero_expanded = zero_fp.repeat_interleave(groupsize, dim=0)[:K, :]    # (K, N)
    
    if debug:
        print(f"  scale_expanded: {scale_expanded.shape}")
        print(f"  zero_expanded: {zero_expanded.shape}")
    
    # Dequantize weights: W_fp = scale * (W_int - zero)
    weight_dequant = scale_expanded * (weight_fp - zero_expanded)  # (K, N)
    
    # Matrix multiplication: output = input @ weight_dequant
    output_data = torch.matmul(input_data, weight_dequant)  # (M, K) @ (K, N) = (M, N)
    
    if debug:
        print(f"[FPINT_EMUL.QCOL_REAL_2SCOMP_TORCH] Output computed, shape: {output_data.shape}")
        if M <= 2 and N <= 4:
            print(f"  Output values:\n{output_data}")
    
    return output_data.to(dtype)



def fpint_gemm_qrow_real_2scomp_torch(
    input_data: torch.Tensor,   # Shape: (M, K), dtype: float16
    weight_data: torch.Tensor,  # Shape: (K, N), dtype: int8
    scale_data: torch.Tensor,   # Shape: (K, NG), dtype: float16
    zero_data: torch.Tensor,    # Shape: (K, NG), dtype: int16 or float16
    groupsize: int,             # Quantization block size in N direction
    debug: bool = False
) -> torch.Tensor:
    """
    FPINT GEMM with row-wise quantization (qrow) using real 2's complement encoding - PyTorch version

    This version expands scale/zero to match (K, N) shape like qcol version.

    Args:
        input_data: FP16 input activations, shape (M, K)
        weight_data: Quantized weights (signed int8), shape (K, N)
        scale_data: FP16 scale factors, shape (K, N//groupsize)
        zero_data: Zero points (int16 or fp16), shape (K, N//groupsize)
        groupsize: Quantization block size in N direction
        debug: Enable debug printing

    Returns:
        output_data: FP16 output, shape (M, N)
    """
    M, K = input_data.shape
    K_w, N = weight_data.shape
    assert K == K_w, f"K mismatch: input {K} vs weight {K_w}"
    
    device = input_data.device
    dtype = input_data.dtype
    
    # NG = number of quantization blocks in N direction
    NG = scale_data.shape[1]
    
    if debug:
        print(f"[FPINT_EMUL.QROW_REAL_2SCOMP_TORCH] M={M}, N={N}, K={K}, NG={NG}, groupsize={groupsize}")
        print(f"  input: {input_data.shape} {input_data.dtype}")
        print(f"  weight: {weight_data.shape} {weight_data.dtype}")
        print(f"  scale: {scale_data.shape} {scale_data.dtype}")
        print(f"  zero: {zero_data.shape} {zero_data.dtype}")
    
    # Ensure weight_data is int8
    if weight_data.dtype != torch.int8:
        weight_data = weight_data.to(torch.int8)
    
    # Ensure zero_data is int16 for proper handling
    if zero_data.dtype == torch.float16:
        zero_data = zero_data.to(torch.int16)
    
    # Convert to float for computation
    weight_fp = weight_data.float()
    scale_fp = scale_data.float()
    zero_fp = zero_data.float()
    
    # Expand scale and zero to match N dimension
    # scale: (K, NG) -> (K, N) by repeating each block groupsize times along dim=1
    scale_expanded = scale_fp.repeat_interleave(groupsize, dim=1)[:, :N]  # (K, N)
    zero_expanded = zero_fp.repeat_interleave(groupsize, dim=1)[:, :N]    # (K, N)
    
    if debug:
        print(f"  scale_expanded: {scale_expanded.shape}")
        print(f"  zero_expanded: {zero_expanded.shape}")
        if K <= 4 and N <= 4:
            print(f"  scale_expanded:\n{scale_expanded}")
            print(f"  zero_expanded:\n{zero_expanded}")
    
    # For qrow: output[m, n] = sum_k( input[m, k] * scale[k, n] * (weight[k, n] - zero[k, n]) )
    # Rearrange: output = (input * scale_expanded.T).T @ (weight - zero_expanded)
    # Or equivalently: output[m, n] = sum_k( (input[m, k] * scale[k, n]) * (weight[k, n] - zero[k, n]) )
    
    # Method 1: Expand input to (M, K, N), broadcast multiplication
    # input_expanded: (M, K, 1) -> (M, K, N)
    # scale_expanded: (K, N) -> (1, K, N)
    # weight_dequant: (K, N) -> (1, K, N)
    
    input_expanded = input_data.unsqueeze(2)  # (M, K, 1)
    scale_expanded_3d = scale_expanded.unsqueeze(0)  # (1, K, N)
    
    # Scaled input: (M, K, N)
    scaled_input = input_expanded * scale_expanded_3d  # (M, K, N)
    
    # Dequantized weight: (K, N) -> (1, K, N)
    weight_dequant = (weight_fp - zero_expanded).unsqueeze(0)  # (1, K, N)
    
    # Element-wise multiply and sum over K: (M, K, N) * (1, K, N) -> (M, N)
    output_data = (scaled_input * weight_dequant).sum(dim=1)  # (M, N)
    
    if debug:
        print(f"[FPINT_EMUL.QROW_REAL_2SCOMP_TORCH] Output computed, shape: {output_data.shape}")
        if M <= 2 and N <= 4:
            print(f"  Output values:\n{output_data}")
    
    return output_data.to(dtype)

