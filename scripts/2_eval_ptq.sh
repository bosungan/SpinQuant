# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# Parse model name from input path (e.g., meta-llama/Llama-2-7b-hf -> Llama_2_7b_hf)
MODEL_NAME=$(echo $1 | sed 's/.*\///' | sed 's/-/_/g')

# Determine dtype (fp16 by default in this script)
DTYPE="fp16"

# Determine K quantization method (asym flag is set, so asym)
K_QUANT="asym"

# Determine V quantization method (asym flag is set, so asym)
V_QUANT="sym"

# Build directory name: {model}_W{w}A{a}KV{kv}_{dtype}_K{k_quant}_V{v_quant}
BUILD_DIR="${MODEL_NAME}_w${2}a${3}kv${4}_${DTYPE}_K_${K_QUANT}_V_${V_QUANT}"
mkdir -p "${BUILD_DIR}/your_output_model_path"

CUDA_VISIBLE_DEVICES=2 torchrun --master_port=29502 --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 32 \
--rotate \
--optimized_rotation_path "${BUILD_DIR}/your_path/R.bin" \
--save_qmodel_path "${BUILD_DIR}/your_output_model_path/consolidated.00.pth"