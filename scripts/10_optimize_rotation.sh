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
BUILD_DIR="${MODEL_NAME}_w${2}a${3}kv${4}"

# Remove existing directory if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "=================================================="
    echo "Removing existing directory: $BUILD_DIR"
    echo "=================================================="
    rm -rf "$BUILD_DIR"
fi

# Create directory structure
echo "=================================================="
echo "Creating build directory: $BUILD_DIR"
echo "=================================================="
mkdir -p "$BUILD_DIR/your_path"
mkdir -p "$BUILD_DIR/your_output_path"
mkdir -p "$BUILD_DIR/your_log_path"

echo "Directory structure created:"
echo "  - $BUILD_DIR/your_path        (for rotation matrices)"
echo "  - $BUILD_DIR/your_output_path (for checkpoints)"
echo "  - $BUILD_DIR/your_log_path    (for training logs)"
echo "=================================================="
echo ""

torchrun --nnodes=1 --nproc_per_node=4 optimize_rotation.py \
--input_model $1  \
--output_rotation_path "${BUILD_DIR}/your_path" \
--output_dir "${BUILD_DIR}/your_output_path/" \
--logging_dir "${BUILD_DIR}/your_log_path/" \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--log_on_each_node False \
--per_device_train_batch_size 8 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 25 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
