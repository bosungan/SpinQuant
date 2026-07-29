# 3-GPU (0,2,3) variant of 10_optimize_rotation.sh
# Usage: sh scripts/10_optimize_rotation_g023.sh <model_path> <w_bits> <a_bits> <kv_bits>
# Uses shared HF cache/token on this machine.
# batch: per_device=11 x 3 GPUs = 33 effective; 33 x 25 steps = 825 >= 800
# (matches original 4-GPU: 8 x 4 = 32 effective, 32 x 25 = 800)

export HF_HOME=/data/hf_cache
export HF_TOKEN=$(cat /data/hf_cache/token)

MODEL_NAME=$(echo $1 | sed 's/.*\///' | sed 's/-/_/g')
DTYPE="fp16"
K_QUANT="asym"
V_QUANT="sym"
BUILD_DIR="${MODEL_NAME}_w${2}a${3}kv${4}_${DTYPE}_K_${K_QUANT}_V_${V_QUANT}"

if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing directory: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi
echo "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR/your_path"
mkdir -p "$BUILD_DIR/your_output_path"
mkdir -p "$BUILD_DIR/your_log_path"

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nnodes=1 --nproc_per_node=3 --master_port=29510 optimize_rotation.py \
--input_model $1  \
--output_rotation_path "${BUILD_DIR}/your_path" \
--output_dir "${BUILD_DIR}/your_output_path/" \
--logging_dir "${BUILD_DIR}/your_log_path/" \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--log_on_each_node False \
--per_device_train_batch_size 11 \
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
--k_groupsize 128 \
--v_groupsize 128 \
