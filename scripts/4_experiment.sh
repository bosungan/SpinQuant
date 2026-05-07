# Usage: sh scripts/4_experiment.sh <model_path> <w_bits> <a_bits> <kv_bits>
# Runs 2x2 zero-shot experiments: --use_custom_kernel x --custom_attention

MODEL=$1
W_BITS=$2
A_BITS=$3
KV_BITS=$4

MODEL_NAME=$(echo $MODEL | sed 's/.*\///' | sed 's/-/_/g')
BUILD_DIR="${MODEL_NAME}_w${W_BITS}a${A_BITS}kv${KV_BITS}_fp16_K_asym_V_sym"
LOG_DIR="${BUILD_DIR}/experiment_logs"
mkdir -p "${LOG_DIR}"

run_exp() {
    local NAME=$1
    local USE_CUSTOM_KERNEL=$2   # "--use_custom_kernel" or ""
    local CUSTOM_ATTENTION=$3    # "--custom_attention"   or ""

    local LOG="${LOG_DIR}/${NAME}.log"

    echo ""
    echo "======================================================"
    echo " Experiment : ${NAME}"
    echo " use_custom_kernel : ${USE_CUSTOM_KERNEL:-off}"
    echo " custom_attention  : ${CUSTOM_ATTENTION:-off}"
    echo " Log               : ${LOG}"
    echo "======================================================"

    mkdir -p "${BUILD_DIR}/${NAME}"

    CUDA_VISIBLE_DEVICES=2 torchrun \
        --master_port=29503 \
        --nnodes=1 \
        --nproc_per_node=1 \
        ptq.py \
        --input_model $MODEL \
        --do_train False \
        --do_eval True \
        --per_device_eval_batch_size 4 \
        --model_max_length 2048 \
        --fp16 True \
        --bf16 False \
        --save_safetensors False \
        --w_bits $W_BITS \
        --a_bits $A_BITS \
        --k_bits $KV_BITS \
        --v_bits $KV_BITS \
        --w_clip \
        --a_asym \
        --k_asym \
        --k_groupsize 128 \
        --v_groupsize 128 \
        --w_groupsize 32 \
        --rotate \
        --optimized_rotation_path "${BUILD_DIR}/your_path/R.bin" \
        --save_qmodel_path "${BUILD_DIR}/${NAME}/consolidated.00.pth" \
        --eval_zero_shot \
        --zero_shot_tasks "boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
        $USE_CUSTOM_KERNEL \
        $CUSTOM_ATTENTION \
        2>&1 | tee "${LOG}"

    echo " Done: ${NAME}"
}

# 2 x 2 experiments
run_exp "no_kernel_no_attn"    ""                     ""
run_exp "kernel_only"          "--use_custom_kernel"  ""
run_exp "attn_only"            ""                     "--custom_attention"
run_exp "kernel_and_attn"      "--use_custom_kernel"  "--custom_attention"

echo ""
echo "======================================================"
echo " All 4 experiments finished."
echo " Logs: ${LOG_DIR}/"
echo "======================================================"
