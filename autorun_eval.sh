#!/bin/bash
# Stage-2-only: reuse existing R.bin, run 4 eval configs (kernel x attention).
# Fix vs first attempt: HF_DATASETS_CACHE points to a writable dir (the shared
# /data/hf_cache/datasets is not writable by this user -> PermissionError on lock).
set -u

export HF_HOME=/data/hf_cache
export HF_TOKEN=$(cat /data/hf_cache/token)
export HF_DATASETS_CACHE=/home/bosungan/hf_datasets_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
# Llama-3 8B (128k vocab) + full quantized model on one GPU OOMs at eval batch 4;
# expandable_segments reduces fragmentation, batch=1 keeps logits memory small.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bosungan/SpinQuant

TR=/home/intern8/SpinQuant/venv/bin/torchrun
MODEL=meta-llama/Meta-Llama-3-8B
BUILD_DIR=Meta_Llama_3_8B_w4a16kv4_fp16_K_asym_V_sym
RBIN="${BUILD_DIR}/your_path/R.bin"
GLOG=/home/bosungan/SpinQuant/autorun_eval.log
TASKS="boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$GLOG"; }

log "=================== EVAL RERUN START ==================="
if [ ! -f "$RBIN" ]; then log "FATAL: R.bin missing ($RBIN). Abort."; exit 1; fi
log "reusing R.bin ($(du -h $RBIN | cut -f1))"
mkdir -p "${BUILD_DIR}/experiment_logs"
rm -f "${BUILD_DIR}/experiment_logs/"*.log   # clear crashed logs (R.bin untouched)

run_eval(){
  local NAME=$1 GPU=$2 PORT=$3 KFLAG=$4 AFLAG=$5
  local LOG="${BUILD_DIR}/experiment_logs/${NAME}.log"
  log "eval start: ${NAME} (GPU ${GPU}, kernel='${KFLAG}', attn='${AFLAG}')"
  CUDA_VISIBLE_DEVICES=$GPU "$TR" --master_port=$PORT --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model "$MODEL" --do_train False --do_eval True \
    --per_device_eval_batch_size 1 --model_max_length 2048 --fp16 True --bf16 False \
    --save_safetensors False \
    --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 \
    --w_clip --a_asym --k_asym --k_groupsize 128 --v_groupsize 128 --w_groupsize 32 \
    --rotate --optimized_rotation_path "$RBIN" \
    --eval_zero_shot --zero_shot_tasks "$TASKS" \
    $KFLAG $AFLAG >> "$LOG" 2>&1
  log "eval DONE: ${NAME} (exit $?)"
}

# GPUs 0,1,2 free (GPU 3 taken by another user).
# GPU 0: fast baseline first, then slowest combo.
( run_eval no_kernel_no_attn 0 29531 "" "";
  run_eval kernel_and_attn   0 29534 "--use_custom_kernel" "--custom_attention" ) &
( run_eval kernel_only       1 29532 "--use_custom_kernel" "" ) &
( run_eval attn_only         2 29533 "" "--custom_attention" ) &
wait

log "=================== ALL EVAL DONE ==================="
for f in "${BUILD_DIR}/experiment_logs/"*.log; do
  echo "===== $(basename $f) =====" | tee -a "$GLOG"
  sed -n '/Zero-shot Results/,/====/p' "$f" 2>/dev/null | tee -a "$GLOG"
done
log "=================== EVAL RERUN END ==================="
