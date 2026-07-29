#!/bin/bash
# Autonomous overnight run: Meta-Llama-3-8B W4A16KV4 FIGNA emulation accuracy.
# Stage 1: learned rotation (R.bin) on GPUs 0,2,3.
# Stage 2: 4 eval configs (kernel x attention) distributed across GPUs 0,2,3.
set -u

export HF_HOME=/data/hf_cache
export HF_TOKEN=$(cat /data/hf_cache/token)
export TOKENIZERS_PARALLELISM=false
cd /home/bosungan/SpinQuant

TR=/home/intern8/SpinQuant/venv/bin/torchrun
MODEL=meta-llama/Meta-Llama-3-8B
BUILD_DIR=Meta_Llama_3_8B_w4a16kv4_fp16_K_asym_V_sym
GLOG=/home/bosungan/SpinQuant/autorun.log

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$GLOG"; }

log "=================== AUTORUN START ==================="
log "env: $TR"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee -a "$GLOG"

fresh_dirs(){ rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR/your_path" "$BUILD_DIR/your_output_path" "$BUILD_DIR/your_log_path" "$BUILD_DIR/experiment_logs"; }

########## STAGE 1: R.bin ##########
run_rotation(){
  local BS=$1 GA=$2
  log "STAGE1 optimize_rotation: per_device=$BS grad_accum=$GA (eff=$((BS*3*GA)), steps=25 -> $((BS*3*GA*25)))"
  CUDA_VISIBLE_DEVICES=0,2,3 "$TR" --nnodes=1 --nproc_per_node=3 --master_port=29510 optimize_rotation.py \
    --input_model "$MODEL" \
    --output_rotation_path "${BUILD_DIR}/your_path" \
    --output_dir "${BUILD_DIR}/your_output_path/" \
    --logging_dir "${BUILD_DIR}/your_log_path/" \
    --model_max_length 2048 --fp16 True --bf16 False --log_on_each_node False \
    --per_device_train_batch_size $BS --gradient_accumulation_steps $GA \
    --logging_steps 1 --learning_rate 1.5 --weight_decay 0. --lr_scheduler_type "cosine" \
    --gradient_checkpointing True --save_safetensors False --max_steps 25 \
    --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym \
    --k_groupsize 128 --v_groupsize 128 >> "${BUILD_DIR}/rotation.log" 2>&1
}

fresh_dirs
run_rotation 11 1
if [ ! -f "${BUILD_DIR}/your_path/R.bin" ]; then
  log "STAGE1 R.bin missing after per_device=11 (likely OOM). Retrying safe config."
  sleep 20
  fresh_dirs
  run_rotation 4 3
fi
if [ ! -f "${BUILD_DIR}/your_path/R.bin" ]; then
  log "STAGE1 FAILED: R.bin still missing. See ${BUILD_DIR}/rotation.log . ABORT."
  exit 1
fi
log "STAGE1 DONE: R.bin created ($(du -h ${BUILD_DIR}/your_path/R.bin | cut -f1))"

########## STAGE 2: eval (4 configs across GPUs 0,2,3) ##########
run_eval(){
  local NAME=$1 GPU=$2 PORT=$3 KFLAG=$4 AFLAG=$5
  local LOG="${BUILD_DIR}/experiment_logs/${NAME}.log"
  log "STAGE2 eval start: ${NAME} (GPU ${GPU}, kernel='${KFLAG}', attn='${AFLAG}')"
  CUDA_VISIBLE_DEVICES=$GPU "$TR" --master_port=$PORT --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model "$MODEL" --do_train False --do_eval True \
    --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 True --bf16 False \
    --save_safetensors False \
    --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 \
    --w_clip --a_asym --k_asym --k_groupsize 128 --v_groupsize 128 --w_groupsize 32 \
    --rotate --optimized_rotation_path "${BUILD_DIR}/your_path/R.bin" \
    --eval_zero_shot \
    --zero_shot_tasks "boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
    $KFLAG $AFLAG >> "$LOG" 2>&1
  log "STAGE2 eval DONE: ${NAME} (exit $?)"
}

# GPU 0: baseline first (fast, guarantees a result) then the slowest combo.
( run_eval no_kernel_no_attn 0 29521 "" "";
  run_eval kernel_and_attn   0 29524 "--use_custom_kernel" "--custom_attention" ) &
# GPU 2: kernel only
( run_eval kernel_only 2 29522 "--use_custom_kernel" "" ) &
# GPU 3: attention only
( run_eval attn_only   3 29523 "" "--custom_attention" ) &
wait

log "=================== ALL STAGES DONE ==================="
log "Accuracy summary (grep from logs):"
grep -riE "acc(,| |_norm|:)" "${BUILD_DIR}/experiment_logs/"*.log 2>/dev/null | tail -80 | tee -a "$GLOG"
log "=================== AUTORUN END ==================="
