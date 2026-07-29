#!/bin/bash
# Autonomous overnight pipeline (optimized FIGNA kernels):
#  Chain A (GPU 2): wait attn_only zeroshot -> attn_only ppl -> kernel_and_attn zeroshot -> kernel_and_attn ppl
#  Chain B (GPU 1): wait kernel_only zeroshot -> kernel_only ppl
# The two zeroshot jobs (kernel_only_opt / attn_only_opt) are ALREADY running; we wait on them.
set -u
export HF_HOME=/data/hf_cache
export HF_TOKEN=$(cat /data/hf_cache/token)
export HF_DATASETS_CACHE=/home/bosungan/hf_datasets_cache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/bosungan/SpinQuant

TR=/home/intern8/SpinQuant/venv/bin/torchrun
MODEL=meta-llama/Meta-Llama-3-8B
BD=Meta_Llama_3_8B_w4a16kv4_fp16_K_asym_V_sym
RBIN="${BD}/your_path/R.bin"
LOGS="${BD}/experiment_logs"
GLOG=/home/bosungan/SpinQuant/pipeline.log
CF="--input_model $MODEL --do_train False --do_eval True --per_device_eval_batch_size 1 --model_max_length 2048 --fp16 True --bf16 False --save_safetensors False --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym --k_groupsize 128 --v_groupsize 128 --w_groupsize 32 --rotate --optimized_rotation_path ${RBIN}"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$GLOG"; }

# wait until logfile shows zero-shot Average (done) OR an error / dead process
wait_zs(){ # $1=logfile $2=port
  while true; do
    grep -qE "Average +: [0-9.]+%" "$1" 2>/dev/null && return 0
    grep -qiE "out of memory|ChildFailedError|AssertionError|Signal 9|Traceback \(most recent" "$1" 2>/dev/null && { log "ERROR while waiting on $1"; return 1; }
    if ! ps -u bosungan -o cmd= | grep -q "master_port=$2"; then
      grep -qE "Average +: [0-9.]+%" "$1" 2>/dev/null && return 0
      log "$1 (port $2) process gone without Average"; return 1
    fi
    sleep 60
  done
}
# wait until logfile shows ppl result OR error / dead process
wait_ppl(){ # $1=logfile $2=port
  while true; do
    grep -qE "wiki2 ppl is" "$1" 2>/dev/null && return 0
    grep -qiE "out of memory|ChildFailedError|AssertionError|Signal 9|Traceback \(most recent" "$1" 2>/dev/null && { log "ERROR while waiting on $1"; return 1; }
    if ! ps -u bosungan -o cmd= | grep -q "master_port=$2"; then
      grep -qE "wiki2 ppl is" "$1" 2>/dev/null && return 0
      log "$1 (port $2) process gone without ppl"; return 1
    fi
    sleep 60
  done
}
run(){ # $1=name $2=gpu $3=port $4=flags $5=evaltype(--eval_zero_shot|--eval_ppl)
  local name=$1 gpu=$2 port=$3 flags=$4 et=$5
  local extra=""; [ "$et" = "--eval_zero_shot" ] && extra="--eval_zero_shot" || extra="--eval_ppl"
  log "launch ${name} on GPU${gpu} (${flags} ${extra})"
  CUDA_VISIBLE_DEVICES=$gpu "$TR" --master_port=$port --nnodes=1 --nproc_per_node=1 ptq.py \
    $CF $flags $extra > "${LOGS}/${name}.log" 2>&1
  log "done ${name} (exit $?)"
}

log "=================== PIPELINE START ==================="

# ---------- Chain B (GPU 1): kernel_only ----------
(
  log "chainB: waiting for kernel_only zeroshot (kernel_only_opt.log, port 29551)"
  if wait_zs "${LOGS}/kernel_only_opt.log" 29551; then
    log "chainB: kernel_only zeroshot DONE -> kernel_only ppl"
    run kernel_only_ppl 1 29561 "--use_custom_kernel" "--eval_ppl"
  else
    log "chainB: kernel_only zeroshot did not complete cleanly; skipping its ppl"
  fi
  log "chainB: finished"
) &
CHAINB=$!

# ---------- Chain A (GPU 2): attn_only -> kernel_and_attn ----------
(
  log "chainA: waiting for attn_only zeroshot (attn_only_opt.log, port 29552)"
  if wait_zs "${LOGS}/attn_only_opt.log" 29552; then
    log "chainA: attn_only zeroshot DONE -> attn_only ppl"
    run attn_only_ppl 2 29562 "--custom_attention" "--eval_ppl"
    wait_ppl "${LOGS}/attn_only_ppl.log" 29562 || log "chainA: attn_only ppl issue (continuing)"
  else
    log "chainA: attn_only zeroshot did not complete cleanly; skipping its ppl"
  fi
  # kernel_and_attn (both) on GPU 2, zeroshot then ppl
  log "chainA: launch kernel_and_attn zeroshot"
  run kernel_and_attn_opt 2 29553 "--use_custom_kernel --custom_attention" "--eval_zero_shot"
  if wait_zs "${LOGS}/kernel_and_attn_opt.log" 29553; then
    log "chainA: kernel_and_attn zeroshot DONE -> kernel_and_attn ppl"
    run kernel_and_attn_ppl 2 29554 "--use_custom_kernel --custom_attention" "--eval_ppl"
  else
    log "chainA: kernel_and_attn zeroshot did not complete cleanly; skipping its ppl"
  fi
  log "chainA: finished"
) &
CHAINA=$!

wait $CHAINB $CHAINA

log "=================== PIPELINE DONE — SUMMARY ==================="
for cfg in kernel_only_opt attn_only_opt kernel_and_attn_opt; do
  echo "===== ${cfg} (zero-shot) =====" | tee -a "$GLOG"
  grep -aoE "^  [a-z_]+ +: [0-9.]+%|Average +: [0-9.]+%" "${LOGS}/${cfg}.log" 2>/dev/null | tee -a "$GLOG"
done
for cfg in kernel_only_ppl attn_only_ppl kernel_and_attn_ppl; do
  echo "===== ${cfg} =====" | tee -a "$GLOG"
  grep -aiE "wiki2 ppl is" "${LOGS}/${cfg}.log" 2>/dev/null | tee -a "$GLOG"
done
log "=================== PIPELINE END ==================="
