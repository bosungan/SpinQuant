#!/usr/bin/env bash
# Llama 3.1 8B W4A16KV4: learned rotation, then baseline and full FIGNA.
# Usage: bash scripts/run_llama31_cached.sh | tee /tmp/llama31-launch.log
# To skip training, set ROTATION_CHECKPOINT to this model's existing R.bin.
# To run only the remaining configuration, set EVAL_CONFIGS=kernel_and_attn.
# To retry only PPL, also set EVAL_METRICS=ppl.
# Each invocation creates a new run; shared model files are only read.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"
PYTHON=/home/jaeyongjang/.conda/envs/spinquant/bin/python
MODEL=/data/hf_cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
ROTATION_GPUS=${ROTATION_GPUS:-0,1,2,3}
EVAL_GPU=${EVAL_GPU:-0}
TRAIN_BATCH=${TRAIN_BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
ROTATION_CHECKPOINT=${ROTATION_CHECKPOINT:-}
read -ra CONFIGS <<< "${EVAL_CONFIGS:-no_kernel_no_attn kernel_and_attn}"
for CONFIG in "${CONFIGS[@]}"; do
    case "$CONFIG" in
        no_kernel_no_attn|kernel_and_attn) ;;
        *) printf 'Unsupported evaluation configuration: %s\n' "$CONFIG"; exit 1 ;;
    esac
done
read -ra METRICS <<< "${EVAL_METRICS:-zero_shot ppl}"
for METRIC in "${METRICS[@]}"; do
    case "$METRIC" in
        zero_shot|ppl) ;;
        *) printf 'Unsupported evaluation metric: %s\n' "$METRIC"; exit 1 ;;
    esac
done
IFS=, read -ra GPU_IDS <<< "$ROTATION_GPUS"
NPROC=${#GPU_IDS[@]}

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
BUILD_DIR="$REPO_DIR/Llama_3.1_8B_w4a16kv4_fp16_K_asym_V_sym/$RUN_ID"
mkdir -p "$(dirname -- "$BUILD_DIR")"
mkdir "$BUILD_DIR"
LOG_DIR=$(mktemp -d /tmp/spinquant-llama31.XXXXXXXX)
printf '%s\n' "$LOG_DIR" | tee "$BUILD_DIR/log_directory.txt"
printf 'Results: %s\nLogs: %s\n' "$BUILD_DIR" "$LOG_DIR"
trap 'rc=$?; printf "FAILED (exit %s), line %s\n" "$rc" "$LINENO" | tee "$BUILD_DIR/status.txt"; exit "$rc"' ERR

export PATH="$(dirname -- "$PYTHON"):$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Dataset downloads and locks must not write into the shared model cache.
export HF_HOME="$REPO_DIR/Llama_3.1_8B_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export WANDB_DISABLED=true
# Compatibility packages are local to this experiment, not the shared conda env.
if [[ -d "$HF_HOME/eval_compat" ]]; then
    export PYTHONPATH="$HF_HOME/eval_compat${PYTHONPATH:+:$PYTHONPATH}"
fi

printf 'preflight\n' | tee "$BUILD_DIR/status.txt"
"$PYTHON" - "$MODEL" <<'PY' | tee "$LOG_DIR/environment.log"
import importlib.metadata
import json
import pathlib
import sys
import torch
import transformers
import accelerate
import datasets
import fast_hadamard_transform
import lm_eval
import tensorboardX

if int(datasets.__version__.split('.')[0]) >= 4:
    raise RuntimeError(
        'The installed datasets version cannot load the legacy PIQA/task scripts. '
        'Resolve evaluation dependencies before starting; requirement.txt specifies datasets==2.20.0.'
    )

model_path = pathlib.Path(sys.argv[1])
print('python:', sys.executable)
print('model:', model_path)
for package in ('torch', 'transformers', 'accelerate', 'datasets', 'lm-eval', 'fast-hadamard-transform'):
    print(package, importlib.metadata.version(package))
assert torch.cuda.is_available(), 'CUDA unavailable in spinquant environment'
print('CUDA devices:', torch.cuda.device_count())
index = json.loads((model_path / 'model.safetensors.index.json').read_text())
for name in sorted(set(index['weight_map'].values())):
    shard = model_path / name
    assert shard.is_file() and shard.stat().st_size > 0, f'Missing shard: {shard}'
    print('shard:', name, shard.stat().st_size)
config = transformers.AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
assert config.rope_scaling['rope_type'] in ROPE_INIT_FUNCTIONS
print('RoPE:', config.rope_scaling)
print('tokenizer length:', len(tokenizer))
# Load every evaluation dataset before spending time on training or GPTQ.
from lm_eval.tasks import TaskManager, get_task_dict
manager = TaskManager()
failed = []
for name in ('boolq', 'piqa', 'social_iqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa'):
    print('Checking task:', name, flush=True)
    try:
        task = get_task_dict([name], task_manager=manager)[name]
        docs = task.eval_docs
        assert len(docs), f'No evaluation samples for {name}'
        print('Task ready:', name, 'samples:', len(docs), flush=True)
    except Exception as exc:
        print('Task failed:', name, type(exc).__name__, str(exc), flush=True)
        failed.append(name)
if failed:
    raise RuntimeError(f'Evaluation dataset preflight failed: {failed}')
print('All 8 evaluation tasks ready.', flush=True)
PY
nvidia-smi | tee "$LOG_DIR/gpu.log"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    printf 'preflight complete\n' | tee "$BUILD_DIR/status.txt"
    exit 0
fi
printf 'model=%s\npython=%s\nrotation_gpus=%s\ntrain_batch=%s\ngrad_accum=%s\neffective_batch=%s\neval_gpu=%s\n' \
    "$MODEL" "$PYTHON" "$ROTATION_GPUS" "$TRAIN_BATCH" "$GRAD_ACCUM" \
    "$((NPROC * TRAIN_BATCH * GRAD_ACCUM))" "$EVAL_GPU" | tee "$BUILD_DIR/run_config.txt"

# Match scripts/10_optimize_rotation.sh defaults: 4 x 8, 25 steps, LR 1.5.
# Training retains its original weight groupsize default; PTQ uses groupsize 32.
if [[ -z "$ROTATION_CHECKPOINT" ]]; then
printf 'rotation\n' | tee "$BUILD_DIR/status.txt"
CUDA_VISIBLE_DEVICES="$ROTATION_GPUS" "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    --log-dir "$LOG_DIR/rotation_workers" --tee 3 optimize_rotation.py \
    --input_model "$MODEL" \
    --output_rotation_path "$BUILD_DIR/rotation" \
    --output_dir "$BUILD_DIR/training" --logging_dir "$BUILD_DIR/training_logs" \
    --model_max_length 2048 --fp16 True --bf16 False --log_on_each_node False \
    --per_device_train_batch_size "$TRAIN_BATCH" --gradient_accumulation_steps "$GRAD_ACCUM" \
    --logging_steps 1 --learning_rate 1.5 --weight_decay 0. --lr_scheduler_type cosine \
    --gradient_checkpointing True --save_safetensors False --max_steps 25 \
    --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 --w_clip --a_asym --k_asym \
    --k_groupsize 128 --v_groupsize 128 | tee "$LOG_DIR/rotation.log"
    ROTATION_CHECKPOINT="$BUILD_DIR/rotation/R.bin"
fi
test -s "$ROTATION_CHECKPOINT"
printf 'rotation_checkpoint=%s\n' "$ROTATION_CHECKPOINT" | tee -a "$BUILD_DIR/run_config.txt"
printf 'eval_configs=%s\neval_metrics=%s\n' "${CONFIGS[*]}" "${METRICS[*]}" \
    | tee -a "$BUILD_DIR/run_config.txt"

COMMON=(
    --input_model "$MODEL" --do_train False --do_eval True
    --output_dir "$BUILD_DIR/evaluation"
    --per_device_eval_batch_size 1 --model_max_length 2048 --fp16 True --bf16 False
    --save_safetensors False --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4
    --w_clip --a_asym --k_asym --k_groupsize 128 --v_groupsize 128 --w_groupsize 32
    --rotate --optimized_rotation_path "$ROTATION_CHECKPOINT"
)
TASKS=boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
for CONFIG in "${CONFIGS[@]}"; do
    FLAGS=()
    if [[ "$CONFIG" == kernel_and_attn ]]; then
        FLAGS=(--use_custom_kernel --custom_attention)
    fi
    # Separate processes: the layerwise PPL evaluator moves layers back to CPU.
    for METRIC in "${METRICS[@]}"; do
        printf '%s %s\n' "$CONFIG" "$METRIC" | tee "$BUILD_DIR/status.txt"
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m torch.distributed.run \
            --standalone --nnodes=1 --nproc_per_node=1 \
            --log-dir "$LOG_DIR/${CONFIG}_${METRIC}_workers" --tee 3 ptq.py \
            "${COMMON[@]}" "${FLAGS[@]}" "--eval_$METRIC" --zero_shot_tasks "$TASKS" \
            | tee "$LOG_DIR/${CONFIG}_${METRIC}.log"
        if [[ "$METRIC" == zero_shot ]]; then
            grep -E 'Average.*\(8 tasks\)' "$LOG_DIR/${CONFIG}_${METRIC}.log" \
                | tee -a "$BUILD_DIR/summary.txt"
        else
            # ptq.py writes PPL through logging to stderr, saved by torchrun.
            grep -m 1 'wiki2 ppl is:' "$LOG_DIR/${CONFIG}_${METRIC}_workers"/*/attempt_0/0/stderr.log \
                | tee -a "$BUILD_DIR/summary.txt"
        fi
    done
done
printf 'complete\n' | tee "$BUILD_DIR/status.txt"
