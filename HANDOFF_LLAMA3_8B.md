# HANDOFF — Llama 3 8B W4A16KV4 FIGNA emulation accuracy 추출

> ⚠️ **이 파일을 다 읽었으면 즉시 삭제하세요** (`rm HANDOFF_LLAMA3_8B.md`).
> 일회성 인수인계 노트이며, 저장소에 남기면 안 됩니다.

## 이 세션에서 하려는 것

- 이 프로젝트는 **SpinQuant를 수정**해서 GEMM/attention 연산을 **FIGNA MXU 하드웨어 emulation** 코드로 대체하고 accuracy를 뽑는 코드다.
- **Llama 2 7B W4A16KV4는 이미 검증 완료** (learned rotation 사용).
- 이제 **Llama 3 8B W4A16KV4**에 대해 동일한 방법론으로 accuracy를 뽑아야 한다.
- 이 서버는 **GPU 사용 가능** → 실제 실행 단계를 진행한다.

## 이전 세션에서 이미 끝낸 것 (코드 변경 완료·검증됨)

1. **GQA 지원 패치** — `utils/figna_utils.py`의 `custom_fp_int_attention`
   - 문제: Llama 3 8B는 GQA (query 32 heads / KV 8 heads). custom attention 경로가 쓰는 **양자화된 int K/V 캐시**에는 `repeat_kv`가 적용된 적이 없어서, query 32 vs KV 8 head 불일치로 `IndexError` 크래시.
   - (참고: `modeling_llama.py:759-760`의 `repeat_kv`는 표준 SDPA 경로용 FP K/V에만 적용됨. custom attention은 `last_k_int`/`v_proj.last_output_int` 캐시를 따로 씀.)
   - 수정: `num_kv_heads = key_int.shape[1]`로 추론, `n_rep = num_heads // num_kv_heads`, 두 GEMM 루프에서 query head `j` → KV head `j // n_rep`로 인덱싱 (repeat_kv를 메모리 복제 없이 재현). MHA(Llama 2)는 `n_rep=1`이라 backward-compatible.
   - CPU mock GEMM으로 검증 완료: GQA 경로 == 명시적 repeat_kv reference (diff=0.0).
2. **오타 수정** — `rotation_utils.py:192` `k_quantiezr` → `k_quantizer` (사용자가 직접 수정 완료).

## 재사용성 판정 (이전 세션 분석)

| 경로 | 파일 | Llama 3 8B |
|---|---|---|
| Rotation training | `train_utils/`, `optimize_rotation.py` | ✅ 그대로 (R2는 head_dim=128 기준, GQA 무관) |
| PTQ/eval 파이프라인 | `eval_utils/`, `ptq.py` | ✅ (GQA 패치 완료) |
| FIGNA GEMM emul | `figna_utils.py` | ✅ shape-agnostic (모든 layer K차원 %32==0: 4096, 14336 등) |
| Hadamard | `hadamard_utils.py` | ✅ `get_hadK(14336)`: `14336%28==0, 512=2^9` 통과 |

- Tokenizer(`ptq.py:55` `LlamaTokenizerFast`)는 문제없을 것으로 추정(사용자 확인).
- head_dim=128, hidden=4096 두 모델 동일. vocab(128256)·intermediate(14336) 차이는 무관/처리됨.

## 실행 순서 (GPU에서 진행)

learned rotation은 **모델별로 달라 재사용 불가** → Llama 3 8B용 R.bin을 **새로 추출**해야 apples-to-apples 비교가 됨.

```bash
# 1) Llama 3 8B용 learned rotation 추출 (기본 4 GPU FSDP, max_steps=25)
sh scripts/10_optimize_rotation.sh <llama3-8b-path> 4 16 4
#    → <MODEL>_w4a16kv4_fp16_K_asym_V_sym/your_path/R.bin 생성
#    (optimize_rotation.py:139에서 output_rotation_path/R.bin 으로 저장)

# 2) FIGNA emul accuracy 추출 — 2×2 (use_custom_kernel × custom_attention)
sh scripts/4_experiment.sh <llama3-8b-path> 4 16 4
#    → 같은 BUILD_DIR의 R.bin을 --optimized_rotation_path로 자동으로 읽음
#    로그: <BUILD_DIR>/experiment_logs/{no_kernel_no_attn,kernel_only,attn_only,kernel_and_attn}.log
```

- `10_`(optimize)와 `2_/4_`(eval) 스크립트의 `BUILD_DIR` 네이밍 규칙이 동일 → R.bin 경로 자동 정합. 수정 불필요.
- `<llama3-8b-path>`는 실제 모델 경로/HF id로 치환 (예: `meta-llama/Meta-Llama-3-8B`). gated repo면 HF 토큰 필요할 수 있음.

## 실행 전 체크포인트

- **GPU 개수**: `10_optimize_rotation.sh`는 `--nproc_per_node=4`. 자원 부족하면 `scripts/11_optimize_rotation_fsdp.sh` 확인 또는 nproc 조정.
- **config 일치**: rotation 추출과 eval의 `w/a/kv/k_groupsize(128)/v_groupsize(128)/asym` 플래그가 같아야 함. 둘 다 `4 16 4`로 넘기면 자동 일치.
- **GQA 최종 확인**: 실제 실행 시 `--custom_attention` 조합(`attn_only`, `kernel_and_attn`)이 크래시 없이 도는지 = GQA 패치 실전 검증. (CPU mock은 통과했으나 실제 CUDA GEMM 경로는 첫 실행에서 확인.)

## 검증 방법

- 4개 로그의 zero-shot accuracy가 정상 범위로 나오는지 확인.
- Llama 2 7B 때의 4조합 수치와 경향 비교 (custom kernel/attention on/off 영향).
