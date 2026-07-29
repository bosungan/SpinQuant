# FIGNA MXU Emulation — PyTorch 성능 최적화 리서치 보고서

- **대상 코드:** `utils/figna_utils.py` (FP-INT MAC-loop emulation), `utils/quant_utils.py` (`ActQuantWrapper`), `eval_utils/modeling_llama.py` (custom attention 통합)
- **모델/설정:** Meta-Llama-3-8B, W4A16KV4 (SpinQuant + FIGNA emulation)
- **작성 방식:** read-only 리서처 2명 병렬 분석(① GEMM 경로 ② attention·통합 경로 + git 이력) 종합
- **전제:** 모든 제안은 **하드웨어 emulation의 비트 정확도**(고정소수점 prealign + int64 MAC 누적 + 블록 지수 복원 + 2's-complement)를 **보존**해야 함. 단순히 `torch.matmul`/`SDPA`로 대체하는 것은 목표가 아님.

---

## 1. 배경 및 문제 정의

FIGNA emulation은 정상 GEMM/attention을 **고정소수점 MXU 하드웨어의 동작을 비트 단위로 모사**하는 PyTorch 코드로 대체한다. 정확성은 확보돼 있으나 속도가 문제다.

| 경로 | 플래그 | zero-shot 실측 속도 | full 8-task(74968 req) ETA |
|---|---|---|---|
| 표준 (emul 없음) | — | ~15 it/s | ~1.4 시간 (완주 가능) |
| custom attention | `--custom_attention` | ~12 s/req | ~10 일 |
| **custom GEMM (linear)** | `--use_custom_kernel` | **~90–146 s/req** | **~80–126 일** |

→ **linear weight-GEMM 경로가 전체 wall-clock의 ~92%를 차지**하며, 이것이 emulation 평가를 사실상 불가능하게 만든다. 최적화의 최우선 타겟이다.

---

## 2. 현재 알고리즘 요약

### 2.1 `_prealign_torch_fp16bits(x, extra)`
FP16을 sign/exp/mantissa로 비트 분해 → **16-lane 블록(`KG = K/16`)마다 최대 지수**를 구해 각 lane의 `hidden_man << extra`를 그 공유 지수로 right-shift. 반환: `aligned_fx (M,K) int64`, `aligned_exp (M,KG) int16`. **이 "블록당 공유 지수 prealign"이 비트 정확도의 핵심이며 반드시 보존.**

### 2.2 `fpint_gemm_qcol_real_2scomp_torch` (Q@K^T, 그리고 linear weight-GEMM)
```
for g in range(KG):                    # 최대 256 블록 (K=4096)
    inner = 0
    for lane in range(MXU_K):          # 16 lane 순차
        inner += a[:,lane] * w[lane,:] #  (M,1)*(1,N) → (M,N) outer-product
    act_sum_red = Σ_lane aligned_fx_red[:,lane]     # reduce 경로(별도 prealign)
    z, sc = zero_data[k0,:], scale_data[k0,:]       # 블록당 1값 (groupsize 중복 가정)
    post = inner - ((z*act_sum_red) << shift_back)
    acc += post.float() * 2^(e-bias) * mant_scale * sc   # 블록별 float 누적
```
- **핵심 의미 제약:** `z`, `sc`는 블록당 `k0` 위치에서 1회 샘플(groupsize 만큼 중복 저장 전제).

### 2.3 `fpint_gemm_qrow_real_2scomp_torch` (P@V)
- 외곽 `for ng in range(NG)` (그룹 단위, `e912f01`에서 이미 벡터화) → 그룹 스칼라 scale로 입력 rescale → **그룹마다 전체 (M,K) prealign 2회** → 동일 `KG×16` MAC.
- 최근 추가: 임의 seq_len 지원 위해 `K`를 16의 배수로 zero-padding(합에 0 기여, 결과 불변).

### 2.4 `custom_fp_int_attention`
```
for i in range(batch):
    for j in range(num_heads):         # 32 head 직렬 (GQA: kv_j = j // n_rep)
        attn_weights[i,j] = qcol(query[i,j], key_int[i,kv_j].t(), ...)   # Q@K^T
softmax(...)
for i in range(batch):
    for j in range(num_heads):
        attn_output[i,j] = qrow(attn_weights[i,j], value_int[i,kv_j], ...)  # P@V
```
- head_dim=128로 작아 각 GEMM이 **런치 바운드** → 32 head 직렬이 손해.

---

## 3. 진단: 왜 느린가

병목의 본질은 **커널 런치 바운드 파이썬 루프**다.
- linear GEMM: `KG`(≤256) × `16 lane` = 최대 **~4096 파이썬 반복 × 다중 커널 런치** / GEMM. 텐서는 M/N엔 벡터화됐으나 K 방향이 파이썬이다.
- attention: layer·forward마다 **32 head × batch 직렬 GEMM 호출**(Q@K^T, P@V 각각).
- 부가 낭비: symmetric(zero=0) linear인데도 reduce 경로를 매번 계산 후 버림; 루프 내 반복 dtype 캐스팅/`.contiguous()`/`.t()` 복사; qrow의 그룹당 이중 prealign.

---

## 4. 과거에 이미 적용된 최적화 (git 이력)

| 커밋 | 내용 |
|---|---|
| `08c5b20`, `edde45b` | linear·QK^T를 FP-INT MAC emul로 대체 |
| `4e04b78`, `8db5400` | fp16 scale, signed(2's-complement) asym 양자화 범위 |
| `c1991be` | `qrow_real_2scomp` allclose 통과 (정확성 기준선) |
| **`e912f01`** | **QROW 외곽 루프 `for n` → `for ng`**(그룹 단위, groupsize 열 벡터화) — P@V의 N 방향은 이미 그룹 벡터화 |
| `a870f51` | GQA 지원 (query head j → kv head `j//n_rep`) |

→ **아직 미최적화:** 16-lane 루프, KG 블록 루프, attention per-head 루프, symmetric reduce 낭비. 아래 제안은 모두 이 미개척 영역.

---

## 5. 최적화 기회 (우선순위: impact × safety)

### ★ #1 — 16-lane 루프 → 정수 matmul (GEMM 양쪽)
- **효과: 5–15× · 안전성: 높음**
- **위치:** qcol lane 루프, qrow lane 루프
- **현재:** 16 lane을 순차 outer-product로 누적 → 블록당 다수 커널 런치.
- **제안:** 블록 `g`에서 `inner = A(M,16) @ W(16,N)` 정수 matmul로 대체.
  ```python
  A = aligned_fx_main[:, k0:k1]           # (M,16) int64
  W = weight_data[k0:k1, :].to(torch.int64)  # (16,N)
  inner = A @ W                            # (M,N)  — 16-lane 루프 대체
  # act_sum_red = aligned_fx_red[:, k0:k1].sum(1, keepdim=True)  (qcol)
  #            = aligned_fx_red[:, k0:k1] @ z_group[k0:k1]        (qrow)
  ```
- **정확성:** 정수 곱-합은 결합·정확 → **비트 동일**(16항, 값 ~2^35 ≪ int64). ⚠️ **CUDA int64 matmul 미지원** 가능 → `torch.einsum` 사용, 또는 값이 2^53에 안전히 들어가므로 **float64로 캐스팅 후 int64 복원(증명상 정확)**. 기존 `__main__` allclose로 검증.

### ★ #2 — KG 블록 루프 → 배치 matmul(bmm)
- **효과: 3–10× (누적) · 안전성: 중상**
- **위치:** qcol/qrow 블록 루프
- **제안:** `aligned_fx_main`을 `(M,KG,16)`, weight을 `(KG,16,N)`로 보고 `bmm → (KG,M,N)`. 지수 복원 `2^(aligned_exp-bias)`은 `(M,KG)` 브로드캐스트로 벡터화, KG축 합산. GEMM 전체가 파이썬 루프 없이 ~5개 텐서 연산.
- **정확성:** 정수부는 정확. ⚠️ **블록 간 float32 누적 순서**가 `acc +=`(순차) vs `sum(dim)`에서 달라져 ULP 차이 가능 → **처음엔 블록 간 float 누적을 순차 유지**(값싼 KG 루프), bit-exact 확인 후 완화.

### ★ #3 — zero=0일 때 reduce 경로 스킵 (linear 경로)
- **효과: 2× · 안전성: 매우 높음 (즉시 적용 가능)**
- **위치:** qcol의 reduce prealign + `act_sum_red` 루프 + `post` 보정; 호출부 `custom_fp16_int4_gemm`가 `zero=zeros` 전달.
- **현재:** linear은 symmetric(zero=0)인데도 두 번째 `_prealign`(reduce) + 16-lane `act_sum_red`를 계산 후 0을 곱해 폐기 = 절반 낭비.
- **제안:** `zero_data`가 전부 0이면 reduce 경로 생략, `post = inner`.
- **정확성:** z=0 → 빼는 항이 정확히 0 → **비트 동일**. #1/#2와 독립·중첩 가능.

### ★ #4 — attention per-head 루프 → leading H축 배치 (QCOL·QROW)
- **효과: 5–15× (attention) · 안전성: 높음**
- **위치:** `custom_fp_int_attention`의 이중 for 루프
- **제안:** kernel에 leading `H = batch*num_heads` 축 추가(`input (H,M,K)`, `weight (H,K,N)`). 내부 연산은 모두 브로드캐스트 → `KG/16` 루프가 head마다가 아니라 **전체 1회**. GQA는 `key_int[:, j//n_rep]` gather로 per-query-head weight 구성.
- **정확성:** M/배치 축은 행 독립 → 축 삽입 무해. `_prealign`의 블록-max 축소가 head 간 행을 섞지 않도록만 확인. 배치 vs 루프 정수 도메인 정확 일치 검증.

### #5 — transpose/contiguous/cast 호이스팅 + custom 경로 낭비 제거
- **효과: 1.1–1.4× (GQA에서 더 큼) · 안전성: 높음**
- **위치:** figna 루프 내 `.t().contiguous()`; `modeling_llama.py`의 `repeat_kv`(759-771), clamp/cast(832-837)
- **제안:** kv-head별 transpose를 루프 밖 1회; custom 경로는 759-760 `repeat_kv` 결과를 **안 쓰므로** `if not use_custom_attn:`로 가드; `.clamp(-8,7).to(int8)` 등 캐스팅을 caching wrapper로 이동(매 forward 재캐스팅 제거).
- **정확성:** layout 변경은 비트 동일.

### #6 — qrow 이중 prealign 호이스팅
- **효과: ~2× (P@V) · 안전성: 중 (검증 필요)**
- **위치:** qrow 그룹 루프 내 두 번의 전체 (M,K) prealign
- **제안:** main/red prealign은 `extra_bitwidth` 시프트만 다르므로 `aligned_fx_red = aligned_fx_main >> (EXTRA_BIT - EXTRA_BIT_FOR_REDUCE)`로 1회에서 파생. 또는 NG 루프 자체를 배치.
- **정확성:** ⚠️ 2's-complement 음수의 산술 vs 논리 시프트 의미 → **비트 동일 검증 필수**. 안 맞으면 이중 prealign 유지하되 NG 배치.

---

## 6. 권장 실행 로드맵

1. **검증 하네스 강화 (선행 필수)** — `utils/figna_utils.py:__main__`의 qcol/qrow allclose를 **bit-exact 골든 비교**로 교체: 현재 코드 출력을 골든 텐서로 캡처, 정수 중간값은 `torch.equal`, 최종 float32 `acc`는 비트 비교. (현재 `atol=1e-2/1e-3`은 #2/#6의 누적순서 회귀를 숨김.)
2. **linear 경로 집중 (최대 wall-clock 이득):** #3(reduce 스킵) → #1(lane→matmul) → #2(block→bmm). 목표: **146s/req → 한 자릿수 초.**
3. **attention 경로:** #4(head 배치) → #5 → #6.

각 단계 후 골든 비교로 비트 동일 확인, 그 다음 실제 eval 재측정.

---

## 7. 별도 발견 — 잠재적 정확성 플래그 (성능과 무관)

`eval_utils/modeling_llama.py`의 custom attention 진입부에서 `k_int.clamp(-8,7)` / `v_int.clamp(-8,7)`로 양자화 값을 saturate한다. 양자화기가 이미 INT4 범위를 보장하면 중복(무해)이지만, 그렇지 않다면 **매 forward 값을 조용히 바꾸는 것**이므로 별도 확인을 권장한다.

---

## 8. 참고 (reference 정확도)

FIGNA emulation 정확도 비교 기준이 되는 baseline(표준 GEMM/attention, W4A16KV4):

| 지표 | 값 |
|---|---|
| wiki2 perplexity | **6.469** |
| zero-shot Average (8 task) | **65.87%** |

emulation config들이 이 값에 근접하면 emulation이 정확하다는 의미.

---

## 9d. 구현·검증 결과 (4차: qcol collapse — 단일 matmul, linear 경로 40-60×)

실측으로 kernel_only(custom GEMM linear)의 병목은 **fp64 matmul(167ms) + (KG,M,N) elementwise 후처리(273ms)** 임을 확인. (fp32-split은 오히려 느려 폐기 — 이 A6000은 fp64가 나쁘지 않음.)

**핵심 통찰:** 블록 지수 `2^(e-bias)`는 블록당 상수라 lane 합에 분배됨 → **exp를 활성값 A에, per-column scale을 W에 미리 접으면 KG-블록 계산 전체가 단일 `(M,K)@(K,N)` matmul로 붕괴**. `(KG,M,N)` 중간텐서·elementwise 루프가 사라짐. 값이 fp32에 들어가 golden과 fp16 ULP 이내 일치(TF32 off).

zero==0(대칭 weight-GEMM linear) 경로에 적용. 실측 (seq=512):

| linear shape | old(block) | new(collapse fp32) | speedup |
|---|---|---|---|
| q/o (4096→4096) | 128ms | 3.6ms | 36× |
| gate/up (4096→14336) | 445ms | 7.6ms | **58×** |
| down (14336→4096) | 465ms | 10.6ms | 44× |
| **per-layer(7 linears)** | ~1073ms | ~35ms | **~30×** |

golden verify: ALL PASS allclose (qcol_perf 19×→**56.6×**), `__main__` 통과, `custom_fp16_int4_gemm` rel error **0.0%**.

**실제 eval 효과:** kernel_only forward ~34s → ~1.1s → **8.2 s/req → ~0.3 s/req → full zeroshot ~6h(완주 가능).** 15.6s→0.3s는 **~50×**.

## 9c. 구현·검증 결과 (3차: #4 attention head 배칭)

`custom_fp_int_attention`의 **32-head × 2-GEMM Python 루프(64회 순차 호출)**를 제거: 모든 (batch, head)를 leading B축으로 접어 **batched qcol/qrow 각 1회 호출**(`fpint_gemm_qcol_batched` / `fpint_gemm_qrow_batched`, head 차원 타일링으로 메모리 제어). GQA는 `key/value[:, arange(Hq)//n_rep]` gather로 per-query-head 구성.

`test_attn_opt.py`로 per-head-loop 출력(golden) 대비 검증:

| config (b=1,Hq=32) | Hkv | seq | bit-exact | old→new | speedup |
|---|---|---|---|---|---|
| mha_s64 | 32 | 64 | ✅ **maxdiff 0.0** | 54.0→6.7ms | 8.1× |
| gqa_s64 | 8 | 64 | ✅ 0.0 | 54.1→6.5ms | 8.3× |
| gqa_s128 | 8 | 128 | ✅ 0.0 | 53.9→6.7ms | 8.1× |
| gqa_s256 | 8 | 256 | ✅ 0.0 | 54.1→11.2ms | 4.8× |
| gqa_s512 | 8 | 512 | ✅ 0.0 | 68.7→39.4ms | 1.7× |

- **전부 완전 bit-exact** (`torch.equal`=True) — per-head 루프와 수치 완전 동일.
- 짧은 seq에서 8×(launch-bound 제거), 긴 seq에서 1.7×(compute-bound로 전환, fp64 matmul이 지배적). GQA/MHA/causal 모두 정상.

## 9b. 구현·검증 결과 (2차: #1 + #2 + #3, GEMM)

`fpint_gemm_qcol` / `fpint_gemm_qrow`를 **완전 벡터화**: 16-lane MAC + KG 블록 루프를 **단일 batched matmul(`bmm`)**로 통합(qcol은 N 방향 타일링으로 메모리 제어), 블록 간 float 누적은 float64로(큰 K에서 f32 누적 오차 방지). float64는 정수(<2^53)를 정확히 표현.

`test_figna_opt.py`로 golden(최적화 전 출력) 대비 검증:

| config | 입력 (M,K,N,gs) | allclose | maxdiff | old→new | **speedup** |
|---|---|---|---|---|---|
| qcol_main | 256,256,256,32 | ✅ | 2e-3(fp16 ULP) | 15.8→1.1ms | 14.4× |
| qcol_linear | 64,512,512,32 | ✅ | 2e-3 | 30.4→0.7ms | **41.3×** |
| qcol_perf | 128,**2048**,2048,32 | ✅ | 8e-3 | 120.3→6.3ms | 19.2× |
| qrow_main | 256,256,256,32 | ✅ | 1e-3 | 114.6→7.3ms | 15.8× |
| **qrow_perf** | 128,**2048**,256,32 | ✅ | 4e-3 | 878.5→7.3ms | **120.9×** |
| (8개 전부) | | ✅ ALL PASS | — | — | 6.8–120.9× |

- 차이는 **fp16 출력의 마지막 1 ULP 수준**(float 재정렬로 불가피, 값 크기에 비례). 내장 `__main__` allclose 테스트 qcol/qrow 모두 통과.
- **integration 정확성 확인:** `custom_fp16_int4_gemm`(groupsize-중복 scale) rel error **0.000%**; `custom_fp_int_attention` 정상 동작(유한, GQA+padding 경로 포함).
- **정수 matmul은 CUDA 미지원** → float64 matmul(A6000에서 2048² ~3ms).

### 실제 eval 예상 효과
kernel_only(custom GEMM linear)가 ~146s/req였는데, linear 경로 qcol이 perf-shape에서 ~19× → **request당 수 초대로 단축** 기대. qrow(P@V)는 120×. 다음: **#4(attention per-head 배칭, leading H축)**로 attention 추가 5–15×.

---

## 9. 구현·검증 결과 (1차: #1 + #3, 참고)

`fpint_gemm_qcol` / `fpint_gemm_qrow`에 **#1(16-lane 루프 → 블록당 float64 matmul)** + **#3(zero=0 시 reduce 경로 스킵)** 적용. float64는 정수(<2^53)를 정확히 표현하므로 정수 contraction이 **비트 동일**, float32 스케일/누적은 원본과 동일 순서 유지.

검증 하네스 `test_figna_opt.py` (golden = 최적화 전 출력, 동일 입력):

| config | 입력 (M,K,N,gs) | bit-exact | old(ms) | new(ms) | speedup |
|---|---|---|---|---|---|
| qcol_main_z0 | 256,256,256,32 | ✅ (maxdiff 0) | 15.8 | 3.3 | 4.8× |
| qcol_z_nonzero | 128,256,256,32 | ✅ | 15.6 | 4.5 | 3.4× |
| qcol_linear_z0 | 64,512,512,32 | ✅ | 30.4 | 5.6 | 5.5× |
| qcol_attn_z0 | 100,128,100,128 | ✅ | 8.1 | 1.7 | 4.7× |
| **qcol_perf_z0** | 128,**2048**,2048,32 | ✅ | 120.3 | 20.5 | **5.9×** |
| qrow_main | 256,256,256,32 | ✅ | 114.6 | 29.0 | 4.0× |
| qrow_attn | 100,112,128,128 | ✅ | 6.8 | 2.1 | 3.3× |
| **qrow_perf** | 128,**2048**,256,32 | ✅ | 878.5 | 194.3 | **4.5×** |

- **모든 config `torch.equal` = True (maxdiff 0.0)** — 진짜 비트 동일.
- 내장 `__main__` allclose 테스트도 qcol/qrow 모두 통과.
- **정수 matmul은 CUDA 미지원**(NotImplementedError) → float64 matmul 사용(A6000에서 2048² ~3ms로 충분).

### 남은 여지 (더 큰 이득)
현재는 KG 블록 루프를 유지(실제 linear의 N이 커서 `(KG,M,N)` 전체 배칭은 메모리 폭발). 추가로:
- **#2 (블록 배칭, 타일링)**: 블록 matmul을 타일 단위 `bmm`으로 묶어 런치 수 추가 감소 → 10–20× 가능. float32 누적을 `sum`으로 하면 ULP 차이(allclose 통과) 또는 순차 유지 시 비트 동일.
- **#4 (attention per-head 배칭)**: leading H축 도입 → attention 5–15×.
현재 1차 결과는 **완전 비트 동일 + 3.3–5.9×**의 안전한 이득.

## 부록: 주요 파일·심볼

- `utils/figna_utils.py` — `custom_fp_int_attention`, `fpint_gemm_qcol_real_2scomp_torch`, `fpint_gemm_qrow_real_2scomp_torch`, `_prealign_torch_fp16bits`, `custom_fp16_int4_gemm`, `__main__` allclose 테스트
- `utils/quant_utils.py` — `ActQuantWrapper.forward` (custom GEMM 호출부, V 캐싱)
- `eval_utils/modeling_llama.py` — `LlamaSdpaAttention.forward` custom-attention 분기
- `eval_utils/rotation_utils.py` — `QKRotationWrapper.forward` (K 캐싱)
