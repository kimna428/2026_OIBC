# TSMixer v0 구현 및 Mac → RTX 3090 Ti 이식 준비 결과

작성일: 2026-11-02. 분석 출발점은 `reports/TSMIXER_REPOSITORY_ANALYSIS_KO.md`, 기준 커밋은 `98c3d778a26dd76336ada7c7dd8977c1ef159183`이다.

## 1. 완료 범위

**TSMixer v0 소스·설정·독립 CLI·테스트·Mac/CUDA 설치 절차·서버 실행 스크립트 구현을 완료했다.** Mac에서 CPU 및 MPS의 제한된 검증을 수행했다. 전체 데이터 학습, 전체 dev 평가, holdout 평가, 제출, 원격 접속 및 GitHub push는 수행하지 않았다. 기존 딥러닝 소스·실험 결과·체크포인트는 변경하지 않았다.

서버의 실제 NVIDIA 드라이버, CUDA wheel, VRAM, 데이터 경로에는 접근하지 않았다. 따라서 서버에서 사전 검사를 통과하기 전까지 전체 학습 실행 가능성을 확정할 수 없다.

## 2. 신규·수정 파일

| 경로 | 내용 |
|---|---|
| `analysis/tsmixer/__init__.py`, `__main__.py` | 독립 패키지와 environment/check-data/prepare-cache/smoke/train/evaluate CLI |
| `analysis/tsmixer/model.py` | Time Mixing, Feature Mixing, TSMixer, future full-fleet API |
| `analysis/tsmixer/config.py` | 독립 설정 검증, 환경변수·CLI override, 루트 작업 경로 검사 |
| `analysis/tsmixer/data.py` | 기존 DayCache/DayDataset 재사용, compact 호환, 제한된 SmokeCache view |
| `analysis/tsmixer/runtime.py` | auto/cpu/mps/cuda, portable batch, 선택적 CUDA AMP, 환경 정보 |
| `analysis/tsmixer/train.py` | weighted L1·AdamW·정확한 가중치 합산 accumulation·dev 선택·epoch resume |
| `analysis/tsmixer/checkpoint.py` | atomic 저장, weights-only 복원 |
| `analysis/tsmixer/evaluate.py` | 발전소 canonical 복원, 기존 후처리·Evaluator 기반 streaming dev 평가 |
| `analysis/tsmixer/inventory.py` | 원본 크기·행 수·schema·streaming SHA-256, 캐시와 원본 hash 연결 검사 |
| `analysis/tsmixer/tests/{__init__,test_tsmixer}.py` | 모델·데이터·축·누수·재로드·CPU/MPS/CUDA·worker 테스트 |
| `analysis/tsmixer/requirements-{common,mac}.txt` | 기존 공통 의존성과 플랫폼별 torch 설치 분리 |
| `analysis/tsmixer/README.md` | 모든 설치·개발·전송·학습·평가·재개 명령과 제약 |
| `configs/tsmixer/base.yaml` | 모델·손실·optimizer·경로·기존 split·seed·후처리 설정 |
| `scripts/tsmixer/check_environment.sh` | Git·Python·장치 확인 |
| `scripts/tsmixer/install_{mac,cuda}.sh` | 별도 venv 설치; 기존 환경 덮어쓰기 거부 |
| `scripts/tsmixer/server_preflight.sh` | Linux/GPU·데이터·기존 캐시 읽기 전용 검사 |
| `scripts/tsmixer/prepare_cache.sh` | 서버에서만 train compact 캐시 준비 |
| `scripts/tsmixer/smoke.sh` | 제한된 smoke CLI 래퍼 |
| `scripts/tsmixer/train_development.sh` | 명시적 --full-train으로만 서버 전체 학습 |
| `reports/tsmixer/data_inventory.json` | 10개 원본 파일의 실제 크기·행 수·schema·SHA-256 |
| `reports/tsmixer/unit_tests.txt`, `cli_checks.json`, `local_verification.json` | 작은 텍스트 검증 증거 |
| `reports/tsmixer/IMPLEMENTATION_REPORT_KO.md` | 이 보고서 |
| `.gitignore` | 기존 규칙을 유지하고 `.venv-tsmixer/` 한 줄 추가 |

기존 `analysis/deep_learning/`, `configs/deep_learning/`, `scripts/gpu/`는 수정하지 않았다. 이전 분석 보고서와 기존 미추적 파일도 수정하지 않았다.

## 3. 모델 구조·파라미터

기본 TSMixer는 **305,865개 파라미터**다.

```text
weather [B,24,90] + weather_missing [B,24,90] + temporal [B,24,8]
static [B,146] + site_index [B]
 → 기존 InputEncoder → [B,24,128]
 → 3 × {
     LayerNorm(feature) → transpose → Time MLP 24→64→24 → transpose → residual
     LayerNorm(feature) → Feature MLP 128→256→128 → residual
   }
 → LayerNorm + 시간별 head → CF [B,24]
```

Time/Feature MLP 내부 차원은 YAML에서 바꿀 수 있다. 정적 조건과 site embedding은 기존 InputEncoder와 동일한 방식으로 시간축에 더한다. v0는 사이트 weighted L1, AdamW(lr=0.001), seed=2026, 서버 epoch 상한30, dropout0.1, fp32 기본이다. group loss는 거부한다.

`forward_fleet`는 향후 사용을 위해 `[B_days,253,24,*]`를 받되 모든 발전소와 canonical site_pos를 강제한다. v0는 이를 그룹 학습에 사용하지 않는다. 임의 부분 배치의 합계를 전체 집합으로 취급하지 않는다.

## 4. 실제 Mac 검증 결과

환경: macOS 15.6.1, arm64, Python 3.13.12, PyTorch 2.15.0, NumPy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, PyYAML 6.0.3. 기존 가상환경의 34개 패키지는 `uv pip check`에서 충돌이 없었다. 기존 uv 환경에는 pip 모듈이 없어 읽기 전용 uv 검사로 확인했으며 패키지를 설치·업그레이드하지 않았다.

| 검증 | 결과 |
|---|---|
| 최종 unit suite | **29개 실행: 28개 통과, CUDA 1개 skip**, 약8.8초 |
| 기본 입력/24시간 출력 | 통과 |
| Time Mixing 축 | MLP 입력 `[B,H,24]`, 한 시간 변화가 다른 시간 출력에 영향: 통과 |
| Feature Mixing 축 | MLP 입력 `[B,24,H]`, 다른 시간에 영향 없음: 통과 |
| finite forward/backward·gradient | CPU·MPS 통과 |
| batch 독립성·정적 특성 반영 | 통과 |
| canonical 복원·부분 fleet 거부 | 통과 |
| 원본 결측·target mask·CF 범위 필터 | 통과 |
| 운영일·전일17시 cutoff·future 통계 제외 | 통과 |
| CPU checkpoint 저장/복원·추론 일관성 | 통과 |
| epoch 경계 optimizer/RNG 복원 | dropout을 켠 상태에서 연속 실행과 CPU bitwise 일치 |
| microbatch accumulation | 결측 수가 다른 1+3 샘플과 4샘플 합산 gradient의 동등성 통과 |
| spawn DataLoader worker1 | 작은 view·2샘플에서 통과 |
| 기존 Evaluator와 streaming dev 평가 정합성 | synthetic 하루 전체253개 예측에서 통과 |
| CLI 차단·shell | Mac full train/cache/full eval, 11step, CPU AMP, missing data, unavailable CUDA, 잘못된 cwd 거부; 셸7개 syntax 통과 |

초기 worker 검사는 샌드박스의 `torch_shm_manager` 소켓 제한으로 실패했다. 같은 작은 테스트를 승인 후 샌드박스 밖에서 실행해 통과했다. MPS도 샌드박스 안에서는 unavailable, 밖에서는 available이었다.

MPS의 `embedding_dense_backward_mps`는 deterministic 구현이 없어 처음에는 backward가 실패했다. 모델 수식은 바꾸지 않았다. MPS smoke에서 `--set training.deterministic=false`를 명시하도록 하고, 기본 deterministic=true의 MPS 실행은 원인을 알리는 오류로 차단한다. CPU·CUDA 기본값은 true다. MPS의 재현성 보장은 낮아지고 CPU/MPS bitwise 일치는 요구하지 않는다. suite에서 CPU/MPS forward는 rtol=2e-3, atol=2e-4로 비교하고 backward의 유한성도 확인했다.

### 소규모 학습

| 실행 | 제한 | 결과 |
|---|---|---|
| 최종 CPU synthetic | batch4, 최대32샘플, 5 update | weighted L1≈0.22776825, 저장/재로드 일치 |
| 최종 CPU 실데이터 compact | 기존 2025-01-30~01-31 캐시 view, batch4, 5 update | weighted L1≈0.30667170, 저장/재로드 일치 |
| MPS synthetic | batch4, 5 update, deterministic=false | weighted L1≈0.23845551, 저장/재로드 일치 |
| 추가 CPU smoke resume | 5→10 update | 재개·재로드 성공. 부분 epoch smoke는 exact trajectory를 주장하지 않음 |

이 loss는 **동작 확인용**이며 모델 성능·정산 점수·CPU/MPS 우열을 뜻하지 않는다. 실데이터 smoke도 마지막2개 학습일 view에서 통계를 fit하므로 전체 학습 통계와 다르다. 모든 smoke checkpoint는 full train/evaluate로 승격할 수 없다.

산출물은 Git 제외 경로 `runs/tsmixer/mac_{synthetic_final,real_final,mps_v0}/` 등에 있다. 원본·기존 캐시를 수정하지 않았으며 **10개 원본의 SHA-256을 작업 중 inventory와 사후 전수 대조해 일치**했다. 전체 데이터셋을 메모리에 적재하지 않았고 대용량 전처리 캐시도 생성하지 않았다.

## 5. 기존 파이프라인 호환성

- 모델 입력·특징 순서 90/8/146, r17/r10/revision, 발전소 ID unknown 정책은 기존 CNN·Transformer와 같다.
- 기존 `fit_normalizer`와 cutoff를 사용한다. 학습 유효 CF `[0,1]`, NaN target 보존, 예보 평균 대체+missing mask, 실측·미래 발전량·SMP/k 모델 입력 금지를 유지한다.
- development cutoff=2025-01-31 17:00 KST, dev 운영일 [2025-02-01,2025-02-28). 기존 holdout 경계를 변경하지 않았고 v0에서는 holdout 선택지 자체를 노출하지 않는다.
- 추론은 253개소를 canonical 순서로 복원한 뒤 기존 야간 후처리·정산을 사용한다. 발전소 오차와 집합 합계 오차를 분리한다.
- full train은 원본 train SMP 전체 day axis와 원본 hash를 검사하므로 Mac의 부분 검사 캐시를 전체 학습에 사용할 수 없다.
- GRU 소스·설정은 여전히 확인하지 못했다. GRU 동일 조건 비교가 완료됐다고 주장하지 않는다.
- 공식 채점기·문제정의서는 없어 로컬 평가 코드 및 제공 README와의 호환 범위로 해석한다.

## 6. Git 상태·전달 범위

납품 branch 이름은 **`feature/tsmixer-v0-mac-cuda`**다. 테스트 통과 후 이번 작업의 `.gitignore`, `analysis/tsmixer/`, `configs/tsmixer/`, `scripts/tsmixer/`, `reports/tsmixer/`만 선택적으로 stage·commit한다. 보고서 자체가 commit에 포함되므로 최종 commit ID는 자기 참조 문자열 대신 다음 명령으로 확인한다.

```bash
git branch --show-current
git log -1 --oneline
git show --stat --oneline HEAD
git status --short
```

작업 시작 때 존재한 다음 미추적 파일은 **추가·수정·삭제하지 않고 보존**한다.

```text
reports/TSMIXER_REPOSITORY_ANALYSIS_KO.md
reports/deep_learning/MODEL_OVERVIEW.md
reports/deep_learning/compact_initial_parity.json
reports/deep_learning/compact_storage_estimate.json
reports/deep_learning/compact_test_parity.json
```

원격 push는 하지 않는다. 사용자 검토 후 Mac에서 직접 실행한다.

```bash
git push -u origin feature/tsmixer-v0-mac-cuda
```

데이터·캐시·checkpoint·venv·임시 로그는 commit에서 제외한다.

## 7. 서버로 옮기는 정확한 절차

### 7.1 코드와 환경

Mac에서 위 push를 직접 수행한 다음 서버에서:

```bash
git clone --branch feature/tsmixer-v0-mac-cuda https://github.com/kimna428/2026_OIBC.git 2026_OIBC_tsmixer
cd 2026_OIBC_tsmixer
git log -1 --oneline
git status --short
nvidia-smi
```

기존 서버 checkout을 강제 reset/clean하지 않는다. Mac commit과 일치하는지 확인한다. 서버 Python도 가능하면3.13을 사용한다.

드라이버와 맞는 공식 torch 버전/index를 확인한 후 설치한다. 다음은 공식 배포 기록의 2.11.0/cu128 **예시**다. 로컬2.15.0과 다르므로 서버 검증이 필수다. 동일 버전 CUDA wheel이 제공되고 호환되면 맞추는 편이 좋다.

```bash
TORCH_VERSION=2.11.0 CUDA_WHEEL_INDEX=https://download.pytorch.org/whl/cu128 \
  bash scripts/tsmixer/install_cuda.sh
source .venv-tsmixer/bin/activate
python -m analysis.tsmixer environment --device cuda
```

설치 선택 근거는 [공식 PyTorch 설치 안내](https://pytorch.org/get-started/locally/)와 [공식 버전별 CUDA wheel 안내](https://pytorch.org/get-started/previous-versions/)다. Mac/CUDA requirements를 한 파일에 강제로 섞지 않았다.

### 7.2 데이터·캐시

```bash
export OIBC_DATA_ROOT=/your/private/data/01_participant_dataset
export OIBC_TSMIXER_CACHE_ROOT="$PWD/runs/tsmixer/cache_station_v1"
bash scripts/tsmixer/server_preflight.sh
```

서버 원본은 `sites.csv`, `smp.csv`, `train/smp.csv`, `train/generation.parquet`, `train/forecast_{wa,wb}.parquet`, `test/forecast_{wa,wb}.parquet`가 필요하다. 파일별 크기·행 수·타입·SHA-256은 `data_inventory.json`에 있다. 실측 파일2개는 선택 항목이다. 데이터가 없으면 별도 승인된 전송이 필요하며 GitHub는 데이터를 전달하지 않는다. 기존 파일은 덮어쓰지 말고 새로운 경로에 `rsync --ignore-existing` 등으로 전송한 다음 checksum 검사를 수행한다. 자세한 예시는 README에 있다.

호환 train 캐시가 있으면 재사용한다. 없으면 서버에서만:

```bash
bash scripts/tsmixer/prepare_cache.sh --dry-run
bash scripts/tsmixer/prepare_cache.sh
```

개인 quota·디스크 여유를 먼저 확인한다. 기존 storage guard의 기본 개인 quota12GiB/cache2GiB를 실제 서버 quota와 혼동하지 않는다. 외부 경로는 자신의 `OIBC_STORAGE_ROOT` 아래에 지정한다.

### 7.3 서버에서 먼저 실행할 검증

```bash
python -m unittest discover -s analysis/tsmixer/tests -v
python -m analysis.tsmixer smoke --device cuda --steps 5 \
  --output runs/tsmixer/cuda_synthetic_001
python -m analysis.tsmixer smoke --device cuda --real-cache --steps 5 \
  --output runs/tsmixer/cuda_real_001
```

CUDA smoke는 기본 batch32, 최대128표본, 최대10 optimizer update다. `run.json`과 `epochs.jsonl`의 free/peak VRAM을 확인한 뒤 batch를 결정한다. OOM이면 새로운 output에서 batch를 줄이고 accumulation을 늘린다. 자동 재시작하지 않는다. CUDA 테스트는 CPU/CUDA fp32 정합성과 fp16 AMP forward 허용 오차·유한 backward를 포함한다. 이 항목은 Mac에서는 skip했다.

### 7.4 사용자가 실행할 전체 학습·개발 평가

```bash
bash scripts/tsmixer/train_development.sh --full-train \
  --output runs/tsmixer/dev_v0_seed2026

python -m analysis.tsmixer evaluate --device cuda \
  --checkpoint runs/tsmixer/dev_v0_seed2026/best.pt \
  --cache-root "$OIBC_TSMIXER_CACHE_ROOT" \
  --output runs/tsmixer/dev_v0_seed2026/eval_dev
```

재개:

```bash
bash scripts/tsmixer/train_development.sh --full-train \
  --output runs/tsmixer/dev_v0_seed2026 \
  --resume runs/tsmixer/dev_v0_seed2026/last.pt
```

동일 설정·소스·cache·장치의 완료 epoch 경계만 정확히 재개한다. 중간 epoch 중단은 마지막 완료 epoch에서 다시 계산한다. 이미 epoch/early-stop recipe가 끝난 run은 재개하지 않는다.

run/checkpoint에는 Git commit·작업 트리, 소스 hash, 설정 fingerprint, cache/feature signature, seed, 환경·normalizer·cutoff·선택 결과를 보존한다. 평가 파일은 `metrics.json`, 발전소/cohort·그룹 CSV, 예측 NumPy, 정산 ledger Parquet이며 모두 Git 제외 run 경로에 저장한다.

## 8. 남은 검증과 최종 판정

**“RTX 3090 Ti 서버에서 TSMixer 전체 학습을 시작할 준비가 되었는가?” → 코드 전달 및 학습 준비 절차는 완료. 실제 전체 학습 시작은 서버 필수 검증 통과를 조건으로 한다.**

우선순위는 다음과 같다.

1. **P0 — 서버 환경**: 실제 driver·CUDA wheel·Python 호환성을 확인하고 CUDA가 인식돼야 한다. Mac에서는 CUDA·3090Ti VRAM을 검증할 수 없었다.
2. **P0 — 데이터·캐시**: 실제 서버의8개 필수 원본과 checksum, 전체 historical compact 캐시, 디스크 quota 검증이 필요하다. GitHub만으로 데이터는 오지 않는다.
3. **P0 — CUDA 기능·메모리**: 서버 unit suite 및 synthetic/실데이터5step smoke를 통과하고 실제 peak VRAM에 따라 batch를 정해야 한다. AMP는 기본 fp32를 검증한 뒤 별도 활성화한다.
4. **P1 — 전체 학습·dev 평가**: 구현한 full train/evaluate orchestration을 실제 전체 cache에서 실행한 결과는 아직 없다. Mac에서 실행하지 않았으며 독립적인 서버 통합 검증 항목이다.
5. **P1 — 비교 한계**: 이미 본3~5월 holdout을 모델 선택에 쓰지 않는다. GRU 동일 조건 비교, 그룹 손실·refit·production·공식 제출은 v0 완료 범위에 포함하지 않는다.

전체 학습을 지금 자동 시작하지 않으며, 위 검증 뒤 사용자가 `--full-train`을 명시해 시작하도록 구성했다.
