# OIBC 2026 TSMixer v0 — Mac 개발 / Linux CUDA 학습

기존 `analysis/deep_learning/`은 수정하지 않는다. 모든 명령은 **저장소 루트**에서 실행한다. Mac에서는 전체 학습·전체 dev 평가·대용량 캐시 생성이 금지되고 코드에서도 차단한다. 기본 `auto`는 macOS CPU, Linux에서 CUDA가 있으면 CUDA를 선택한다. MPS는 명시적으로 선택한다.

## 모델과 데이터 계약

- 기본 모델: hidden=128, mixer=3, time hidden=64, feature hidden=256, dropout=0.1. 총 **305,865 parameters**.
- 기존 `InputEncoder`: 기상 90개 + 기상 결측 mask 90개 + 시간 8개, 정적 146개 + site embedding. 모델은 정규화된 `[B,24,90]`, `[B,24,8]`, `[B,146]`, `[B]`를 받아 `[B,24]` CF를 낸다.
- 각 블록: feature LayerNorm → `[B,H,24]`에서 시간 MLP 24→64→24 → residual → feature LayerNorm → `[B,24,H]`에서 특징 MLP 128→256→128 → residual. 정적 조건은 encoder에서 모든 시간에 더한다. batch를 혼합하는 연산은 없다.
- `forward_fleet`는 향후 그룹 손실용 `[B_days,253,24,*]` 인터페이스다. 모든 canonical site_pos가 있을 때만 허용하며 부분 발전소 배치는 거부한다. **v0 학습은 사이트 weighted L1만** 사용한다.
- 운영일 D 01:00~D+1 00:00, 발행 D−1 17:00 KST. round17 30개 + round10 30개 + revision 30개를 유지한다. 실측 기상·발전량 이력·SMP·k는 모델 입력에 넣지 않는다. SMP·k는 기존 weighted L1과 정산에서만 사용한다.
- `DayCache`/`DayDataset` 및 compact station-month mmap, `fit_normalizer`, `compute_loss`, `apply_postprocess`, `Evaluator`를 재사용한다. 학습 CF `[0,1]` 필터, target NaN mask, train-only 표준화, 기본 train_calibrated 야간 후처리를 유지한다.
- 설정은 `configs/tsmixer/base.yaml`. AdamW lr=0.001, seed=2026, 서버 최대30 epoch, patience6, fp32 기본. 사이트 가중 loss의 분자·분모를 합산해 accumulation하므로 서로 다른 결측 비율의 microbatch도 올바르게 가중한다.
- development cutoff=2025-01-31 17:00, dev 운영일 [2025-02-01,2025-02-28). 기존 2025년 3~5월 holdout은 이미 관찰한 구간이므로 v0 CLI에서 평가/튜닝 대상으로 노출하지 않는다. v0는 development만 지원하고 refit/production/앙상블은 제공하지 않는다.

## 설치: Mac

권장 Python은 양쪽 모두 **3.13**이다. 로컬 검증 환경은 Python 3.13.12 / torch 2.15.0 / NumPy 2.5.3이며 결과 보고서에 나머지 버전을 기록했다. 기존 `.venv`를 덮어쓰지 않고 새 환경을 만든다.

```bash
bash scripts/tsmixer/install_mac.sh
source .venv-tsmixer/bin/activate
bash scripts/tsmixer/check_environment.sh --device cpu
python -m unittest discover -s analysis/tsmixer/tests -v
python -m analysis.tsmixer smoke --device cpu --steps 5 --output runs/tsmixer/mac_smoke_001
```

선택적 MPS 검증:

```bash
python -m analysis.tsmixer environment --device mps
python -m analysis.tsmixer smoke --device mps --steps 5 \
  --set training.deterministic=false --output runs/tsmixer/mac_mps_001
```

`mps.is_available()`이 false면 명확한 오류로 종료한다. 자동 CPU fallback이나 수학 변경은 하지 않는다. MPS embedding backward는 deterministic 구현이 없어 기본 deterministic=true에서는 명확한 오류를 내며, 위 명령처럼 선택적 테스트에서만 false를 명시한다. 이는 재현성 보장을 낮추며 모델 수식을 바꾸지는 않는다. 테스트 suite의 MPS 테스트는 이 모드를 명시해 CPU 대비 rtol=2e-3/atol=2e-4와 backward를 검사한다. 연산 지원 오류가 나면 traceback을 보존하고 CPU 테스트로 확인한다. Mac CPU/MPS smoke는 batch≤4, sample≤32, optimizer update≤10, thread 기본2, worker0이다.

## GitHub로 전달

데이터·캐시·checkpoint·가상환경·로그는 Git으로 전달하지 않는다. 이 작업은 push하지 않는다. feature branch의 변경을 먼저 검토한 뒤 Mac에서 직접 수행한다.

```bash
git status --short
git log -1 --oneline
git push -u origin feature/tsmixer-v0-mac-cuda
```

서버에서는 새 checkout을 권장한다. 기존 서버 작업 트리를 reset/clean하거나 데이터를 덮어쓰지 않는다.

```bash
git clone --branch feature/tsmixer-v0-mac-cuda https://github.com/kimna428/2026_OIBC.git 2026_OIBC_tsmixer
cd 2026_OIBC_tsmixer
git log -1 --oneline
git status --short
```

이미 checkout이 있다면 변경을 먼저 확인하고 `git fetch origin feature/tsmixer-v0-mac-cuda` 후 검토하여 switch/pull한다. Mac과 서버 commit을 비교한다.

## 설치: Linux / RTX 3090 Ti

공통 의존성은 `requirements-common.txt`가 기존 딥러닝 requirements를 참조한다. CUDA torch는 Mac requirements에 포함시키지 않는다. NVIDIA 드라이버와 호환되는 공식 wheel index와 버전을 **명시적으로** 선택한다. 다음은 공식 배포 기록에 있는 2.11.0/cu128 예시이며, 로컬 검증 torch 2.15.0과는 다르다. 같은 버전의 CUDA wheel이 제공되고 드라이버가 호환되면 `TORCH_VERSION`을 로컬 버전으로 맞춘다. 버전이 다르면 서버 테스트 통과가 필수다.

```bash
nvidia-smi
TORCH_VERSION=2.11.0 CUDA_WHEEL_INDEX=https://download.pytorch.org/whl/cu128 \
  bash scripts/tsmixer/install_cuda.sh
source .venv-tsmixer/bin/activate
python -m analysis.tsmixer environment --device cuda
python -m pip freeze > /tmp/oibc-tsmixer-server-environment.txt
```

설치 스크립트는 기존 `.venv-tsmixer`를 덮어쓰지 않는다. 다른 경로는 `OIBC_TSMIXER_VENV`, Python 실행 파일은 설치 시 `OIBC_BOOTSTRAP_PYTHON`, 실행 시 `OIBC_PYTHON`으로 지정할 수 있다. 설치 실패 시 driver/wheel/Python 조합을 수정하고 새 venv 이름으로 재시도한다. torchvision/torchaudio는 필요 없다.

공식 설치 근거: [PyTorch 설치 선택기](https://pytorch.org/get-started/locally/), [공식 과거 버전의 CUDA 설치 명령](https://pytorch.org/get-started/previous-versions/), [공식 MPS 설명](https://docs.pytorch.org/docs/stable/notes/mps.html). 설치 명령은 서버에서 검증해야 하며 이번 Mac 작업에서는 설치나 CUDA 실행을 하지 않았다.

## 서버 데이터 준비와 검증

필수 원본 파일은 다음 8개다. 경로는 `OIBC_DATA_ROOT` 아래 상대 경로다.

```text
sites.csv
smp.csv
train/smp.csv
train/generation.parquet
train/forecast_wa.parquet
train/forecast_wb.parquet
test/forecast_wa.parquet
test/forecast_wb.parquet
```

`train/observed_wa.parquet`, `train/observed_wb.parquet`는 inventory에 기록하지만 TSMixer 학습에는 필요 없다. 공식 test 예측은 v0 범위 밖이지만 기존 prepare의 입력 계약상 test 예보·SMP도 요구한다.

`reports/tsmixer/data_inventory.json`에는 파일별 크기, 행 수, 실제 schema, SHA-256이 있다. CSV string dtype 명칭은 pandas 버전별 차이가 있으므로 비교는 컬럼명·행 수·크기·hash를 사용하고, Parquet schema는 타입까지 비교한다. 검사는 데이터를 변경하지 않으며 SHA-256은 1MiB 청크로 읽는다.

```bash
export OIBC_DATA_ROOT=/your/private/data/01_participant_dataset
export OIBC_TSMIXER_CACHE_ROOT="$PWD/runs/tsmixer/cache_station_v1"
bash scripts/tsmixer/server_preflight.sh
```

데이터가 없으면 명령은 실패한다. GitHub clone으로 데이터는 생기지 않는다. 승인된 서버 경로에 원본 데이터를 별도로 전송해야 한다. 예를 들어 서버에 **새 수신 디렉터리**를 만들고 Mac에서 다음처럼 기존 파일 덮어쓰기를 방지한다. HOST와 경로는 실제 값으로 바꾼다. 비밀번호/키를 저장소에 넣지 않는다.

```bash
rsync -av --ignore-existing 'handover 2/01_participant_dataset/' USER@HOST:/your/private/new_dataset_directory/
```

`--ignore-existing`은 기존 파일을 보존하지만 손상된 기존 파일도 건너뛴다. 전송 후 checksum 검사를 반드시 실행하고 mismatch 파일은 별도 새 경로로 받아 원인을 확인한다. 삭제나 덮어쓰기 명령은 제공하지 않는다.

## 서버 캐시

기존 호환 캐시가 있으면 그 부모 디렉터리를 `OIBC_TSMIXER_CACHE_ROOT`로 지정한다. `check-data --cache-root`는 train 캐시의 manifest/array schema/feature order와 모든 원본 입력 hash를 검사한다. 완전 학습은 원본 train SMP의 전체 운영일 축도 일치해야 하므로 부분 검사 캐시를 전체 학습에 사용할 수 없다.

```bash
python -m analysis.tsmixer check-data --data-root "$OIBC_DATA_ROOT" \
  --cache-root "$OIBC_TSMIXER_CACHE_ROOT" --checksums \
  --reference reports/tsmixer/data_inventory.json
```

캐시가 없으면 **Linux CUDA 서버에서만** 생성한다. `prepare-cache`는 train compact 캐시만 만들고 test 캐시를 만들지 않는다. 기존 완성 캐시는 검증 후 재사용하고, 불완전하거나 충돌하는 경로는 삭제하지 않고 오류로 끝낸다.

```bash
bash scripts/tsmixer/prepare_cache.sh --dry-run
bash scripts/tsmixer/prepare_cache.sh
```

기존 storage guard의 기본 개인 quota는 12GiB, cache budget은 2GiB다. 서버 실제 개인 quota를 확인한다. 외부 출력 경로를 쓰면 `OIBC_STORAGE_ROOT`를 **자신의 quota 디렉터리**로 지정한다. 필요한 경우에만 실제 승인된 quota에 맞춰 `OIBC_QUOTA_GIB`를 설정하고 cache budget은 `--cache-max-gib`로 지정한다. quota 회피나 기존 파일 삭제는 하지 않는다.

## 서버에서 전체 학습 전에 실행

```bash
python -m unittest discover -s analysis/tsmixer/tests -v
python -m analysis.tsmixer smoke --device cuda --steps 5 \
  --output runs/tsmixer/cuda_synthetic_001
python -m analysis.tsmixer smoke --device cuda --real-cache --steps 5 \
  --cache-root "$OIBC_TSMIXER_CACHE_ROOT" --output runs/tsmixer/cuda_real_001
```

real smoke는 학습 cutoff 이전 마지막 2개 운영일의 view만 만들고 그 view에서 정규화한다. 원본 cache signature는 유지하지만 체크포인트·결과를 **smoke-only**로 표시한다. 정규화가 전체 학습과 다르므로 성능·초기 모델로 승격할 수 없다. MPS/CPU는≤32개, CUDA는≤128개 사이트/일 표본만 사용한다. 캐시를 생성하거나 수정하지 않는다.

CUDA smoke의 batch는 설정값(상한128)이며 기본32다. `epochs.jsonl`의 `peak_allocated_bytes`, `peak_reserved_bytes`, 환경의 free VRAM을 확인한다. 필요하면 새 output 이름으로 `--set training.batch_size=16 --set training.accumulation_steps=2`를 시험한다. smoke는 train forward/backward만 확인하므로 전체 dev 평가의 추가 메모리 여유도 남긴다. OOM 시 자동 학습·자동 재시도는 하지 않는다.

Mac에서 기존의 **작은 실데이터 캐시**가 있을 때만 같은 real smoke를 CPU에서 허용한다. 전체 원본을 메모리에 올리거나 새 대용량 캐시를 만드는 명령은 아니다.

## 사용자가 명시적으로 실행하는 development 전체 학습

위의 환경·데이터·캐시 검사 및 CUDA smoke가 통과한 뒤 직접 실행한다. 아래 명령은 이번 Mac 작업에서 실행하지 않았다.

```bash
bash scripts/tsmixer/train_development.sh --full-train \
  --output runs/tsmixer/dev_v0_seed2026
```

메모리 설정 예:

```bash
bash scripts/tsmixer/train_development.sh --full-train \
  --output runs/tsmixer/dev_v0_bs16_acc2 \
  --set training.batch_size=16 --set training.accumulation_steps=2 \
  --set training.num_workers=2
```

AMP는 기본 꺼져 있다. 동일 seed의 fp32 CUDA smoke와 비교해 유한 loss·gradient·예측 오차를 먼저 확인한 뒤 별도 실험에서 `--set training.precision=fp16` 또는 `bf16`을 사용한다. bf16 미지원이면 자동 대체하지 않는다. mixed precision과 fp32의 bitwise 일치는 요구하지 않는다. `run.json`과 `epochs.jsonl`로 실제 장치·설정·VRAM·loss를 확인한다.

재개는 **완료한 epoch 경계의 last.pt**에서 가능하다. optimizer/scaler/RNG/DataLoader generator 상태, 설정·코드·cache 서명이 같아야 한다. 중간 epoch가 끊겼다면 마지막 완료 epoch부터 다시 계산한다. smoke의 partial-epoch resume는 정확한 연속 trajectory를 보장하지 않으며 모델 성능용으로 사용할 수 없다.

```bash
bash scripts/tsmixer/train_development.sh --full-train \
  --output runs/tsmixer/dev_v0_seed2026 \
  --resume runs/tsmixer/dev_v0_seed2026/last.pt
```

동일 output에서 설정을 바꾸는 resume는 거부한다. early stop/epoch recipe가 완료된 run도 재개하지 않는다. checkpoint 저장 중 남은 `.pt.tmp`가 있으면 사용자가 확인하고 새 run 경로를 선택한다.

개발 구간 평가:

```bash
python -m analysis.tsmixer evaluate --device cuda \
  --checkpoint runs/tsmixer/dev_v0_seed2026/best.pt \
  --cache-root "$OIBC_TSMIXER_CACHE_ROOT" \
  --output runs/tsmixer/dev_v0_seed2026/eval_dev
```

평가는 batch 예측을 canonical site/day로 복원하고 기존 후처리·Evaluator에 전달한다. 출력은 `predictions.npy`, `metrics.json`, `site_and_cohort_metrics.csv`, `group_metrics.csv`, 완전 fleet 정답이 있으면 `settlement_ledger.parquet`다. 발전소 CF/kWh MAE와 두 집합의 정산금·페널티를 구분한다. 같은 output에 덮어쓰지 않는다.

`run.json`/checkpoint에는 Git commit·작업 트리·소스 hash·config fingerprint·cache/feature signature·seed·환경·normalizer·cutoff·checkpoint 경로를 기록한다. dirty tree이면 Git commit만으로 재현할 수 없으므로 학습 전 변경을 검토·커밋한다. runtime 결과와 checkpoint는 `/runs/` 아래에 보존하며 Git에서 제외된다.

## CLI와 제한

```bash
python -m analysis.tsmixer --help
python -m analysis.tsmixer train --help
```

모든 설정은 별도 YAML 또는 `--set section.key=value`로 바꿀 수 있다. data/cache 경로는 `--data-root`, `--cache-root`도 지원한다. 장치는 `--device auto|cpu|mps|cuda`. 임의 모델 입력·group loss·알 수 없는 architecture 설정은 오류로 처리한다. v0 loss는 weighted L1으로 고정한다. 단위 테스트 외에서는 synthetic 체크포인트로 실데이터 성능을 평가할 수 없다.

GRU 구현은 현재 저장소에서 발견되지 않았다. 따라서 CNN·Transformer와 입력·평가 계약은 맞췄지만 GRU와 동일 조건임을 확인했다고 주장하지 않는다. 공식 채점기/문제정의서가 없으므로 기존 로컬 정산 구현을 재사용했다는 범위에서 호환성을 해석한다.

## Mac에서 원격 학습 제어

기존 학습·평가 CLI를 변경하지 않고 SSH 실행·상태 조회·결과 회수를 추가했다. 최초 설정과 명시적 캐시/검증 옵션은 [원격 실행 안내](../../scripts/tsmixer/REMOTE_README_KO.md)를 따른다.

```bash
bash scripts/tsmixer/remote_setup.sh
bash scripts/tsmixer/remote_launch.sh
bash scripts/tsmixer/remote_status.sh
bash scripts/tsmixer/remote_fetch.sh <experiment_id>
```

기본 setup은 검사 전용이다. launch는 사용자가 직접 실행할 때만 기존 `--full-train` 학습과 development 평가를 시작한다. 설정 파일과 회수 결과는 Git에서 제외하며, SSH·GPU 서버 통합 검사는 사용자가 설정을 마친 뒤 명시적으로 수행한다.

## Google Colab GPU

[`train_colab.ipynb`](../../train_colab.ipynb)는 기본 synthetic5step smoke로 시작하며, Drive 원본을 로컬 디스크로 복사하고 기존 compact 전처리·학습·dev 평가를 호출한다. 선택적 `--backup-root`로 완료 epoch마다 검증된 generation을 저장하고 정상 checkpoint를 복원한다. 원래 constant LR을 유지하면서 scheduler 상태를 checkpoint에 추가했다.

셀1~12 실행 순서, 전체 학습 전환, Drive 경로와 Resume 조건은 [Colab 한국어 안내](../../scripts/tsmixer/colab/README_KO.md)를 따른다. 기존 checkpoint는 보존하며 이전 소스의 실험은 그 commit에서 재개한다. Colab 변경은 GitHub에 반영한 다음 노트북에서 clone해야 한다.
