# OIBC GPU 딥러닝 구현 및 서버 실행 가이드

이 문서는 **코드 구현과 로컬 검사 결과**를 기록한다. 원격 접속, RTX 3090 Ti 학습, 전체 데이터 학습, 새로운 예측 성능 측정은 수행하지 않았다. 새 모델의 성능은 서버에서 학습·평가한 뒤 기록해야 한다.

## 1. 기존 프로젝트와 연결

검토 대상은 `reports/data_quality.md`, `reports/validation_protocol.md`, `reports/feature_pipeline.md`, `reports/baseline_lightgbm.md`, `reports/third_experiment/main_report.md`와 `report.md`, `analysis/feature_pipeline.py`, `validation.py`, `lightgbm_baseline.py`, `third_experiment.py`, 기존 manifest/config/prediction 스키마, 제공 데이터 README다.

- 피처 결합: 기존 `feature_pipeline.build_chunk`, `join_weather`, `weather_columns`를 재사용한다. 기존 `analysis/datasets/{train,dev,holdout,test}.parquet`와 manifest가 있으면 원본·산출물 해시를 확인한 뒤 **기존 round17 피처/정답을 재사용**한다. 없으면 동일 builder로 월별 생성한다. round10과 수정량만 추가한다.
- 피처 계약: `configs/deep_learning/feature_contract.json`은 실제 3차 manifest에서 추출한 B 계열 예보 allowlist와 출처 해시다. 서버에 대형 3차 피처 파일이나 LightGBM/CatBoost 설치는 필요 없다.
- 정산: 기존 `validation.EvaluationWindow`, `aggregate_site_predictions`, `score_groups`, `hourly_weights`, `validate_composition`을 직접 호출한다. 과거의 Jan–Mar Dev/Apr–May Holdout 고정 상수는 새 분할에 사용하지 않고, 3차의 운영일 fold를 사용한다.
- 기준 배정: 3차와 같은 용량 기준 균형 배정을 `sites.csv`에서 결정적으로 재생성한다. 기존 `reports/expanding/composition.csv`가 있으면 일치 여부도 검사한다. 그룹별 최소 20,000 kW 및 253개소의 정확한 분할을 검증한다. **3차의 OOF 최적화 배정을 기본 학습에 가져오지 않는다.** 그 배정은 2025-02-28까지의 정답을 이용했으므로 더 이른 학습에 쓰면 안 된다.

참고 기준은 3차 `B_1m_revision + train_calibrated`의 봄 CF MAE 약 0.03587, 균형 배정 페널티 약 50,896,936원, OOF 최적화 배정 페널티 약 50,843,982원이다. **이는 기존 LightGBM 결과이며 새 딥러닝 결과가 아니다.** 평가 기간, 관측 마스크, 배정, 학습 종료, 후처리가 다르면 직접적인 개선율을 계산하지 않는다. 기본 DL 학습은 과거 전체 유효 관측을 사용하므로 100만 행 LightGBM과 학습량도 다르다.

## 2. 모델과 실제 파라미터 수

외부 사전학습 가중치는 사용하지 않는다. 수치는 예보 90개, 시간 피처 8개, 정적 입력 146개, 253개 발전소 기준의 실제 `sum(p.numel())`이다. 전체 크기 열거는 meta device에서 수행해 큰 학습을 실행하지 않는다.

| 모델 | small | base | large |
|---|---:|---:|---:|
| ConvNeXt 1D | 2,015,793 | 5,214,241 | 9,979,009 |
| Weather Transformer | 7,950,657 | 16,176,385 | 29,785,281 |

Cross-Site base는 **3,991,233개**, Cross-Site Attention을 끈 Shared CNN ablation은 **2,411,713개**다. Group Loss fine-tuning은 base와 같은 파라미터 수다. 설정별 상세는 [parameter_counts.json](parameter_counts.json)에 있다.

- CNN: depthwise Conv1d → LayerNorm → pointwise expansion/GELU/projection → residual. Site Embedding과 Static Encoder를 시간별 표현에 결합한다. kernel/width/depth/expansion/dropout을 YAML로 바꿀 수 있다.
- Transformer: weather projection + 학습 위치 인코딩 + encoder layers + 정적/발전소 조건 + 시간별 head. d_model/heads/layers/feedforward/dropout을 설정한다. 24시간 예보 전체에 양방향 attention을 허용한다.
- Cross-Site: shared CNN으로 각 발전소의 24시간을 인코딩하고 시간 평균으로 만든 **253개 토큰**에만 attention을 적용한다. 6,072개 토큰의 full attention은 없다. 시간별 표현과 site context를 decoder에서 결합한다. `cross_site_cnn_only.yaml`, `cross_site_base.yaml`, `cross_site_group.yaml`로 세 구조를 구분한다.

세 모델 모두 Site Embedding dropout으로 정적 피처 의존 경로를 학습한다. 학습 이력이 없는 ID는 임의의 미학습 임베딩 대신 **고정된 0번 zero embedding**을 사용한다. 학습에서 보지 못한 기상지점의 one-hot은 0 벡터로 처리하며, 학습 중 station one-hot dropout도 적용한다.

### 공통 텐서 계약

| 항목 | CNN / Transformer | Cross-Site |
|---|---|---|
| weather | `[B,24,90]` | `[B_days,253,24,90]` |
| weather_missing (bool) | 동일 shape | 동일 shape |
| temporal | `[B,24,8]` | `[B_days,253,24,8]` |
| static | `[B,146]` | `[B_days,253,146]` |
| site_index / metadata_valid | `[B]` | `[B_days,253]` |
| CF 예측 / target / loss_mask | `[B,24]` | `[B_days,253,24]` |
| group_complete (bool) | 정산 감사용 `[B,24]` | `[B_days,24]` |

weather 90개는 원래 30개 round17 + 30개 round10 + 30개 `(17−10)` 수정량이다. 모델 내부에서 weather와 90개 결측 플래그, 시간 8개를 연결해 188개 채널을 투영한다. 시간 입력은 hour/month/dayofyear, 시·연중 sin/cos, 근사 태양고도다. static 146개는 용량·좌표·거리 5개와 해당 결측 플래그 5개, WA/WB 지점 one-hot 136개(unknown 예약 열 포함)다. 실제 사이트 순서는 manifest의 정렬된 `site_ids`이며, embedding ID와 출력 위치 `site_pos`는 분리한다. metadata_valid는 용량·위경도의 유한성을 뜻하며, 거리·지점 결측은 별도 플래그/unknown 인코딩으로 처리한다. 용량은 기존 validator로 양수도 확인한다. 이 마스크는 타깃 관측 여부와 독립적이다.

## 3. 데이터 준비와 누수 방지

운영일 D는 D 01:00부터 D+1 00:00까지다. 발행 시각은 D−1 17:00 KST이며 round10은 같은 전일 10시다. `source_audit.jsonl`에 월별 발행 계약과 키 검사 결과를 기록한다. 실제 예보 갱신 시각 열은 없으므로 제공 README의 발행 계약을 따른다. 좌표는 제공된 비식별 좌표이며 태양고도는 기존 코드와 같은 근사식이다.

모델 입력은 explicit allowlist만 허용한다. energy/CF 정답/SMP/k/관측 여부/그룹 완전성/발전 이력은 모델에 넣을 수 없다. SMP와 k는 손실과 평가에서만 사용한다. 실제 기상 관측 파일은 새 pipeline에서 읽지 않는다.

정규화 평균/표준편차는 **해당 학습 종료 전의 유효 학습 샘플**에서만 산출한다. 정적 수치 통계는 그 시점에 관측 이력이 있는 사이트에서 계산한다. NaN 피처는 학습 평균으로 대체해 표준화 값 0이 되며, 기상·정적 결측 플래그를 별도로 제공한다. 기상 원단위는 유지한다(A 풍속 m/s, B km/h; A 적설 mm, B cm). 검증/test 통계로 채우지 않는다.

타깃은 CF이며 원래 energy와 CF를 별도 float64 cache로 보존한다. 학습에서는 음수/CF>1과 정보 마감 이후 정답을 제외한다. 마지막 부분 운영일의 18~24시 미래 정답은 **학습 batch에서도 NaN으로 차단**한다. 검증에는 원본 유효 정답을 사용한다. 최초 관측 전, 내부 공백, 마지막 관측 후, 이력 없음, test 무정답은 `missing_kind`로 구분한다. 캐시나 batch의 NaN 타깃을 0으로 보간하지 않는다. 손실 내부 `where(mask, y, 0)`는 마스킹 산술용 임시 값이다.

캐시는 월별로 만들며 `.npy` memory map을 worker에서 읽기 전용으로 다시 연다. training worker에 전체 pandas 데이터프레임을 복제하지 않는다. 4.4년+test의 90개 weather cache는 압축하지 않은 디스크 배열이므로 전체 캐시에 약 5 GB 전후가 필요하다(피처/기간에 따라 변동). GPU 메모리와는 다른 수치다. 중단된 준비 작업은 manifest가 없는 불완전 캐시로 남으며 자동으로 덮어쓰지 않는다. 새 cache 경로를 지정한다.

### 기본 개발/최종 평가 구분

기본 YAML의 outer fold는 3차 Fold 5(2025년 3~5월)다.

| 단계 | 정답 학습 종료 | 선택/평가 |
|---|---|---|
| development | 2025-01-31 17:00 | 운영일 2/1~2/27의 dev로 early stopping/설정 선택 |
| finetune | 같은 1/31 17:00 | site pretrain에서 시작, group loss 별도 실험; 같은 dev로 선택 |
| refit | 2025-02-28 17:00 | dev가 선택한 epoch·단계별 optimizer/loss recipe를 처음부터 재학습; early stopping 없음 |
| production | 2025-05-31 17:00 | 같은 고정 recipe를 test 첫 발행 직전까지 재학습; test 정답 사용 없음 |

봄을 평가할 때는 refit checkpoint를 사용한다. production checkpoint로 봄을 다시 평가하는 것은 거부한다. 개발 점수는 모델 선택에 사용했으므로 독립적인 성능이 아니다. **2025년 봄은 이미 여러 차례 관찰한 구간이므로 완전히 새로운 미관측 holdout도 아니다.**

`train --fold 0..5`는 3차의 6개 outer fold를 사용하고, 각 outer 첫 발행 전에 끝나는 내부 월별 dev를 설정한다. 원래 YAML의 `dev_start/dev_end`를 직접 조정할 수도 있다. 정산 가능한 시각이 없는 dev에서 `penalty_per_hour` early stopping은 실패한다. 이때 명시적으로 `selection_metric: site_cf_mae`를 설정하거나 더 적절한 과거 개발 구간을 선택한다. 미래 구간의 정답을 가져오지 않는다.

Group Loss는 **253개소 전부의 사용 가능한 학습 정답**이 있는 시각에만 적용한다. 기본 site loss pretrain 후 별도 fine-tuning config를 사용한다. 해당 cutoff 전에 완전 관측 시각이 없으면 group fine-tuning은 중단한다. 더 이른 fold에서 이 문제가 생길 수 있다. 기본 배정은 정답 없이 만든 균형 배정이다. custom loss 배정은 `composition_path`와 `available_at`을 요구하며, 학습 마감 이후 정답으로 선택한 배정은 거부한다. 평가 배정은 모든 모델에서 cache의 동일한 기준 배정을 사용한다.

## 4. 손실, 학습 및 체크포인트

- Site: L1, Huber/δ, SMP×k×capacity 가중 L1. Huber는 δ로 나눠 CF 스케일로 정규화한다.
- Group: `|Σ capacity×(predCF−trueCF)| / group_capacity`를 완전 관측 시각에만 사용한다. 가격·시간 가중치와 group_capacity로 가중 평균해 CF 스케일로 맞춘다. 최종 `alpha*site + beta*group`, 기본 alpha=1/beta=0, group fine-tuning 예시는 beta=0.2다.
- k는 **구간 끝 시각의 시가 11~15이면 3**, 나머지 1이다. 00시를 운영일로 옮긴 시가 아니라 기존 `hourly_weights()`를 따른다.
- AdamW, warmup+cosine scheduler, clipping, gradient accumulation, early stopping, 선택적 non-reentrant gradient checkpointing을 지원한다.
- CUDA `auto` precision은 실제 `is_bf16_supported()`를 검사해 BF16 또는 FP16을 선택한다. FP16에는 GradScaler를 사용한다. 손실 합산은 float32다.
- NaN/Inf 입력·출력·손실·gradient는 중단한다. 유효 supervised mass가 없는 batch는 업데이트하지 않는다.
- `steps.jsonl`/`epochs.jsonl`에 train loss, raw-forward validation loss, CF/정산 검증 지표, LR, 단계, GPU peak allocated/reserved를 기록한다.
- `last.pt`/`best.pt`는 atomic save다. model/optimizer/scheduler/scaler, Python/NumPy/Torch/CUDA RNG, shuffle generator와 batch cursor, normalizer, recipe/선택 출처를 저장한다. `weights_only=True`로 읽는다.
- 재개는 같은 config/data/정밀도에서 허용한다. batch/LR 등을 바꾸면 새 실험을 만든다. 동일 CPU 환경에서 중간 1-update checkpoint를 재개한 2-update 결과와 연속 2-update 가중치가 정확히 일치함을 검사했다. GPU/다른 버전 간 bitwise 일치는 아직 확인하지 않았다.
- group fine-tuning은 optimizer를 새로 시작한다. refit에서는 site pretrain과 group fine-tuning의 **각 epoch 수와 각 optimizer 설정**을 보존한다.

평가 지표는 site CF/kWh MAE, 그룹별 MAE, 공식 산식의 페널티 W/정산금 S, 신규/기존(학습 관측 2,160시간 기준), 주간/야간/황혼, 계절별 MAE다. 완전 관측 시각이 없으면 W/S는 `null`, `score_available=false`이며 0점으로 기록하지 않는다. 모델/배정마다 관측 마스크를 바꾸지 않는다. 기본 후처리는 3차와 같은 training-only night calibration이며, `postprocess: none/night_only/train_calibrated`로 바꿀 수 있다.

## 5. 서버에서 처음 실행할 순서

아래는 **사용자가 이미 SSH로 서버에 접속하고 자신의 저장소 디렉터리로 이동한 상태**에서 실행한다. 서버 주소·키·원격 경로는 포함하지 않는다. Codex는 접속하지 않았다.

### 5.1 코드와 환경

```bash
git pull --ff-only
python3 --version
nvidia-smi
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
```

드라이버와 호환되는 wheel 인덱스/버전을 [공식 PyTorch 설치 선택기](https://pytorch.org/get-started/locally/)에서 확인한다. 로컬 CPU 검사는 torch 2.15.0/Python 3.13에서 했지만 서버 CUDA wheel의 버전이나 드라이버를 추측해 고정하지 않았다. 구현은 torch >=2.5,<3 API를 사용한다.

```bash
read -r -p '공식 CUDA wheel 인덱스 URL: ' PYTORCH_INDEX_URL
export PYTORCH_INDEX_URL
read -r -p '그 인덱스에서 지원하는 torch 버전: ' OIBC_TORCH_VERSION
export OIBC_TORCH_VERSION
scripts/gpu/install_dependencies.sh
```

`PYTORCH_INDEX_URL`은 `https://download.pytorch.org/whl/` 아래의 실제 공식 CUDA 인덱스를 입력한다. torchvision/torchaudio, 외부 모델, 별도 학습 데이터는 필요 없다. 설치 환경은 `/tmp/oibc-gpu-environment.txt`에 저장된다. 보존하려면 `OIBC_ENV_LOCK`을 원하는 파일로 설정한다.

### 5.2 데이터와 캐시

```bash
read -r -p '제공 sites.csv가 있는 실제 데이터 디렉터리: ' OIBC_DATA_ROOT
export OIBC_DATA_ROOT
export OIBC_CACHE_ROOT="$PWD/analysis/deep_learning/cache"
export OIBC_RUN_ROOT="$PWD/runs/deep_learning"
export OIBC_PYTHON="$PWD/.venv-gpu/bin/python"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
scripts/gpu/check_environment.sh
python -m analysis.deep_learning.prepare
python -m analysis.deep_learning.check_data --scan --verify-originals
```

데이터가 Git에 없으면 제공 파일을 사용자의 실제 서버 데이터 경로에 먼저 복사한다. 누락 파일은 `prepare --check-only`가 전체 경로와 준비 방법을 출력하며 자동 생성하지 않는다. 완성된 train/test 캐시를 옮긴 경우 원본 없이도 `check_data --scan`과 학습은 가능하다. 이 경우 원본이 필요한 `prepare --check-only`/`--verify-originals`는 생략하고, 저장된 source hash를 함께 보존한다.

선택적으로 원격 CUDA의 **tiny synthetic** 점검을 먼저 한다. 이 명령도 성능 실험은 아니다.

```bash
python -m analysis.deep_learning.smoke --device cuda --precision auto --output runs/deep_learning/cuda_smoke
```

### 5.3 tmux와 모델 학습

```bash
tmux new -s oibc_gpu
source .venv-gpu/bin/activate
scripts/gpu/train_cnn.sh
scripts/gpu/train_transformer.sh
scripts/gpu/train_cross_site.sh
```

또는 `scripts/gpu/train_all.sh`로 세 기본 모델을 순서대로 실행한다. tmux는 `Ctrl-b d`로 분리하고 `tmux attach -t oibc_gpu`로 다시 연결한다. 새 tmux shell에서 가상환경·데이터 환경변수가 유지되는지 확인한다. 하나의 GPU에서 train/evaluate/predict는 같은 process lock을 사용한다. 이미 실행 중이면 다른 명령은 중단해 동시 학습 OOM을 방지한다.

시작 batch는 CNN 128, Transformer 64(accumulation 2), Cross-Site 2일(accumulation 4)이다. **24 GB에서의 실제 peak memory/처리량은 아직 측정하지 않았다.** large 모델이나 worker/AMP 조건에 따라 조절한다.

```bash
tail -f "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/epochs.jsonl"
python -m analysis.deep_learning.train --config configs/deep_learning/cnn_base.yaml --resume "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/last.pt"
```

가중치·로그는 모델/config name·seed·outer fold·phase별 폴더로 분리된다. 다른 설정을 시험할 때 YAML의 name을 바꾸거나 `--run-dir`을 지정한다. `*.local.yaml`은 Git에서 제외돼 서버 전용 설정에 사용할 수 있다.

### 5.4 그룹 손실 별도 fine-tuning

```bash
python -m analysis.deep_learning.train \
  --config configs/deep_learning/cross_site_group.yaml \
  --init-checkpoint "$OIBC_RUN_ROOT/cross_site_base_seed2026_fold5_development/best.pt"
```

base site pretrain과 fine-tuning을 dev에서 비교해 채택 여부를 결정한다. 그룹 손실이 항상 개선된다고 가정하지 않는다. `cross_site_cnn_only.yaml`로 cross-site attention 없는 대조군도 학습할 수 있다.

### 5.5 개발 평가, 고정 recipe 재학습, 봄 평가

```bash
python -m analysis.deep_learning.evaluate \
  --checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/best.pt" --split dev

python -m analysis.deep_learning.train \
  --config configs/deep_learning/cnn_base.yaml --phase refit \
  --selection-checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/best.pt"

python -m analysis.deep_learning.evaluate \
  --checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_refit/best.pt" --split holdout
```

Transformer/Cross-Site도 해당 config/경로로 동일하게 실행한다. Group Loss 후보를 선택했다면 group config와 fine-tuned selection checkpoint를 사용해 두 단계를 재학습한다. `scripts/gpu/evaluate_models.sh checkpoint1.pt checkpoint2.pt ...`는 기본 dev 평가이며, refit 모델의 봄 비교에는 `OIBC_EVAL_SPLIT=holdout`을 설정한다. 봄 평가 후 설정을 고르면 그 구간을 최종 독립 평가로 재사용할 수 없다.

### 5.6 앙상블 가중치 선택과 적용

```bash
python -m analysis.deep_learning.ensemble fit \
  --checkpoints \
  "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/best.pt" \
  "$OIBC_RUN_ROOT/transformer_base_seed2026_fold5_development/best.pt" \
  "$OIBC_RUN_ROOT/cross_site_base_seed2026_fold5_development/best.pt" \
  --output "$OIBC_RUN_ROOT/ensemble_dev.json"
```

dev 공식 페널티/시간으로 nonnegative simplex 가중치를 고정한다. holdout/refit/production checkpoint로 가중치를 고르는 것은 거부한다. 적용 시 대응하는 refit/production checkpoint의 `selected_dev_signature`가 일치해야 한다.

```bash
python -m analysis.deep_learning.ensemble predict \
  --weights "$OIBC_RUN_ROOT/ensemble_dev.json" \
  --checkpoints \
  "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_refit/best.pt" \
  "$OIBC_RUN_ROOT/transformer_base_seed2026_fold5_refit/best.pt" \
  "$OIBC_RUN_ROOT/cross_site_base_seed2026_fold5_refit/best.pt" \
  --split holdout --output "$OIBC_RUN_ROOT/ensemble_holdout.parquet"
```

앙상블은 모델을 하나씩 GPU에 올려 예측하므로 세 모델 가중치를 동시에 GPU에 유지하지 않는다. 고정 가중치로 만든 site 예측과 group 예측을 보존하며, 이후 가중치를 봄 정답으로 재탐색하지 않는다.


고정 앙상블 자체의 봄 평가 CLI도 제공한다. `ensemble fit`에 사용한 dev component와 연결되는 refit checkpoint를 같은 순서로 전달한다.

```bash
python -m analysis.deep_learning.ensemble evaluate \
  --weights "$OIBC_RUN_ROOT/ensemble_dev.json" \
  --checkpoints \
  "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_refit/best.pt" \
  "$OIBC_RUN_ROOT/transformer_base_seed2026_fold5_refit/best.pt" \
  "$OIBC_RUN_ROOT/cross_site_base_seed2026_fold5_refit/best.pt" \
  --split holdout --output "$OIBC_RUN_ROOT/ensemble_holdout_metrics.json"
```


### 5.7 제출용 재학습·추론

모델/recipe를 dev에서 정한 상태로 제공 역사 자료를 test 첫 발행까지 확장한다.

```bash
python -m analysis.deep_learning.train \
  --config configs/deep_learning/cnn_base.yaml --phase production \
  --selection-checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/best.pt"

python -m analysis.deep_learning.predict \
  --checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_production/best.pt" \
  --split test --output "$OIBC_RUN_ROOT/cnn_test.parquet"
```

생성물은 site `.parquet`, group `.forecast.csv`, `.composition.csv`, 추론용 `.npy`, 메타 `.json`이다. 공식 test는 G1/G2 각각 SMP의 8,760시각, timezone-aware ts, 비음수 forecast_kwh를 검증한다. **test 평가 점수는 계산하지 않는다.** 앙상블 test 추론은 각 production checkpoint를 같은 고정 ensemble JSON에 연결한다. 로그·환경 lock·config.json·normalizer와 checkpoint를 함께 보관한다.

## 6. 로컬 확인 범위와 재현

```bash
.venv/bin/python -m unittest discover -s analysis -p 'test_*.py' -v
.venv/bin/python -m analysis.deep_learning.smoke --extended --output runs/deep_learning/fresh_cpu_smoke
.venv/bin/python -m analysis.deep_learning.environment --parameters
```

로컬에서는 실제 CUDA가 없으며 CPU float32만 확인했다. 일반 CPU 학습 명령은 거부한다. `--smoke-test --device cpu --max-steps 1/2`만 허용하고, synthetic/debug checkpoint의 실험 승격도 제한한다. 정상 데이터가 없을 때 이 synthetic fixture를 자동으로 사용하는 경로는 없다.

검사 범위는 세 모델/tiny·실제 small config의 dummy forward/backward, 결측 타깃 gradient, 완전 관측 group loss, 미래 정보 마감, 정상 00시 경계, 253개소 입력 순서, all-invalid metadata attention, zero unknown embedding, worker mmap 재개방, 체크포인트·RNG 재로드, 실제 CLI 1→2 update 재개와 연속 실행 일치다. 추가로 refit/production/test, group fine-tuning 및 2단계 recipe, dev ensemble/frozen-refit 추론을 synthetic 입력으로 검사했다. 진단 loss와 테스트용 정산 수치는 실제 모델 성능으로 보고하지 않는다.

실제 제공 데이터에서는 **2일·12,144 site-hours의 전처리만** 실행했다. 기존 Parquet 재사용 경로와 원본 builder 경로의 weather/시간/정답/가격/가중치/결측 분류 배열이 정확히 일치했다. 해당 실제 데이터로 학습하지 않았다. [real_feature_parity.json](real_feature_parity.json), [unit_tests.log](unit_tests.log), [local_verification.json](local_verification.json)을 참조한다.

## 7. 예상 환경 문제

| 현상 | 조치 |
|---|---|
| `torch.cuda.is_available() == False` | CPU wheel/드라이버/컨테이너 GPU 노출 여부를 확인하고 공식 selector로 맞는 CUDA wheel을 설치한다. 서버에 대한 자동 수정·접속은 없다. |
| CUDA wheel 설치 불가 | 선택 버전이 해당 Python/인덱스에 있는지 확인한다. 로컬 macOS wheel을 Linux 서버에 복사하지 않는다. |
| CUDA/AMP kernel 오류 | `environment --cuda-probe`와 tiny CUDA smoke로 먼저 확인한다. BF16 미지원은 FP16으로 전환하며 실제 dtype을 environment.json에서 확인한다. |
| GPU OOM | 모델별 batch를 줄이고 accumulation을 늘린다. gradient checkpointing을 켜고 다른 작업이 GPU를 점유하는지 확인한다. config 변경은 새 실험으로 수행한다. |
| DataLoader 메모리/worker 오류 | num_workers를 0~2로 낮춰 점검한다. worker는 mmap을 다시 열며 전체 pandas 데이터는 포함하지 않는다. |
| 파일·manifest 없음 | 원본 또는 완성 cache를 실제 경로에 준비한다. 명시된 누락 목록을 해결한 뒤 진행한다. synthetic 대체는 하지 않는다. |
| 정산 선택 지표 없음 | 완전 관측 시각이 있는 과거 dev를 쓰거나 site_cf_mae를 선택한다. 점수 0으로 대체하지 않는다. |
| group fine-tuning 불가 | 학습 마감 전에 253개소 완전 관측 시각이 있는지 확인한다. 없으면 site pretrain만 가능하다. |
| Resume signature/AMP 불일치 | 원래 설정·cache·정밀도로 재개한다. 설정이 바뀌면 새 run으로 시작한다. |
| deterministic CUDA 연산 오류 | CUBLAS_WORKSPACE_CONFIG와 설치 버전을 확인하고 해당 kernel의 결정론 지원을 점검한다. 검증 없이 재현성을 보장한다고 하지 않는다. |

PyTorch API 근거: [공식 AMP 문서](https://docs.pytorch.org/docs/stable/amp.html), [gradient checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html), [BF16 지원 검사](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html), [안전한 checkpoint load](https://docs.pytorch.org/docs/stable/generated/torch.load.html). GPU kernel/VRAM/최적 batch/실제 성능은 원격 서버에서 확인할 사항이다.

## 8. 파일 안내

- `analysis/deep_learning/config.py`, `features.py`, `prepare.py`, `dataset.py`, `check_data.py`: config·피처 계약·월별 cache·메모리 매핑·사전 검사.
- `models/common.py`, `convnext1d.py`, `weather_transformer.py`, `cross_site_attention.py`: 공통 encoder와 3종 모델.
- `losses.py`, `runtime.py`, `checkpoint.py`, `train.py`: 마스킹·학습·AMP·재현·재개·단계별 recipe.
- `metrics.py`, `inference.py`, `evaluate.py`, `predict.py`, `ensemble.py`: 공식 평가·공통 추론·제출 검증·개발 전용 앙상블.
- `environment.py`, `smoke.py`, `tests/test_pipeline.py`: 환경/규모 확인·명시적 synthetic CLI 검사·단위 테스트.
- `configs/deep_learning/*.yaml`, `feature_contract.json`: 9종 설정과 기존 manifest에서 추출한 피처.
- `scripts/gpu/*.sh`: 설치·환경 점검·모델별/순차 학습·평가(실행 권한 설정).
- `.gitignore`: 새 cache, runs, checkpoint, 가상환경, local config를 제외한다. 기존 추적 파일을 삭제하거나 추적 해제하지 않았다.

Git push/commit과 SSH 접속은 사용자가 수행한다. 학습 결과와 큰 checkpoint는 기본적으로 Git 대상에서 제외된다.

GPU 최소 의존성만 설치한 서버에서 단위 테스트를 실행하려면 `python -m unittest discover -s analysis/deep_learning/tests -v`를 사용한다. 저장소 전체의 기존 테스트에는 별도의 LightGBM/CatBoost 의존성도 필요하다. 재현 시 같은 코드 버전을 사용한다. 각 run의 environment.json에 패키지 버전과 소스 파일 해시도 저장된다.

로컬 sandbox에서는 PyTorch `torch_shm_manager`의 IPC가 차단될 수 있다. 이 경우 worker 테스트는 공유 메모리가 허용된 환경에서 실행하고, 제한 환경의 smoke는 `num_workers=0`으로 수행한다. 컨테이너 서버에서도 worker 오류가 나면 공유 메모리/IPC 제한을 확인한다. 이를 모델 또는 CUDA 학습 성공으로 해석하지 않는다.
