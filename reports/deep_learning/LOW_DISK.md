# 12GB 개인 저장 공간용 데이터 파이프라인

모델 3종과 입력 계약은 그대로 두고 기본 준비 형식을 `station_month_v1`로 변경했다. 원본 데이터 삭제, 기존 캐시 변환·삭제, 원격 접속, 전체 데이터 학습은 실행하지 않았다. 기존 legacy 캐시도 읽을 수 있다.

## 저장량과 동작

실제 제공 SMP의 전체 운영일 수와 sites의 관측소 매핑으로 계산한 **예상치**다. 전체 캐시는 로컬에서 생성하지 않았다. .npy 헤더·manifest를 위한 여유분을 포함한다. GB는 10⁹ bytes, GiB는 2³⁰ bytes다.

|구분|기존 발전소별 mmap|월별 관측소 캐시|
|---|---:|---:|
|Train|4,005,068,576 bytes|730,400,192 bytes|
|Test|907,669,560 bytes|166,196,880 bytes|
|합계|4.913 GB / 4.575 GiB|0.897 GB / 0.835 GiB|

약 **81.7% 감소**한다. 예상치가 2GiB 미만이므로 원본을 매 배치마다 읽는 별도 Parquet streaming 경로는 필요하지 않다. 대신 월별 mmap을 필요 시 열고 worker마다 최근 2개월의 mmap 핸들만 유지한다. 운영일·날짜 범위를 `--start`/`--end`로 제한해 준비할 수도 있다. 검증일이 빠진 캐시는 기존 검증의 연속 날짜 검사를 통과하지 못한다.

- WA/WB별 `[일, 관측소, 24, round17 변수 + round10 변수]` float32를 한 번만 저장한다. 같은 관측소의 발전소 값이 실제로 다르면 저장을 중단한다.
- 90개 모델 피처는 기존 순서로 모은다. 수정량은 기존과 같은 **float32 subtraction**으로 계산한다. 기상 NaN 자체를 저장해 별도 결측 마스크 파일은 만들지 않는다. 배치에서 기존 bool 마스크를 생성한다.
- 8개 시간·태양고도 피처는 기존 `temporal_features`를 그대로 호출한다. 정적 자료와 발전소 순서도 그대로다.
- 원본 energy는 공식 평가의 정밀도를 보존하기 위해 float64를 유지하고, CF는 기존처럼 float64 energy / 원래 용량으로 복원한다. NaN은 그대로 남는다. SMP도 float64를 유지한다. `missing_kind`와 공식 k는 uint8이다. float16 저장은 사용하지 않는다.
- CNN/Transformer weather `[B,24,90]`, Cross-Site weather `[B_days,253,24,90]`를 유지한다. 기상·메타데이터·정답·집합완전관측 마스크를 합치지 않는다.
- D 01:00~D+1 00:00, D-1 17:00 KST, train-only normalization, as-of 검사, 공식 정산 함수는 변경하지 않았다.

## 저장 공간 제한

기본값은 개인 quota 12GiB, cache 2GiB, 결과 4GiB, 파일시스템 및 quota의 비상 여유 0.5GiB다. 서버 제한이 **십진수 12GB**면 아래 명령처럼 `OIBC_QUOTA_GIB=11.175870895`를 사용한다. 결과 cap은 기본/설정된 `OIBC_RUN_ROOT` 전체에 적용한다. 별도 명시 출력 경로는 그 부모 디렉터리에 같은 cap을 적용하며, 어느 경우에도 개인 quota 검사는 유지한다. 결과는 되도록 한 `OIBC_RUN_ROOT` 아래 모은다.

`OIBC_STORAGE_ROOT`는 **자신의 quota에 속하는 파일을 모두 포함하는 개인 디렉터리**로 지정한다. 공유 상위 디렉터리나 다른 사용자의 디렉터리를 지정하면 안 된다. 기본은 저장소 디렉터리다. 경로 밖에 있는 자신의 quota 사용량은 `OIBC_EXTERNAL_USED_GIB`에 별도 합산할 수 있다. quota를 서버 API에서 자동 조회하지는 않는다. 실제 서버 `quota -s`/관리자 수치와 대조해야 한다. 심볼릭 링크를 따라 다른 디렉터리를 스캔하지 않는다. 원본이 공유 읽기 전용 위치라 개인 quota에 포함되지 않는 경우에도 원본은 그대로 읽으며 캐시·결과 출력만 개인 경로에 쓴다.

준비 시작 전 전체 예상 바이트와 현재 사용량, 실제 파일시스템 여유를 검사한다. 월 저장 직전에도 검사한다. 부족하면 명확한 오류로 중단한다. manifest는 원본 해시 재확인 뒤에만 활성화한다. 기존 캐시가 있으면 덮어쓰지 않는다. 중단된 자신의 부분 캐시는 자동 삭제하지 않으므로 확인 후 새 경로를 선택하거나 직접 정리한다.

학습은 기존 `last.pt`/`best.pt`만 갱신하고 epoch별 체크포인트를 만들지 않는다. Adam 상태를 포함한 두 체크포인트와 atomic 교체용 임시 파일 공간을 사전 예약 검사한다. 각 실제 체크포인트 저장도 tensor 바이트에 여유를 더해 재검사한다. 저장 실패 시 기존 last/best를 보존한다. 로그는 파일당 64MiB 상한이며 예측 mmap과 제출 export도 쓰기 전에 크기를 검사한다. 다른 사용자 파일, 원본, 기존 실험 파일을 자동 정리하는 기능은 없다. quota 검사는 검사 사이에 다른 프로세스가 동시에 공간을 차지하는 것을 예약·차단하지는 못한다. 실제 OS quota가 최종 제한이다.

## 검증 결과

[train parity](compact_train_parity.json): 실제 2025-01-31~02-02의 3일, 18,216 site-hours. 월 경계를 포함하고 train/dev 마스킹을 각각 검사했다. [test parity](compact_test_parity.json): 실제 2025-06-01~02의 2일, 12,144 site-hours. legacy와 compact를 같은 원본 builder로 만들었다. [초기 검사](compact_initial_parity.json)는 기존 round17 Parquet 재사용 경로와도 2일 동등성을 확인했다.

weather/temporal/target/energy/SMP/k/missing_kind 최대 절대 오차는 **전부 0**, NaN 위치·정규화 통계·정규화된 Tensor·모든 배치 마스크는 정확히 같다. Test 정답은 전부 NaN이며 점수를 만들지 않았다. 소규모 캐시의 `check_data --scan --verify-originals`도 통과했다.

[단위 테스트](compact_unit_tests.txt)는 기존 75개와 추가 13개를 합쳐 **88개 통과**했다. 모델 3종에 같은 배치를 넣은 forward/backward 값과 gradient 동등성, 부분 정답과 cutoff, 월별 mmap pickle/spawn worker, 관측소 중복 불일치 거부, 디스크 부족/외부 경로/예산/체크포인트 보존을 포함한다. 이 결과는 GPU 학습 성능 검증이 아니다.

실제 소규모 디스크 측정은 train 3일 **7,521,995 → 1,430,837 bytes**, test 2일 **5,038,026 → 977,457 bytes**였다. 전체 예상치의 근거는 [용량 dry-run](compact_storage_estimate.json)에 저장했다.

|CPU warm read, 5회 중앙값|기존 ms/sample|개선 형식 ms/sample|
|---|---:|---:|
|dev site-day|0.177|1.308|
|dev fleet-day|2.686|7.087|
|test site-day|0.173|1.291|
|test fleet-day|2.446|7.040|

로딩에는 역정규화가 아니라 **원래 입력 복원 계산**이 추가된다. parity JSON의 타이밍은 소수 샘플의 단일 CPU 프로세스 warm read이므로 GPU 처리량을 나타내지 않는다. 디스크 읽기는 줄지만 CPU 로딩 시간은 증가한다. 기존 `num_workers=4`, pinned memory를 유지하고 실제 서버에서 worker 수를 조절한다. 각 worker는 최대 2개월의 파일을 mmap하며 전체 253개 발전소 weather Tensor를 프로세스 시작 때 복제하지 않는다. train sample 인덱스·정규화 적합에 필요한 정답은 날짜별로 읽는다.

## 서버 실행 명령

GitHub에 push한 후 서버의 저장소 디렉터리에서 기존 CUDA smoke를 통과한 가상환경을 활성화한다. 아래 입력값은 자신이 아는 실제 경로를 넣는다. 코드는 SSH나 다른 서버 경로를 추측하지 않는다.

```bash
# 이미 tmux 안이라면 새 세션 명령은 생략한다. 아래 환경변수는 세션 안에서 설정한다.
tmux new -s oibc-lowdisk
git pull --ff-only
read -r -p 'GPU smoke를 통과한 가상환경 디렉터리: ' OIBC_VENV
source "$OIBC_VENV/bin/activate"
read -r -p '개인 quota에 속한 파일들을 포함하는 개인 디렉터리: ' OIBC_STORAGE_ROOT
read -r -p '기존 대회 데이터 디렉터리: ' OIBC_DATA_ROOT
export OIBC_STORAGE_ROOT OIBC_DATA_ROOT
export OIBC_CACHE_ROOT="$OIBC_STORAGE_ROOT/oibc-cache-station-v1"
export OIBC_RUN_ROOT="$OIBC_STORAGE_ROOT/oibc-runs"
export OIBC_GPU_LOCK="$OIBC_STORAGE_ROOT/.oibc-single-gpu.lock"
export OIBC_QUOTA_GIB=11.175870895
export OIBC_CACHE_MAX_GIB=2
export OIBC_RESULTS_MAX_GIB=4
export OIBC_MIN_FREE_GIB=0.5
# 개인 경로 밖에서 같은 quota를 쓰는 파일이 있다면 그 사용량(GiB)도 합산한다.
export OIBC_EXTERNAL_USED_GIB=0
mkdir -p "$OIBC_STORAGE_ROOT/tmp"
export TMPDIR="$OIBC_STORAGE_ROOT/tmp"

# 기존 cache 폴더를 재사용하지 않는다. 부족하면 생성 전에 실패한다.
bash scripts/gpu/prepare_compact.sh
```

이 스크립트는 저장 공간 검사 → 전체 크기 dry-run → 월별 캐시 준비 → 원본 해시/캐시 검사를 순서대로 수행한다. prepare는 전체 캐시 크기의 임시 복사본을 만들지 않는다. 날짜별 결과 대신 월별 .npy를 직접 쓴다.

```bash
# 위에서 환경변수를 설정한 같은 tmux 세션 안에서, 한 GPU에 순차 실행한다.
bash scripts/gpu/train_cnn.sh
bash scripts/gpu/train_transformer.sh
bash scripts/gpu/train_cross_site.sh
```

또는 기존 `scripts/gpu/train_all.sh`로 순차 학습한다. 재개는 동일 config/phase/cache로 실행한다.

```bash
bash scripts/gpu/train_cnn.sh \
  --resume "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/last.pt"
python -m analysis.deep_learning.evaluate \
  --checkpoint "$OIBC_RUN_ROOT/cnn_base_seed2026_fold5_development/best.pt" \
  --split dev
```

**기존 legacy 캐시로 학습한 checkpoint의 exact resume에 새 compact 캐시를 바로 끼우지 않는다.** feature signature는 같지만 저장 manifest/signature가 달라 기존 normalizer/cache 및 resume provenance 검사가 거부한다. 새 캐시로 실험을 시작하면 같은 seed/샘플 순서/정규화 입력을 유지한다. 이미 시작한 실험의 재개는 원래 캐시로 하고, 이전 캐시/파일을 임의로 삭제하지 않는다.

작은 범위의 동등성을 서버에서 재현할 때는 서로 다른 새 경로에 `--layout legacy`와 `--layout compact`로 같은 날짜 범위를 준비한 뒤 다음처럼 실행한다. 전체 legacy 캐시를 만들 필요는 없다.

```bash
python -m analysis.deep_learning.prepare --partition train \
  --start 2025-01-31 --end 2025-02-03 --layout legacy \
  --cache-root "$OIBC_STORAGE_ROOT/parity-legacy"
python -m analysis.deep_learning.prepare --partition train \
  --start 2025-01-31 --end 2025-02-03 --layout compact \
  --cache-root "$OIBC_STORAGE_ROOT/parity-compact"
python -m analysis.deep_learning.parity \
  --legacy-cache "$OIBC_STORAGE_ROOT/parity-legacy/train" \
  --compact-cache "$OIBC_STORAGE_ROOT/parity-compact/train" \
  --output "$OIBC_RUN_ROOT/parity.json"
```

테스트 명령은 `python -m unittest discover -s analysis -p 'test_*.py' -v`다. 로컬 macOS 샌드박스는 torch shared-memory IPC를 차단해서 spawn 테스트만 외부 sandbox 실행이 필요했다. 서버에서는 기존 PyTorch/CUDA 환경 그대로 실행한다. 신규 의존성은 없다. 실제 전체 캐시 바이트는 prepare manifest의 storage 및 `check_data` 출력으로 확인한다. 실제 GPU 학습 속도·VRAM·장시간 동작은 이번 로컬 검증 범위에 포함하지 않는다.

## 변경 파일

- 신규: `analysis/deep_learning/compact.py`, `storage.py`, `parity.py`, `tests/test_compact.py`, `scripts/gpu/prepare_compact.sh`, 이 문서와 `compact_*` 검증 JSON/log.
- 수정: `prepare.py`, `dataset.py`, `check_data.py`, `checkpoint.py`, `inference.py`, `predict.py`, `ensemble.py`, `runtime.py`, `train.py`, `tests/test_pipeline.py`, `reports/deep_learning/README.md`.
- 모델 소스, config 아키텍처, `features.py` 계산 함수, `analysis/validation.py`는 수정하지 않았다.
