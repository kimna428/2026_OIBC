# OIBC TSMixer — Google Colab 실행 순서

저장소 루트의 [`train_colab.ipynb`](../../../train_colab.ipynb)를 Google Colab에서 열고 **1~12번을 위에서 아래로** 실행한다. 기본값은 synthetic5step smoke다. 이번 개발 작업에서는 Colab 접속·Drive Mount·T4 실행·전체 학습을 수행하지 않았다.

## 먼저 준비할 것

1. 이번 변경을 검토하고 GitHub에 commit/push한다. 노트북의 `REPO_URL`과 첫 실행 `REPO_REF`를 자신의 저장소/branch에 맞춘다. 기본 branch는 `feature/tsmixer-v0-mac-cuda`다. private 저장소는 별도 인증이 필요하며 token을 노트북·Git에 저장하지 않는다.
2. Colab에서 런타임 유형을 GPU로 선택한다. T4 등 GPU 종류와 사용 가능 시간은 고정 보장이 아니다. 셀5가 실제 GPU·CUDA·VRAM을 확인한다. torch는 Colab의 CUDA 설치를 유지하고 공통 패키지만 설치한다.
3. 실제 학습 시 아래8개 파일을 Drive에 별도로 준비한다. GitHub Clone은 데이터를 전달하지 않는다. 기본 synthetic smoke에는 원본이 필요 없다.

```text
MyDrive/OIBC2026/
├── data/01_participant_dataset/
│   ├── sites.csv
│   ├── smp.csv
│   ├── train/{smp.csv,generation.parquet,forecast_wa.parquet,forecast_wb.parquet}
│   └── test/{forecast_wa.parquet,forecast_wb.parquet}
└── runs/<RUN_ID>/                  # 노트북이 만드는 백업·세션 정보
```

실측 기상2개는 학습 입력이 아니다. 제공 데이터 checksum·schema·행 수는 `reports/tsmixer/data_inventory.json`에 있다. 노트북은 Drive 원본을 수정하지 않고 필요한8개만 로컬로 스트리밍 복사한다.

## 셀 실행 순서

| 번호 | 셀 기능 | 기본 smoke | 전체 development |
|---:|---|---|---|
| 1 | 모드·ID·경로 설정 | `MODE="smoke"`, 5step, `REAL_DATA_SMOKE=False` | `MODE="development"`, `ALLOW_FULL_TRAIN=True`, 새 RUN_ID |
| 2 | Drive Mount·저장된 commit 확인 | 계정 승인 필요 | 동일 |
| 3 | GitHub Clone·정확한 commit checkout | 처음엔 설정 ref, 재접속엔 저장된 commit | 동일 |
| 4 | 공통 의존성 설치·환경 기록 | CUDA torch를 CPU wheel로 바꾸지 않음 | 동일 |
| 5 | GPU/디스크·경로 확인 | GPU 없으면 작은 CPU smoke 가능 | Linux CUDA 필수 |
| 6 | 설정·세션 ID 고정 | smoke 설정 epoch2, 실제 종료는5~10step 제한 | 원래 모델·seed2026·최대30epoch |
| 7 | Drive → 로컬 원본 복사 | 기본은 생략 | SHA-256 검증 후 복사/기존 동일 파일 재사용 |
| 8 | 로컬 compact 캐시 준비 | 기본은 생략 | 기존 prepare·feature contract 재사용 |
| 9 | 마지막 정상 백업 복원 | smoke 상태도 재개 가능 | 완료 epoch부터 재개 |
| 10 | 학습·epoch별 Drive 백업 | 최대5~10 optimizer update | 기존 full-train CLI 호출 |
| 11 | 검증·CSV | 작은 synthetic 평가 정합성 검사 | 기존 dev 평가·발전소/G1/G2 CSV |
| 12 | 최종 Drive 백업·산출물 확인 | checksum 확인 | metrics·CSV·로그·checkpoint 보관 |

모든 코드 셀은 일반 Python이다. `!`/shell 치환에 경로를 넣지 않고 argv 목록으로 전달한다. Notebook에는 실행 출력·계정 정보·token을 포함하지 않았다.

## 작은 smoke부터 실행

셀1의 기본값을 유지하고 전체 셀을 순서대로 실행한다.

```python
MODE = "smoke"
ALLOW_FULL_TRAIN = False
REAL_DATA_SMOKE = False
SMOKE_STEPS = 5
RUN_ID = "tsmixer_colab_smoke_v1"
RESUME = True
```

이 모드는 작은 synthetic fixture만 사용한다. checkpoint reload·학습/복원·백업의 동작 확인용이며 실제 정산 성능이 아니다. 실제 데이터 로더도 확인하려면 **새 ID**로 `REAL_DATA_SMOKE=True`를 지정한다. 이때 원본8개를 로컬로 복사한 후 기존 준비기로 2025-01-30~01-31의 이틀 캐시만 만든다. small-cache의 정규화 통계는 테스트용이며 본 학습 checkpoint로 승격할 수 없다.

## 전체 학습으로 전환

smoke를 통과한 다음 셀1을 다음처럼 바꾸고 **새 RUN_ID**로 1번부터 재실행한다.

```python
MODE = "development"
ALLOW_FULL_TRAIN = True
RUN_ID = "tsmixer_colab_dev_v1"
RESUME = True
```

기본 `configs/tsmixer/colab.yaml`은 모델128/3blocks/dropout0.1, AdamW lr0.001, batch32, accumulation1, worker0, 최대30epoch, seed2026, fp32다. 원래 v0의 일정한 학습률을 유지하면서 constant `LambdaLR`의 상태만 명시적으로 저장한다. T4에서 bf16을 강제하지 않는다. AMP는 검증 후 새 실험 설정에서 선택적으로 사용한다.

OOM이면 새 실험 ID에서 batch를 줄이고 필요하면 accumulation을 늘린다. 설정을 바꾸고 같은 checkpoint를 "정확한 재개"로 취급하지 않는다. GPU·PyTorch·NumPy 버전이 바뀐 재개는 bitwise 동일성을 보장하지 않는다. 필요하면 백업의 환경 기록으로 기존 환경을 재현한다.

전체 학습은 노트북이 기존 명령을 호출한다.

```bash
python -m analysis.tsmixer train --config CONFIG.yaml --device auto \
  --full-train --output LOCAL_RUN --backup-root DRIVE_RUN
```

이 명령을 Mac에서 실행하면 기존 Linux CUDA guard가 거부한다. Mac에서는 smoke/단위 테스트만 실행한다.

## 기존 데이터·평가 계약

- 운영일 D 01:00~D+1 00:00, 예보 사용 가능 시각 D−1 17:00 KST.
- round17 30개 + round10 30개 + revision 30개, 시간8개·정적146개와 기존 InputEncoder.
- 기존 `DayCache`/`DayDataset`/compact 구조, cutoff 이전 통계, target NaN mask, CF `[0,1]` 학습 필터, weighted L1, train_calibrated 후처리 유지.
- 실측 기상·미래 발전량·SMP·k를 모델 입력에 추가하지 않음. SMP/k는 기존 loss·정산에서만 사용.
- G1/G2는 기존 캐시의 고정 용량 균형 composition을 그대로 사용.
- **cutoff=2025-01-31 17:00 KST**, dev 운영일 **[2025-02-01,2025-02-28)**. 마지막 시각은2/28 00시다.
- Holdout 평가나 공식 test 추론은 이 노트북에서 실행하지 않는다.

`final prediction CSV`는 **best checkpoint의 development 예측 CSV**다. test 제출 CSV가 아니다. 현재 단일 development 실험에는 독립적인 cross-validation OOF가 없으므로 OOF라고 이름 붙이지 않는다. metadata에 `independent_oof=false`를 명시한다. dev를 모델 선택에도 사용하므로 독립 일반화 성능으로 해석하지 않는다.

## Checkpoint와 안전 백업

매 완료 epoch마다 로컬 `last.pt`와 개선 시 `best.pt`를 저장한다. checkpoint에는 다음이 포함된다.

- 모델·AdamW optimizer·constant scheduler·AMP scaler 상태
- 완료 epoch·global step·best score·early-stop counter
- Python/NumPy/Torch CPU/CUDA RNG, DataLoader generator, 필요 시 MPS RNG
- train-only normalizer, cache/feature signature, source hash, 설정 fingerprint
- Git commit·환경, dev 결과·선택 종료 시각·smoke 여부

그 뒤 Drive에 고유한 새 generation을 만든다. 파일 복사 → 크기/SHA-256 및 checkpoint 상태 검증 → manifest → 마지막 `COMMITTED.json` marker 순서다. 복사가 중단된 generation은 복원 후보가 아니다. FUSE가 fsync 자체를 지원하지 않는 경우만 재읽기 검증으로 진행하고, 실제 I/O 오류는 무시하지 않는다. 과거 세대를 자동 삭제·덮어쓰기하지 않는다. 재접속 시 최신 정상 세대를 고르며 손상된 최신 세대는 건너뛴다. 모든 세대가 손상됐으면 오류로 멈추고 새 학습으로 조용히 대체하지 않는다.

백업 실패 시 학습도 멈춘다. 로컬 정상 checkpoint와 이전 정상 Drive 세대는 보존한다. 지원되는 파일시스템에서는 `fsync`를 사용하고 항상 재읽기 checksum·완료 marker를 사용하지만 mounted Drive/FUSE의 원격 영속성은 서비스 보장 범위를 따른다. 하드 런타임 종료 시 마지막 미완료 epoch나 아직 Drive에 반영되지 않은 작업까지 복구된다고 보장하지 않는다.

이전 v0 checkpoint와 결과는 수정하지 않는다. 소스 hash 계약과 Scheduler 상태가 달라졌으므로 **기존 v0 실험은 원래 commit에서 재개**하고, 본 변경은 새 Colab 실험에 사용한다.

## 런타임 재접속·Resume

1. 같은 RUN_ID·MODE·설정·로컬 경로 문자열을 유지하고 새 GPU 런타임에서 노트북을 다시 연다.
2. 1~12번을 순서대로 실행한다. 셀2가 `session_ref.json`의 기존 commit을 선택하고 셀3이 그 commit을 checkout한다.
3. 원본은 다시 Drive→로컬 복사하고 캐시는 같은 소스·layout·옵션으로 로컬에서 재생성한다. 원본과 캐시는 Drive checkpoint 백업에 포함하지 않는다.
4. 셀9가 파일·상태·checksum을 검증한 last/best·로그·설정을 새 로컬 run으로 복원한다.
5. 셀10이 마지막 완료 epoch 이후부터 재개한다. 끊긴 epoch는 다시 계산한다. 완료/early-stop 상태면 학습을 반복하지 않고 평가로 이동한다.
6. 셀11은 기존 평가가 best.pt checksum과 맞는지 확인한다. 불완전/이전 best의 평가는 별도 진단 폴더로 보존한 뒤 새 dev 평가를 실행한다.

정확한 재개에는 같은 소스·설정·cache signature가 필요하다. 경로를 바꿔 fingerprint가 달라지거나 원본/캐시가 다르면 거부한다. synthetic smoke는 캐시를 백업하지 않고 같은 fixture를 재생성한다. 부분 epoch에서5→10step으로 확장한 smoke는 exact trajectory를 주장하지 않으며, 완료 epoch 경계의 CPU optimizer/scheduler/RNG 재개는 unit test에서 bitwise 검증했다.

## Drive 산출물

```text
MyDrive/OIBC2026/runs/<RUN_ID>/
├── session_ref.json                    # 소스 commit·설정·모드 고정
└── snapshots/epoch_.../
    ├── COMMITTED.json, manifest.json   # 완료 marker·파일 SHA-256
    ├── last.pt, best.pt
    ├── run.json, epochs.jsonl, summary.json
    ├── config.yaml, environment_freeze.txt, console.log
    ├── evaluation_console.log
    └── eval_dev/
        ├── metrics.json, predictions.npy
        ├── site_and_cohort_metrics.csv, group_metrics.csv
        ├── settlement_ledger.parquet
        └── csv/
            ├── development_site_predictions.csv
            ├── development_group_forecast.csv
            ├── composition.csv
            └── metadata.json
```

파일은 단계 성공 여부에 따라 존재한다. epoch 백업에는 구조화 학습 로그 `epochs.jsonl`이 포함되고, 일반 stdout 로그·환경·설정은 셀10 종료/예외 처리 및 최종 백업에서 추가한다. 강제 런타임 종료 시 아직 flush되지 않은 stdout은 잃을 수 있다. 백업을 유지하므로 Drive 여유 용량을 확인한다. 원본·캐시·실험 결과·노트북 계정 출력은 Git에 추가하지 않는다.

## 근거와 미검증 항목

- [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html): GPU 자원·런타임 수명은 가변적이고 Drive의 많은 작은 I/O 대신 로컬 저장소 사용이 적절한 근거다.
- [Colab 공식 Drive I/O 예제](https://colab.research.google.com/notebooks/io.ipynb): `drive.mount` 사용 절차.
- [PyTorch LambdaLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html): optimizer와 scheduler 상태를 함께 보존한다.

실제 Google 계정 승인, GitHub 접근 권한, Colab CUDA/T4, full dataset 캐시 시간·메모리, Drive/FUSE 백업 및 런타임 재접속 통합 검사는 사용자가 Colab에서 확인해야 한다. 이번 Mac 검증은 정적 검사·로컬 백업 시뮬레이션·작은 CPU/MPS 테스트 범위다.
