# 실제 실행·검증 기록 — 2026-11-04

작업 위치: `/Users/kimna/workspace/2026_3_2/2026_OIBC`.
브랜치: `experiment/tsmixer-year-backtest`. 시작 commit은 `8dc66a6`이다.
요청에 적힌 단축 저장소 경로는 존재하지 않아 현재 환경의 실제 저장소를 사용했다.

## 실제 접근한 자료

- `/Users/kimna/Downloads/oibc_repro_sj_2026_11_03/`: run.py, config.json, manifest.json,
  code/{pipeline,prepare,neural,weather,temporal,helpers}.py, dataset/README.md,
  validation/final 모델 파일. manifest 존재 파일33개의 실제 hash를 검사했다.
- `/Users/kimna/Downloads/oibc_repro_sj_2026_11_03.zip`: 목록과 hash만 읽었다. 원본 수정/압축해제는 하지 않았다.
- `/Users/kimna/Downloads/oibc_train_source_2026_11_03/`: README/config/manifest/run/code/validator 등
  manifest20개 전부 hash를 검사하고 해당 학습·전처리 차이를 읽었다.
- `/Users/kimna/Downloads/{Clustering_sj.md,group_sj.json,report_sj_2.md}`.
- 현재 저장소의 TSMixer model/year_backtest/runtime/config/tests, 원본 baseline.py.
- `handover 2/01_participant_dataset/{sites.csv,train/generation.parquet,train/smp.csv}`:
  정답·mask·가격축을 실제로 재구성했다. 이 원본3개의 hash는 팀원 manifest와 일치한다.
- `reports/tsmixer/year_backtest/seed2026_001/`의 metrics, Epoch 로그/선택 결과, 두 composition, 보고서.
  실제 `runs/tsmixer/year_backtest/seed2026_001/`와 연간 predictions는 로컬에 없다.
  development/smoke prediction 파일은 존재를 확인했지만 연간 비교 입력으로 사용하지 않았다.

모든 팀원 파일은 읽기 전용으로 취급했다. 두 번의 감사 사이 존재 파일의 byte hash는 모두 그대로다.
원본 폴더의 run.py/prepare/predict/train을 실행하거나 joblib 모델을 로드하지 않았다.
BiGRU checkpoint만 Torch weights_only로 CPU에서 tensor 구조·Epoch를 확인했다.

## 실제 실행한 주요 명령

파일 위치/코드 조사는 `rg --files`, `rg -n`, `cat`, `sed`, 폴더 목록과 Git 읽기 명령으로 수행했다.
Python 표준 zipfile로 archive 목록을 읽었으며 임의 스크립트를 실행하지 않았다.
주요 재현 명령은 다음과 같다.

```bash
.venv/bin/python -m analysis.tsmixer.comparison \
  --package /Users/kimna/Downloads/oibc_repro_sj_2026_11_03 \
  --train-source /Users/kimna/Downloads/oibc_train_source_2026_11_03 \
  --output runs/tsmixer/comparison/local_audit_20261104_001

.venv/bin/python -m unittest analysis.tsmixer.tests.test_comparison -v
.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v

.venv/bin/python -m analysis.tsmixer.comparison \
  --package /Users/kimna/Downloads/oibc_repro_sj_2026_11_03 \
  --train-source /Users/kimna/Downloads/oibc_train_source_2026_11_03 \
  --output runs/tsmixer/comparison/local_audit_20261104_002

.venv/bin/python -m analysis.tsmixer.comparison --help
bash -n scripts/tsmixer/compare_teammate.sh
git diff --check
git status --short
```

001은 최초 감사, 002는 원본 CRLF CSV hash 및 기존 TSMixer mask 대조를 추가한 최종 감사다.
기존 결과를 덮어쓰지 않고 별도 디렉터리로 남겼다.
최초 신규 fixture에서 두 그룹 관측 용량이 같아 ratio/macro가 같았는데 차이를 기대한 assertion이 실패했다.
서로 다른 관측 용량을 갖게 fixture를 수정해 두 산식을 구분하는 테스트로 만들었고 재실행은 통과했다.

최종 결과: **총61개 중60개 통과, CUDA 전용1개 skip**, 실행11.700초.
신규9개 테스트는 모두 fixture 기반이다. 실제 모델 예측/잔차/alpha 성능을 검증한 테스트가 아니다.
기존 multiprocessing 회귀 테스트가 요구하는 shared memory를 위해 승인된 sandbox 외부에서 전체 suite를 실행했다.
기존 suite에는 제한된 합성 CPU/MPS forward/backward와 5~10 update smoke가 포함된다.
기존 모델 재학습이나 전체 GPU 학습을 실행한 것은 아니다.

신규 테스트 범위:

| 항목 | 근거 |
|---|---|
| ID/그룹 완전성 | unknown/duplicate/missing/단일그룹/20MW미달 거부; 복원 소스 불일치 거부 |
| 시간/순서 | UTC→KST, shuffled long rows, TSMixer site축 역순, 1시간축 이동 거부 |
| shape/관측 mask | 관측 정답의 예측 누락 거부, 무정답 예측 누락 허용, empty 정산 unavailable |
| 정산 | SMP×k×0.1 hand oracle, S=매출−W, 오차 상쇄, 잘못된 k 거부 |
| 지표 | 같은 예측의 그룹 변경에서 site NMAE 불변; ratio와 macro 구분 |
| 기준 점수 | W_ref baseline 재계산, SMP 변경 시 호환 거부, 반복 평가 재현 |
| 미래 유입 | OOF 평가연도 정답 및 양쪽 모델의 발행 후 cutoff 거부; 미래 정답이 학습이력 cohort에 영향 없음 |
| 분석 | 동일 행 residual/앙상블8개 후보/실패사례 출력; 독립 평가 false 유지 |
| 기존 TSMixer | 기존 모델/개발 데이터/체크포인트/Colab/누수/경계 테스트 회귀 통과 |

## 실행하지 못했거나 실행하지 않은 작업

| 작업 | 상태/이유 |
|---|---|
| 연간 예측 공통 재평가 | 양쪽 saved year predictions 부재; 숫자 생성 안 함 |
| component별 MAE/W/S | 검증 component 예측 부재 |
| 실제 잔차 상관/시간별 장단점 | 예측 부재; 모델 metrics만으로 역산 불가 |
| alpha별 성능 | 동일 행의 두 예측 부재; fixture 결과를 실제로 제시하지 않음 |
| 분모 기준의 실제 재현 | 원래 baseline 연간 예측 부재; config 상수만으로 검증하지 않음 |
| 그룹 생성 코드/잔차상관 조건 | 원래 build_balanced_vpp.py와 OOF/감사자료 누락 |
| 저장 모델 재추론 | repro records/prepared 누락; 전처리 재생성은 teacher18개 학습을 포함 |
| 별도 GRU2개 공동학습 실험 | 코드/체크포인트/예측 미발견 |
| 원격 접속/rsync | 임의 수행하지 않음; 정확한 수신 명령은 비교 README에 작성 |
| 전체 GPU 학습/패키지 설치/환경 변경 | 수행하지 않음 |
| Git commit/push | 수행하지 않음 |

## 추가 자료와 다음 실행

서버 TSMixer 결과는 [비교 README](../../analysis/tsmixer/comparison/README_KO.md)의 rsync 명령으로 수신한다.
팀원에게 최소 validation component predictions 또는 selected_stack_all_oof.parquet를 받으면
현재 비교 CLI로 재학습 없이 재평가·잔차·alpha 탐색을 할 수 있다.
학습 provenance 확인에는 validation preprocessing/로그와 teacher cutoff 기록이 필요하다.
OOF 선택과 W_ref 검증에는 별도의 과거 TSMixer OOF 및 원래 baseline 예측이 필요하다.
복원된 그룹 CSV는 원본 manifest와 정확히 일치하므로 더 이상 필요 파일로 요구하지 않는다.

## 추가 파일과 Git 상태

이번 작업은 기존 tracked 파일을 수정하지 않고 다음 파일을 새로 추가했다.

- `analysis/tsmixer/comparison/{__init__,__main__,core,io,audit,diagnostics}.py`
- `analysis/tsmixer/comparison/README_KO.md`
- `analysis/tsmixer/tests/test_comparison.py`
- `scripts/tsmixer/compare_teammate.sh`
- `reports/tsmixer/TEAMMATE_COMPARISON_KO.md`
- `reports/tsmixer/ENSEMBLE_ANALYSIS_KO.md`
- `reports/tsmixer/IMPROVEMENT_ROADMAP_KO.md`
- `reports/tsmixer/COMPARISON_EXECUTION_KO.md`
- `reports/tsmixer/COMPARISON_EVIDENCE_20261104.json`

새 ignored 실행 폴더001/002에는 복원 CSV3개, 감사JSON, 가용성/기존보고값/공통계약JSON,
미계산 이유가 담긴 diagnostics/ensemble JSON, header-only summary.csv, 실행기록/상태보고서가 있다.
모델 예측이 없어 성능 hourly CSV나 실제 상관 CSV가 생성된 것으로 주장하지 않는다.
최종002의 teammate_composition.csv 파일 hash를 별도로 재검사해 manifest 일치를 확인했다.

시작 시 이미 있던 다음 untracked 파일은 건드리지 않았다:
`reports/TSMIXER_REPOSITORY_ANALYSIS_KO.md`, `reports/deep_learning/MODEL_OVERVIEW.md`,
`reports/deep_learning/compact_initial_parity.json`, `compact_storage_estimate.json`, `compact_test_parity.json`.
기존 TSMixer 모델·설정·학습 스크립트·체크포인트·캐시에 diff가 없다.
원본 팀원 코드/모델/데이터는 저장소로 복사하거나 Git에 추가하지 않았다.
