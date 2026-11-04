# 저장 예측 공통 비교 도구

이 모듈은 평가·기술통계만 수행한다. 학습, 모델 추론, 팀원 Python import, joblib 역직렬화,
원격 접속, 설치를 호출하지 않는다. checkpoint 감사는 `torch.load(weights_only=True, map_location='cpu')`만 사용한다.
팀원 폴더와 기존 TSMixer 결과는 읽기 전용이며 출력은 새 `runs/tsmixer/comparison/<이름>`에만 쓴다.
기존 출력이 있으면 거부한다. 대용량 파일은 기존 Git ignore 및 storage.result_guard 정책을 따른다.

## 현재 로컬 자료로 감사 실행

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_teammate.sh \
  --package /Users/kimna/Downloads/oibc_repro_sj_2026_11_03 \
  --train-source /Users/kimna/Downloads/oibc_train_source_2026_11_03 \
  --clustering /Users/kimna/Downloads/Clustering_sj.md \
  --output runs/tsmixer/comparison/local_audit_20261104_003
```

없는 예측을 재생성하지 않는다. 데이터 감사·그룹 복원·가용성 상태를 기록하고, 성능 결과는 unavailable로 남긴다.
`audit.json`에 전체 manifest 누락 목록, 존재 파일의 실제/기대 hash, 데이터 hash,
그룹 복원 근거, 체크포인트 tensor 구조를 기록한다. `metrics.json`의 imported/reported 항목은 재계산값이 아니다.

## TSMixer 서버 산출물을 Mac으로 받기

아래 명령은 이번 작업에서 **실행하지 않았다**. Mac 실제 저장소 루트에서 실행한다.
신규 경로를 만들고 `--ignore-existing`으로 기존 파일을 덮어쓰지 않는다. 이미 같은 이름의 import 폴더가 있다면 새 이름으로 바꾼다.

```bash
cd /Users/kimna/workspace/2026_3_2/2026_OIBC
mkdir -p runs/tsmixer/year_backtest/import_seed2026_001
rsync -av --ignore-existing \
  --include='/predictions.npy' --include='/prediction_axis.json' \
  --include='/metrics.json' --include='/run.json' --include='/epoch_selection.json' \
  --include='/sites.csv' --include='/REPORT_KO.md' --include='/*composition.csv' \
  --include='/selection/' --include='/selection/best.pt' --include='/selection/epochs.jsonl' \
  --include='/refit/' --include='/refit/last.pt' --include='/refit/epochs.jsonl' \
  --exclude='*' \
  hai-gpu-209:/home/user/2026_OIBC/runs/tsmixer/year_backtest/seed2026_001/ \
  runs/tsmixer/year_backtest/import_seed2026_001/
```

비교에는 predictions.npy, prediction_axis.json, run.json과 그룹 CSV가 필요하다.
체크포인트와 epochs는 감사용이며 비교를 위해 다시 추론하지 않는다.
rsync 후 adapter가 파일 hash, 축, shape, 단위, run.json refit cutoff를 검사한다.
다운로드 파일 자체의 학습 provenance는 별도 run/로그 감사로 확인해야 한다.

## 팀원에게 필요한 최소 파일

가장 작은 경로는 `reference/selected_stack_all_oof.parquet` 또는 validation의 `site_predictions.parquet`이다.
필수 열은 `site_id`, timezone 포함 `ts`, `lightgbm`, `xgboost`, `bigru`이며 값은 **kWh**다.
발전량 정답 열 `energy`가 있으면 원본 정답·마스크와 일치하는지도 검사한다.
`fold`가 있으면 `year_2024`만 사용한다. 다른 프로필을 자동 추정하지 않는다.
3개 모델 중 일부 열만 있으면 존재 모델만 계산하며 앙상블은 3개 모두 있을 때만 만든다.
`forecast_kwh`가 함께 있으면 검증 프로필의 가중치·배율로 재구성한 앙상블과 일치해야 한다.
추가 GRU 2개 실험은 코드/모델/검증 예측이 없으므로 이번 adapter가 가정하지 않는다.

정확한 기존 모델 재추론까지 필요하면 누락된 `records/validation/full_outer_preprocessing.json`,
`prepared/`와 관련 metadata, 검증 운량 teacher의 학습 cutoff 기록도 필요하다.
기존 repro의 `prepare`는 기상 보조모델 18개를 새로 학습하므로 이번 단계에서 실행하지 않는다.
그룹 생성 로직 감사에는 `provenance/original_sources/build_balanced_vpp.py`와 당시 OOF/감사자료가 필요하다.
누락 원본 CSV는 이번에 manifest와 동일한 바이트로 복원하여 추가 요청이 필요하지 않다.

## 예측을 확보한 뒤 공통 평가

`/path/team_validation.parquet`는 받은 파일의 실제 경로로 바꾼다.

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_teammate.sh \
  --package /Users/kimna/Downloads/oibc_repro_sj_2026_11_03 \
  --train-source /Users/kimna/Downloads/oibc_train_source_2026_11_03 \
  --tsmixer-run runs/tsmixer/year_backtest/import_seed2026_001 \
  --teammate-predictions /path/team_validation.parquet \
  --output runs/tsmixer/comparison/common_year_001
```

서버에 코드와 필요한 자료를 별도 전달한 경우 동일 명령을 `/home/user/ai_env/bin/python`으로 실행할 수 있다.
서버의 Downloads 경로는 자동 추정하지 않는다. `--package`와 `--train-source`에는 실제 전달 경로를 지정한다.

## 평가 계약

- 모든 시각을 KST로 변환한 뒤 `[2024-06-01 01:00, 2025-06-01 01:00)`의 8760시간을 사용한다.
- TSMixer `[365,253,24]` CF를 `[시간,발전소]`로 전치하고 axis의 site_id에 따라 재정렬한 뒤 kW 용량을 곱한다.
  날짜는 운영일 00시, 실제 종료 시각은 1~24시다. 1시간 보정/이동을 자동 적용하지 않는다.
- 정답은 원본 generation float64, 관측 0 유지, NaN 미보간이다. 팀원 prepared의 float32 반올림은 별도 차이로 기록한다.
- 모든 관측 정답에 유한한 예측이 있어야 한다. 예측 누락 때문에 평가 행을 줄이지 않는다.
- observed-only는 같은 관측 발전소들의 예측과 정답을 그룹별 합산한다. 정답 없는 그룹·시간은 생략한다.
- site NMAE는 관측행별 `100×|예측−정답|/용량` 평균이다. MAE는 kWh 단위다.
- **공통 group NMAE**는 팀원 코드대로 `100×Σ|그룹오차|/Σ관측그룹용량`이다.
  기존 TSMixer 방식인 그룹·시간별 비율의 단순 평균은 `group_nmae_macro_pct`로 함께 기록한다.
- `W=Σ0.1×SMP×k×|그룹오차|`, k는 종료 시각 11~15시 3, 그 외 1이다.
- `S=Σ관측실제×SMP−W`, 시간당 페널티는 W/유효 고유시간이다. 그룹 수를 곱한 시간 수로 나누지 않는다.
- complete-fleet는 모든 253개 정답이 있는 시간만 사용하는 별도 참고 결과다.
- contract에 시간·발전소순서·정답·관측 mask·SMP·k·그룹 hash와 산식을 기록한다.
  동일 평가기 아래 두 모델 수치는 비교할 수 있어도 학습/모델선택의 독립성까지 보장되지는 않는다.

`internal_score=100×(1−W/223892580.02742994)`는 기본적으로 provisional이다.
기준 분모 검증은 별도의 **제공 baseline 검증 예측**을 `--baseline-predictions /path/baseline.parquet`로 넣는다.
필수 예측 열은 forecast_kwh다. 원래 sites.current_group A→G1/B→G2에 공통 observed-only 계약을 적용해
기준 W를 1원 이내 재현해야 reference_verified가 된다. config의 상수나 팀원 기대 점수만으로 활성화하지 않는다.
이는 분모의 수치 호환성 검증이며 baseline 학습 provenance/공식 대회 인증을 의미하지 않는다.

## 오차·앙상블 산출물

`summary.csv`, 모델×그룹×평가모드 hourly.csv/groups.csv, `metrics.json`,
`site_errors.csv`, `error_slices.csv`(시각/월/계절/규모/과거관측이력/태양고도),
`residual_correlations.csv`(kWh와 CF), `group_residual_correlations.csv`,
상대 모델이 이기는 그룹·시간 상위25개 `cases_*.csv`, `ensemble_sweep.csv`,
`ensemble.json`, `diagnostics.json`, `audit.json`, `execution.json`, `REPORT_KO.md`를 기록한다.
실측 일사량은 이 도구에 없으므로 태양고도>45도를 **일사량 실측으로 표시하지 않는다**.
상관계수는 같은 관측행의 residual로 계산하며 상수 잔차는 undefined로 남긴다.

alpha 0/0.05/0.10/0.15/0.20/0.30/0.40/0.50은 모두 탐색용으로 기록한다.
이 기간의 최소 W 가중치를 자동 채택하지 않는다.
과거 OOF가 확보되면 `--oof-predictions /path/pre202406_oof.parquet`를 추가할 수 있다.
필수 열은 `site_id,ts,energy,teammate,tsmixer,teammate_trained_until,tsmixer_trained_until,smp,k`다.
두 모델 cutoff는 각 예측의 정규화/보조모델/보정까지 포함하는 최대 학습 시점이어야 한다.
모든 정답 ts는 2024-05-31 17시 이하여야 하며 각 cutoff는 해당 운영일 발행 이전이어야 한다.
2024년 이전 OOF의 W로 alpha를 선택해 연간 검증에 고정 적용한 결과는 별도 저장한다.
OOF cutoff 열의 진실성은 원래 학습 기록으로 별도 검증해야 한다.
공통 그룹도 OOF fold보다 나중에 생성됐다면 그룹 선택 편향이 남으므로 독립 평가라고 주장하지 않는다.

## 테스트

```bash
.venv/bin/python -m unittest analysis.tsmixer.tests.test_comparison -v
.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v
bash -n scripts/tsmixer/compare_teammate.sh
```

신규 fixture 테스트와 실제 자료 감사의 범위를 구분한다. fixture의 완벽 예측/상관/앙상블 수치는 실제 모델 성능이 아니다.

## 01_state 및 세 모델 비교 확장 (2026-11-04)

별도 진입점 `python -m analysis.tsmixer.comparison.three`를 추가했다.
입력 어댑터·cutoff 계약·drop-in inbox·66/231개 convex grid·OOF 절차와 실행 명령은
[THREE_MODELS_KO.md](THREE_MODELS_KO.md)에 있다. 기존 두 모델 CLI와 그 산출물은 보존한다.
현재는 TSMixer 연간 예측이 로컬에 있어 새 경로에서 직접 재평가할 수 있다.
이전 감사 당시 없었던 파일이 도착한 것이며, 이전 보고서의 관측 당시 상태를 덮어쓰지 않는다.
