# 세 공급자 비교: tsmixer / sj_ensemble / state01

기존 `python -m analysis.tsmixer.comparison` 및 그 테스트/산출물을 유지한다.
새 진입점은 `python -m analysis.tsmixer.comparison.three`와 `scripts/tsmixer/compare_three_models.sh`다.
기존 core.Grid/evaluate, keyed alignment, 잔차 slice·상관·case 분석을 재사용한다.
외부 프로젝트 import/실행, pickle/joblib/Torch 체크포인트 로드, 학습, 원격 접속은 하지 않는다.
모든 결과는 새로운 `runs/tsmixer/comparison/<run>`에 저장하며 기존 디렉터리가 있으면 종료한다.

## 지금 실행 가능한 감사 및 TSMixer 재평가

저장소 루트에서:

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --output runs/tsmixer/comparison/three_local_002
```

기본 경로는 현재 Mac의 Downloads/01_state 및 SJ 패키지, 기존 복원 SJ CSV,
`runs/tsmixer/year_backtest/seed2026_001/`다. 다른 머신에서는 해당 CLI 경로를 명시한다.
TSMixer predictions.npy/axis/run.json이 있어야 실제 TSMixer 점수가 계산된다.
SJ와 state01 입력이 없으면 재학습하지 않고 unavailable을 기록한다.

## 추가 예측을 넣는 표준 inbox

기본 inbox는 `runs/tsmixer/comparison_inputs/`다. 필요한 경우 아래 폴더를 만들고 **받은 파일**을 넣는다.
원본 팀원 폴더에 저장하거나 기존 실행 결과를 수정하지 않는다.

```text
runs/tsmixer/comparison_inputs/
  sj_validation.parquet
  sj_validation.json          # 선택: 학습 provenance; 없으면 config 선언만 확인했다고 기록
  state01_validation.parquet
  state01_validation.json     # 필수: 01_state 원본 main_all의 cutoff 문제 때문에 자동 추정 금지
```

입력은 timezone 있는 `ts`, `site_id`를 키로 한다. UTC 표기도 같은 시각이면 KST로 변환한다.
키 중복/미지ID/시간이동/예측 결측으로 관측행 감소를 허용하지 않는다.
CSV도 명시적 `--sj-predictions`/`--state01-predictions`로 받을 수 있다.
태양 아래0, 야간0 등 모델 후처리 값을 유지하며 평가용 보정을 새로 fit하지 않는다.

SJ 표는 기존 adapter 형식인 kWh 단위 `lightgbm,xgboost,bigru` 열을 받는다.
셋 다 있으면 검증 배율 .9612887427622985와 .25/.25/.50으로 `sj_ensemble`을 만든다.
`forecast_kwh`가 함께 있으면 일치 검사를 한다. 최종 앙상블 `forecast_kwh`만 있어도 평가 가능하지만
component 재구성/배율 검증은 못 했다고 해석한다. 일부 component만 있으면 그것만 별도 평가한다.
`fold`가 있으면 year_2024만 사용한다. 단독 component 이름은 sj_lightgbm/sj_xgboost/sj_bigru다.

state01 표는 기본 `kwh` 열이다. 메타데이터로 prediction_column을 명시할 수 있다.
기본 출력명은 state01이다. 같은 표의 추가 열을 다음처럼 component로 노출할 수 있다.

```json
"components": {
  "state01_lgb": "lgb_n75",
  "state01_xgb": "xgb_n100",
  "state01_cloud": "cloud_l05",
  "state01_all": "all_l08"
}
```

숫자 배열의 행 순서는 가정하지 않고 `(ts,site_id)`로 재정렬한다.
모든 원본 관측 정답에 예측이 있어야 한다. 정답 NaN을0으로 채우거나 그 누락 때문에 평가행을 줄이지 않는다.
정답 energy 열을 함께 받으면 raw mask와 정답값도 대조한다.

## state01 provenance 계약

**현재 공유 원본 main_all의 예측에는 아래 계약을 허위로 붙이면 안 된다.**
원본은 2024-06-01 00시까지 학습하고, outage를 전체 generation에서 정하고,
auxiliary 범위를 validation 예보까지 포함해 계산한다. 이 문제를 고친 별도 과거 검증이 필요하다.
또는 실제 run 기록이 다른 안전한 학습을 입증해야 한다. 현재는 그 기록과 예측이 모두 없다.
최종 test 모델은 2025년5월까지 학습했으므로 과거1년 독립 검증용으로 받지 않는다.

JSON 예시는 형식 설명이며 실제 학습을 입증하는 파일이 아니다.

```json
{
  "profile": "year_validation",
  "protocol": "cutoff_safe_v1",
  "timezone": "Asia/Seoul",
  "timestamp_convention": "interval_end",
  "units": "kWh",
  "prediction_column": "kwh",
  "prediction_start": "2024-06-01 01:00:00+09:00",
  "prediction_end_exclusive": "2025-06-01 01:00:00+09:00",
  "training_cutoff": "2024-05-31 17:00:00+09:00",
  "normalization_cutoff": "2024-05-31 17:00:00+09:00",
  "auxiliary_cutoff": "2024-05-31 17:00:00+09:00",
  "outage_detection_cutoff": "2024-05-31 17:00:00+09:00",
  "calibration_cutoff": null,
  "selection_target_end": null,
  "inference_inputs": ["forecasts", "solar_geometry", "calendar", "site_metadata"],
  "leakage_controls": {
    "outages_fit_before_cutoff": true,
    "aux_bounds_fit_before_cutoff": true,
    "no_target_inputs": true
  },
  "prediction_sha256": "받은_예측파일의_실제_SHA256"
}
```

마지막 문자열은 실제 hash로 바꾼다. Python `hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()`로 계산한다.
JSON만으로 학습을 증명하지 않는다. adapter 결과는 `declared_protocol_bounds_checked`,
`independently_audited=false`로 표시한다. 로그/학습코드/normalizer/aux/outage 이력은 별도 검토해야 한다.
쓰지 않는 auxiliary/calibration/selection 단계만 null을 허용한다. 필수 training/normalizer/outage는 생략하지 않는다.
units=capacity_factor를 명시하면 canonical site 용량으로 kWh 변환한다. 단위는 추정하지 않는다.

## 원래 seed별 파일을 받는 경우

검증 예측이 아래 native 구조로 올 수도 있다. test/ 폴더를 대신 넣지 않는다.

```text
<run>/main_all/gbdt/pred_lgb_n75_s0.parquet
<run>/main_all/gbdt/pred_xgb_n100_s0.parquet
<run>/main_all/gru/pred_cloud_l05_s0.parquet
<run>/main_all/gru/pred_all_l08_s0.parquet
# metadata.seeds에 명시된 모든 seed에 대해 같은 파일 필요
```

각 파일은 `site_id,ts,kwh`다. 명시된 모든 seed/4개 component가 있어야 원래의 .1/.1/.4/.4 혼합을 만든다.
누락 component를0으로 채우거나 남은 seed로 가중치를 재정규화하지 않는다.
위 provenance에 `seeds:[0,1,...]` 및 `prediction_files:{"main_all/gbdt/pred_lgb_n75_s0.parquet":"<sha256>",...}`를 추가한다.
각 seed 파일의 hash와 모든 관측행을 개별 검사한 뒤 평균한다.

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --sj-predictions /실제/SJ_validation.parquet \
  --state01-run /실제/수정된검증run \
  --state01-metadata /실제/comparison_provenance.json \
  --output runs/tsmixer/comparison/three_native_001
```

최종 혼합 parquet를 받았다면 native-run 대신 다음 명령을 사용한다.

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --sj-predictions /실제/SJ_validation.parquet \
  --state01-predictions /실제/state01_validation.parquet \
  --state01-metadata /실제/state01_validation.json \
  --grid-step 0.10 \
  --output runs/tsmixer/comparison/three_received_001
```

## 공통 지표·가중치·OOF

공통 evaluator는 SJ 배정을 primary로 쓰고 TSMixer baseline/metadata를 각각 별도로 재집계한다.
ratio/macro 그룹 NMAE, Site NMAE/MAE, W/S/매출/시간당W, 유효시간/관측행, 잠정 내부점수를 저장한다.
W는 `0.1*SMP*k*abs(sum_observed(pred−actual))`다. 01_state 원래의 W(0.1 없음)를 직접 재사용하지 않는다.
기준 W_ref는 기존 baseline 예측을 `--baseline-predictions`로 넣어 같은 원래 A/B/mask/시간에서 재현하기 전까지 미검증이다.
complete-fleet는 별도 참고 평가로 유지한다.

0.10 simplex는66개(단독3, pair27, triple36), 0.05는231개다.
`a+b+c=1`, 음수 없음; 양의 가중치 모델이 없으면 그 조합은 unavailable이다.
0 가중치의 없는 모델 때문에0×NaN이 생기지 않게 처리한다. 가중치를 자동 정규화하거나 최적점을 채택하지 않는다.
모든 연간 grid는 탐색용이다. component는 진단/별도 평가되지만3-way grid의 축은3개 family만 사용한다.

`--oof-predictions /실제/preyear_oof.parquet`에 `--oof-composition`과 `--oof-group-provenance`를 함께 지정하면 이전 OOF에서만 가중치를 선택한다.
필수 열은 다음과 같다.

```text
site_id, ts, energy, smp, k,
tsmixer, sj_ensemble, state01,
tsmixer_trained_until, sj_ensemble_trained_until, state01_trained_until
```

모든 ts≤2024-05-31 17시, 각 trained_until≤해당 운영일 발행 시각이어야 한다.
trained_until은 constituent/normalizer/aux/outage/보정/선택에 사용한 정보의 최대시점이다.
OOF의 W로 고정한 weights를 이후 연간 예측에 적용한 결과는 oof_selection.json에 분리한다.
OOF 그룹 provenance JSON은 `{"basis":"metadata_only","composition_sha256":"<canonical membership hash>"}`
또는 basis=historical_residuals와 cutoff를 기록한다. 후자는 **가장 이른 OOF 발행 이전**으로 제한한다.
canonical membership hash는 core.mapping_hash(pd.read_csv(path))다. 원본 파일 byte hash와 다르다.
현재 SJ 배정이 OOF fold 뒤의 잔차로 만들어졌다면 이 검사를 통과하지 못하며 소급 선택을 거부한다.
하나의 이전 고정 그룹 또는 metadata 그룹으로 선택한다. fold별 가변 그룹 최적화는 아직 구현하지 않았다.
선언 JSON만으로 실제 그룹 생성 provenance를 증명하지는 않는다.
기존 연간 구간이 구조/seed 선택에 반복 사용됐다는 한계도 유지한다. 독립 최종 평가라고 표시하지 않는다.

## 산출물과 비용

- availability.csv/metrics.json: 모델별 available/unavailable/ineligible 사유와 공통 계약.
- summary.csv 및 모델×그룹×모드 hourly/groups.csv: 실제 계산된 결과만.
- error_slices.csv/site_errors.csv/residual_correlations.csv/group_residual_correlations.csv.
- site_residual_correlations.csv: 발전소별 모델쌍 잔차 상관/반대 부호 비율.
- group_complementarity.csv: G1/G2별 모델쌍 상관/반대 부호/각 W 및 50:50 혼합 W(탐색).
- cases_*.csv: 상대 모델의 성공·실패 시간.
- high_penalty_contributions.csv: 배정별 상위50개 어려운 그룹·시간과 각 모델의 signed residual/W.
- ensemble_grid.csv/ensemble.json: 가중치별 점수와 누락 조합. 그룹별 오차 상쇄율도 기록.
- oof_selection.json: OOF 선택과 고정 연간 적용을 분리.
- audit.json/execution.json/REPORT_KO.md: 외부 source hash/구조/데이터 비교/실행 기록.

저장 예측 계산은 CPU만 쓴다. 모든3개와 component가 있을 때 메모리·CSV가 늘며,
0.05 grid는0.10보다 많은 반복 평가를 한다. 초기에는0.10으로 제한하는 편이 비용을 줄인다.
원본 모델 학습/캐시 재생성은 하지 않는다. 특히 original main.py --quick도 전체 격자 생성을 하므로 호출하지 않는다.

```bash
.venv/bin/python -m unittest analysis.tsmixer.tests.test_three_comparison -v
.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v
bash -n scripts/tsmixer/compare_three_models.sh
```
