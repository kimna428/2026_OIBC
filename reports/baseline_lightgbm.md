# OIBC LightGBM 통합 발전량 베이스라인

## 실행 결과

253개 발전소를 하나의 LightGBM Regressor로 학습했다. 기존 작업 2의 round=17 예보 77개 피처와 작업 1 평가 함수를 사용했다. 실제 실행 결과:

| split | model | mae_kwh | rmse_kwh | cf_mae | cf_rmse | revenue_krw | penalty_krw | score_krw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | lightgbm | 5.425745 | 24.445967 | 0.030638 | 0.065901 | 1675042422.014020 | 45976209.319027 | 1629066212.694993 |
| dev | climatology | 10.154824 | 35.010047 | 0.059485 | 0.118438 | 1675042422.014020 | 126801752.867262 | 1548240669.146758 |
| holdout | lightgbm | 6.494788 | 22.828271 | 0.037835 | 0.075925 | 1598579787.536722 | 38381222.444239 | 1560198565.092483 |
| holdout | climatology | 12.729370 | 38.391274 | 0.076355 | 0.140548 | 1598579787.536722 | 115834662.257781 | 1482745125.278941 |

에너지 오차는 개별 발전소 관측 행을 pooled한 kWh MAE/RMSE다. CF 오차는 각 행의 `(pred_kwh−actual_kwh)/capacity_kw`로 계산한 무차원 MAE/RMSE이며 CF MAE×100은 설비용량 대비 퍼센트포인트다. 정산은 고정 집합 G1/G2의 점수를 합산한 원(KRW)이다. [전체 수치와 11~15시 가중 지표](../analysis/results/lightgbm_baseline/comparison.csv).

dev: 통계 모델 대비 LightGBM MAE 46.57% 감소, 집합 정산 페널티 63.74% 감소.

holdout: 통계 모델 대비 LightGBM MAE 48.98% 감소, 집합 정산 페널티 66.87% 감소.

두 구간 모두 원시 CF>1 LightGBM 예측은 0건이었다. 상한 clipping은 적용하지 않았다. 실제 0 근처에서도 큰 양의 예측 잔차가 남는 사례가 있어, 이후 주간 무발전 사례와 발전소별 오차를 별도 검토할 필요가 있다.

## 학습·재현 설정

Train은 ts≤2025-01-01 00:00 KST, Dev는 2025-01-01 01:00~2025-04-01 00:00, Holdout은 2025-04-01 01:00~2025-06-01 00:00이다. 우측닫힘 운영일 경계를 변경하지 않았다. 유한 실제 target 3,475,694개만 학습했고, 관측 기간은 2021-01-19 13:00:00+09:00 ~ 2025-01-01 00:00:00+09:00이다.

고정 300 estimators, learning_rate=0.05, num_leaves=31, min_child_samples=100, L2=1, seed=20261030, CPU 4 threads, deterministic/force_col_wise를 사용했다. Dev/Holdout 모두 fit의 eval_set에 전달하지 않았으며 조기 종료·파라미터 탐색·Holdout 기반 선택을 하지 않았다. Train으로만 학습한 모델을 그대로 두 구간에 평가했다. 모델 학습 68.86초, 전체 실행 416.29초.

설치된 LightGBM은 4.7.0이다. 최초 import는 libomp 부재로 실패했으며 Homebrew libomp 설치 후 실행했다. GPU 확인 결과는 [gpu_check.json](../analysis/results/lightgbm_baseline/gpu_check.json)에 기록했다. CPU를 사용한다.

[모델](../analysis/results/lightgbm_baseline/model.txt), [정확한 파라미터·피처 순서·범주·버전·입력 해시](../analysis/results/lightgbm_baseline/run_manifest.json). 저장한 모델을 다시 로드하여 1,000개 학습 입력의 예측이 일치함을 확인했다. 새 추론에도 동일 feature_columns 순서와 site_categories를 적용한다.

## 피처·target·후처리

site_id는 253개 고정 범주형이다. capacity, 위경도, 기상 지점까지 거리, 운영일 달력/주기, WA/WB 예보와 결측 플래그를 사용했다. 운영일 시각은 1~24, 월/dayofyear는 `(ts−1ns)` 날짜 기준이다. 실측 기상·lag/rolling·외부 데이터·target 통계 피처를 사용하지 않는다. SMP/k/실제 발전량/target 유무/target 이상치 플래그는 모델 입력에서 제외했다. 입력 float64는 모델에 전달할 때만 float32로 변환하며 원본·Parquet은 변경하지 않는다. 예보 NaN은 LightGBM native missing 처리에 맡긴다.

목표는 energy/capacity_kw이다. 음수 target 및 CF>1 관측을 유지했다. 예측 CF를 0 이상으로 제한하고 capacity_kw를 곱해 kWh로 복원했다. **용량 상한 clipping=False**이며 기본값은 미적용이다. `--clip-capacity`로 상한 CF=1을 선택할 수 있으나 이번 비교에서는 설정을 변경하지 않았다. 별도의 야간 고정 후처리는 없다. 따라서 낮은 야간 양의 예측이 남을 수 있다.

## Train 통계 비교 모델

발전소×운영일 월×운영일 hour의 Train 평균 CF를 사용한다. 셀 표본이 24개 미만이면 전체 발전소의 Train 월×hour 평균, 없으면 Train hour 평균, 다시 없으면 Train 전체 평균으로 대체한다. fit에서 Train 경계 이후 데이터가 들어오면 실패한다. Dev/Holdout target으로 통계를 갱신하지 않는다. [셀별 통계](../analysis/results/lightgbm_baseline/climatology_site.csv), [구간별 fallback 사용량](../analysis/results/lightgbm_baseline/climatology_fallback.csv). 동일 음수/상한 후처리를 적용한다.

## 세부 성능과 정산

시간대별 오차는 KST **ts 끝 시각**, 월별 오차는 **운영일 월**로 집계한다. `*_hour_metrics.csv`, `*_month_metrics.csv`, `*_site_id_metrics.csv`에 MAE/RMSE/CF 오차/가중 오차가 있다. `*_site_metrics.csv`는 작업 1의 개별 평가 함수 결과다.

`weighted_error_krw=Σ SMP×k×|site error|`, `peak_weighted_error_krw`는 그중 KST ts 11~15시만 합산한 값이다. 각각 weighted MAE는 대응 Σ(SMP×k)로 나눈 kWh다. 이 **개별 발전소 가중 오차**는 집합 내 오차 상쇄가 없으므로 집합 정산 페널티와 다르다. 집합 정산은 작업 1의 `aggregate_site_predictions`와 `score_groups`로만 계산한다.

기준 배정은 sites.current_group의 A→G1, B→G2이며 용량은 {'G1': 20465.655, 'G2': 22189.715} kW다. 두 그룹이 전체 253개를 정확히 분할하고 각각 20,000 kW 이상인지 검증했다. 두 모델은 같은 배정과 고정 정산 시각을 사용한다. Dev 2,160시간, Holdout 1,464시간 모두 실제값 완전으로 제외 0시간이다. 실제 그룹 발전량에 의존하는 매전수익이 두 모델에서 같음을 검증했다. 그룹별 금액과 매 시각 ledger는 `*_group_metrics.csv`, `*_settlement_ledger.csv`에 남겼다.

## 예측 파일과 그래프

[Dev LightGBM 예측](../analysis/results/lightgbm_baseline/dev_lightgbm_predictions.csv), [Holdout LightGBM 예측](../analysis/results/lightgbm_baseline/holdout_lightgbm_predictions.csv). 각각 site_id, ts(KST), actual_kwh, pred_kwh, capacity_kw와 원시/후처리 CF를 포함한다. 비교 모델 예측도 같은 폴더에 있다. 모든 예측/실제는 원래 kWh 단위다.

![실제·예측 비교](../analysis/results/lightgbm_baseline/actual_vs_predicted.png)

위 그래프는 전체 발전소의 완전한 일별 합계이며 아래는 개별 시간 관측의 밀도도다.

![시간·월별 MAE](../analysis/results/lightgbm_baseline/errors_by_time.png)

![학습 gain 중요도](../analysis/results/lightgbm_baseline/feature_importance.png)

[전체 feature importance](../analysis/results/lightgbm_baseline/feature_importance.csv)는 gain과 split count를 포함한다. 상관된 기상 변수 사이 중요도가 분산될 수 있고 gain은 인과 효과가 아니다.

## 실행 및 검증

```sh
uv pip install --python .venv/bin/python -r analysis/lightgbm_requirements.txt
# macOS에서 libomp가 없는 경우: brew install libomp
MPLCONFIGDIR=/tmp/oibc-lgb-mpl .venv/bin/python analysis/lightgbm_baseline.py
.venv/bin/python -m unittest discover -s analysis -p test_lightgbm_baseline.py -v
.venv/bin/python analysis/audit_lightgbm_baseline.py
```

기존 데이터셋/원본 해시를 학습 전에 검증했고 원본은 실행 후에도 해시가 같다. target 관측만 학습, Train 경계, feature/audit 분리, 253개소, 고정 정산 시각, 예측 유한·비음수, kWh 복원, 평가 함수와 MAE/RMSE 일치, 모델 재로드 예측 일치를 확인했다. 테스트 출력은 [test_results.txt](../analysis/results/lightgbm_baseline/test_results.txt)에 기록했다. 신규 단위 테스트 4개와 기존 평가 테스트 11개가 통과했다. 독립 감사는 원본 발전량 전수 대조, 역사적 SMP와 k, 정산 내역 재계산, 저장 모델로 두 구간의 표본 예측 재현을 확인하며 [verification.json](../analysis/results/lightgbm_baseline/verification.json)에 결과를 기록한다. Test 예측 생성이나 Train+Dev 재학습은 이번 단계에 포함하지 않는다.
