# OIBC 로컬 검증 및 정산 프로토콜

## 근거와 기존 구현

데이터 안내 `handover 2/01_participant_dataset/README.md`의 정산식·단위·구성 조건 및 요청한 기간을 사용한다. 기존 `reports/data_quality.md`에서는 발전소별 편입 시점이 다르고 내부 결측은 7시간임을 확인했다. `analysis/forecast_baseline.py`는 개별 발전소 CF/energy 오차와 가중 오차를 계산하지만 0.10 정산 페널티나 집합 구성을 평가하지 않는다. 기존 모델·EDA를 재실행하거나 복제하지 않고 `analysis/validation.py`를 추가했다. SHA-256은 기존 EDA의 `sha` 함수를 재사용한다.

제공 폴더에 공식 `scoring.py`와 문제정의서 원문은 없어 공식 프로그램과의 직접 대조는 할 수 없다. 제공 README와 사용자 지정 정산식에 일치하도록 구현했다. 실제 모델 예측이 제공되지 않아 임의의 모델 점수는 작성하지 않았다. 테스트용 가상 수치는 공식 성능으로 보고하지 않는다.

## 시간 분리와 관측 완전성

모든 입력 시각은 timezone-aware여야 한다. UTC 등 명시적 시간대는 Asia/Seoul로 변환하고 naive·NaT·정시가 아닌 값은 거부한다. 1시간 발전량은 우측닫힘이며 00시는 전날 운영일 마지막 구간이다.

- Train: ts ≤ 2025-01-01 00:00+09:00.
- Dev: 2025-01-01 01:00 ~ 2025-04-01 00:00+09:00 (90 운영일).
- Holdout: 2025-04-01 01:00 ~ 2025-06-01 00:00+09:00 (61 운영일).

| 구간 | 시작 ts (KST) | 종료 ts (KST) | 기대 시간 | 정산 사용 | 제외 | 관측 행 |
|---|---|---|---:|---:|---:|---:|
| train | 2021-01-01 01:00:00+09:00 | 2025-01-01 00:00:00+09:00 | 35064 | 178 | 34886 | 3475694 |
| dev | 2025-01-01 01:00:00+09:00 | 2025-04-01 00:00:00+09:00 | 2160 | 2160 | 0 | 546480 |
| holdout | 2025-04-01 01:00:00+09:00 | 2025-06-01 00:00:00+09:00 | 1464 | 1464 | 0 | 370392 |

SMP 시간축은 역사적 `train/smp.csv`의 연속된 정시 격자다. 시작은 역사적 SMP 최초 시각이며 발전량 최초 관측 이전도 포함한다. 발전량 4,392,566행을 세 구간에 중복·누락 없이 배정했다. Dev/Holdout은 각각 2,160/1,464시간임을 검증했다. 제출용 `smp.csv`는 학습 구간 가격으로 사용하지 않고 k 규칙 일치 검증에만 사용했다.

[구간별 완전성 요약](validation/completeness_summary.csv). 각 구간의 `*_hourly_completeness.csv`에 매 시각 관측 발전소 수·결측 수·포함 여부·제외 이유를 저장했고, `*_site_coverage.csv`에 발전소별 관측률을 저장했다. 제외 사유 `missing_or_nonfinite_actual`은 행 부재 또는 NaN/±inf이다. 편입 이전과 내부 결측의 구분은 EDA 보고서를 참조한다. 이 평가에서는 이유와 무관하게 253개 전체 관측이 있어야 완전 집합 정산에 포함한다.

`*_settlement_times.csv`는 **원관측만으로 정한 고정 정산 시각 집합**이다. 예측이나 구성에 따라 바뀌지 않는다. [manifest.json](validation/manifest.json)에 입력과 시각 파일 해시를 저장했다. 동일 입력에서 `build_windows()`를 재사용하면 동일 집합이다. 비교 시 해시가 같은지 확인한다. Train 마스크도 감사 목적으로 제공하지만 모델 선택은 Dev로 수행하고 Holdout은 최종 평가에 보존한다.

## 함수와 입력 계약

`analysis/validation.py`는 학습이나 파일 쓰기 없이 DataFrame을 받아 결과를 반환한다.

- `build_windows(generation, prices, sites)`: `generation=[site_id,ts,energy]`, `prices=[ts,smp]`, `sites=[site_id,capacity_kw,...]`. 중복 관측/SMP 키·알 수 없는 발전소·불연속 SMP·잘못된 가격을 거부한다. 구간별 실제값·가격·고정 마스크를 반환한다. 발전량 결측은 유지한다.
- `validate_composition(composition, sites)`: `[site_id,group]`가 253개소를 정확히 한 번씩 G1/G2로 분할하고 각 용량이 20,000 kW 이상인지 검증한다. 참조 A→G1/B→G2 용량은 {'G1': 20465.655, 'G2': 22189.715} kW이다. 이를 최적 배정으로 간주하지 않는다.
- `aggregate_site_predictions(predictions, composition, sites, window)`: `[site_id,ts,pred_kwh]`를 그룹으로 합산한다. 고정 정산 시각의 253개소 예측이 모두 유한해야 한다. 개별 음수 예측은 보존하지만 최종 그룹 예측은 정산 함수에서 비음수를 요구한다.
- `score_groups(predictions, composition, sites, window)`: `[group,ts,pred_kwh]` 입력으로 그룹별 지표 DataFrame, 합산 결과 dict, 시간별 정산 ledger를 반환한다. 두 그룹 모두 고정 시각의 예측이 있어야 하며 누락·비유한·음수·중복 키는 실패한다. 해당 구간 안의 제외 시각 예측은 허용하되 점수에 사용하지 않는다. 구간 밖 입력은 거부한다. 정산 시각이 0개면 점수 계산 불가로 실패한다.
- `evaluate_sites(predictions, window)`: 완전 집합 마스크와 별도로, **실제 유한 관측이 있는 모든 발전소-시각 행**에서 개별 MAE/RMSE와 pooled 지표를 계산한다. 관측 target에 대한 예측 누락은 실패한다. 미관측 target은 채우지 않는다. 관측이 없는 발전소는 관측 수 0과 NaN 오차를 반환한다.

예측에 결측이 있으면 평가 시간을 줄이지 않고 실패한다. 여러 모델이 평가할 수 있는 시각만 취하는 사후 교집합 방식은 쓰지 않는다. 입력 중복은 평가 구간의 제외 시간에 있더라도 거부한다. 동일 구성은 평가 구간 전체에서 고정된다.

## 정산식과 단위

각 g∈{G1,G2}, 고정 시각 집합 T에 대해:

- `revenue_krw = Σ SMP(t) × actual_g(t)` — 매전수익, 원(KRW).
- `penalty_krw = 0.10 × Σ SMP(t) × k(t) × |pred_g(t) − actual_g(t)|` — 오차 페널티, 원(KRW).
- `score_krw = revenue_krw − penalty_krw` — 최종 정산 수익/점수, 원(KRW).

SMP는 원/kWh, 실제·예측은 해당 1시간 kWh이며 k와 0.10은 무차원이다. k는 **구간 끝 ts의 KST 시가 11~15일 때 3**, 나머지는 1이다. 운영일 변환이나 구간 중앙 시각으로 k를 정하지 않는다. G1/G2의 revenue·penalty·score를 합산한다. 원자료의 음수 target이나 CF>1은 수정하지 않는다.

MAE와 RMSE 단위는 kWh이며 그룹별 시간 오차를 계산한다. 합산 dict의 MAE/RMSE는 두 그룹의 2×|T|개 행을 pooled한 값으로, 두 그룹 총발전량 오차와 다르다. `used_hours`는 그룹 공통 시각 수, `group_hour_rows`는 그 두 배다. 수익은 실제 발전량에만 의존하므로 전체 분할과 시각이 같으면 구성에 불변이다. 페널티는 집합 내 오차 상쇄 때문에 구성에 따라 달라질 수 있다.

## 실행 및 재사용

저장소 루트에서 기존 `.venv`와 `analysis/requirements.txt`를 사용한다. 추가 의존성은 없다.

```sh
.venv/bin/python -m unittest discover -s analysis -p test_validation.py -v
.venv/bin/python analysis/build_validation_protocol.py
```

Python 재사용 예시(실제 예측 파일이 필요함):

```python
from analysis.validation import build_windows, aggregate_site_predictions, score_groups, evaluate_sites
windows = build_windows(generation, historical_smp, sites)
w = windows['dev']  # 최종 평가에는 'holdout'
group_pred = aggregate_site_predictions(site_pred, composition, sites, w)
by_group, total, ledger = score_groups(group_pred, composition, sites, w)
by_site, site_total = evaluate_sites(site_pred, w)
```

`site_pred`는 선택한 구간만 전달한다. `evaluate_sites`는 완전 집합 시각 밖의 관측에도 예측을 요구하므로 개별 평가는 전체 구간 관측 행을 준비한다.

## 검증

단위 테스트는 완전 예측의 페널티 0, 수작업 계산 가능한 가중 페널티와 금액 단위, 배정 변경 시 수익 불변, 분할 누락·중복·용량 부족, 관측/예측/가격 키 중복, 예측 결측·비유한·음수, 실제 결측 시 고정 마스크, 개별 관측만 평가, UTC 변환·naive 거부·기간 경계·k 경계를 확인한다. 테스트 출력은 [test_results.txt](validation/test_results.txt)에 저장한다.

실데이터 감사는 전체 행의 단일 구간 배정, SMP 격자, 구간 시간 수, 참조 구성 용량, 제출 k와 규칙 일치, 원본 4개 파일 실행 전후 SHA-256 불변을 통과했다. 결과는 manifest에 기록했다. 원본을 수정하지 않는다.
