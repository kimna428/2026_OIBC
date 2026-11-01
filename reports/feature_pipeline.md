# OIBC 예보 피처 파이프라인

## 입력과 기존 코드

기존 `forecast_baseline.py`의 station/ts 인덱스 조회 방식을 검토했다. 기존 builder는 CatBoost 의존, 관측 target 기준, 원시 ts 달력 특징, 결측 플래그 부재 및 NaN 대체를 전제로 하므로 기존 실험 재현성을 유지하고 별도 `analysis/feature_pipeline.py`로 확장했다. 작업 1의 `local_hours`, `split_labels`, `validate_sites`를 재사용한다. 원본 8개 파일의 실행 전후 SHA-256 일치를 확인했다. 실측 기상·외부 자료·사전학습 모델은 사용하지 않는다.

## 실행과 산출물

```sh
.venv/bin/python -m unittest discover -s analysis -p test_feature_pipeline.py -v
.venv/bin/python analysis/build_feature_datasets.py
```

기존 `analysis/requirements.txt` 환경을 사용하며 추가 의존성은 없다. 출력은 `analysis/datasets/{train,dev,holdout,test}.parquet`와 [manifest](../analysis/datasets/manifest.json)다. 원관측이 없는 행도 energy/target_cf=NaN으로 보존한다. test target은 모두 NaN이다. 발전소별 추론에 필요한 SMP 시각×253개소만 월별로 생성하며 전체 기간 Cartesian merge는 수행하지 않는다. 모든 join은 고정 길이 index lookup 또는 metadata map이다. 한 번에 한 달을 만들고 ParquetWriter로 순차 저장한다.

```text
            rows  target_present  negative_energy  cf_gt_1  sites  target_missing
train    8871192         3475694               40        8    253         5395498
dev       546480          546480                0        0    253               0
holdout   370392          370392                0        1    253               0
test     2216280               0                0        0    253         2216280
```

[데이터셋 행·target 요약](features/dataset_summary.csv), [월별 입력/출력 행 수·메모리·처리시간](features/chunk_processing.csv).

총 실행 50.47초, 프로세스 peak RSS 1717.8 MiB, 최대 월별 최종 DataFrame 71.1 MiB. RSS는 파일 읽기·Arrow·중간 배열 등을 포함한 OS 측정 최대값이다. 최종 DataFrame 크기는 전체 프로세스 메모리와 다르다.

## 결합·가용성·결측

Train WB에서 정시가 아닌 :55 예보 92행을 발견했다. 원본 시각을 반올림하지 않고 정확한 시각 일치로만 결합하므로 이 행들은 정시 target에 결합되지 않는다. [해당 키 목록](features/off_hour_forecast_keys.csv).

4개 예보 파일의 전체 round에 대해 `(station,round,ts)` 유일성을 검증한 뒤 round=17만 읽는다. sites의 wa_station/wb_station을 통해 해당 provider 예보를 조회한다. 중복·미지 발전소·SMP 결측·행 수 변화·최종 `(site_id,ts)` 중복은 즉시 실패한다. 네 데이터셋의 전체 Arrow 스키마와 feature 순서/형식을 일치시켰고 각각 253개소를 유지한다.

원시 train 시간축 전체(Train/Dev/Holdout 포함)와 test에 대한 결합 요약:

```text
                    rows  missing_rows  any_missing_rows  join_success_rate  any_missing_rate
source provider                                                                              
test   wa        2216280           188               188           0.999915          0.000085
       wb        2216280             0                 0           1.000000          0.000000
train  wa        9788064         16954             16954           0.998268          0.001732
       wb        9788064         38890             42838           0.996027          0.004377
```

[시각별 provider 행 결합 성공률](features/hourly_join_quality.csv), [발전소별 예보 결측률](features/site_join_quality.csv), [변수별 결측률](features/feature_missingness.csv). 행 결합 성공과 값 완전성은 다르다. `*_row_missing`은 예보 행 부재, 각 `*_missing`은 변수 NaN, `*_any_missing`은 provider 중 하나 이상 NaN이다. 예보 값은 원래 단위의 float64로 보존하고 NaN을 채우지 않는다. inf 발견 시 중단한다. LightGBM의 native NaN 처리를 사용할 수 있으며 이 단계에서는 학습/중앙값 추정/스케일링을 수행하지 않는다.

데이터 안내에 round=17은 대상일 전날 17시 발행 예보라고 명시되어 있다. 실제 발행시각/수정시각 열은 없으므로 그 계약을 따른다. 사후 갱신 여부를 독립적으로 검증할 수는 없다. 같은 날 미래의 실측이나 발전량을 입력으로 사용하지 않는다.

## 피처와 target 계약

모델 입력은 manifest의 **feature_columns만** 사용한다. categorical_features는 site_id이며 전체 sites 메타에서 고정한 253개 범주로 동일하게 저장한다. station ID는 조회에 사용하고 모델 입력에서는 제외했다. capacity_kw, lat/lon, wa_dist/wb_dist와 WA 10개·WB 20개 모든 예보 변수를 포함한다. A 풍속은 **m/s**, B 풍속은 **km/h**이며 변환하지 않았다. 거리 단위는 m이다. WA temp/temp_max/temp_min의 중복 값도 원본대로 유지한다.

시각은 모두 Asia/Seoul. operating_day=(ts−1ns)의 날짜이며 00시는 전날 마지막 구간이다. operating_hour는 **1~24 구간 끝 번호**(00시=24)이다. month/dayofyear는 operating_day 기준이다. 시간 sin/cos는 2π×operating_hour/24, 연중 sin/cos는 2π×(dayofyear−1)/(해당 연도의 365 또는 366일)이다. 이력에서 학습한 값은 없다.

energy(kWh)와 target_cf=energy/capacity_kw를 float64로 보존한다. 복원은 pred_cf×capacity_kw → kWh이다. negative_energy와 cf_gt_1은 **감사 전용**이며 target/energy/target_present와 함께 모델 입력에서 제외한다. SMP와 k도 정산용 감사 열이며 입력 피처에서 제외한다. 음수·CF>1 target은 삭제/클리핑하지 않는다. 결측 target 행은 학습 시 `target_present`로 제외하며 편입 전을 0으로 학습하지 않는다. lag/rolling/target 기반 집계는 없다.

## 작업 1 경계와 사용법

Train ts≤2025-01-01 00:00, Dev 2025-01-01 01:00~2025-04-01 00:00, Holdout 2025-04-01 01:00~2025-06-01 00:00 KST를 재사용한다. Test는 제출 smp.csv 시간축이다. 과거 SMP 전체 시간축에 target을 붙이므로 편입 전 target 누락도 명시된다. 관측된 target 총수는 원본 generation 행 수와 정확히 일치한다.

```python
from analysis.feature_pipeline import load_dataset
import json
from pathlib import Path
root = Path('analysis/datasets')
m = json.loads((root / 'manifest.json').read_text())
d = load_dataset(root / 'train.parquet')
fit = d.loc[d.target_present]
X = fit[m['feature_columns']]
y = fit.target_cf
# pred_kwh = pred_cf * inference.capacity_kw.to_numpy()
```

전체 파일을 메모리에 올리기 어려우면 `pyarrow.parquet.ParquetFile(...).iter_batches()`를 사용한다. `load_dataset(..., features_only=True)`는 입력 피처만 읽는다. Dev/Holdout의 모델 예측은 작업 1 `aggregate_site_predictions`와 `score_groups`에 `[site_id,ts,pred_kwh]`로 전달한다. 정산 비교 시 `reports/validation/*_settlement_times.csv` 고정 시각을 그대로 유지한다.

## 검증

입력 예보 전체 키, 원본 target 키, 시간대/정시, 결합 행 수, 출력 키·schema, 전체 253개소, 관측 target 개수 보존, Dev/Holdout 행 수, Test target 없음, 원본 해시 불변을 실행 중 검증했다. 테스트 결과는 [test_results.txt](features/test_results.txt)에 저장한다. 모델 학습이나 예측 점수는 이 단계에서 생성하지 않는다.
