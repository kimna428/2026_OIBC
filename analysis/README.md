# 발전량 관측 및 결측 복원 실험

실행 위치: 저장소 루트. Python 3.13 환경에서 실행했습니다.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r analysis/requirements.txt
MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/analyze_generation.py
MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/build_report.py
```

결과는 `analysis/results/report.html` 또는 `report.md`에서 확인합니다.
입력 파일을 읽기만 하며 실행 전후 SHA-256을 검증합니다. 관측값의 CF 파생
파일과 집계 CSV, 그래프만 저장하며 최종 imputation 파일은 만들지 않습니다.

분류는 시작부 장기결측(>24시간)을 우선하고, 나머지를 1~3시간, 4~24시간,
25시간 이상으로 나눕니다. 시작/내부/종료 위치도 별도 열로 기록합니다.
전체 분모는 `train/smp.csv`의 시간축이며 활성 구간 관측률은 첫·마지막
실제 관측 사이를 분모로 사용합니다.

복원 실험은 개별 발전소의 독립적인 artificial missing이며 같은 실험 내에서
대상 블록의 관측값을 사용할 수 없습니다. 시작부 실험은 그 이전 해당
발전소 관측도 사용할 수 없습니다. 기상·미래 관측은 사후 복원에 사용할 수
있는 것으로 가정합니다. 실험 조건, 복원률, 공통 시각 오차, 한계는 보고서에
기재됩니다. 실제 결측 길이의 빈도와 artificial missing 표본의 길이별 빈도는
서로 다르므로 전체 벤치마크 평균을 실제 결측 기대오차로 해석하지 않습니다.

## Forecast baseline

```sh
uv pip install --python .venv/bin/python -r analysis/forecast_requirements.txt
MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/test_forecast_baseline.py
MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/forecast_baseline.py
MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/audit_forecast_baseline.py
```

결과 위치는 `analysis/results/forecast_baseline/`입니다. CatBoost global
model 1개에 round=17 예보 30개, 시간 특징 7개, 메타 특징 8개를 사용합니다.
학습은 2024-06-01 01:00 KST 이전 실제 관측이며, 이후 마지막 1년이 holdout입니다.
random split이나 holdout 기반 조기종료를 사용하지 않고 400회 고정 학습합니다.
모델이 학습한 발전소와 학습 관측이 없는 신규 발전소를 따로 집계합니다.

발전량 결측은 복원하지 않고 존재하는 유한 관측만 target으로 사용합니다.
원본 음수 target이나 CF>1도 변경하지 않습니다. 예보 결측은 학습 중앙값으로
처리하며 관측 기상은 읽지 않습니다. test의 예보 schema에도 존재하는 변수만
사용하며 station/ts/round 키 유일성과 원본 SHA-256 불변성을 확인합니다.

`validation_predictions.parquet`는 요청한 7개 열을 포함합니다.
`residual_kwh = actual_kwh - pred_kwh`로 양수 bias는 과소예측입니다.
`site_metrics.csv`에는 CF/energy MAE, RMSE, bias, 11~15시 MAE,
SMP 가중 error 합계·정규화 MAE 및 학습 관측 수, 신규 여부가 있습니다.
`weighted_error = Σ SMP × k × |residual_kwh|`, `k=3`(KST 11~15시),
나머지 `k=1`이며 `weighted_mae_kwh = weighted_error / Σ(SMP × k)`입니다.
정산식의 0.10은 이 지표에 곱하지 않습니다.

추론 CF는 음수만 0으로 제한하며 야간 고정이나 상한 clipping은 없습니다.
`model.cbm`은 검증용 train-only 모델입니다. 재사용할 때
`feature_manifest.json`에 있는 특징 순서, 범주, train 중앙값을 유지해야 합니다.
`build_features()`는 test 예보에도 동일하게 적용할 수 있습니다.

잔차 상관 CSV와 공통 관측 수 CSV는 all-hour 및 11~15시를 각각 저장합니다.
동시 관측을 사용하고 NaN을 0으로 채우지 않습니다. 아직 G1/G2 최적화나
전체 자료 재학습·test 예측·제출 파일 생성은 하지 않습니다. 이 holdout residual로
향후 그룹을 최적화한 뒤에는 별도의 시간 구간에서 최종 성능을 검증해야 합니다.

`audit_forecast_baseline.py`는 저장된 검증 예측을 원본 관측과 전수 대조하고,
설비용량 환산·가중치·residual·시점 분리·공통 관측 수를 확인합니다.
저장 모델을 다시 읽어 256개 시각/발전소 표본의 예측을 재현하고 원본 해시와
보고서 링크를 검증합니다. 결과는 `verification.json`, 잔차 분위수 및
기상지점 공유 여부별 주간 상관 통계는 `residual_summary.json`입니다.

## Fixed local validation and settlement

The reusable `validation.py` implements the Jan–Mar Dev and Apr–May Holdout
protocol, fixed complete-fleet masks, G1/G2 composition validation, KRW settlement,
and separate site-level errors. The earlier forecast baseline's split is unchanged;
use these new windows explicitly for subsequent experiments.

```sh
.venv/bin/python -m unittest discover -s analysis -p test_validation.py -v
.venv/bin/python analysis/build_validation_protocol.py
```

See `reports/validation_protocol.md` for input contracts, units and examples.
Audits and fixed scoring timestamps are in `reports/validation/`. No model score
is reported without actual forecasts.

## Round-17 feature datasets

```sh
.venv/bin/python -m unittest discover -s analysis -p test_feature_pipeline.py -v
.venv/bin/python analysis/build_feature_datasets.py
```

`feature_pipeline.py` provides the shared chunk builder and Parquet loader.
Outputs are `analysis/datasets/{train,dev,holdout,test}.parquet`; select model
inputs using `manifest.json`'s `feature_columns`. Targets, target-derived flags,
SMP and k are audit columns, excluded from model inputs. Missing targets remain
NaN; filter `target_present` for fitting. See `reports/feature_pipeline.md`.

## LightGBM fixed baseline

```sh
uv pip install --python .venv/bin/python -r analysis/lightgbm_requirements.txt
# macOS runtime if missing: brew install libomp
MPLCONFIGDIR=/tmp/oibc-lgb-mpl .venv/bin/python analysis/lightgbm_baseline.py
.venv/bin/python -m unittest discover -s analysis -p test_lightgbm_baseline.py -v
```

Uses the Task 2 datasets and Task 1 fixed masks, with 300 fixed CPU rounds and
no early stopping or validation-based model selection. Model, Dev/Holdout site
predictions, metrics and settlement ledgers are saved in
`analysis/results/lightgbm_baseline/`. Capacity clipping defaults to disabled;
`--clip-capacity` enables CF <= 1 for a separate rerun. Negative forecasts are
always floored at zero. See `reports/baseline_lightgbm.md` for measured results.

## Six-fold day-ahead expanding pipeline

```sh
# Existing environment includes LightGBM, scipy, pandas, pyarrow, matplotlib.
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python -m unittest discover -s analysis -p test_expanding_pipeline.py -v
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python analysis/expanding_pipeline.py
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python analysis/audit_expanding_pipeline.py
# Full-row experiment: use a separate output directory.
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python analysis/expanding_pipeline.py --max-train-rows 0 --trees 300 --out reports/expanding_full
```

The measured default run uses a shared deterministic 200,000-row training sample
per fold and 60 trees. This is a compute-limited baseline; all validation observations
are evaluated. Outputs, models, configuration, raw input hashes, V0–V3 datasets,
separate single-hour interpolation additions, per-fold predictions, and reports
are in `reports/expanding/`. Read `report.md` and `dataset_manifest.json` before
using the outputs. V0/V1/V2 feature matrices are reproducibly derived by the builder;
V3 stores the combined feature matrix and uses explicit feature allowlists.
Train with `train_eligible` and the fold cutoff together; never use target/audit
columns as model inputs. Validation uses original finite targets even if flagged.
The audit command enriches the report and independently checks all 42 prediction
files against raw observations. Run it once after each pipeline run.

For a fresh environment, inspect installed packages first, then install
`analysis/expanding_requirements.txt` if needed. This includes CatBoost because
its existing module supplies the shared forecast builder; the new fitted models
use LightGBM. `--resume` reuses prepared data and existing fold models after
checking raw input hashes, training settings, and model feature order. It still
recomputes metrics and audits. Date aggregation uses unique operational dates
to avoid expensive per-row timezone formatting.

```sh
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python analysis/expanding_pipeline.py --resume
# Audit a separately configured full-data run:
MPLCONFIGDIR=/tmp/oibc-expanding-mpl .venv/bin/python analysis/audit_expanding_pipeline.py --out reports/expanding_full
```

The audit report also includes an observed-row-weighted hourly CF MAE table
and plot (`hourly_performance.csv/png`), explicit fold periods and input inventory.
`official_group_metrics.csv` contains G1/G2 MAE, penalties and scores on the
same fixed complete-fleet timestamps as the settlement ledger.

## Third experiment: reliability, scale, forecast rounds and OOF grouping

```sh
# No new dependency installation was needed in the current environment.
MPLCONFIGDIR=/tmp/oibc-third-mpl .venv/bin/python -m unittest discover -s analysis -p test_third_experiment.py -v
MPLCONFIGDIR=/tmp/oibc-third-mpl .venv/bin/python analysis/third_experiment.py
MPLCONFIGDIR=/tmp/oibc-third-mpl .venv/bin/python analysis/audit_third_experiment.py
```

Outputs live in `reports/third_experiment/`. `report.md` describes all experiments,
selection limitations, raw data assumptions and score masks. The default run fits
96 fixed-seed LightGBM models, including tuning, 200k/1m/full training and four
forecast round settings, with three postprocessing policies evaluated per main
experiment. Each fit is checkpointed and a repeated command reuses matching
checkpoints. Input hashes, feature order and hyperparameters must match.

To run stages separately, use `--stage audit`, `--stage tune`,
`--stage development`, then `--stage holdout`. Run them in that order. The final
stage requires existing `selection_lock.json` and `composition_lock.json`;
it never selects parameters or groups from holdout results. A different experiment
configuration should use a new `--out` directory. The feature cache uses only
provided forecasts/metadata and causal history; the independent auditor rebuilds
sample feature values from raw sources and recomputes all main experiment scores.

`OOF_B.parquet` and `OOF_C.parquet` contain only folds 0–4 and can be used for
G1/G2 research; `selected_B_holdout.parquet` and `selected_C_holdout.parquet`
are held out from selection. Group optimization uses only complete-fleet OOF
hours; partial target periods are never filled with zero. The older stage-two
report already exposed the spring holdout, so stage three offers a frozen
procedure but cannot claim a completely new unseen holdout.

Strict historical parameter selection is recorded in `parameter_schedule.json`:
each fold only uses tuning windows fully ended before its first issuance. Fold 0
uses the fixed initial grid; fold 5 can use all tuning folds 0–3. Prediction files
include `parameter_source_max` in addition to the fitting cutoff. Model/group
selection excludes the last seven development hours after 2025-02-28 17:00 KST;
`scores_selection.csv` is distinct from full-period development scores. OOF and
March bias-calibration training also respect this final issuance cutoff.

The final preferred B/C recipe is locked using development penalties only in
`preferred_candidate_lock.json` before holdout fits. C's rolling-history result
remains conditional on incoming operational generation: the supplied test folder
contains no generation stream. B can use the supplied test forecasts alone.

Forecast round comparisons use the same maximum-one-million-row sample (r17,
r10, both, and both plus revisions); r17 alone is also compared at 200k and all
eligible rows. The independent audit additionally re-infers the selected C models
with generation frozen at the first issuance. This no-stream sensitivity is saved
separately as `offline_C_metrics.csv`, `offline_C_scores.csv`, and
`offline_C_selected_holdout.parquet`; it does not change frozen selections.
Fresh environments can use `third_requirements.txt` after checking installed
versions. The existing workspace required no installation.
