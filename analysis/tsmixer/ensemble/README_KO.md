# Walk-forward 및 convex 앙상블

프로토콜: [WALKFORWARD_PROTOCOL_KO.md](../../../reports/tsmixer/WALKFORWARD_PROTOCOL_KO.md)

서버 설치·smoke·Fold 학습·가중치 고정·평가 명령: [GPU_EXECUTION_PLAN_KO.md](../../../reports/tsmixer/GPU_EXECUTION_PLAN_KO.md)

실제 가용 자료 및 한계: [ENSEMBLE_RESULTS_KO.md](../../../reports/tsmixer/ENSEMBLE_RESULTS_KO.md)

Mac 경량 실행:

```bash
export OIBC_PYTHON="$PWD/.venv/bin/python"
bash scripts/tsmixer/run_walkforward.sh audit --output runs/tsmixer/ensemble/walkforward_v1/audit_new
bash scripts/tsmixer/run_walkforward.sh smoke --fixture --updates 5 --output runs/tsmixer/ensemble/walkforward_v1/smoke_new
"$OIBC_PYTHON" -m unittest analysis.tsmixer.tests.test_walkforward -v
```

기본 config의 run_root 하위에만 산출물을 저장한다. 기존 경로는 평가 시 거부하며 학습은 `--resume`과 동일 recipe가 있어야 재개한다. 완료 hash가 다르면 실패한다. 새 설정/코드는 새 실행 루트를 사용한다.

외부 예측은 CSV 또는 Parquet의 `ts,site_id,prediction`이다. timestamp는 timezone이 명시된 구간 종료 KST, site_id는 원본 canonical 값이다. 순서가 달라도 키로 정렬한다. 매 관측 정답행 예측은 유한한 비음수여야 한다. 단위 `capacity_factor`이면 용량으로 kWh 변환한다. 예측 파일만으로 OOF 여부를 확인할 수 없으므로 JSON에 아래 필드와 실제 증거를 제공해야 한다.

- `model`: tsmixer / sj_ensemble / state01
- `fold`: F1 / F2 / F3
- `protocol`: past_only_walkforward_v1, `smoke_only`: false
- `start`, `end_exclusive`: folds.py의 정확한 timezone 포함 문자열
- `timestamp_convention`: interval_end, `units`: kWh 또는 capacity_factor, `column`: prediction
- `training_cutoff`, `normalization_cutoff`, `selection_cutoff`: timezone 포함, 해당 Fold cutoff 이하
- `auxiliary_cutoff`, `outage_cutoff`, `calibration_cutoff`: 해당 단계 cutoff 또는 미사용 시 null
- `calibration_factor`: 1 또는 과거 추정 provenance가 확인된 양수
- `inference_inputs`: forecasts/solar_geometry/calendar/site_metadata/predicted_weather의 사용 항목
- `prediction_sha256`: 실제 파일 SHA256
- `data_hashes`: actual_grid가 원본 정답/SMP/메타데이터에서 산출한 source_hashes와 동일한 사전

자체 학습 산출물에는 데이터·소스·환경·피처·정규화·체크포인트 hash를 자동 기록한다. 외부 JSON의 선언만으로 원본 학습 기록이 증명되는 것은 아니다. 검토되지 않은 01_state 원본, 기존 TSMixer legacy F3, smoke를 새 OOF로 등록하면 안 된다.

`score --components`는 개별 component 진단이며 component와 가족 앙상블을 동시에 simplex 축으로 넣지 않는다. 월별/계절별/시간별/규모별 진단과 각 그룹 시간별 SMP/k/예측오차/페널티 ledger를 저장한다. 이 ledger로 고가격 시간대의 모델 기여도를 확인할 수 있다. F3는 grid 탐색을 차단한다.
