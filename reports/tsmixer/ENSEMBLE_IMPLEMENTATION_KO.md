# 앙상블 구현 및 검증

새 구현은 analysis/tsmixer/ensemble에 분리했다. folds/data는 시간 경계와 train-only view, features/teacher/datasets는 팀원 피처와 과거 보조학습, models/trees/trainer는 모델별 선택·refit, predictions/persistence는 정렬·provenance·재시작, blend/weight_selection/evaluation/run은 공통 평가와 가중치 고정을 담당한다. 설정은 configs/tsmixer/ensemble_walkforward.yaml이다.

모델 계열은 tsmixer, sj_ensemble, state01이다. SJ는 .25 LightGBM+.25 XGBoost+.5 BiGRU, state01은 .1 LightGBM+.1 XGBoost+.4 운량 GRU+.4 기상14 GRU다. 기존 Windows 프로그램이나 pickle/joblib을 실행/로드하지 않고 검토한 소스에서 필요한 순수 피처 함수를 프로젝트 안으로 옮겼다. 원본은 변경하지 않았다.

TSMixer는 기존 모델/손실/train_epoch/predict를 재사용한다. SJ는 86개 트리 피처와 101개 XGB 피처, 수치/결측/범주 embedding을 사용하는 BiGRU다. state01은 45입력→192 projection→2층 양방향 GRU192→발전량 및 보조 head다. state 손실은 masked site/group 가격·용량 가중 오차+보조 손실이다. 상세 파라미터는 configs/tsmixer/ensemble_sj_features.json 및 models.py에 고정되어 있다.

01_state 수정: 17시 이후 정답을 제거하고 outage는 과거만으로 계산한다. 보조학습 forecast 범위도 train-only fit하고 실측은 cutoff까지 사용한다. 유효한 표준화0과 결측을 구분한다. 추론에는 실제 발전량/실측을 전달하지 않는다. 기존의 누락 예측0 평가를 재사용하지 않는다. SJ 연도 보정계수는 가져오지 않고 raw factor1로 시작한다.

이식 구현은 기존 저장 모델과 bitwise 동등하다고 검증되지 않았다. state는 float32 lazy 입력, day batch1, 내부 Epoch 선택을 사용하며 원본 고정30 Epoch/float16 cache와 다르다. SJ는 train-only teacher, 과거 운영일 기반 zero-spell 처리를 사용한다. 원본 패키지의 정리된 캐시와 행 필터가 완전히 같다고 주장하지 않는다. seed2026, teacher의 명시된 seed42를 각각 기록한다.

예측은 Parquet의 ts/site_id/prediction(kWh)와 같은 이름 JSON이다. JSON은 model/fold/protocol/start/end_exclusive, interval_end, 단위, training/normalization/auxiliary/outage/calibration/selection cutoff, 허용 추론 입력, 보정 계수, 데이터 hash, 파일SHA를 포함한다. 다른 데이터판, 미래 cutoff, 중복/누락 site, 시간 이동, 관측행 예측 NaN, 음수, smoke 결과는 거부한다. CF 입력은 용량을 곱해 변환한다. 기존 NPY는 기존 비교 어댑터로 읽을 수 있지만 provenance를 사실과 다르게 작성해 OOF로 승격하면 안 된다.

모든 조합은 site 예측을 먼저 convex 혼합한 뒤 그룹 합산한다. 0.1 simplex66개+정확한 동일가중치1개를 F1/F2에서만 평가한다. .1/.8/.1 baseline과 단독/두모델 조합도 포함된다. 가족과 component를 동시에 grid 축에 넣지 않는다. --components는 별도 진단이다. 누락 모델은 unavailable이며0대체/재정규화하지 않는다.

재시작은 --resume으로 recipe/source/config/데이터/환경 hash를 대조한다. Epoch별 optimizer/scheduler/RNG와 정규화 JSON, outage 판정, tree/teacher 단계 hash, 완료 manifest를 보존한다. 완성된 동일 작업은 재생성하지 않으며 손상/다른 recipe는 거부한다. 새로운 설정은 새 OIBC_ENSEMBLE_ROOT를 사용한다. 실제 학습과 smoke는 Linux CUDA에서만 허용한다.

기존 코드의 유일한 호환성 수정은 comparison/io.py의 SHA 계산에 Python3.10 fallback을 추가한 것이다. 기존 값은 동일하다. .gitignore에 XGBoost .ubj를 추가했다. 기존 체크포인트/연간 결과/원본 팀원 폴더는 수정하지 않았다.

검증 결과와 실행 명령은 ENSEMBLE_RESULTS_KO.md, GPU_EXECUTION_PLAN_KO.md를 참고한다. 실제 CUDA 전체 학습/트리 전체 학습의 완주는 이번 Mac 작업에서 검증하지 않았다.
