# RTX 3090 Ti 실행 계획

아래 명령은 사용자가 서버에서 실행한다. 이번 작업에서는 서버 접속/학습을 실행하지 않았다. 서버에 Codex는 필요 없다. 소스와 경량 설정을 전달하고 팀원 복원 배정 CSV는 ignored runs 경로에 별도 전달한다. 프로젝트 코드 전달 예시는 아래와 같다(자동 실행하지 않음).

```bash
# Mac 프로젝트 루트: 대용량 데이터/가중치/기존 결과를 제외하고 필요한 코드만 전달
rsync -av analysis/tsmixer/ensemble/ hai-gpu-209:/home/user/2026_OIBC/analysis/tsmixer/ensemble/
rsync -av analysis/tsmixer/comparison/ hai-gpu-209:/home/user/2026_OIBC/analysis/tsmixer/comparison/
rsync -av configs/tsmixer/ensemble_walkforward.yaml configs/tsmixer/ensemble_sj_features.json hai-gpu-209:/home/user/2026_OIBC/configs/tsmixer/
rsync -av scripts/tsmixer/run_walkforward.sh scripts/tsmixer/run_ensemble.sh hai-gpu-209:/home/user/2026_OIBC/scripts/tsmixer/
rsync -av analysis/tsmixer/requirements-walkforward.txt hai-gpu-209:/home/user/2026_OIBC/analysis/tsmixer/
# 대상 배정 디렉터리가 이미 존재하는지 서버에서 확인 후 전달
rsync -av runs/tsmixer/comparison/local_audit_20261104_002/teammate_composition.csv hai-gpu-209:/home/user/2026_OIBC/runs/tsmixer/comparison/local_audit_20261104_002/
```

```bash
cd /home/user/2026_OIBC
export OIBC_PYTHON=/home/user/ai_env/bin/python
export OIBC_DATA_ROOT='/home/user/oibc_dataset/handover 2/01_participant_dataset'
export OIBC_TSMIXER_CACHE_ROOT=/home/user/2026_OIBC/runs/tsmixer/cache_station_v1
export OIBC_ENSEMBLE_ROOT=/home/user/2026_OIBC/runs/tsmixer/ensemble/walkforward_v1
export OIBC_STORAGE_ROOT=/home/user/2026_OIBC
export OIBC_RUN_ROOT="$OIBC_ENSEMBLE_ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi
"$OIBC_PYTHON" -c 'import torch,pandas,numpy,pyarrow,lightgbm,xgboost; print(torch.__version__,torch.cuda.is_available(),lightgbm.__version__,xgboost.__version__)'
bash scripts/tsmixer/run_walkforward.sh audit --output "$OIBC_ENSEMBLE_ROOT/audit_server_001"
# 한 번에 한 모델: smoke는 NN update5, OOF 등록 금지
bash scripts/tsmixer/run_walkforward.sh smoke --fold F1 --model tsmixer --updates 5 --resume
bash scripts/tsmixer/run_walkforward.sh smoke --fold F1 --model sj_ensemble --updates 5 --resume
bash scripts/tsmixer/run_walkforward.sh smoke --fold F1 --model state01 --updates 5 --resume
# 로그의 준비 시간/RSS/GPU peak를 확인한 뒤, 필요한 하나만 선택하여 실행
bash scripts/tsmixer/run_walkforward.sh train --full-train --fold F1 --model tsmixer --resume
bash scripts/tsmixer/run_walkforward.sh train --full-train --fold F1 --model sj_ensemble --resume
bash scripts/tsmixer/run_walkforward.sh train --full-train --fold F1 --model state01 --resume
```

원본 데이터/compact cache 경로는 기존 서버 배치에 맞춘 값이다. audit에서 존재/축/원본 hash를 확인한다. cache를 덮어쓰거나 재생성하지 않는다. 의존성이 없으면 먼저 requirements-walkforward.txt를 검토하고 기존 환경을 변경할지 결정한다. Mac에는 설치하지 않았다.

기본 seed1개, fp32, state daybatch1/SJ batch256, max30Epoch/patience6이다. 3090Ti24GB에서 실측 VRAM/학습시간은 미측정이며 성공을 보장하지 않는다. state는 발전소253개를 하루 단위로 공유 GRU에 투입한다. 트리는 CPU이며 full matrix 준비 RAM이 커질 수 있어32GiB 이상 가용 RAM을 권장하되 실제 peak를 smoke/단일 job로 확인한다. smoke는 전체 tree fit을 하지 않으므로 CPU tree 시간/메모리는 별도 미검증이다. 기존4GiB 결과/2GiB cache 등 저장 정책을 자동 확대하지 않으며 quota 실패 시 원본을 삭제하지 않고 필요한 저장 예산을 검토한다. 각 모델×Fold마다 내부 선택+refit이 필요하므로 일괄9개 실행 대신 하나씩 진행한다.

```bash
# F1 결과로 선택: F2/F3를 아직 보지 않는다
bash scripts/tsmixer/run_ensemble.sh score --fold F1 --output "$OIBC_ENSEMBLE_ROOT/score_F1_v1"
bash scripts/tsmixer/run_ensemble.sh select --inputs "$OIBC_ENSEMBLE_ROOT/score_F1_v1/metrics.json" --strategy site --output "$OIBC_ENSEMBLE_ROOT/weights/F1_site.json"
# F2 OOF는 위 train 명령의 --fold F2로 필요한 모델만 생성한 뒤
bash scripts/tsmixer/run_ensemble.sh apply --fold F2 --weights "$OIBC_ENSEMBLE_ROOT/weights/F1_site.json" --output "$OIBC_ENSEMBLE_ROOT/confirm_F2_site"
bash scripts/tsmixer/run_ensemble.sh score --fold F2 --output "$OIBC_ENSEMBLE_ROOT/score_F2_v1"
bash scripts/tsmixer/run_ensemble.sh select --inputs "$OIBC_ENSEMBLE_ROOT/score_F1_v1/metrics.json" "$OIBC_ENSEMBLE_ROOT/score_F2_v1/metrics.json" --strategy site --output "$OIBC_ENSEMBLE_ROOT/weights/F1F2_site.json"
# F3 OOF 생성 후: F3 결과로 가중치를 다시 바꾸지 않는다
bash scripts/tsmixer/run_ensemble.sh apply --fold F3 --weights "$OIBC_ENSEMBLE_ROOT/weights/F1F2_site.json" --output "$OIBC_ENSEMBLE_ROOT/final_F3_site"
bash scripts/tsmixer/run_ensemble.sh score --fold F3 --output "$OIBC_ENSEMBLE_ROOT/score_F3_v1"
# 고정 baseline: 계수 순서 tsmixer, sj_ensemble, state01
bash scripts/tsmixer/run_ensemble.sh fixed --coefficients .1 .8 .1 --output "$OIBC_ENSEMBLE_ROOT/weights/manual.json"
bash scripts/tsmixer/run_ensemble.sh fixed --coefficients .3333333333333333 .3333333333333333 .3333333333333333 --output "$OIBC_ENSEMBLE_ROOT/weights/equal.json"
bash scripts/tsmixer/run_ensemble.sh apply --fold F3 --weights "$OIBC_ENSEMBLE_ROOT/weights/equal.json" --output "$OIBC_ENSEMBLE_ROOT/final_F3_equal"
bash scripts/tsmixer/run_ensemble.sh summary --inputs "$OIBC_ENSEMBLE_ROOT/score_F1_v1/metrics.json" "$OIBC_ENSEMBLE_ROOT/score_F2_v1/metrics.json" "$OIBC_ENSEMBLE_ROOT/score_F3_v1/metrics.json" --output "$OIBC_ENSEMBLE_ROOT/summary_v1"
```

penalty/joint 전략은 위 select의 --strategy와 별도 weights/output 이름만 변경한다. apply는 F3 grid를 수행하지 않는다. 고정 baseline도 F1/F2/F3 각각 apply하여 비교한다. 새 결과 폴더 이름을 사용하며 평가 폴더는 덮어쓰지 않는다.

기존 OOF 확보 시 학습 대신 아래로 등록한다. 메타데이터는 실제 cutoff/전처리/보정 증거로 작성해야 한다. 경로만 넣거나 기존 원본 예측에 프로토콜 이름을 붙이는 것으로 검증되지 않는다.

```bash
bash scripts/tsmixer/run_ensemble.sh import --fold F1 --model sj_ensemble --prediction /absolute/path/F1.parquet --metadata /absolute/path/F1.json --output "$OIBC_ENSEMBLE_ROOT" --resume
```

현재 필요한 자료는 SJ/state01 각 Fold별 발전소 예측과 전처리·모델선택·보정 cutoff 기록, 가능하면 내부 OOF 및 그룹 생성 로그다. 확보되면 hash/축/provenance 검사 후 누락 Fold만 학습한다.
