# TSMixer v0 과거 1년 검증

모든 명령은 저장소 루트에서 실행한다. 기존 development CLI/설정/체크포인트는 수정하지 않는다.
`runs/tsmixer/dev_v0_seed2026_tmux`는 입력·출력으로 사용하지 않는다.
현재 작업 시작 브랜치는 `feature/tsmixer-v0-mac-cuda`였으며 요청된 브랜치로 자동 전환하지 않았다.
Git push는 수행하지 않는다.

## 시간과 학습

| 단계 | 정답 사용 상한 KST | 평가 운영일 |
|---|---|---|
| Epoch 선택 | 2024-03-31 17:00 | 2024-04-01~2024-05-31 |
| 새 모델 재학습 | 2024-05-31 17:00 | 선택된 Epoch 수만큼 고정 학습 |
| 최종 평가 | 학습/선택 없음 | 2024-06-01~2025-05-31 |

운영일 D는 D 01:00부터 D+1 00:00까지이며 발행 시점은 D−1 17:00이다.
최종 검증 8760시간의 키는 `(2024-06-01 00:00, 2025-06-01 00:00]` KST이다.
경계일 17시 이후 정답은 학습 loss와 정상화 fit의 유효 정답에서 제외한다.
기상은 기존 발행시점 round10/17 예보만 사용한다. 실측 기상은 사용하지 않는다.
정규화·야간 보정은 각 단계 cutoff 이전 정답과 예보로 다시 fit한다.
최종 예측용 데이터 뷰는 정답을 NaN으로 가리고 SMP/k도 읽지 않는다.

모델 구조/입력/AdamW/lr/seed2026/FP32/학습 CF 유효범위는 기존 v0와 같다.
**요청에 따라 SMP/k를 정산 전용으로 제한했다.** 기존 weighted L1에 SMP=k=1을 전달하여
용량 가중 L1을 유지한다. 기존 v0의 가격 가중치와 달라지는 의도된 예외다.
Epoch 선택은 가격 없는 관측 발전소 CF MAE이며 그룹 배정과 무관하다.
두 단계 모두 동일 seed를 다시 설정하고 모델·optimizer를 처음부터 생성한다.
중단 시 부분 산출물을 보존한다. 자동 resume는 지원하지 않으며 새 출력 경로로 재실행한다.

## RTX 3090 Ti 실행

기존 CUDA 가상환경과 전체 원본 데이터가 준비된 서버에서 실행한다. 설치는 기존 README의 CUDA 절차를 따른다.
아래 `/srv/USER/...`는 본인 실제 경로로 바꾼다. 공용 디렉터리를 개인 quota root로 지정하지 않는다.

```bash
source .venv-tsmixer/bin/activate
export OIBC_PYTHON="$PWD/.venv-tsmixer/bin/python"
export OIBC_DATA_ROOT=/srv/USER/oibc-data
export OIBC_STORAGE_ROOT=/srv/USER/oibc
export OIBC_TSMIXER_CACHE_ROOT="$OIBC_STORAGE_ROOT/cache_station_v1"
export OIBC_TSMIXER_RUN_ROOT="$OIBC_STORAGE_ROOT/runs/tsmixer"
export OIBC_RUN_ROOT="$OIBC_TSMIXER_RUN_ROOT"
python -m unittest discover -s analysis/tsmixer/tests -v
python -m analysis.tsmixer environment --device cuda
```

기존 compact cache가 전체 원본 day axis, 90/8/146/253 피처 계약, manifest 서명,
원래 용량 균형 배정, 발행 감사 파일을 충족하면 **읽기 전용으로 그대로 재사용**한다.
학습 실행 전 자동 검사하며 불충족하면 쓰기 전에 종료한다.
Mac에 있는 `runs/deep_learning/compact_real_check/train`은 2025-01-30~31의 2일뿐이므로
스키마는 호환되지만 1년 검증에는 재사용할 수 없음을 실제 확인했다.

전체 캐시가 없을 때만 **새 경로**에 생성한다. 기존 캐시에 이 명령을 실행하지 않는다.
캐시 생성은 서버에서만 한다. 정책 기본값은 개인 전체 12GiB, 캐시 2GiB, 결과 4GiB,
여유 0.5GiB이며 실제 quota를 확인한 경우에만 환경변수를 조정한다.

```bash
export OIBC_TSMIXER_CACHE_ROOT="$OIBC_STORAGE_ROOT/cache_year_backtest_v1"
python -m analysis.deep_learning.prepare \
  --data-root "$OIBC_DATA_ROOT" --cache-root "$OIBC_TSMIXER_CACHE_ROOT" \
  --partition train --layout compact --cache-max-gib 2 --dry-run
python -m analysis.deep_learning.prepare \
  --data-root "$OIBC_DATA_ROOT" --cache-root "$OIBC_TSMIXER_CACHE_ROOT" \
  --partition train --layout compact --cache-max-gib 2
```

팀원 **원본** CSV(`site_id,group`, G1/G2)를 지정한다. `Clustering_sj.md`, `group_sj.json`은
현재 작업 폴더에 제공되지 않아 배정을 추정하지 않았다. 기존 `reports/expanding/composition.csv`는
baseline 검증용이며 팀원 최적화 배정으로 취급하지 않는다.

```bash
bash scripts/tsmixer/train_year_backtest.sh --full-train \
  --config configs/tsmixer/year_backtest.yaml \
  --output "$OIBC_TSMIXER_RUN_ROOT/year_backtest/seed2026_groups_001" \
  --teammate-composition /srv/USER/team_original/composition.csv \
  --metadata-candidate
```

원본 CSV를 아직 받지 못했다면 `--teammate-composition`만 생략할 수 있다. 보고서에 팀원 결과 없음으로 표시한다.
기존 출력 디렉터리가 있으면 항상 거부한다. 설정의 `split`은 기존 설정 스키마 호환용이며
이 모듈에서는 사용하지 않는다. 날짜는 별도 고정 `plan()`으로 관리하여 Holdout 보호 로직을 우회하지 않는다.

## 그룹과 평가 정의

- baseline: 기존 cache의 용량 균형 배정. 원래 balanced_composition과 같아야 한다.
- teammate: 원본 composition.csv를 직접 로드한다. JSON/문서 기반 추정 배정은 사용하지 않는다.
- tsmixer_metadata_candidate: 용량 균형 초기 배정에서 용량 불균형과 기상 관측소별 용량 노출 불균형을 줄이는
  고정 seed 2000회 swap 탐색. TSMixer의 station 입력 구성을 고려한 실험 후보이며,
  정산 성능 최적화가 완료된 배정이라는 뜻은 아니다. API 입력은 발전소 메타데이터와 seed뿐이다.
  검증 실제값/잔차/가격/예측값으로 후보를 선택하지 않는다. 미래 확장 시에도 같은 제한을 지킨다.

모든 배정은 전체 253개 정확히 1회, G1/G2, 그룹별 20,000kW 이상을 검증한다.
같은 최종 예측 NPY를 재집계하며 그룹 때문에 재학습하지 않는다.

`complete_fleet`는 253개 실제 발전량이 모두 유한한 시간만 정산한다.
`observed_only`는 그룹별로 관측된 발전소의 실제와 예측을 동일하게 부분 합산한다.
한 그룹에 정답이 하나도 없는 시간은 그 그룹의 정산을 생략한다.
0 보간은 하지 않는다. 후자는 팀원 구현과 확인 전까지 **팀원과 동일한 평가 방식이라고 주장하지 않는다**.
시간 수는 하나 이상의 그룹이 평가된 고유 시간 수, group_hour_rows는 실제 정산된 그룹×시간 행 수다.
각 모드에 관측 행 수, 유효 시간, 그룹별 시간 수를 따로 저장한다.

발전소 NMAE는 관측행의 `100×|예측−실제|/발전소용량` 평균이다.
그룹 NMAE는 `100×|그룹예측−그룹실제|/해당 관측 발전소 용량합` 평균이다.
complete-fleet에서는 해당 용량합이 고정 그룹 용량이다.

`W = Σ 0.1×SMP×k×|그룹예측−그룹실제|`, k는 시각 11~15시=3, 나머지=1.
`S = Σ 실제×SMP − W`(순정산금), 원시 매출은 revenue_krw로 별도 기록한다.
시간당 페널티는 W/유효 고유시간이며 정확한 시간별 값은 hourly.csv에 있다.
내부점수는 `100×(1−W/223892580.02742994)`이며 기준 일치 미확인 시 비교 불가다.

팀원 기준을 확인했다면 `--reference-contract /path/team_reference.json`을 추가할 수 있다.
JSON은 `{"penalty_reference":223892580.02742994,"contract":{...}}` 형식이며 contract 필드는
metrics.json의 해당 assignment/mode의 contract와 같다. 팀원 평가기에서 독립적으로
생성/검증한 계약만 사용한다. 본 실행의 계약을 복사했다는 이유만으로 팀원 방식이 검증되지는 않는다.
그룹 배정 SHA256, 유효 시간 SHA256, 원래 관측 mask SHA256, 시작/끝,
관측행 수, 그룹시간 수, 부분합/용량분모 규칙까지 모두 일치해야 비교 가능으로 표시한다.

## 산출물

새 결과 폴더 아래 `selection/best.pt`, `selection/last.pt`, `selection/epochs.jsonl`,
`epoch_selection.json`, `refit/last.pt`(고정 Epoch 최종 모델), `refit/epochs.jsonl`,
`run.json`(코드 hash/환경/설정/캐시 감사), `predictions.npy`(CF), `prediction_axis.json`,
각 배정 composition.csv, 각 배정×평가모드의 hourly.csv/groups.csv,
발전소별 sites.csv, metrics.json, REPORT_KO.md를 생성한다.
대용량 산출물은 기존 ignore 대상 runs 또는 개인 외부 스토리지에만 저장한다.
최종 모델과 실제 점수는 GPU 실행 후 생성된다. Mac에서 점수를 만들어 보고하지 않는다.

## 로컬 검증

```bash
.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v
bash -n scripts/tsmixer/train_year_backtest.sh
```

2026-11-03 Mac 경량 검증: 52개 중 51 통과, CUDA 전용 1개 skip.
최초 sandbox 실행의 shared-memory 권한 오류는 sandbox 밖 동일 테스트 재실행으로 통과했다.
신규 테스트는 시간/발행 경계, 정규화의 미래 접근 차단, 가격 없는 학습 배치,
예측 정답 차단, 그룹 무결성/최소용량/seed 재현성,
NaN 정답 미보간, 두 평가 마스크 차이, 내부점수 비교 계약 불일치를 검사한다.
전체 학습은 실행하지 않았다.
