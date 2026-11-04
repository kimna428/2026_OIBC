# TSMixer 개선 실험 우선순위 — 2026-11-04

현재 근거로는 모델 크기 확대나 BiGRU 전환을 먼저 할 이유가 부족하다.
TSMixer는 약30.6만, 팀원 BiGRU는 약5.17만 파라미터이며 피처·손실·마스크가 다르다.
다음 실험의 성공 기준은 공통 observed-only 계약에서 **고정 recipe의 W 감소**이며,
TSMixer 단독 순위가 같더라도 팀원 앙상블에 추가했을 때 W가 줄면 가치가 있다.
확인된 점수 개선은 아직 없다. 아래 자원 수치는 계획용 범위이며 벤치마크가 아니다.

## 공통 통제와 비용 기준

- 기존 모델/설정/체크포인트를 수정하지 않고 새 variant와 새 출력 경로를 쓴다.
- 내부 selection은 cutoff 2024-03-31 17시, 4~5월 검증; 선택 Epoch를 고정해 cutoff 2024-05-31 17시로 새로 refit한다.
- 연간 실제/잔차로 feature·보정·그룹·seed·alpha를 골랐다면 탐색으로 표시한다.
- 그룹 A/B/C를 모두 보되 주 배정 하나를 사전 지정한다. 팀원 final 배정은 manifest hash로 고정한다.
- 평가 실제값은 clipping/0채움/의심0 제거를 하지 않는다. 학습 mask 변경과 평가 mask 변경을 분리한다.
- 기존 v0 연간 실험은 **SMP/k 정산 전용** 계약이다. 현재 구현을 바꾸지 않는다.
- 확인된 refit 로그는 1 Epoch당 2,897 optimizer updates, 2 Epoch 총5,794회다.
  로그에 시간/VRAM 측정값이 없어 GPU 소요시간은 실측하지 못했다.
- 시간 예산은 첫 50~100 update의 데이터로더 포함 wall time과 peak allocated/reserved를 서버에서 측정해 외삽한다.
  예: 1 update가 t초라면 refit2 epoch는 약5794t초이며 selection·평가·I/O를 별도 더한다.
  가령 t=.02~.10초라면 refit 계산부 약1.9~9.7분이지만 이는 **조건부 계산 예시**다.

## 우선 1 — 저장 예측으로 보완성 확인, 과거 OOF로 소량 앙상블 선택

**변경:** 현재 구현된 비교 도구로 alpha 0/.05/.10/.15/.20/.30/.40/.50을 모두 분석한다.
내부1년 곡선은 탐색으로만 보고, cutoff 이전 OOF가 있으면 그 자료의 W로 alpha를 고정한다.
보정도 동일하게 과거 OOF에서만 추정하는 별도 후속 후보로 둔다. 현재 코드는 새 보정계수를 학습하지 않는다.

**이유:** 모델 구조가 달라 site MAE가 나빠도 그룹 잔차 상쇄에 기여할 수 있다.
기존 metadata 그룹의 보고된 W 개선은 약0.091%로 작다. 그룹의 미세 조정보다 독립 오차를 섞는지 먼저 확인한다.
단순 잔차 상관이 낮다는 것만으로 정산 개선을 보장하지 않는다.

**코드 범위:** comparison adapters/core/diagnostics는 구현·테스트 완료.
추가로 필요한 것은 팀원 validation/OOF, TSMixer 연간 예측 및 과거 OOF다.
OOF가 없으면 시간순 fold별 TSMixer 학습 드라이버가 별도로 필요하며 아직 구현하지 않았다.

**GPU/메모리:** 저장 예측 분석은 GPU0, Mac CPU 사용. 8760×253 float64 배열 하나는 약17.7MB다.
component·잔차·태양위치·DataFrame을 포함한 peak RSS는 수백MB~1GB를 계획하되 실측은 안 했다.
OOF를 새로 만들면 2~3개 과거 fold 학습 비용이 추가되므로 첫 분석과 구분한다.

**검증/누수:** 같은 mask/가격/배정. OOF의 양쪽 모델 cutoff는 각 발행 이전이어야 한다.
그룹이 OOF fold 이후 잔차로 선택됐다면 metadata-only 그룹이나 fold별 이전 그룹으로 대조한다.
팀원은 이미 연간 검증을 반복 탐색했으므로 이 구간에서의 개선을 독립 최종 성능으로 홍보하지 않는다.

**성공:** 과거 OOF에서 고정한 alpha가 이후 구간의 W를 alpha0보다 줄이고 개선이 특정 월 하나에만 의존하지 않는다.
**실패/보류:** 연간 탐색상 alpha0이 계속 우세하거나 OOF 선택 alpha가 이후 계절에 악화한다.
당장 OOF가 없으면 탐색 결과 이상으로 결론 내리지 않는다.

**실행:** 비교 README의 rsync와 공통 평가 명령은 지금 사용 가능한 정확한 명령이다.
`--oof-predictions`는 실제 OOF가 있을 때만 넣는다. 이 작업에서는 서버 접속/학습을 실행하지 않았다.

## 우선 2 — 전처리·피처·학습 mask를 한 번에 하나씩 대조

**변경 순서:**

1. 현90/8/146 입력을 유지하면서 train-only 물리단위/robust scale와 안전clip을 별도 variant로 대조한다.
2. round17 우선/10 보완/운영일 내부 짧은 보간/과거 동일시각 carry와 source-age flag를 별도 피처 계약으로 추가한다.
3. 태양 azimuth/coszenith·구간별 야간 판정·제공사 차이·주간 문맥을 개별 feature bundle로 대조한다.
4. sunlit-only, 높은 태양고도의 의심0 제거, 7일 연속 관측0 제외를 **각각** 독립 mask 실험으로 검증한다.
   기존 CF [0,1] 필터도 팀원 CF clipping과 혼동하지 말고 raw 분포부터 확인한다.

한 번에 모두 바꾸면 이득의 원인을 구분할 수 없다. 가장 작은 pilot 하나를 내부 검증에서 고른 후 다음으로 간다.

**이유:** 팀원은 unit 변환, median/고정 물리scale, ±8clip, forecast fill, 유효 주간 supervision을 쓴다.
현재 TSMixer는 다른 단위/standard scale/야간 포함 loss를 쓰므로 입력과 최적화 난이도의 차이가 클 수 있다.
BiGRU 자체의 우위라는 가설보다 먼저 검증할 수 있다.

**코드 범위:** 새 `analysis/tsmixer/experiments/`의 feature-contract/normalizer/mask와 별도 trainer/config가 필요하다.
현재 v0 `open_cache`는 90/8/146 계약을 엄격히 검사한다. 새 피처를 기존 compact cache에 덮어쓰거나
검사 조건을 완화하지 않는다. 원본90을 읽는 변환 view 또는 새로운 서명/경로의 cache를 쓴다.
기존 TSMixer 모델 클래스/개발 CLI/학습 설정은 보존한다.

**GPU/메모리:** scale/mask만 바꾸면 기존 모델과 거의 같은 계산량을 예상한다.
계획용 VRAM2~6GiB, 1개 pilot은 selection 최대6 + refit 최대6 epoch, 대략3.5만 update 이내로 시작한다.
기존 recipe의 최대30/early-stop6 결과와 같은 수렴 예산이 아니므로 pilot 결과는 초기 선별용이다.
6 epoch에서 아직 개선 중이면 충분한 epoch로 재검증한다. 첫 pilot 시간 측정 후 나머지 개수를 정한다.
피처 추가는 입력 projection과 I/O가 늘며 cache 저장예산도 별도 검사한다.

**부작용/누수:** 오래된 예보 carry가 부정확할 수 있어 age를 함께 기록한다.
하루24시간 문맥은 같은 발행 예보만 사용한다. 이후 운영일 예보를 보간 원천으로 쓰지 않는다.
관측0 제거는 실제 저발전/정지 이벤트를 학습에서 지워 과대예측을 키울 수 있다.
이것을 평가0 제거로 해결하면 안 된다. 중앙값/clip/feature 선택도 train-only다.

**성공:** 내부 검증에서 W/MAE/bias와 미학습48개 cohort가 함께 개선되거나 W가 줄고 연간 고정 적용에서 유지된다.
**실패:** site MAE만 개선하면서 peak 그룹 W가 커지거나 의심0 많은 발전소에 일관된 과대 편향이 생긴다.
복수 seed 확인은 후보1개로 줄인 후 한다.

**서버 명령 상태:** 새 전처리 variant/trainer는 이번 분석 범위에서 구현하지 않았다.
현재 train_year_backtest.sh에 가상의 scale/mask 플래그를 붙여 실행하면 안 된다.
아래 재현 명령은 기존 recipe+복원 그룹의 **새 기준 실행**에만 유효하며 이 실험의 대체가 아니다.
실험 trainer 구현·경량 테스트 후에 그 실제 CLI로 별도 명령을 확정해야 한다.

## 우선 3 — 손실과 Epoch 선택을 목표 지표에 맞춘 별도 실험

**변경:** 용량 가중 site L1과 CF MAE를 먼저 대조하고,
그 다음 작은 계수의 관측 그룹합 절대오차를 더하는 variant를 평가한다.
시간대 강조를 시험한다면 timestamp에서 사전 정의한11~15시 가중치를 별도 실험 recipe로 선언한다.
기존 백테스트의 SMP/k 정산 전용 정책과 분리하고 검증 SMP·정답으로 학습 가중치를 추정하지 않는다.
현재 v0의 loss/selection을 수정한 것은 아니다.

**이유:** 용량 가중 site error의 합과 그룹합의 절대오차는 다르다.
같은 방향으로 발생하는 오차를 줄이거나 서로 상쇄하도록 학습하면 W에 유리할 수 있다.
site CF MAE가 최저인 Epoch가 그룹 W 최저일 필요도 없다.
기존 Epoch2 선택이 강한 규제/마스크/선택지표 때문인지도 함께 조사할 수 있다.

**코드 범위:** 새 full-fleet 또는 동시시간 sampler, observed target mask, group-loss와 별도 checkpoint schema가 필요하다.
기존 모델의 forward_fleet와 공통 evaluator를 재사용할 수 있다.
반드시 **그룹 오차를 먼저 합산한 뒤 abs**를 취한다. site microbatch별 abs의 합은 상쇄를 잃으므로 다른 loss다.
기존 development CLI의 그룹loss 금지/계약을 느슨하게 만들지 않는다.

**GPU/메모리:** full fleet는 site batch32 대비 최소253/32≈7.9배 샘플 activation을 유지한다.
계획용4~12GiB에서 day batch1~2로 시작해 peak를 측정한다. 3090Ti24GiB에서 가능할 것으로 보지만
측정 없이 batch4 이상의 적합성을 보장하지 않는다. 동일 site-hour 수 기준 계산량은 비슷해도
I/O·큰 batch·동기화·optimizer step 수가 달라 wall time을 다시 측정한다.
먼저 loss gradient fixture/100 update pilot 후 selection 최대6 epoch로 후보를 줄인다.

**부작용/누수:** 그룹 특화로 다른 배정에서 악화하거나 site 성능이 손상될 수 있다.
결측 site를0으로 loss에 넣지 않는다. 모든253개 정답 완전관측만 학습하면 초기 신규 발전소가 배제되므로
관측합 loss와 site loss를 함께 설계해야 한다. 팀원 그룹 생성 cutoff도 감사한다.
SMP 기반 loss는 원래 요청의 정산 전용 제약과 상충하므로 이번 실험으로 자동 도입하지 않는다.

**성공:** 미리 고정한 배정에서 이후 W가 줄고 다른 배정/계절에서 큰 악화가 없으며,
단독 또는 고정alpha 앙상블이 우세하다. **실패:** 특정 그룹/짧은 계절만 개선하거나 낮은 site loss와 함께
연간 W가 악화하면 계수/설계를 재검토한다. 연간을 보고 계수를 계속 바꾸면 탐색으로 표시한다.

**서버 명령 상태:** 이 trainer도 아직 구현하지 않았다. 기존 v0 설정의 group.enabled를 true로 바꾸거나
존재하지 않는 train variant 명령을 제시하지 않는다. 실행 가능한 공통 비교 명령과 기존 기준 재현 명령만 아래에 둔다.

## 보류할 구조 후보와 보조학습

- TSMixer 폭/블록 확대: 현재 크기가 GRU보다 이미 크다. 규제/입력 통제 후에 검토한다.
- BiGRU/CNN/Transformer 교체: 동일 피처·mask·loss의 작은 모델 ablation으로만 구조 효과를 해석한다.
- 예측 운량 teacher: XGB처럼 예보→실측운량 학습으로 추가 정보가 생길 수 있다. teacher도 각 cutoff 이전에만 fit하고
  검증기간에는 고정해야 한다. 보조기상 teacher18개를 만드는 현재 prepare는 가벼운 전처리가 아니다.
- TSMixer 운량/14기상 multi-task: 표현학습의 정규화 가능성은 있지만 결측·단위·관측소 중복 가중과 목표 충돌이 있다.
  현재 패키지의 PV-only GRU 성능을 multi-task 효과라고 설명할 근거가 없다. 후순위 별도 실험으로 둔다.
- 전체 하향 배율: 과대 bias가 공통 관측 peak에 존재하는지 확인한 뒤 과거 OOF로만 추정한다.
  NMAE를 악화시키면서 W를 줄일 수도 있으므로 둘을 기록한다. 연간 실제합으로 배율을 맞추지 않는다.

## 현재 유효한 RTX 3090 Ti 서버 명령

다음은 **작성만 했으며 실행하지 않았다**. 먼저 새 비교 코드 및 원본 자료를 서버에 별도 전달해야 한다.
TEAM_REPRO/TEAM_SOURCE는 실제 복사한 패키지 경로, TEAM_PRED는 받은 검증 예측 경로를 지정한다.

```bash
cd /home/user/2026_OIBC
export OIBC_PYTHON=/home/user/ai_env/bin/python
export TEAM_REPRO=/실제/팀원재현패키지경로
export TEAM_SOURCE=/실제/팀원최소소스경로
export TEAM_PRED=/실제/검증예측.parquet
"$OIBC_PYTHON" -m unittest analysis.tsmixer.tests.test_comparison -v
"$OIBC_PYTHON" -m analysis.tsmixer.comparison \
  --package "$TEAM_REPRO" --train-source "$TEAM_SOURCE" \
  --tsmixer-run runs/tsmixer/year_backtest/seed2026_001 \
  --teammate-predictions "$TEAM_PRED" \
  --output runs/tsmixer/comparison/server_common_year_001
```

기존 recipe를 새 폴더에서 다시 기준 학습해야 할 경우에만 아래 명령을 사용한다.
현재 저장 결과가 있으므로 **이번 단계에는 필요 없고 실행하지 않았다**.
기존 서버의 OIBC_DATA_ROOT/OIBC_TSMIXER_CACHE_ROOT/OIBC_STORAGE_ROOT 설정을 유지하고
캐시 전체축/서명/쿼터를 먼저 확인한다. 복원 CSV를 TEAM_COMPOSITION에 지정한다.

```bash
cd /home/user/2026_OIBC
export OIBC_PYTHON=/home/user/ai_env/bin/python
export TEAM_COMPOSITION=/실제/복원된/teammate_composition.csv
bash scripts/tsmixer/train_year_backtest.sh --full-train \
  --config configs/tsmixer/year_backtest.yaml \
  --teammate-composition "$TEAM_COMPOSITION" --metadata-candidate \
  --output runs/tsmixer/year_backtest/seed2026_reference_groups_002
```

미구현 개선 실험의 실행 명령을 존재하는 것처럼 만들지 않았다.
다음 GPU 실험의 첫 결정은 자료 확보 후 alpha 보완성 분석이며,
재학습을 시작한다면 위 우선2의 단일 scale/mask pilot부터 별도 trainer로 구현하는 것이다.
