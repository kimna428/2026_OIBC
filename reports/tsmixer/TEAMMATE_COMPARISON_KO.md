# 팀원 모델 분석과 TSMixer 공통 평가 — 2026-11-04

현재 확인된 차이는 아키텍처뿐 아니라 피처·스케일링·학습 마스크·손실·보정·그룹 NMAE 산식에 걸쳐 있다.
팀원 구성은 **LGB 25% + XGB 25% + 발전량 전용 BiGRU 50%**로 확인됐다.
그러나 양쪽 연간 예측 파일이 로컬에 없어 공통 평가 성능은 아직 계산할 수 없다.
이 문서의 기존 성능 표는 저장 보고값이며 이번 재계산 결과가 아니다.

실제 감사 산출물은 `runs/tsmixer/comparison/local_audit_20261104_002/`에 있다.
실행법과 정확한 rsync 명령은 [비교 도구 안내](../../analysis/tsmixer/comparison/README_KO.md)에 있다.

## 1. 실제 접근한 패키지와 검증 범위

작업 저장소는 `/Users/kimna/workspace/2026_3_2/2026_OIBC`, 브랜치는
`experiment/tsmixer-year-backtest`다. 요청에 적힌 `/Users/kimna/w/2026_3/2026_OIBC`는 없다.
팀원 폴더는 다음 두 개이며 원본 데이터 ZIP과 구분했다.

| 자료 | 실제 상태 | 확인 결과 |
|---|---|---|
| Downloads/oibc_repro_sj_2026_11_03 | manifest 104개 중 33개 존재, 71개 누락 | 존재한 33개 SHA256 전부 일치 |
| 같은 이름 ZIP | 48개 엔트리 | reference/records/prepared/루트 README 없음; 추출 문제로 단정할 수 없음 |
| Downloads/oibc_train_source_2026_11_03 | manifest 20개 전부 존재 | SHA256 20개 전부 일치 |
| Downloads/Clustering_sj.md | 존재 | 253개 배정과 거리·관측소 통계 검증 |
| Downloads/group_sj.json | 존재 | 그룹 규모·기존 CSV hash 대조 |
| Downloads/report_sj_2.md | 존재 | 11월2일의 과거 LGB/CatBoost/GRU 실험 설명; 현재 XGB 구성과 분리 |

repro에는 `run.py`, config, code 6개, dataset, validation의 LGB/XGB/BiGRU 모델이 있다.
final에는 LGB/XGB, BiGRU inner/outer, 운량 teacher 5개가 있다.
`reference/selected_stack_all_oof.parquet`, 검증/제출 예측, 전처리 통계, 학습 로그, 제출 validator는 없다.
존재하는 dataset/README.md는 대회 데이터 설명이다. 공유 재현 패키지 루트 README로 취급하지 않았다.
누락 목록과 모든 존재 파일 hash는 audit.json에 보존했다.

## 2. 모델·설정과 체크포인트 증거

| 항목 | 실제 코드/설정 | 아티팩트 확인 |
|---|---|---|
| LightGBM | CF regression_l1, 500 rounds, leaves63, min_data200, lr .06, bagging .8/freq1, feature .8, L2=0, seed42, threads6 | validation/final 텍스트 모델 모두 500 Tree, 입력86개와 L1/lr/leaves/seed 확인 |
| XGBoost | CF reg:absoluteerror, 500 trees, lr .06, depth7, min_child_weight20, subsample .8, colsample .8, lambda5, hist, categorical, max_cat_to_onehot4, n_jobs6, random_state2026, device cpu | validation/final joblib 존재 및 hash 일치. pickle을 실행하지 않아 내부 객체 파라미터 재검증은 안 함 |
| BiGRU | 24시간, numeric80 + missing80 + 범주 embedding22 → Linear182→64 → 2층 양방향 GRU(hidden32/방향, dropout .05) → residual+LayerNorm64 → Linear64→1 | validation 51,677개, epoch9; final 52,137개, epoch14. 실제 tensor shape와 pv head 확인 |
| 범주 | site_id8차원, WA/WB 각4차원, source3종 각2차원; 미관측 category=0 | validation vocab sizes [206,54,63,4,4,4], final [254,62,74,4,4,4] |
| GRU 학습 | AdamW .001, wd .0001, batch256, clip1, max40/patience6, cosine 최소 .0001, inner2개월 | 코드 확인. 원래 학습 기록과 preprocessing JSON은 누락 |

BiGRU의 실제 손실은 **학습 mask에 대한 CF 절대오차 평균**이다. 용량·SMP·k 가중치와 그룹 loss는 없다.
`cloud_lambda`는 variant='bigru'에서 0이고, 모델의 두 번째 출력은 학습 head가 없는 고정 0이다.
SETTINGS에 cloud_weight=.1이 남아 있어도 실제 운량 공동학습이라고 해석하지 않는다.
final의 inner/outer 두 파일은 Epoch 선택과 재학습 단계이며 두 GRU 예측 앙상블이 아니다.

주의할 seed 증거 차이가 있다. config의 `seed2026_chosen_after_repeated_validation=true`와 달리
배포 neural.py의 set_seed/SETTINGS는 **42**다. checkpoint에는 seed가 없으므로 원래 checkpoint의 seed를
코드만으로 확정하지 않는다. 트리 XGB seed2026과 GRU seed도 혼동하지 않는다.
재학습 시 checkpoint 정밀 재현을 보장할 수 없고 원래 학습 로그가 필요하다.

### A와 B 실험의 구분

A의 모델 가중치 25/25/50은 config와 pipeline.predict에 일치한다.
전체 앙상블 공통 배율은 final **0.966293519325946**, validation **0.9612887427622985**다.
모델별 별도 곱셈 보정값은 발견되지 않았다. 각 모델의 비음수/야간 처리는 보정 배율과 구분한다.
validation 배율의 원래 OOF/추정 cutoff 기록은 누락되어 2024-06 이전 정답만으로 정했는지 확인하지 못했다.
run.py calibrate는 모든 제공 OOF의 sunlit peak 행을 합산하는 **final용** 재현 경로이며 validation 배율을 재추정하지 않는다.
이 경로를 그대로 연간 검증 보정에 사용하면 검증 정답이 들어갈 수 있으므로 실행하지 않았다.

B의 LGB/XGB20% + GRU2개80%, 발전량+운량/14기상 공동 예측은 조사한 폴더에 구현/체크포인트가 없다.
최종 구성을 B로 설명하거나 B의 성능을 현재 패키지에 귀속하지 않는다.
과거 report_sj_2.md는 **LGB25/CatBoost25/GRU50, seed42**를 기술하며 또 다른 단계의 보고서다.

## 3. 입력·전처리와 TSMixer의 차이

| 항목 | 팀원 | 현재 TSMixer 연간 실험 |
|---|---|---|
| 예보 | round17 우선, 10 보완, 같은 운영일 내부 최대3시간 보간, 마지막 원본 과거일 동일시각 carry | round17/10 각각30 + revision30, 합90개; 원본 결측 mask |
| 단위 | B 풍속/돌풍 km/h→m/s, 교차 제공사 강설 차이만 B cm→mm | 기존 v0 원본 단위 유지 |
| 물리 파생 | 풍향 sin/cos, 이슬점차·돌풍차, 제공사 차이/원본 여부 | raw 예보와 revision 중심 |
| 시간 문맥 | 이전/다음 시각, 하루 평균/std/max; XGB에는 운량 추세13개 | 24시간 mixer가 문맥을 학습 |
| 운량 보조 | XGB만 추가15개(추세13 + 예측 운량평균/가용성2), 합101개 | 없음 |
| 기본 입력 | LGB/GRU86개(수치80 + 범주6) | 기상90 + 시간8 + 정적146 + ID |
| 태양 | 구간 중앙 elevation/coszenith/azimuth sin/cos, 구간 시작·중앙·끝의 야간 판정 | midpoint elevation 및 train-only 야간 임계 보정 |
| 범주/정적 | site/station/source 임베딩을 시각마다 입력; capacity/lat/lon/dist는 수치 | ID embedding·station one-hot·capacity/좌표/거리·결측 flags |
| 수치 처리 | train-only 변환 median, 물리단위 scale, fallback max(IQR,std,1), missing mask, ±8clip | train-only mean/std와 missing mask |
| 학습 mask | sunlit, 긴 0발전 연속구간 제외, elevation>15의 관측0 제외 | CF [0,1]만 학습, 관측0 유지, 모든 시각 |
| loss | site CF MAE | capacity 가중 L1; 연간 실험은 SMP=k=1 |
| 규제 | GRU dropout .05, wd .0001, cosine, site embedding dropout 없음 | dropout .1, site dropout .1, wd .01, constant lr |
| 크기 | 검증 BiGRU 약5.17만 파라미터 | TSMixer 305,865 파라미터 |

같은 24시간 예보와 발전소 간 공유 파라미터를 사용하는 두 모델이다.
BiGRU는 양방향 recurrent 문맥을, TSMixer는 시간축 MLP와 feature MLP를 쌓는다.
양방향은 미래 **발전량** 입력이 아니라 전날 발행된 하루 예보 전체를 읽는 구조이므로 그 자체가 누수는 아니다.
현재 점수 차이를 모델 종류 하나로 설명할 수 없다. 피처/마스크/loss를 맞춘 ablation이 먼저다.

### 실측 기상·정답 유입 검토

- `prepare.daily_weather`는 observed_wb의 **11~15시 운량 평균**을 별도 LGB teacher의 학습 정답으로 사용한다.
  결측/충돌/범위 밖 운량을 처리하고 5시간 관측이 모두 있어야 정답으로 인정한다.
- teacher 입력은 24시간 기상예보·연결 WA 예보·계절/관측소 코드다. 실측 운량은 입력 열이 아니다.
- 일반 prepare는 분기별 held-forward teacher18개를 **새로 학습**한다. 따라서 이번 작업에서는 실행하지 않았다.
- validation_mean은 **202406 teacher를 검증 1년 내내 고정**해 XGB 운량평균 입력을 덮어쓴다.
  이후 분기 teacher의 실측 정답을 validation 추론에 사용하지 않으려는 코드 경로가 있다.
  다만 teacher 학습 기록이 없으므로 저장 teacher의 실제 학습 cutoff는 독립적으로 확인하지 못했다.
- BiGRU에는 운량 추세/teacher 피처가 들어가지 않는다(기본86개). forward는 numeric/missing/category만 받는다.
- target/energy는 tensors에 담기지만 forward 인자로 전달하지 않으며 손실·평가 mask에만 사용한다.
  cache의 NaN 정답을 0으로 바꾸는 것은 masked loss 내부용이다. 공통 평가에서는 원본 NaN을 유지한다.
- zero_spells는 cutoff까지의 관측만 검사한다. 검증 NMAE/정산에서는 관측0을 제거하지 않는다.
- repro.prepare는 reference layout/통계가 누락되어 바로 재현할 수 없다. 최소 소스 prepare는 raw key에서 layout을 복원하지만
  보조모델 학습을 포함하며 저장 모델의 원래 category/normalizer와 동일하다는 보장은 별도 검증이 필요하다.

이는 코드 데이터흐름 감사다. 누락된 학습 provenance까지 검증해 완전한 무누수를 인증한 것은 아니다.

## 4. 최종 G1/G2 복원

원본 CSV는 없었다. 최소 source config의 site→group 253개와 Clustering_sj.md의 전체 표를 각각 파싱해 비교했다.
중복/누락/미지ID/G1·G2 명칭/그룹20MW 조건 모두 통과했다.

| 그룹 | 발전소 수 | 용량 kW |
|---|---:|---:|
| G1 | 127 | 21,334.270 |
| G2 | 126 | 21,321.100 |

복원 CSV를 CRLF로 저장한 SHA256은
`92f82612b50e9447dcfa0f32a336ee2c7bd7596cea1a5530b961caa7aea26ea5`다.
repro manifest의 누락된 `reference/composition.csv`와 group_sj.json의 원본 CSV hash에 **정확히 일치**한다.
따라서 단순히 용량이 비슷한 다른 배정이 아니라 공유 패키지의 최종 배정 바이트를 복원했다.
새 산출물은 `runs/tsmixer/comparison/local_audit_20261104_002/teammate_composition.csv`다.

| 관계 | 쌍 수 | 팀원 다른 그룹 분리율 | TSMixer 균형 | TSMixer metadata |
|---|---:|---:|---:|---:|
| 10km 이내 | 938 | 55.8635% | 52.3454% | 51.3859% |
| 30km 이내 | 2,835 | 52.5573% | 51.4991% | 51.3580% |
| 같은 WA | 959 | 55.3702% | 52.4505% | 54.1189% |
| 같은 WB | 717 | 56.7643% | 53.6960% | 52.4407% |

위 값은 **이번에 원본 메타데이터로 재계산**했고 팀원 문서 반올림값과 일치한다.
좌표는 제공 좌표상 거리이며 실제 행정구역을 의미하지 않는다.

문서는 30km 관계/10km 감쇠, 같은 관측소, elevation>15·공통500시간 이상의 양의 CF잔차 상관,
수축 bias ±0.1%p 분류, 성향별 수·용량·예상오차 균형, 절반±25kW 제약을 설명한다.
그룹 생성 이력은 **2024-05-31 17시까지 9계절 LGB/CatBoost/GRU OOF**라고 명시하며 현재 XGB 구성 전의 배정이다.
`build_balanced_vpp.py`, candidate_history_audit, 원래 OOF가 없어 이 목적함수·잔차상관·편향 균형의 구현/시간 경계는 재검증하지 못했다.
문서 내용과 최종 배정/기하 통계의 일치는 확인했지만 실행 코드를 확인한 것으로 표현하지 않는다.

## 5. 공통 평가와 검증된 정렬 근거

시간은 `[2024-06-01 01:00, 2025-06-01 01:00)` KST다. 00시 값은 이전 운영일 24시다.
TSMixer `[365,253,24]` CF를 day→hour→site로 전치하고 축 파일의 site ID를 기준으로 용량을 곱한다.
임의의 1시간 이동은 금지한다. 모든 관측 정답에 예측이 있어야 하며 누락 예측 때문에 평가 행을 줄이지 않는다.

원본 metadata/generation/SMP의 실제 hash는 팀원 manifest와 일치한다.
원본에서 재구성한 관측행 **2,170,645**, 평가시간 **8,760**, 전체253개 완전관측시간 **3,802**다.
TSMixer 보고서의 관측 mask SHA256도 원본 배열의 기존 day/site/hour 순서로 재구성해 일치했다.
유효 CF [0,1] 학습 이력이 없는 발전소는 cutoff 기준 **48개**다.

팀원 prepared는 energy/capacity를 float32로 저장하고 공통 평가기는 원본 float64를 유지한다.
검증 실제 발전량의 float32 반올림 최대 차이는 **0.000048828125 kWh**다.
이 차이는 기록하지만 팀원 기대 W까지 1e-6원 단위 동일 재현을 주장하지 않는다.

공통 평가 정의:

- 발전소 NMAE: 관측행 `100×|pred−actual|/capacity` 평균.
- observed-only: 각 그룹·시간에 **같은 관측 발전소**의 예측과 실제만 합산; 정답0 유지, NaN 미보간.
- 공통 그룹 NMAE: 팀원 방식 `100×Σ|그룹오차|/Σ관측그룹용량`.
- 기존 TSMixer 그룹 NMAE: `mean(100×|그룹오차|/관측그룹용량)`; 별도 macro 항목으로 보존.
- W: `Σ0.1×SMP×k×|그룹오차|`, 종료 시각11~15시 k3, 나머지1.
- S: `Σ관측실제×SMP−W`. 팀원 기존 score()는 S를 반환하지 않으므로 없는 값을 추정해 채우지 않는다.
- 내부점수: `100×(1−W/223892580.02742994)`; 기준 baseline 검증 예측이 없어 현재 잠정치.

계약 hash는 정답값/mask/시간/발전소순서/SMP/k/그룹/산식을 모두 포함한다.
그룹 NMAE의 비율 평균 차이는 W를 바꾸지 않는다. 그룹 배정 변경은 발전소 NMAE를 바꾸지 않는다.
complete-fleet와 observed-only의 W/점수는 유효 시간이 다르므로 서로 순위 비교하지 않는다.

## 6. 성능 표 — 기존 보고값, 이번 재계산 아님

| 모델/배정/모드 | 발전소 NMAE% | 그룹 NMAE% | W 원 | S 원 | 시간당W | 내부점수 | 시간/관측행 |
|---|---:|---:|---:|---:|---:|---:|---|
| TSMixer 균형 observed | 4.018044589 | 2.592506422 macro | 265,885,394.2495 | 7,231,698,092.6482 | 30,352.2140 | −18.755786 잠정 | 8,760 / 2,170,645 |
| TSMixer metadata observed | 4.018044589 | 2.588873156 macro | 265,643,168.9391 | 7,231,940,317.9585 | 30,324.5627 | −18.647598 잠정 | 8,760 / 2,170,645 |
| TSMixer 균형 complete | 3.690239337 | 2.328533046 macro | 101,542,037.5931 | 3,281,873,341.6560 | 26,707.5322 | 54.646984 참고값 | 3,802 / 961,906 |
| TSMixer metadata complete | 3.690239337 | 2.326099327 macro | 101,485,933.5549 | 3,281,929,445.6942 | 26,692.7758 | 54.672042 참고값 | 3,802 / 961,906 |
| 팀원 앙상블/팀원 observed | 3.287997241 기대값 | 1.672499632 ratio 기대값 | 176,133,675.5585 기대값 | 계산 불가 | 계산 불가 | 21.331169 기대값·잠정 | 파일 부재로 검증 불가 |
| LGB 단독 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 예측 부재 |
| XGB 단독 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 예측 부재 |
| BiGRU 단독 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 예측 부재 |
| 별도 GRU1/GRU2 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 계산 불가 | 구현/모델/예측 미제공 |

TSMixer MAE(kWh), 공통 ratio 그룹 NMAE, TSMixer의 팀원 배정 성능, 팀원 각 모델의 A/B/C 배정 성능은
원래 연간 예측이 없으므로 계산 불가다. 기존 metrics에는 MAE가 없다.
팀원 앙상블 숫자는 config.validation_expected에서 읽었으며 이번 평가를 통과한 값이 아니다.

기존 TSMixer 보고서 안에서만 배정 효과를 분리하면 metadata 후보의 W 감소는 **242,225.3104원(0.09110%)**이다.
이는 모델 재학습 없이 같은 예측을 재집계한 기존 보고값 간 차이다. 구조 개선을 대신할 큰 근거로 읽기 어렵다.
현재 두 모델의 표를 빼서 공정한 성능 개선량/우열을 확정하지 않는다. 배정과 group NMAE 산식이 다르다.

## 7. 검증 독립성 및 다음 결정

검증 profile의 cutoff/start/end는 TSMixer와 일치한다. GRU는 앞선2개월로 Epoch 선택 후 새 outer 모델 학습을 설계했다.
TSMixer 선택 Epoch2는 저장 epoch_selection과 확인했지만 refit/last.pt는 로컬에 없어 모델 자체를 재검증하지 않았다.
팀원 config는 반복 검증 후 선택했음을 명시하고 pipeline.score도 independent_new_holdout=False로 반환한다.
이 1년은 양쪽 차이를 분석하는 공통 내부 구간이며 새로운 독립 최종 평가가 아니다.

우선 서버의 TSMixer 예측을 rsync로 받고 팀원의 검증 site predictions 또는 OOF를 확보한다.
그 후 같은 모델 예측으로 A/B/C 그룹 재집계 → 오차 상관 → alpha 탐색 순서로 진행한다.
공통 evaluator와 테스트는 준비됐지만 누락 파일 때문에 실제 재계산이 끝나지 않은 상태임을 유지한다.
