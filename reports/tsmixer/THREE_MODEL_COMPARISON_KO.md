# TSMixer · SJ · 01_state 통합 비교 — 2026-11-04

현재 실제 재계산 가능한 모델은 **TSMixer v0**다.
SJ와01_state의 검증 예측은 없으므로 세 모델의 우열·잔차 상관·최적 앙상블은 아직 판단할 수 없다.
TSMixer 예측은 이번 조사 시점에 로컬에 존재해 **SJ 최종 배정까지 직접 재평가**했다.

이전 TEAMMATE_COMPARISON_KO.md의 “TSMixer 연간 예측 없음”은 당시의 가용성 기록이다.
현재 predictions.npy/axis/run.json과 체크포인트 폴더가 도착한 것이므로 이전 보고서/산출물을 수정하지 않았다.
SJ 예측 누락 상태는 그대로다. 01_state의 두 보조 GRU 설명은 SJ 패키지의 실제 PV-only GRU와 다른 프로젝트에 해당한다.

## 1. 구조·데이터·목적 비교

| 항목 | TSMixer v0 | SJ | 01_state |
|---|---|---|---|
| family | 시간/feature MLP mixer3블록 | LGB25/XGB25/BiGRU50 | LGB10/XGB10/cloudGRU40/allGRU40 가능한 혼합 |
| NN 크기 | 305,865 | 검증51,677 | 코드상1,122,051 /1,126,671 |
| GRU | 없음 | 2층, 방향당32, projection64, residual | 2층, 방향당192, projection192, 보조 head |
| 시계열 문맥 | 같은 운영일24시간 | 같은 운영일24시간+파생 문맥 | 같은 운영일24시간 |
| 입력 | 예보90+mask, 시간8, 정적146, ID | 기본86(수치80+범주6), XGB101 | GRU45, 트리24 |
| 발전소 표현 | ID embedding+metadata/station one-hot | ID/station/source embedding | ID 없음; logcap/lat/lon |
| 예보 처리 | 10/17/revision 별도, 원본 단위 | 17우선/10보완/짧은 보간/과거carry, B풍속단위 변환 | 10/17 별도, train 전지점결손112시간 제거; test WA264행 보완 |
| 정상화 | train mean/std, mask | 변환 median/물리scale/clip8 | train mean/std, 일부 cloud 기반 missing flag; X float16 |
| 발전량 과거값 입력 | 없음 | 없음 | 없음 |
| 실측 기상 | 사용 안 함 | 과거 WB cloud teacher 정답→XGB 예측 운량 피처 | 훈련 때2/14변수 보조 정답; 추론 입력 아님 |
| NN loss | 용량가중 L1, 연간실험 SMP/k는 정산 전용 | PV CF MAE | .5site+.5group의 SMP/k/cap가중 L1+보조λ |
| train target/mask | CF[0,1], 결측mask, 관측0 유지 | 주간/긴0·의심0 제외 | CFclip[0,1.2], 주간/비정지; tree정지타깃 L2대체 |
| 출력 | CF→kWh, 비음수/train-only 야간보정 | 비음수/태양야간0, 공통하향배율 | GRU1.2sigmoid/태양야간0, tree음수0; 공통배율 없음 |
| 원래 정산 그룹 | 용량 균형/metadata | 최적화SJ G1/G2 | current_group A→G1/B→G2 |
| cutoff | 2024-05-31 17시, run 기록 확인 | config17시; 모델/보정 기록 추가 필요 | 원본main_all 2024-06-01 00시, 요구보다7h 늦음 |
| 모델 선택 | 내부4~5월 siteCFMAE, 선택2epoch | 검증용9epoch 저장 모델, 반복 검증 이력 | 고정30epoch 마지막; 과거탐색 선택값/REF상수 |
| seed/추론 비용 | seed2026 모델1개 | component3개, 검증GRU1개 | 기본6seed×2GRU + seed/round별tree평균 |

01_state는 보조실측 감독과 그룹 loss를 TSMixer/SJ와 다르게 사용한다.
성능 차이를 GRU192 대 Mixer 구조만의 효과로 해석할 수 없다.
실측 보조 정답 자체는 미래 추론입력과 구분되지만 원본 cutoff/outage/필터 통계 문제는 별도로 해결해야 한다.
현재 원본 main_all은 공통 계약에 부적합하다. 자세한 근거는 [모델 분석](STATE01_MODEL_ANALYSIS_KO.md)에 있다.

## 2. 공통 기준과 실제 입력 검사

공통 시간은 `[2024-06-01 01:00,2025-06-01 01:00)` KST, 원본253개 발전소,
동일 raw generation/mask/SMP/k다. 00시는 이전 운영일의24시이며 자동1시간 이동은 하지 않는다.
observed-only는 정답이 관측된 같은 발전소의 예측과 실제만 그룹 합산한다. NaN을0으로 채우지 않는다.
예측의 관측행 누락은 거부한다. complete-fleet는 별도 참고 모드다.

실제 검사 결과:

- 원본 관측행 **2,170,645**, 유효시간 **8,760**, 전체253개 완전관측 **3,802시간**.
- state01 정제 데이터의 같은 연도 값·mask는 원본과 정확히 같다.112개 삭제시간은 모두2021~22년에 있다.
- TSMixer `[365,253,24]` CF와 axis site ID를 검사하고 `[시간,site]`로 전치/재정렬해 원래 용량을 곱했다.
- run.json refit cutoff가2024-05-31 17시임을 확인했다. 모델을 다시 추론/재학습하지 않았다.
- 공통 baseline W는 기존 보고값265,885,394.24949712원을 재현했다.
- SJ 배정은 이전 복원 CSV를 재사용했고253개/최소20MW 조건을 재검사했다.

공통 ratio 그룹 NMAE는 `100×Σ|그룹오차|/Σ관측그룹용량`이며,
macro는 `mean(100×|그룹오차|/관측그룹용량)`이다. 둘을 다른 열에 저장한다.
`W=Σ.1×SMP×k×|그룹오차|`, k는 종료시각11~15시3, 나머지1.
S는 관측실제×SMP 매출 합−W다. 시간당 페널티는 W/유효 고유시간이다.
내부점수는100×(1−W/223892580.02742994)이며 **분모의 baseline 예측이 없어 모두 잠정치**다.
TSMixer baseline 배정 점수가 예전 값과 같다는 사실은 기준 baseline 모델의 W_ref를 검증한 것이 아니다.

## 3. 직접 재계산한 TSMixer — observed-only

| 배정 | Site NMAE% | Site MAE kWh | 그룹 ratio% | 그룹 macro% | 그룹 MAE kWh |
|---|---:|---:|---:|---:|---:|
| SJ 최종(주 기준) | 4.018044589 | 6.594384113 | 2.567720506 | 2.585278930 | 526.955155 |
| TSMixer 균형 | 4.018044589 | 6.594384113 | 2.571847138 | 2.592506422 | 527.802034 |
| TSMixer metadata | 4.018044589 | 6.594384113 | 2.569495964 | 2.588873156 | 527.319519 |

| 배정 | W 원 | S 원 | 시간당W 원 | 잠정 내부점수 | 시간/관측행 |
|---|---:|---:|---:|---:|---|
| SJ 최종 | **265,331,397.7142** | **7,232,252,089.1835** | 30,288.972342 | **−18.508348** | 8,760 /2,170,645 |
| TSMixer 균형 | 265,885,394.2495 | 7,231,698,092.6482 | 30,352.213955 | −18.755786 | 같음 |
| TSMixer metadata | 265,643,168.9391 | 7,231,940,317.9585 | 30,324.562664 | −18.647598 | 같음 |

같은 예측에서 그룹만 바꾼 결과다. SJ 배정의 W는 기존 균형보다 **553,996.5353원(0.208359%)** 적다.
Site NMAE/MAE는 그대로다. 모델 개선이나 앙상블 개선으로 표현하지 않는다.
세 후보 중 이 구간의 SJ 배정이 가장 낮은 W였지만 독립적인 새 구간에서의 보장은 아니다.
원래01_state 참조 배정은134/119개,20,465.655/22,189.715kW다.
이 배정의 자체 source 점수와 위의SJ 배정 점수는 다른 그룹 조건이다.

## 4. complete-fleet 참고 결과

모든253개 정답이 있는3,802시간/961,906행만 사용한다. observed-only와 서로 점수를 순위 비교하지 않는다.
Site NMAE3.690239337%, Site MAE6.327825909kWh는 세 배정에 동일하다.

| 배정 | 그룹 ratio% | 그룹 macro% | W 원 | S 원 | 시간당W 원 |
|---|---:|---:|---:|---:|---:|
| SJ 최종 | 2.325060641 | 2.325026258 | 101,229,494.8393 | 3,282,185,884.4098 | 26,625.327417 |
| TSMixer 균형 | 2.328544484 | 2.328533046 | 101,542,037.5931 | 3,281,873,341.6560 | 26,707.532244 |
| TSMixer metadata | 2.326264142 | 2.326099327 | 101,485,933.5549 | 3,281,929,445.6942 | 26,692.775790 |

## 5. 아직 비교 불가능한 모델과 자료

| 공급자 | 실제 saved year 예측 | 공통 평가 | 필요한 자료 |
|---|---|---|---|
| tsmixer | 있음 | 위 표 직접 계산 | 추가 학습 불필요 |
| sj_ensemble | 없음 | unavailable | validation kWh/component 예측과 학습·보정 이력 |
| sj_lightgbm/xgboost/bigru | 없음 | unavailable | 각 component 예측 |
| state01 | 없음 | unavailable; 원본 프로토콜도 부적합 | cutoff/outage/aux통계를 바로잡은 과거검증 예측+provenance |
| state01_lgb/xgb/cloud/all | 없음 | unavailable | 같은 안전한 run의 component/seed별 예측 |

SJ config의 기대값3.287997241%/W176,133,675.5585원/내부21.331169는 이전 문서에 보존한다.
지금 재계산한 숫자가 아니며 이 표의 공정 비교 결과에 편입하지 않는다.
01_state 코드REF의17.53/17.72점은 다른 L2 분모·7시간늦은 학습·원래A/B·자체W10배 산식의 과거 인용이다.
NMAE/W금액/S를 알 수 없으므로 역산하지 않는다.

현재 단일모델 TS 잔차 slice/발전소별 오차/상쇄율/고페널티 시각은 실제 계산했다.
모델 간 잔차 상관 CSV는 비교할 다른 모델이 없어 header-only다.
가중치 grid에는 TS 단독만 계산되고 SJ/state01에 양의 가중치가 필요한65개 조합은 unavailable이다.

## 6. 구현·실행·테스트와 보존

기존 core/io/diagnostics 및 두 모델 CLI를 유지하고 다음을 추가했다.

- `comparison/providers.py`: SJ 새 이름/단독앙상블, state01 최종표·native seed별 adapter, 단위·cutoff·hash 검사.
- `comparison/three_diagnostics.py`: 발전소별/각 그룹별 잔차 상관·반대부호·50:50혼합W 진단.
- `comparison/multi_ensemble.py`: 66/231 simplex, unavailable 조합, OOF 선택과 OOF 그룹시점 검사.
- `comparison/state01_audit.py`: 소스 AST literal/파일hash/ZIP/정제 데이터/해석적 파라미터 수 감사.
- `comparison/three.py`, `scripts/tsmixer/compare_three_models.sh`: 독립 CLI, 새 출력 경로.
- `comparison/THREE_MODELS_KO.md`, README 링크 추가, `tests/test_three_comparison.py`.
- 이번3개 보고서와 `THREE_MODEL_EVIDENCE_20261104.json`.

실제 전체 suite69개 중 **68개 통과, CUDA1개 skip**(최종14.765초).
OOF 그룹 시점·명시한 provenance 누락·발전소/그룹 상쇄 진단을 보강한 최종 코드로 전체 suite를 재실행했다.
신규8개 fixture도 모두 통과했다.
기존 suite의 제한된 합성 CPU/MPS smoke 외에 실제 모델 학습은 없다.
신규 fixture는 실제 모델 성능을 검증한 결과가 아니다. 실제 데이터 평가는 위TS 표와 정답/그룹 감사에 한정된다.

검사 범위는 ID/키 중복·누락/UTC-KST/shape/정답mask/단위/예측hash/negativeweights/누락seed,
SMP/k/그룹20MW/그룹변경 site NMAE 불변/최종모델·늦은cutoff·oracle입력거부/OOF시점이다.
기존 회귀 테스트는 shared-memory 권한이 필요해 승인된 sandbox 외부에서 실행했다.

실행 결과는 새 `runs/tsmixer/comparison/three_audit_20261104_001/` 등 초기 감사와 최종 `three_audit_20261104_final/`에 있다.
최종 실행은 원본01_state 파일을 전후 재hash해 동일함을 확인하고 소스/예측hash를 기록한다.
TSMixer 기존 `seed2026_001`, SJ복원002 및 기존 모델/캐시/설정/학습스크립트는 덮어쓰지 않았다.
Git commit/push·원격 접속·설치·외부 코드/모델 실행은 하지 않았다.
시작 시부터 존재하던 다른 untracked 보고서도 수정하지 않았다.

실행 명령과 입력 JSON은 [세 모델 사용법](../../analysis/tsmixer/comparison/THREE_MODELS_KO.md)에 있다.
TSMixer 추가 학습부터 시작할 단계가 아니다. 누락 예측 및 안전한 검증 provenance 확보가 다음 작업이다.

실제로 실행한 주요 명령은 다음과 같다.

```bash
.venv/bin/python -m analysis.tsmixer.comparison.three --output runs/tsmixer/comparison/three_audit_20261104_final
.venv/bin/python -m unittest analysis.tsmixer.tests.test_three_comparison -v
.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v
bash -n scripts/tsmixer/compare_three_models.sh
git diff --check
git status --short
```

초기 감사001~003도 보존했다. 최종 source/hash·수치 근거는
[THREE_MODEL_EVIDENCE_20261104.json](THREE_MODEL_EVIDENCE_20261104.json)에 요약했다.
기존 두 모델 비교 Python 파일들의 hash는 이전 local_audit_20261104_002 실행기록과 모두 일치한다.
새 함수/진입점과 README 안내만 추가했으며 원래 평가 산식/CLI를 바꾸지 않았다.
