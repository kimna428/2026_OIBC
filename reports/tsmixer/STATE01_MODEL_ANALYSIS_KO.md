# 01_state 모델 분석 — 2026-11-04

01_state는 **LGB/XGB 평균20% + 보조학습 BiGRU2종 평균80%**를 실제 구현한다.
실측 운량/기상은 보조 정답이며 추론 입력은 예보·태양·달력·메타데이터다.
그러나 저장 모델/예측/검증 로그가 없고, 원본 검증 프로토콜에는 cutoff와 미래 통계 문제가 있어
현재 상태를 공정한 2024-06~2025-05 검증 후보로 인정하지 않는다.
점수가 낮아서 제외한 것이 아니라 재현 자료와 시간 경계를 충족하지 못한 상태다.

## 1. 파일과 실행 구조

실제 경로는 `/Users/kimna/Downloads/01_state/01_state/`의
`main.py, common.py, gru.py, gbdt.py, preprocess.py, report.py, requirements.txt`다.
바깥 `dataset/`에는 정제 데이터가 있다. 파일21개(DS_Store 제외)의 hash를 기록했다.
외부코드는 한 줄도 실행/import하지 않았고 모델 역직렬화도 하지 않았다.
`/Users/kimna/Downloads/01_state.zip`의25개 엔트리도 조사했다. 압축을 풀거나 원본을 수정하지 않았다.
폴더와ZIP 모두 README/학습 checkpoint/work 결과/검증 예측/점수CSV는 없다.

main은 gbdt/gru/eval/test를 순서대로 실행한다. 기본 seed0~5, 보조2종, 30epoch다.
`--quick`도 전체 격자를 만들므로 실행하지 않았다.
원래 코드는 src/01_state/ 배치를 가정한다. 현재 포장 경로에서 common.ROOT의 parents[2]는
`/Users/kimna/Downloads`가 되어 기본 dataset 경로가 잘못된다. 실행 결과가 없다는 사실과 구분되는 포장 문제다.
Mac에서는 GRU가 CPU를 선택하며 MPS 자동 사용 경로는 없다. 새 비교 도구는 예측표만 읽는다.

## 2. 실제 구현된 모델

### 트리 BS

- 라운드10/17별 별도 모델을 fit하고 CF 예측을 kWh로 바꿔 평균한다.
- LGB75 trees: L1, learning_rate .06, leaves127, min_child_samples200,
  subsample .8/freq1, colsample .8, reg_lambda0. CPU threads 최대16.
- XGB100 trees: reg:absoluteerror, learning_rate .06, max_leaves127, max_depth0,
  lossguide, hist, min_child_weight200, subsample/colsample .8, lambda0.
  CUDA 빌드이면 device=cuda로 설정한다. 실제 GPU 가용성까지 검사하는 조건은 아니다.
- 두 모델 모두 주간(elev>0) CF target=clip(energy/cap,0,1.2), sample weight=SMP×k×cap를 평균1로 정규화한다.
- 장기정지 target은 정지 표본을 뺀 L2 LightGBM600 trees의 예측으로 대체한다.
  이 L2는 별도 기준선과 target imputation용이며 최종 BS의 세 번째 혼합 성분이 아니다.
- BS는 LGB/XGB .5/.5, seed/round도 균등 평균한다. 각 출력은 밤0/음수0이며 위쪽 clip은 없다.

### 두 보조학습 GRU

| 항목 | cloud_l05 | all_l08 |
|---|---|---|
| 입력 | 24×45 | 같음 |
| projection | Linear45→192 + GELU | 같음 |
| recurrent | GRU192→192/방향, bidirectional, 2층, interlayer dropout .1 | 같음 |
| PV head | LayerNorm384 + Linear384→1, 1.2×sigmoid | 같음 |
| 보조 head | LayerNorm384 + Linear384→2 | LayerNorm384 + Linear384→14 |
| 보조 손실 λ | .5 | .8 |
| 파라미터 | **1,122,051** | **1,126,671** |

파라미터 수는 소스 구조로 해석적으로 산출했고 독립적인 작은 Torch layer fixture로 대조했다.
checkpoint에서 확인한 수치가 아니다. 발전소별로 모델을 만들지 않고 전체 발전소가 파라미터를 공유한다.
발전소 ID/station embedding은 없다. 용량·위경도의 정적 채널이 발전소 차이를 표현한다.

optimizer=AdamW lr.001/wd.01, OneCycleLR(max_lr .001,pct_start .1), clip1,
배치8일×253발전소, 30epoch의 **마지막 모델**을 사용한다. Early stopping/best checkpoint 선택은 없다.
seed0~5로 각각 fit한다. seed별 모델 선택이나 보정 탐색의 실제 로그는 제공되지 않았다.

주 손실은 `.5×site weighted L1 + .5×group weighted L1`이다.
site/group 모두 SMP×k×capacity와 주간/관측/비정지 mask를 사용하며 CF scale로 정규화한다.
group은 원래 A/B에 대해 **예측과 실제를 먼저 합한 뒤 절대값**을 취한다.
보조 정규화 실측 L1에 각 λ를 곱해 추가한다. 관측소 중복에 대한 보조 sample weight 보정은 없다.
같은 관측소를 공유하는 발전소 수만큼 보조 정답이 반복될 수 있다.
보조 loss는 PV정답 관측 mask와 별개로 주간 실측 mask를 쓰므로, 발전량 이력이 없는 발전소도
과거 연결 관측소의 실측이 있으면 보조 감독학습에 참여할 수 있다. 신규 발전소에서 유리할 수 있는 가설이지만
실제 예측이 없어 효과는 확인하지 못했다.

정규화된 보조 정답이0이면 결측으로 간주하는 `o!=0` mask를 사용한다.
따라서 진짜 평균과 같은 유효 관측0까지 제외할 수 있다. 별도의 원래 결측 mask를 유지하는 구현과 다르다.

### 혼합과 checkpoint 구분

report.combos는 다음을 구현한다.

- BS 단독, 각 GRU 단독.
- .2BS+.8cloud, .2BS+.8all.
- 두 변형이 있으면 .2BS+.4cloud+.4all.

마지막 혼합을 component별로 펼치면 **.1LGB+.1XGB+.4cloudGRU+.4allGRU**다.
λ .5/.8은 학습 보조손실 계수이며 혼합 가중치가 아니다.
SJ처럼 공통 하향 보정 배율은 없다. PV sigmoid/야간0·비음수는 출력 제약이다.

코드는 seed별 예측 parquet를 저장한다. GRU의 test 학습에서만 state_dict를 저장하고 validation에서는 기본적으로 저장하지 않는다.
GBDT는 fit 후 예측만 저장하며 모델 객체를 저장하지 않는다.
현재 공유본에는 그 결과 파일 자체가 없으므로 어떤 variant/checkpoint가 실제 최종 선택됐는지 확정하지 않는다.
`bs02_gru_avg` 등의 이름은 생성 가능한 코드 경로이며 존재하는 산출물이 아니다.

## 3. 데이터와 사용 단계

### 예보/정적 입력

기상 변수는 WA6 + WB9=15개다.

- WA: cloud/temp/humidity/rain/wind_speed/ground_press.
- WB: cloud/temp/rel_hum/uv_idx/precip_prob/total_liq/vis/wind_speed/dew_point.

각 라운드15+cloud 기반 WA/WB 결측표시2 =17개, 두 라운드34개다.
태양 elevation/90, cosz, azimuth sin/cos, doy sin/cos=6개,
시간 sin/cos=2개, 정적 log(cap)/lat/lon=3개를 더해 **45개**다.
예보 mean/std는 training grid의 두 라운드를 합쳐 fit하고 결측은 정규화0(평균)으로 바꾼다.
전 변수별 결측표시는 아니며 cloud가 없는지로 각 제공사 결측을 나타낸다.
정적 값은 알려진 전체253개 메타데이터로 표준화한다. 입력/cache는 float16, 실제 forward는 float32다.
B 풍속을 m/s로 바꾸는 SJ 방식의 변환은 없다. GBDT는 기본15+태양/달력/용량/좌표9=24개 FEATS다.

타깃은 발전량/cap, 학습 중0~1.2 clip, 결측은 계산용0과 mask로 분리한다.
추론에는 과거 발전량 lag나 실측 기상/발전량 정답이 들어가지 않는다.
PV 실제는 target/mask/정지 탐지, SMP/k는 학습 가중치/정산에 쓰고 X의 채널에는 넣지 않는다.
원래 A/B는 group loss와 정산에만 사용한다. 그 one-hot은 GRU 입력이 아니다.

### 실측 보조 정답

cloud_l05는 WA/WB cloud2개다. all_l08은 위15개에서 실측 없는 WB precip_prob을 뺀14개다.
WB total_liq의 보조 정답은 observed WB precip_1h를 매핑한다.
observed_like_forecast 함수명과 oracle 원본 주석이 있어도 현재 dataflow는 `tr['obs'] → aux loss`이며 X로 연결되지 않는다.
발전량·기상 정답을 미래 추론 입력으로 쓰는 oracle 실행 경로는 현재 main의 aux choices에 없다.
따라서 정보 이용 방식 자체는 **훈련 때의 보조 감독학습**으로 분석한다.
하지만 아래 시간 경계/필터 문제는 별개로 해결해야 한다.

### 정제 데이터 실제 검사

정제 generation은 **4,389,192행**, 원본은4,392,566행이다.
전 지점·두 라운드에 예보가 없는112시간(2021-01-20~2022-08-16)을 모든 training 표에서 제거했다.
공통 평가연도에서 제거한 시간은 **0**이다. 해당 연도 정제 정답과 원본은 값·mask 모두 정확히 일치한다:
2,170,645행 /8,760시간 /완전관측3,802시간.

test WA264행은 같은 지점·시각의 다른 라운드로 보완한다.
그 대상은2025~26 제출 예보이며 현재 과거 검증연도의 성능 변경과 혼동하지 않는다.
night_mask.csv와 generation_daytime은 전체 이력 발전량에서 만들었지만 main 모델은 사용하지 않는다.
외부 다운로드/추가 외부 데이터 호출은 발견하지 못했다. 제공 좌표/관측소/예보/실측/발전량을 사용한다.

## 4. 학습·검증 시간과 누수 위험

| fold | 학습 시작 | 코드상 마지막 학습 정답 시각 | 예측 구간(종료 시각) |
|---|---|---|---|
| inner | 2021-06-01 01:00 | 2023-06-01 00:00 | 2023-06-01 01:00~2024-06-01 00:00 |
| main | 2022-06-01 01:00 | 2024-06-01 00:00 | 2024-06-01 01:00~2025-06-01 00:00 |
| main_all | 2021-02-01 01:00 | 2024-06-01 00:00 | 위와 같음 |
| test | 2021-02-01 01:00 | 2025-06-01 00:00 | 2025-06-01 01:00~2026-06-01 00:00 |

모든 시각은 KST다. `gen_tr = ts < p_from`이므로 요구 cutoff2024-05-31 17시보다 **7시간 늦다**.
정제 데이터에서 이 구간에 실제로 **1,435행**, 첫18시·마지막다음날00시의 정답이 존재한다.
GRU auxiliary는 같은 training grid 끝의 실측까지 포함한다. 예보 정규화도 이 grid를 쓰지만,
cutoff 이전에 발행된 예보의 대상 시각이17시 이후라는 사실 자체는 누수가 아니다.
여기서 문제는 실제 발전량/실측 보조 정답의 늦은 포함과 이후 검증연도 분포를 사용하는 별도 필터다.

또한 다음 문제가 있다.

1. load_data에서 outage_days(gen)를 **전체2025년5월까지의 발전량으로 먼저** 계산한다.
   그 뒤 날짜만 잘라도 cutoff 경계를 가로지르는3일 run의 과거 부분이 미래 실제에 의해 정지로 판정될 수 있다.
   GBDT imputation/GRU mask가 이 목록을 사용한다.
2. 실측 이상치 경계는 fa_all/fb_all의 전체 이력 min/max에10% 여유를 둔다.
   main_all에서는 미래 검증연도의 예보 분포가 과거 auxiliary target 필터 선택에 들어갈 수 있다.
   직접 미래 실측 입력은 아니지만 cutoff-safe 통계 fit이 아니다.
3. assert_same_day는 대상일 일치를 검사할 뿐17시 발행 이전 데이터만 fit했다는 보장은 아니다.
4. 30epoch/폭/λ/BS비중은 이전 scratch 실험의 선택값이다. report의 REF도 같은 구간의 과거 실험을 인용한다.
   inner→main_all의 expanding 검증은 구현됐지만 모델/계수 선택의 완전 독립 OOF 증거는 없다.
5. report.components는 GRU key 누락을0으로 채우고 common.score는 outer-merge 뒤 실제/예측 누락을0으로 채운다.
   같은 observed key가 모두 있을 때는 큰 문제가 드러나지 않지만 누락 예측이 있어도 정산이 끝날 수 있다.
   새 공통 adapter는 관측행 예측 누락을 거부한다.

기존 main_all은 대부분 학습정답이 검증 이전이라도 **이번 엄격한 cutoff 계약을 만족하지 않는다**.
2025년5월까지 학습한 test 모델을 과거1년에 쓰는 것은 더 직접적인 in-sample 혼합이므로 받지 않는다.
현재 세 모델 CLI는 state01의 cutoff/normalizer/aux/outage/선택/보정 및 예측 hash를 명시한 별도 기록을 요구한다.
원본 문제를 해결하지 않은 예측에 안전한 프로토콜 이름만 붙여서는 안 된다.

## 5. 평가 산식과 보고된 점수

common.score는 W=`ΣSMP×k×|group error|`다. `PENALTY=.10` 상수는 있지만 score에서 쓰지 않는다.
따라서 W 단위는 공통 페널티의 **10배**이며 공통 W로 변환할 때0.1이 필요하다.
score는 group 실제/예측을 observed site 기준으로 합산하려고 설계했지만 누락0처리 문제는 위와 같다.

자체 내부점수는100×(1−W/W_L2), main_all 분모는 main의2년 학습 L2 seed 평균이다.
같은 numerator/denominator에0.1을 곱하면 자체 정규화 점수는 같지만,
이 W_L2가 SJ의223,892,580.02742994원과 같다는 증거는 없다.
비율 점수를 SJ/TSMixer의 내부점수로 그대로 비교하지 않는다.

| 코드의 과거 REF | inner 점 | main_all 점 | 증거 수준 |
|---|---:|---:|---|
| BS | 7.46 | 6.79 | source 상수, 주석상seed6 |
| .2BS+.8cloud_l05 | 15.77 | 17.53 | source 상수, 주석상seed0/1 |
| .2BS+.8all_l08 | 16.38 | 17.72 | source 상수, 주석상seed0/1 |
| .2BS+.8GRU 보조없음 | 미기재 | 16.82 | source 참고 상수 |
| .2BS+.4cloud+.4all | 미기재 | 미기재 | 혼합 코드만 존재 |

Site NMAE/Group NMAE/MAE/W금액/S/시간당 페널티/실제 유효시간·행 수가 담긴 실행 결과는 없다.
위 수치는 저장 예측 재계산도, 현재6seed 실행 결과도 아니다.
표의 점수로 실제 numerator/분모를 역산하지 않았다.

## 6. 활용 판단

- A(단독 정산 강점): 미확인. 원래 source 점수는 다른 기준이다.
- B(낮은 잔차상관): 미확인. 공통 예측 없이는 계산할 수 없다.
- C(특정 날씨/시각): 실측 보조학습이 이점을 줄 가능성은 있으나 실제 slice 검증 전에는 추천하지 않는다.
- D(신규 발전소): ID가 없어 unseen embedding 문제는 없지만 보조학습·정적 공유의 실제 효과는 미확인이다.
- E(추가 비용): 저장 예측 혼합은 CPU만 필요하다. 원래 모델 추론은2종×6seed의 큰 GRU라 단일 TSMixer보다 무겁다.

공통 목표는 잔차상관 자체가 아니라 동일 mask/그룹/SMP/k에서 W 감소다.
현재 TSMixer 추가 학습은 필요 없다. SJ 검증 예측과 **올바른 시간 경계의 state01 검증 자료**를 먼저 확보한다.
기존 원본 코드밖에 없다면 state01의 cutoff/누수 문제를 고친 별도 검증 실험이 추후 필요할 수 있으며 이번에는 수행하지 않았다.
