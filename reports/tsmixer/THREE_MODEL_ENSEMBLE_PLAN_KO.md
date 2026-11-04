# 세 모델 앙상블 검증 계획 — 2026-11-04

현재 실제 계산된 혼합은 **TSMixer 단독**뿐이다.
SJ/state01 예측이 없으므로 모델 간 상관계수나 혼합 개선량을 만들어 제시하지 않는다.
01_state 원본 프로토콜을 그대로 실행한 예측은 cutoff가 늦으므로 파일이 도착하더라도 안전한 연간 검증으로 자동 인정하지 않는다.

## 구현된 후보와 분석

family 이름은 `tsmixer`, `sj_ensemble`, `state01`이다.
같은 timestamp/site_id/actual-mask/그룹/가격으로 정렬 후 평가한다.
SJ component3개와 state01 component4개도 단독 지표/오차 분석에 포함할 수 있다.
component를 family와 함께 중복 가중하는 탐색은 현재 하지 않는다.

| 후보 | 지원 | 현재 실데이터 상태 |
|---|---|---|
| SJ 단독 | simplex 꼭짓점 | unavailable |
| 01_state 단독 | 꼭짓점 | unavailable/원본 프로토콜 부적합 |
| TSMixer 단독 | 꼭짓점 | 실제 점수 계산 |
| SJ+TSMixer | 변 | unavailable |
| SJ+01_state | 변 | unavailable |
| 01_state+TSMixer | 변 | unavailable |
| 세 모델 | 내부 | unavailable |

0.10 간격은 단독3+두모델27+세모델36=66개,0.05는231개다.
음수 또는 합1이 아닌 weights는 거부하며, 없는 모델의 양의 가중치는 unavailable로 표시한다.
없는 모델을0예측으로 대체하거나 남은 모델로 재정규화하지 않는다.
0 가중치 모델의 NaN은 혼합 계산에 넣지 않는다.

각 조합은 site NMAE/MAE, ratio/macro group NMAE, W/S/시간당W, 잠정 내부점수,
유효시간·관측행 수를 SJ 최종/균형/metadata 각각에 기록한다.
가중치별 내부점수의 기준은 동일한W_ref지만 아직 분모 호환이 미검증이므로 provisional이다.
원래01_state의0.1없는 W와 L2비율 점수는 이 grid의 숫자로 재사용하지 않는다.

## 진단과 가치 판단

저장 예측이 들어오면 다음이 자동 생성된다.

- 같은 관측행의 kWh/CF residual Pearson, 발전소별 모델쌍 상관/반대 부호 비율.
- G1/G2 각각의 residual Pearson/반대 부호/각 모델 W와50:50 혼합 W(탐색).
- 시간대/월/계절, 규모(<100,100~500,500+kW), cutoff 이전 유효관측0/90일미만/90일이상 cohort의 NMAE/MAE/bias.
- 과대/과소 예측 행 수, 그룹 오차 상쇄율, 상대 성공/실패 그룹·시간 상위25개.
- 고페널티 그룹·시간 상위50개와 각 모델의 signed residual/페널티를 나란히 기록.
- 태양고도>45도 slice는 기하학적 proxy이며 실측 고일사량 성능이라고 표현하지 않음.

A(단독W낮음), B(잔차다름), C(특정상황강점), D(신규사이트강점), E(비용낮음)를 모두 보되
추천의 최종 근거는 고정 조건의 **W 감소**다.
낮은 잔차상관만으로 추천하지 않고, SMP×k가 큰 시간의 그룹 절대오차가 줄었는지 확인한다.
예측값이 없는 현재는 B/C/D의 실제 우열을 판단할 수 없다.

## 시간순 선택 절차

1. **설명/탐색:** 현재1년의66개 곡선을 모두 기록한다. 이 구간의 최저W 가중치를 자동 채택하지 않는다.
2. **과거 OOF 확보:** 각 모델의 정답 이전 cutoff로 fit한 공통 과거예측을 받는다.
   site/ts/energy/smp/k와3개 예측, 각 family의 *_trained_until을 기록한다.
   이 cutoff는 normalizer/보조모델/aux필터/outage/보정/모델선택까지 포함한 최대시점이다.
3. **그룹도 고정:** 현재SJ 그룹은2024년5월까지 잔차로 정했으므로 이전OOF 전체에 소급해 선택하면 누수가 남는다.
   metadata-only 또는 가장 이른OOF 발행 이전 자료로 만든 고정 그룹을 사용한다.
   새 CLI는 별도 OOF그룹 CSV/provenance를 요구하고 미래 잔차로 선택한 그룹은 거부한다.
4. **OOF에서만 선택:** OOF의W 최소 weights를 고정한다. 모든OOF 정답≤2024-05-31 17시,
   각 모델 cutoff≤각 운영일 발행 이전이어야 한다. JSON/열의 선언 진실성은 원래 학습기록으로 별도 확인한다.
5. **이후 기간 고정 적용:** 선택 weights를 연간 예측에 적용해 oof_selection.json에 분리 저장한다.
   family 예측이 누락되면 적용 불가이며 weights를 바꾸지 않는다.
6. **독립 최종 평가:** 구조/seed/보정/weights를 고르는 데 사용하지 않은 이후 정답 구간에서 recipe를 고정 평가한다.
   현 연간 구간은 이미 반복 탐색됐으므로 이 단계의 새 독립 시험으로 표시하지 않는다.

현재 과거 TSMixer OOF도 없으므로2~5의 실데이터 결과는 없다.
새 OOF를 만들기 위한 GPU 학습은 이번 범위에서 실행하지 않았다.

## 자료를 받으면 실행할 명령

지금 감사/TS 재평가만:

```bash
cd /Users/kimna/workspace/2026_3_2/2026_OIBC
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --output runs/tsmixer/comparison/three_local_002
```

새 예측을 inbox에 넣은 뒤(기존 결과는 새 이름으로):

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --inbox runs/tsmixer/comparison_inputs --grid-step 0.10 \
  --output runs/tsmixer/comparison/three_received_002
```

필요한 inbox 파일은 SJ validation parquet, state01 validation parquet 및 cutoff-safe provenance JSON이다.
state01 native seed파일로 받으면 다음처럼 위치를 명시한다.

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --sj-predictions /실제/SJ검증예측.parquet \
  --state01-run /실제/수정된검증run \
  --state01-metadata /실제/state01_provenance.json \
  --output runs/tsmixer/comparison/three_native_002
```

과거OOF까지 받았을 때:

```bash
OIBC_PYTHON="$PWD/.venv/bin/python" bash scripts/tsmixer/compare_three_models.sh \
  --inbox runs/tsmixer/comparison_inputs \
  --oof-predictions /실제/세모델과거OOF.parquet \
  --oof-composition /실제/OOF이전고정배정.csv \
  --oof-group-provenance /실제/OOF그룹provenance.json \
  --output runs/tsmixer/comparison/three_oof_fixed_001
```

표준 열/단위/JSON/hash 형식은 [사용법](../../analysis/tsmixer/comparison/THREE_MODELS_KO.md)에 있다.
위의`/실제/...`는 제공되는 파일의 실제 경로로 바꾼다. 없는 예측/학습을 실행한 것처럼 간주하지 않는다.

## 비용과 다음 학습 판단

현재 비교는 CPU이며 TSMixer 재학습은 필요 없다. 데이터가 이미 저장돼 있으면 파일 추가 후 재평가하면 된다.
0.10 grid부터 시작해 더 촘촘한weights가 필요한 근거가 있을 때만0.05를 탐색한다.
많은 component가 있으면 메모리/CSV가 늘지만 모델 forward/optimizer는 실행하지 않는다.

원래01_state는 기본6seed×2GRU=12개 NN을 혼합하고, default inner/main_all/test를 모두 돌리면
GRU학습이36회다. 폭192/2층 GRU는약112만 파라미터로, 단일 TSMixer보다NN당약3.7배 크다.
전체 training/prediction float16 격자를 GPU에 올리고8일×253개를 한 배치로 처리한다.
소스의 RTX5070 약3시간 주석은 작성자의 설명이며 이번에 측정한 비용이 아니다.
Mac에서 전체 학습·기존원본 main.py·quick·설치 명령을 실행하지 않았다.

만약 안전한state01 검증 모델이 없다면 다음 단계는 **01_state에 한정한** cutoff17시/과거전용outage/
과거전용aux필터 수정과 별도 과거검증이다. TSMixer부터 다시 학습하는 것보다 이 자료의 유효성을 먼저 확보한다.
C의 잔차 보완성이나새 loss/보조학습 효과를 확인한 후에만 TSMixer 개선 재학습의 우선순위를 결정한다.
