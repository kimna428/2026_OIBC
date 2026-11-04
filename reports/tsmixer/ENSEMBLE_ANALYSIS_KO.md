# TSMixer 추가 앙상블 분석 — 2026-11-04

실제 연간 예측 파일이 양쪽 모두 로컬에 없어 **잔차 상관계수와 alpha별 실제 성능은 계산 불가**다.
팀원 config의 기대 점수와 TSMixer의 저장 metrics만으로 상관·상쇄·유리한 시각을 역산하지 않는다.
`runs/tsmixer/comparison/local_audit_20261104_002/ensemble.json`은 unavailable 상태와 이유를 기록한다.
fixture 테스트 결과는 실제 모델 실험으로 제시하지 않는다.

## 이미 구현한 분석

[비교 모듈](../../analysis/tsmixer/comparison/README_KO.md)은 동일 관측행으로 정렬한 뒤 다음을 산출한다.

| 분석 | 계산 방법 | 해석 한계 |
|---|---|---|
| site 잔차 상관 | pred−actual의 kWh 및 CF Pearson | kWh는 대형 발전소 영향 큼; 둘을 함께 본다 |
| 그룹 잔차 상관 | 같은 배정/그룹/시간으로 merge한 잔차 | 부분 관측의 그룹 용량 변화는 계약과 함께 기록 |
| 시간별·월별·계절별 | NMAE/MAE/평균 bias/과대·과소 행 수 | 같은 구간 안의 설명 통계 |
| 규모·미학습 발전소 | <100/100~500/≥500kW; cutoff 기준0/1~2159/2160+ 학습 시간 | 미래 관측이 cohort 분류에 들어가지 않음 |
| 높은 태양 위치 | midpoint elevation>45도 | 실측 일사량이 없으므로 실제 고일사량이라고 부르지 않음 |
| 그룹 오차 상쇄 | 1−Σ|그룹잔차|/Σ|site잔차| | 다른 그룹과 직접 비교할 때 site mask가 같아야 함 |
| 상대 성공/실패 사례 | 그룹·시간 페널티 차이 양방향 상위25개 | 시각별 성공이 구조적 일반화라는 뜻은 아님 |

상관은 야간0이 다수인 전체 표본과 주간 subset의 오차 slice를 함께 해석한다.
현재 모듈의 상관 CSV는 전체 관측행 기준이며 주간만의 상관은 추가 분석이 필요하다.
원하는 방향의 상관을 얻기 위해 임의로 시각을 제외하지 않는다.

## alpha 실험

`P=(1−alpha)P_team+alpha P_tsmixer`, alpha는 0/.05/.10/.15/.20/.30/.40/.50이다.
각 alpha마다 기존 균형/팀원/metadata 배정으로 site NMAE, ratio/macro group NMAE,
MAE, W, S, 시간당 페널티, 잠정 내부점수, 유효시간/관측행을 기록한다.
가중치를 바꿔도 정답 mask·가격·그룹은 그대로다.

낮은 site MAE나 낮은 잔차 상관만으로 W 개선이 보장되지는 않는다.
최종 목적은 SMP×k가 큰 시간대의 그룹 절대잔차를 줄이는 것이다.
TSMixer 단독이 열세여도 팀원의 큰 오차를 반대 방향으로 보완한다면 작은 alpha의 가치가 있을 수 있다.
이것은 가능성 설명이며 실제 상관/개선량은 미확인이다.

## 모델 선택과 독립 검증의 구분

1. **탐색:** 현재 검증1년의 alpha 전체 곡선을 기록한다. 최소 W alpha를 자동 선정하거나 배포하지 않는다.
2. **시간순 OOF 선택:** 2024-05-31 17시 이전의 두 모델 OOF를 확보한다.
   같은 fold cutoff 이전으로 normalizer·보정·teacher·그룹을 고정하고 fold 뒤에서 예측한다.
   마지막2~3계절은 가중치 선택 이후의 안정성 확인용으로 남기는 방안이 좋다.
3. **고정 적용:** OOF W로 선택한 alpha를 현재 연간 검증에 고정 적용한다.
   모듈은 각 row의 양쪽 모델 cutoff를 발행 시각과 검사하고 평가연도 정답을 OOF로 받지 않는다.
4. **독립 최종 평가:** 지금까지 구조·seed·마스크·alpha 선택에 사용하지 않은 정답 구간에서 고정 recipe를 평가한다.
   현 공유 자료에는 이런 구간의 두 모델 예측/정답이 없다. 2025~26 제출 예측은 정답이 없으므로 성능 평가에 쓸 수 없다.

그룹 배정은 현재 2024-05까지 이력으로 만든 것이다. 그보다 앞선 OOF에 이 배정을 소급 적용해 alpha를 고르면
그룹이 미래 fold의 잔차를 이미 반영했을 수 있다. 제공 metadata만의 그룹 또는 각 fold 이전으로 생성한 그룹을
쓰는 대조가 필요하다. 코드의 cutoff 열 검사는 실제 학습 provenance의 진실성을 자동 증명하지 않는다.

## 필요한 추가 파일

- TSMixer 연간 predictions.npy, prediction_axis.json, run.json: 서버에서 가져온다.
- 팀원 validation site_predictions.parquet 또는 selected_stack_all_oof.parquet: 3개 component 예측 포함.
- OOF 가중치 선택을 하려면 cutoff 이전 TSMixer OOF도 필요하다. 현재 TSMixer1년 모델의 in-sample 과거 예측으로 대체하지 않는다.
- 기준 W 호환 검증을 위해 원래 A/B baseline의 같은 연도 예측도 받는다.

자료 확보 전에는 실제 alpha별 표를 빈칸 대신 unavailable 사유로 남겼다.
실행 명령은 비교 README의 공통 평가 명령에 `--oof-predictions`를 선택적으로 추가한다.
