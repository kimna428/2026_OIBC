# 실행 결과와 현재 가능한 결론

2026-11-04 Mac CPU에서 원본 데이터 감사 및 기존 F3 참고 평가를 수행했다. 새 프로토콜의 실제 OOF는 세 모델×세 Fold 모두 unavailable이다. 따라서 새로운 앙상블 성능, 선택 가중치, 잔차 상관, Fold 일반화 개선 수치는 아직 산출할 수 없다. 임의 점수나0 예측을 생성하지 않았다.

저장 감사: runs/tsmixer/ensemble/walkforward_v1/audit_final_20261104/audit.json 및 legacy_F3_reference.json. 데이터 규모는 WALKFORWARD_PROTOCOL_KO.md 표에 기록했다. SJ는 weights가 있지만 검증 예측/준비 입력/모든 manifest 파일이 없으며 원본 체크포인트만으로 과거 OOF를 증명할 수 없다. state01은 코드와 데이터만 확인되며 검증 체크포인트/연간 예측이 없다.

중요한 새 발견: 기존 TSMixer F3 내부 Epoch 선택은 2024-06-01 00시까지의 정답을 포함해 최종17시 cutoff보다7시간 늦었다. 저장 predictions는 변경하지 않고 legacy_reference_not_strict_OOF로 구분한다. 새 내부 선택 후 Epoch가 기존2와 같고 소스/설정/cache/정규화가 일치하는 경우에만 기존 refit 가중치를 재사용할 수 있다. 기존 predictions 자체를 strict OOF로 등록하지 않는다.

기존 F3는 참고로만 재계산했다. site NMAE 4.018044589%, site MAE 약6.594384113kWh이다. 용량 균형 W=265,885,394.2495원, S=7,231,698,092.6482원; SJ 배정 W=265,331,397.7142원, S=7,232,252,089.1835원이다. 이는 새 프로토콜의 독립 평가가 아니며 팀원 모델과 공정 비교한 점수가 아니다. W_ref 계약 증명이 없어 내부점수는 null/unverified로 출력한다. 신규 구현은 final SJ 배정의 작은 개선을 학습할 가중치 선택에 사용하지 않는다.

가장 비용 효율적인 다음 실험은 F1의 단일 seed TSMixer/SJ/state01 smoke 후 누락 OOF를 한 계열씩 생성하는 것이다. F1에는 과거 관측 발전소가45개뿐이므로 신규 발전소 일반화와 01_state 보조학습의 안정성이 중요하다. F1 성능만으로 선택한 가중치를 F2에 고정 적용한 결과를 먼저 확인한다. F3 W만 보고 TSMixer 추가 학습의 가치를 결정할 증거는 없다.

이번 실제 실행: unittest 신규12개, 전체 회귀 suite, 작은 CPU fixture smoke(각5update), 실제 데이터 audit, 예측이 없는 score/F3 fixed apply의 unavailable 경로를 실행했다. fixture 학습/앙상블 점수는 실제 모델 성능이 아니다. 실제 원본 외부 프로그램 실행, pickle/joblib 로드, 패키지 설치, 원격 접속, GPU 전체 학습, commit/push는 수행하지 않았다. 정확한 전체 테스트 결과는 아래 검증 기록에 추가한다.

## 검증 기록

- `.venv/bin/python -m unittest analysis.tsmixer.tests.test_walkforward -v`: 신규12개 통과.
- `.venv/bin/python -m unittest discover -s analysis/tsmixer/tests -v`: 전체81개 실행,80개 통과, CUDA1개 skip. 기존 다중 프로세스 DataLoader 테스트는 sandbox 공유 메모리 제한으로 첫 실행이 실패하여 승인된 sandbox 밖 실행으로 통과했다. MPS 경량 기존 테스트는 통과했다.
- 실제 데이터 `ensemble.walkforward audit` 최종 산출물에서 F1/F2/F3 시간축·관측 행·그룹 무결성을 검사했다. 외부 팀원 프로그램은 실행하지 않았다.
- `ensemble.run score --fold F1`은 검증된 예측이 없어 individual/trials 빈 결과를 저장한다. 가중치 선택을 실제 수행했다는 의미가 아니다.
- CPU fixture는 작은 네트워크이며 GPU 전체 모델의 시간/VRAM 추정치로 사용하지 않는다. 로컬 XGBoost가 없어 tree end-to-end 실행도 하지 않았다.

## 변경 범위와 Git

이번 추가 소스는 `analysis/tsmixer/ensemble/` 전체, `requirements-walkforward.txt`, `test_walkforward.py`, `configs/tsmixer/ensemble_walkforward.yaml`, `ensemble_sj_features.json`, `scripts/tsmixer/run_walkforward.sh`, `run_ensemble.sh`, 네 보고서다. comparison/io.py에는 Python3.10 SHA fallback이 있다. `.gitignore`에 `.ubj` 제외를 추가했다. 이전 비교 소스·보고서는 기존 미추적 작업과 함께 작업 트리에 남아 있다. commit/push/staging은 하지 않았다. ignored runs 내부 실제 결과는 Git에 포함되지 않는다.
