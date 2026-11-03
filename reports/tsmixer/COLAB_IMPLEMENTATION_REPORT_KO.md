# TSMixer Colab 실행 환경 구축 결과

작성일: 2026-11-03. 기준 소스: `1762676` (`feature/tsmixer-v0-mac-cuda`).

## 구현 결과

저장소 루트에 **`train_colab.ipynb`**를 추가했다. 한국어 실행 README는 **`scripts/tsmixer/colab/README_KO.md`**다. Notebook은 미실행 상태이며25개 셀(설명 포함),12단계로 구성했다.

| 파일 | 역할 |
|---|---|
| `train_colab.ipynb` | GitHub Clone·Drive Mount·의존성·GPU·로컬 데이터·캐시·학습·재개·dev 평가·백업 |
| `configs/tsmixer/colab.yaml` | 기존 모델·loss·split을 유지한 Colab 설정, constant scheduler |
| `analysis/tsmixer/persistence.py` | 검증 가능한 generation 백업·손상/불완전 세대 fallback 복원 |
| `analysis/tsmixer/colab.py` | 경로 인자 기반 로컬 데이터 복사·세션 commit 고정·로그·development CSV |
| `analysis/tsmixer/train.py` | 선택적 epoch 백업 hook, constant scheduler 상태, 완료/선택 상태 보존 |
| `analysis/tsmixer/__main__.py` | `--backup-root`, `--export-csv`, 평가 checkpoint hash |
| `analysis/tsmixer/config.py` | 기존 constant LR 외의 scheduler 거부 |
| `analysis/tsmixer/tests/test_colab.py` | 백업·복원·복사·CSV·scheduler/RNG·notebook 테스트20개 |
| `analysis/tsmixer/README.md` | Colab 실행 안내 링크 추가 |
| `reports/tsmixer/colab_tests.txt` | 기존+신규 unit 결과 |
| `reports/tsmixer/colab_remote_regression.txt` | 기존 원격 제어 회귀 테스트 결과 |
| `reports/tsmixer/colab_verification.json` | 정적 검사·변경 범위·smoke 상태 증거 |

## 기존 코드 분석과 유지한 계약

기존 `runtime.device_for(auto)`는 Linux에서 CUDA가 있으면 CUDA를 고른다. Colab도 이 경로로 T4 등을 사용하며 특정 GPU 이름을 요구하지 않는다. 모델 코드에 Colab 전용 경로·Google API import를 추가하지 않았다. Drive·`/content` 경로는 노트북 설정에만 있다. Mac CPU/MPS 및 Linux GPU 사용 방식을 유지한다.

기존 `train.py`는 이미 epoch마다 모델/optimizer/scaler/RNG와 normalizer를 저장하고 epoch 경계 resume를 지원했다. 누락된 scheduler는 **constant LambdaLR**로 명시해 원래 lr0.001을 바꾸지 않았다. 실제 `train_epoch`의 AdamW 업데이트·weighted L1·accumulation·gradient clipping은 변경하지 않았다.

변경하지 않은 항목:

- TSMixer hidden128·3blocks·dropout0.1, 입력 weather90/time8/static146와 기존 InputEncoder.
- 기존 원본·feature contract·round17/round10/revision·compact cache·DayDataset.
- train-only 정규화, NaN target mask, CF `[0,1]` 학습 필터, SMP/k의 loss·정산 사용.
- 운영일01시~다음00시, 전일17시 예보 발행 계약.
- **cutoff=2025-01-31 17:00 KST**, **dev 운영일 [2025-02-01,2025-02-28)**.
- 기존 G1/G2 배정과 train_calibrated 후처리, 기존 정산 Evaluator.
- Mac full-train 금지 및 Linux CUDA/명시적 `--full-train` 조건.

## Notebook 기본 동작

기본은 `MODE="smoke"`, `ALLOW_FULL_TRAIN=False`, `REAL_DATA_SMOKE=False`,5step이다. 작은 synthetic 데이터로 확인하며 원본 복사·대용량 캐시 생성·전체 학습을 생략한다. 전체 학습은 MODE와 허용 flag를 직접 바꾸고 새 RUN_ID를 사용해야 한다.

실데이터를 사용할 때는 Drive 원본8개를 로컬 디스크에 스트리밍 복사하고 checksum을 검증한다. 실제 학습 I/O는 로컬 compact cache에서 발생한다. 기본 Colab CUDA torch는 유지하며 공통 dependencies만 설치한다. fp32 기본이며 T4에서 bf16을 강제하지 않는다.

## Checkpoint·Drive 백업·Resume

새 checkpoint는 모델·optimizer·**scheduler**·AMP scaler·완료 epoch·global step·best score·early-stop counter, Python/NumPy/Torch CPU/CUDA RNG·DataLoader generator·normalizer·cache/feature/source/config signature·환경·Git commit을 포함한다. MPS 상태도 기존대로 보존한다.

매 완료 epoch 후 별도 새 generation에 last/best·run.json·epochs.jsonl 등을 복사하고 검증한다. manifest와 마지막 COMMITTED marker가 있어야 복원 후보가 된다. SHA-256·크기·checkpoint 상태를 모두 확인하며 최신 손상/불완전 백업은 건너뛴다. 모든 세대가 손상되면 오류로 멈춘다. 과거 백업·기존 local run을 덮어쓰거나 삭제하지 않는다.

Drive가 fsync를 지원하면 사용하고, fsync 미지원일 때만 재읽기 checksum/marker로 진행한다. 실제 I/O 오류는 무시하지 않는다. mounted Drive/FUSE가 원격 영속성을 절대 보장하는 것은 아니며, 마지막 미완료 epoch·아직 flush되지 않은 작업까지 살릴 수는 없다. 백업 실패 시 학습을 멈추고 이전 정상 세대를 보존한다.

재접속 시 같은 RUN_ID/설정으로1~12번을 재실행한다. 저장된 `session_ref.json`으로 원래 source commit을 checkout하고, 데이터를 다시 로컬에 준비한 뒤 마지막 정상 checkpoint에서 이어간다. 데이터·cache는 백업하지 않으며 같은 signature로 재생성해야 한다. synthetic smoke도 cache를 재생성해 복원할 수 있다.

이전 v0 checkpoint는 수정하지 않았다. source hash와 scheduler 상태가 달라졌으므로 기존 실험은 원래 commit에서 재개하고 새 Colab 실험에는 본 변경을 사용한다. CUDA 장치 종류나 라이브러리 버전 변경 시 bitwise 동일성은 보장하지 않는다.

## Validation·예측·로그 산출물

전체 학습 완료 후 기존 development 평가 CLI를 호출한다. metrics.json·평가 CSV·정산 ledger·canonical 예측 NumPy를 저장하고, postprocessed 예측에서 다음 CSV를 만든다.

- `development_site_predictions.csv`: 발전소·KST ts·CF·kWh.
- `development_group_forecast.csv`: 기존 G1/G2·ts·발전량 합계.
- `composition.csv`: 사용한 기존 배정.
- `metadata.json`: dev 역할, cache signature, 행 수, `independent_oof=false`.

현재 단일 development 실험은 독립적인 OOF를 만들지 않는다. 이 CSV는 **development 최종 예측이며 공식 test 제출물이 아니다.** Holdout 평가·공식 test 추론은 실행하지 않는다. 기존 평가 결과가 best.pt checksum과 다르거나 불완전하면 진단 폴더로 보존하고 다시 평가한다.

epoch 구조화 로그는 매 백업에 포함하고, 일반 console log·YAML·환경 freeze는 셀 종료/예외 처리 및 최종 백업에 추가한다. 원본·cache를 제외한 최종 metrics/CSV/log/checkpoint를 Drive generation에 보관한다.

## Mac 검증 결과

- **전체 TSMixer/Colab unit49개:48개 통과, CUDA1개 skip**, 약11.4초. 기존 CPU·MPS 검사 포함.
- 신규20개 검사: 완료 marker·중단/손상 fallback·Drive 오류·fsync 미지원·overwrite/경로 차단·세션 pin·원본 복사·CSV·로그·scheduler/RNG resume·미실행 notebook AST.
- 기존 원격 제어 회귀37개 모두 통과. 실제 SSH/서버 연결은 없고 네트워크 부분은 모의 처리했다.
- 기본305,865파라미터 모델의 CPU CLI5step·백업·복원 후10step까지 재개를 실행했다. 학습률0.001 유지, checkpoint 재로드 정합성 통과. 작은 smoke의 partial-epoch 확장은 exact trajectory를 주장하지 않는다.
- 완료 epoch 경계의 optimizer/scheduler/RNG 재개는 dropout을 켠 CPU 테스트에서 bitwise 일치했다.
- Notebook25개 셀의 JSON 구조·id·미실행 상태·Python AST와 smoke 기본값을 확인했다. 별도 nbformat 패키지는 설치돼 있지 않아 그 validator는 실행하지 않았다.
- 대규모 학습, 원본 전체 메모리 적재, 새 대용량 cache 생성, Google 계정/Drive 접속, Colab/T4 실행, 원격 GPU 학습, holdout/test 예측은 하지 않았다.

## 사용자 실행 순서

1. 이번 변경을 GitHub에 반영한 다음 `train_colab.ipynb`를 Colab에서 연다.
2. GPU 런타임을 선택하고 셀1의 저장소·Drive 경로·RUN_ID를 확인한다.
3. **기본 smoke로1~12번 실행**한다.
4. 실제 데이터8개를 Drive에 준비한다. full 학습에는 새 RUN_ID, MODE=development, ALLOW_FULL_TRAIN=True로1번부터 재실행한다.
5. 런타임이 끊기면 같은 설정으로1번부터 실행한다. 원래 commit과 정상 checkpoint를 자동 선택한다.
6. Drive의 `runs/<RUN_ID>/snapshots/`에서 완료 marker·manifest가 있는 최신 정상 결과를 확인한다.

세부 셀별 표·폴더 구조·오류 복구는 `scripts/tsmixer/colab/README_KO.md`를 따른다. 실제 Colab CUDA·VRAM, full cache 시간/메모리, Drive/FUSE·재접속 동작은 사용자가 Colab에서 통합 검증해야 한다.

공식 참고: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Drive I/O 예제](https://colab.research.google.com/notebooks/io.ipynb), [PyTorch LambdaLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html).
