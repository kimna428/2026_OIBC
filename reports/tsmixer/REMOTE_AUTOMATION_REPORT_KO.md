# TSMixer 원격 GPU 학습 자동화 추가 결과

작성일: 2026-11-02. 기준 구현: `67224e4` (`feature/tsmixer-v0-mac-cuda`).

## 1. 완료 범위

Mac에서 SSH로 기존 RTX 3090 Ti 서버의 학습을 시작하고, 상태를 조회하고, 종료한 결과를 가져오는 제어 계층을 추가했다. 서버에 Codex를 설치하지 않는다.

**기존 모델·weighted L1·split·정규화·학습 루프·평가기·기본 YAML·기존 서버 스크립트는 변경하지 않았다.** 기존 README에는 새 원격 안내 링크만 추가했고 `.gitignore`에는 로컬 설정과 회수 결과 제외 규칙을 추가했다.

이번 작업에서 실제 서버 SSH 접속, GitHub 조회/push, 원격 GPU 학습, holdout 평가, 공식 test 추론, 원본 전송·캐시 생성은 수행하지 않았다. 기존 unit suite의 작은 CPU/MPS 검사만 재실행했다.

## 2. 추가 파일과 기능

| 파일 | 기능 |
|---|---|
| `scripts/tsmixer/remote_setup.sh` | 기본 검사 전용. SSH·Git·Python/CUDA·데이터·기존 캐시 확인. 별도 옵션으로만 sync/tests/smoke/cache 요청 |
| `scripts/tsmixer/remote_launch.sh` | 사용자 직접 실행 시 push된 commit 확인·서버 fast-forward·준비 검사·고유 ID·detached worker 시작 |
| `scripts/tsmixer/remote_status.sh` | 최근/지정 실험 ID·PID·단계·epoch·loss/dev 지표·GPU/VRAM·로그20줄·종료 코드 |
| `scripts/tsmixer/remote_fetch.sh` | 종료 실험의 소형 결과만 rsync 및 SHA-256 검증 후 Mac에 보관 |
| `scripts/tsmixer/remote_control.py` | Mac stdlib 제어기. 설정 검증, JSON/안전 quoting, 실제 pushed commit 확인, 결과 경로·무결성 검증 |
| `scripts/tsmixer/remote_agent.py` | 서버 stdlib endpoint. 기본 setup은 read-only, 명시적 fast-forward/준비/launch/export 처리 |
| `scripts/tsmixer/remote_worker.py` | 기존 학습 wrapper와 기존 development 평가 CLI를 순차 호출하는 detached orchestration |
| `configs/tsmixer/remote.example.json` | 실제 주소·키·비밀번호가 없는 설정 예시 |
| `scripts/tsmixer/REMOTE_README_KO.md` | 최초 설정·사용법·실패 처리·서버 통합 검사 안내 |
| `scripts/tsmixer/tests/test_remote.py` | SSH/rsync/GPU 모의 검사 및 로컬 Git/rsync 통합 검사 |
| `reports/tsmixer/remote_automation_tests.txt` | 최종 원격 자동화 테스트 기록 |
| `reports/tsmixer/remote_tsmixer_regression.txt` | 기존 TSMixer unit 회귀 검사 기록 |
| `reports/tsmixer/remote_verification.json` | 변경 범위·검증 결과·원격 실행 미수행 기록 |

새 학습기·평가기는 작성하지 않았다. Python 제어 소스는 `scripts/tsmixer/`에 둬서 기존 checkpoint가 사용하는 `analysis/tsmixer/` 소스 hash 계약도 유지했다.

## 3. 원격 동작과 안전성

### 기본 setup

`remote_setup.sh` 기본 실행은 Git fetch, Python 설치, 데이터 전송, 캐시 생성, smoke 또는 학습을 실행하지 않는다. 기존 `server_preflight.sh`를 통해 환경과 데이터를 검사하며, 문제 시 필요한 준비 절차를 출력한다. Python 환경이 없으면 자동 설치하지 않는다.

사용자가 명시적으로 요청하는 옵션은 다음과 같다.

- `--sync`: push된 Mac commit으로 fetch 및 `merge --ff-only`.
- `--tests`: 기존 TSMixer unit suite 실행.
- `--smoke`: 기존 CUDA synthetic5step 및 캐시가 있으면 실데이터5step 실행.
- `--prepare-cache`: 기존 `prepare_cache.sh` 호출.

tracked 변경, 다른 branch, 분기된 Git history, 활성 원격 실험이 있으면 갱신을 거부한다. 기존 OIBC GPU lock을 보유한 수동 실행도 Git 갱신 전에 검사한다. reset/clean/rebase/강제 push/기존 결과 삭제는 하지 않는다. 서버의 개인 미추적 파일을 보존하며, fast-forward가 미추적 파일과 충돌하면 Git 자체가 중단한다.

### launch와 자동 평가

Mac HEAD·실제 push된 origin branch commit·로컬 제어 소스의 Git blob이 일치해야 한다. 서버도 같은 branch와 commit으로 맞춘다. 새 고유 실험 디렉터리를 원자적으로 생성하며 중복 ID를 거부한다.

worker는 새 session, 닫힌 stdin, 파일로 연결된 stdout/stderr를 사용해 SSH stdio와 분리된다. 기존 `train_development.sh --full-train --config configs/tsmixer/base.yaml`을 실행하고, 성공한 경우에만 기존 `analysis.tsmixer evaluate`를 `best.pt`와 새 `eval_dev` 경로로 실행한다.

기존 조건은 그대로다.

```text
seed = 2026
cutoff = 2025-01-31 17:00 KST
dev 운영일 = [2025-02-01, 2025-02-28)
```

Holdout·공식 test 호출은 없다. `completed`, `training_failed`, `evaluation_failed`, `worker_failed`, `launch_failed`를 구분하고 학습·평가 종료 코드를 따로 기록한다. 학습0/평가비정상 종료를 학습 실패로 합치지 않는다. 갑작스러운 worker 소실은 `interrupted_or_lost`로 표시하고 종료 코드를 추정하지 않는다. PID 시작 시각과 남은 process group도 확인한다.

**Validation loss 표시 한계:** 기존 v0는 validation loss를 기록하지 않는다. 학습기를 변경하거나 값을 지어내지 않고 `validation_loss=null`과 설명을 출력한다. 대신 기존 development CF/kWh MAE·정산 페널티 지표를 표시한다.

### fetch

종료한 정상·실패 실험을 회수한다. 기본 allowlist는 YAML, run/epochs/metrics JSON, 평가 CSV, 한국어 summary, 학습/평가/worker 로그, Git·환경·상태 정보다. best.pt/last.pt는 경로·크기·SHA-256만 기록한다. `.pt`, `.npy`, Parquet·원본·캐시는 복사하지 않는다.

서버는 별도 export snapshot을 생성한다. Mac은 rsync 종료 코드, 파일별 SHA-256·크기, 허용 경로, 중복·symlink를 검사한 뒤에만 최종 결과 경로로 전환한다. 파일64MiB·전체128MiB 상한이 있다. 실패 시 partial을 남기고 성공으로 처리하지 않는다. 기존 결과는 덮어쓰지 않는다.

```text
reports/tsmixer/remote_results/<experiment_id>/
```

이 경로와 기본 `.tsmixer-remote.json`은 Git 제외다. custom 설정 파일도 저장소 밖 또는 별도 ignore 경로에 보관해야 한다.

## 4. Mac에서 실제로 수행한 검증

| 항목 | 결과 |
|---|---|
| 신규 원격 자동화 suite | **37개 모두 통과**, 약3.3초 |
| 기존 TSMixer suite | **29개 중28개 통과, CUDA1개 skip**, 약9초 |
| Shell syntax | 기존7개+신규4개, 총11개 통과 |
| 잘못된 SSH 설정·host option·명령 삽입·부적절한 경로·ID | 검출 |
| Mac 미push commit·제어 소스 불일치·서버 commit/branch mismatch | 검출 |
| 중복 실험·활성 실험의 checkout 갱신 | 거부 |
| 정상·학습 실패·평가 실패·실행 중·소실 상태 | 모의 worker/status 검사 통과 |
| 인자 quoting | 공백·작은따옴표·`$()`·semicolon을 JSON/argv의 리터럴로 전달함을 확인 |
| 셸 wrapper 연결 | 가짜 SSH/Git 실행 파일로 setup/status/launch 연결 확인; 실제 SSH는 호출하지 않음 |
| 실제 로컬 Git | 임시 bare 저장소만 사용해 fast-forward·개인 미추적 파일 보존·분기된 history 갱신 거부 확인 |
| 실제 로컬 rsync | Mac openrsync 2.6.9-compatible의 옵션·file allowlist를 임시 디렉터리 전송으로 확인 |
| 전송 실패·손상·기존 결과·symlink | 실패 검출·partial 보존·덮어쓰기 금지 확인 |
| 원본 학습·평가 코드 변경 여부 | 기준 commit 대비 diff 없음 |

오프라인 Git fixture에서는 임시 로컬 경로의 bare repository를 origin으로 사용했다. GitHub나 실제 서버에는 연결하지 않았다. rsync도 실제 검사에서는 로컬 경로끼리만 전송했다. GPU 동작은 원격 자동화 테스트에서 모의 처리했고, 기존 unit suite는 Mac에서 CPU/MPS로 실행했다.

검증 환경: 로컬 제어 테스트 Python3.15.0, 기존 모델 suite Python3.13.12/PyTorch2.15.0. 제어기는 stdlib만 사용하고 최소 bootstrap Python3.9를 검사한다. 서버 모델 환경은 기존 CUDA 설치 절차와 Python 버전 정책을 따른다.

## 5. Git 상태와 push

기존 `feature/tsmixer-v0-mac-cuda`에서 이번 신규 스크립트·테스트·설정 예시·문서·작은 검증 기록과 README/ignore 변경만 선택적으로 커밋한다. 본 보고서가 포함된 실제 납품 commit은 다음 명령으로 확인한다.

```bash
git log -1 --oneline
git show --stat --oneline HEAD
git status --short
```

기존 사용자 미추적 파일5개는 stage·수정·삭제하지 않았다.

```text
reports/TSMIXER_REPOSITORY_ANALYSIS_KO.md
reports/deep_learning/MODEL_OVERVIEW.md
reports/deep_learning/compact_initial_parity.json
reports/deep_learning/compact_storage_estimate.json
reports/deep_learning/compact_test_parity.json
```

GitHub push는 사용자가 직접 실행한다.

```bash
git push -u origin feature/tsmixer-v0-mac-cuda
```

## 6. 사용자 설정과 정확한 실행 순서

### 최초 설정

Mac SSH alias를 `~/.ssh/config`에 등록하고 호스트 키·공개키 인증을 직접 확인한다. 예시는 `scripts/tsmixer/REMOTE_README_KO.md`에 있다. 비밀번호·키는 JSON이나 Git에 넣지 않는다.

```bash
cp configs/tsmixer/remote.example.json .tsmixer-remote.json
chmod 600 .tsmixer-remote.json
```

다음7개 실제 값을 설정한다.

```text
SSH_HOST
REMOTE_REPO_DIR
REMOTE_BRANCH
REMOTE_DATA_ROOT
REMOTE_CACHE_ROOT
REMOTE_PYTHON
REMOTE_RUN_ROOT
```

환경변수로도 지정할 수 있고 환경변수가 JSON보다 우선한다. `REMOTE_BOOTSTRAP_PYTHON`(기본 python3), 외부 경로용 `REMOTE_STORAGE_ROOT`, 확인한 quota인 `REMOTE_QUOTA_GIB`는 선택이다. 서버 경로는 정규화된 절대 경로이며 run root에는 symlink나 데이터·캐시와의 중첩을 허용하지 않는다.

기존 서버 checkout·CUDA 환경·데이터가 있으면 그대로 사용한다. 없으면 사용자가 새 경로에 clone, 기존 `install_cuda.sh`로 명시적 환경 설치, 별도 안전 데이터 전송을 먼저 수행한다. 서버에 Codex는 필요 없다.

### 최초 점검·명시적 검증

```bash
bash scripts/tsmixer/remote_setup.sh
bash scripts/tsmixer/remote_setup.sh --sync --tests --smoke
```

캐시가 없으면 위 두 번째 명령 대신 다음을 명시적으로 실행한다.

```bash
bash scripts/tsmixer/remote_setup.sh --sync --prepare-cache --tests --smoke
```

### 매 실험 실행·조회·회수

```bash
git push -u origin feature/tsmixer-v0-mac-cuda
bash scripts/tsmixer/remote_launch.sh
bash scripts/tsmixer/remote_status.sh
bash scripts/tsmixer/remote_fetch.sh <experiment_id>
```

launch가 출력한 ID를 보관한다. 특정 실험 조회는 `remote_status.sh <experiment_id>`, 직접 ID를 지정한 시작은 `remote_launch.sh tsmixer_v0_trial001`이다. 응답을 놓쳤다면 새 launch 전에 status로 확인한다.

## 7. 실제 서버에서 남은 통합 검증

1. SSH alias·호스트 키·키 인증, 비대화형 SSH stdout의 JSON 전달, bootstrap Python·GitHub 읽기 권한.
2. 설정한 branch·경로·권한·개인 quota, NVIDIA 드라이버·CUDA torch·RTX3090Ti VRAM.
3. 원본8개 checksum·전체 historical compact 캐시·기존 preflight/unit/CUDA smoke 통과.
4. SSH 연결 종료 후 detached worker가 계속 실행되는지 확인. 서버의 재부팅·세션 정리 정책·SIGKILL은 별도 운영 문제다.
5. 전체 학습 성공 후 자동 dev 평가, 두 종료 코드·한국어 summary와 metrics 생성 확인.
6. 실제 Mac/서버 rsync 조합의 quoting·전송·SHA-256 검증과 기본 대용량 제외 확인.

**판정:** 원격 실행 기능과 오프라인 검증을 완료했다. 사용자가 설정·push·최초 서버 통합 검사를 마치면 Mac에서 launch 한 번으로 학습과 development 평가를 시작하고, fetch 한 번으로 소형 결과를 회수할 수 있다. 실제 원격 접속과 전체 학습은 이번 작업에서 수행하지 않았다.
