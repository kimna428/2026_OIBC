# TSMixer Mac → RTX 3090 Ti 원격 실행

개발·수정은 Mac에서만 하고, 서버에는 Codex를 설치하지 않는다. 서버는 기존 Git checkout·GPU Python 환경·데이터·캐시를 재사용한다. 모델·손실·split·정규화·학습기·평가기에는 변경이 없다. 모든 Mac 명령은 프로젝트 루트에서 실행한다.

## 1. 최초 로컬 설정

SSH alias를 Mac의 `~/.ssh/config`에 직접 등록한다. 실제 접속 정보나 키를 저장소에 넣지 않는다.

```sshconfig
Host oibc-gpu
    HostName YOUR_SERVER_HOST
    User YOUR_USER
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

최초 호스트 키와 공개키 인증은 사용자가 평소 SSH 절차로 확인한다. 스크립트는 `BatchMode=yes`를 사용하며 비밀번호 입력이나 host-key 검사 비활성화를 하지 않는다. 포트는 SSH config의 `Port`에 넣는다.

```bash
cp configs/tsmixer/remote.example.json .tsmixer-remote.json
chmod 600 .tsmixer-remote.json
```

`.tsmixer-remote.json`을 편집한다. JSON은 shell로 source/eval하지 않으므로 `$HOME`, `~`, 명령 치환을 확장하지 않는다. 서버 경로는 **실제 절대 경로**를 적는다. `REMOTE_RUN_ROOT`는 심볼릭 링크 없는 경로여야 하며, 저장소 안에 둘 경우 Git에서 제외되는 `runs/` 아래에 둔다. 데이터·캐시 경로와 겹치면 거부한다.

| 항목 | 의미 |
|---|---|
| `SSH_HOST` | SSH alias 또는 user@host |
| `REMOTE_REPO_DIR` | 이미 존재하는 서버 checkout 루트 |
| `REMOTE_BRANCH` | 기본 `feature/tsmixer-v0-mac-cuda`; Mac·서버가 같은 branch여야 함 |
| `REMOTE_DATA_ROOT` | 제공 데이터8개 필수 파일이 있는 디렉터리 |
| `REMOTE_CACHE_ROOT` | 호환 train compact 캐시의 부모 디렉터리 |
| `REMOTE_PYTHON` | 기존 CUDA venv의 Python 실행 파일 |
| `REMOTE_RUN_ROOT` | 실험별 디렉터리를 생성할 서버 경로 |
| `REMOTE_BOOTSTRAP_PYTHON` | 선택; torch 없이 제어 코드를 실행할 Python≥3.9, 기본 `python3` |
| `REMOTE_STORAGE_ROOT` | 선택; 외부 run/cache 경로를 쓸 때 기존 storage guard에 전달할 개인 quota 루트 |
| `REMOTE_QUOTA_GIB` | 선택; 실제 확인한 개인 quota만 지정 |

동일 이름의 환경변수가 JSON 값보다 우선한다. 다른 설정 파일은 `--config /path/to/private.json` 또는 `OIBC_TSMIXER_REMOTE_CONFIG`로 지정한다. 커스텀 설정도 Git 밖에 보관한다. 로컬 제어기는 stdlib만 사용하며 `OIBC_LOCAL_PYTHON`으로 Python≥3.9를 선택할 수 있다. 기존 학습과 서버 Python은 가능하면3.13을 유지한다.

필요한 도구: Mac은 Python·Git·OpenSSH·rsync, 서버는 Git·rsync·bash·NVIDIA 드라이버·기존 CUDA Python 환경이다. tmux는 필요하지 않다.

## 2. GitHub 전달과 최초 서버 점검

사용자가 직접 push한다. 스크립트는 push를 하지 않는다.

```bash
git push -u origin feature/tsmixer-v0-mac-cuda
bash scripts/tsmixer/remote_setup.sh
```

기본 setup은 SSH 연결, 원격 branch/commit, Python/CUDA, 원본 checksum, 기존 캐시를 검사한다. **fetch·설치·데이터 전송·캐시 생성·smoke·전체 학습을 자동 실행하지 않는다.** 환경 오류 시 `checks_passed=false`와 준비 안내를 출력하고 종료 코드2를 반환한다. 캐시나 commit이 맞지 않으면 `ready_to_launch=false`를 표시한다.

서버 checkout이 이전 commit이라면 아래 옵션으로 사용자가 명시적으로 갱신·검증한다.

```bash
bash scripts/tsmixer/remote_setup.sh --sync --tests --smoke
```

캐시도 없다면 생성까지 명시적으로 요청한다.

```bash
bash scripts/tsmixer/remote_setup.sh --sync --prepare-cache --tests --smoke
```

- `--sync`: Mac HEAD가 실제 origin branch에 push됐는지 확인한 뒤 서버를 fetch/fast-forward한다. dirty tracked 파일, branch 불일치, 분기된 history, 활성 원격 실험이나 기존 OIBC GPU lock이 있으면 중단한다. reset/clean/rebase/강제 덮어쓰기는 하지 않는다.
- `--tests`: 기존 TSMixer unit suite를 서버 CUDA 환경에서 실행한다.
- `--smoke`: 기존 CUDA synthetic5step, 캐시가 있으면 기존 실데이터5step도 실행한다. 결과는 고유 `setup_*` 디렉터리에 남긴다.
- `--prepare-cache`: 기존 `prepare_cache.sh`를 호출한다. 기존 캐시를 지우거나 덮어쓰지 않는다.

서버 Python이나 데이터가 없으면 사용자가 별도로 준비해야 한다. 기존 `analysis/tsmixer/README.md`의 `install_cuda.sh`, 원본8개 목록, 별도 안전 전송과 checksum 절차를 따른다. Python 설치·데이터 전송은 이 원격 도구에서 실행하지 않는다. 서버 checkout 자체가 없다면 새 경로에 clone하는 최초 준비도 사용자가 한다. 기존 checkout의 branch가 다르면 변경을 보존한 상태에서 직접 정리한다.

## 3. 매 실험: 학습 시작

환경 검증을 마친 뒤 Mac에서 push하고 다음 한 명령을 실행한다.

```bash
bash scripts/tsmixer/remote_launch.sh
```

또는 의미 있는 고유 ID를 직접 지정한다.

```bash
bash scripts/tsmixer/remote_launch.sh tsmixer_v0_trial001
```

기본 ID는 `tsmixer_UTC시각_랜덤8자리`다. 기존 ID는 재사용·덮어쓰기하지 않는다. ID에는 영문·숫자·underscore·hyphen만 허용한다.

실행 순서:

1. Mac tracked 파일이 깨끗하고 HEAD가 실제 push된 origin branch와 일치하는지 확인한다. 개인 미추적 파일은 보존한다.
2. 서버의 활성 실험·작업 트리·branch를 확인하고 `git fetch` + `merge --ff-only`로 같은 commit을 맞춘다. 서버 원본·CUDA·전체 historical train 캐시 준비를 검사한다. 캐시가 없으면 안내 후 실패하며 자동 생성하지 않는다.
3. 기본 `configs/tsmixer/base.yaml`, seed2026, 기존 development split을 확인한다. 새 실험 폴더에 설정·Git 정보·job/state JSON을 기록한다.
4. `start_new_session=True`, stdin 폐쇄, stdout/stderr 파일 연결을 사용한 detached worker를 시작한다. SSH가 끊겨도 학습 프로세스는 SSH stdio를 사용하지 않는다.
5. worker는 기존 `train_development.sh --full-train`을 호출한다. 학습 성공 시 **기존** `python -m analysis.tsmixer evaluate --device cuda`를 best.pt와 새 `eval_dev` 경로로 실행한다.
6. 종료 상태·종료 코드·`summary.md`를 기록한다. `completed`, `training_failed`, `evaluation_failed`, `worker_failed`, `launch_failed`를 구분한다.

학습 성공 후 평가 실패이면 `train_exit_code=0`, `evaluation_exit_code!=0`, `status=evaluation_failed`가 남는다. 학습 실패이면 평가하지 않는다. Development cutoff는 2025-01-31 17:00 KST, 평가 운영일은 [2025-02-01,2025-02-28) 그대로다. Holdout 평가·공식 Test 추론 명령은 호출하지 않는다.

한 번의 응답을 놓쳤다면 재실행 전에 `remote_status.sh`로 확인한다. 서버의 세션 정리 정책·재부팅·SIGKILL까지 detached 프로세스가 견디는 것은 아니므로 최초 실제 통합 검사에서 SSH 종료 후 지속 동작을 확인한다. 진행 중에는 다른 터미널에서도 checkout·학습 코드를 바꾸지 않는다.

## 4. 상태 확인

```bash
bash scripts/tsmixer/remote_status.sh
bash scripts/tsmixer/remote_status.sh tsmixer_v0_trial001
```

ID를 생략하면 가장 최근 실험을 조회한다. 출력은 JSON으로 ID·PID·실행 여부·단계·최근 epoch·train loss·dev 지표·GPU 사용률/VRAM·최근 로그20줄·종료 코드·최종 상태를 포함한다. 기존 v0에는 validation loss 로그가 없으므로 값을 만들지 않고 `validation_loss=null` 및 설명을 표시한다. 대신 기존 `development` CF/kWh MAE·정산 페널티를 보여준다.

PID 시작 시각도 함께 검사한다. worker가 사라졌지만 상태가 미완료면 `interrupted_or_lost`로 표시하고 종료 코드를 추정하지 않는다. worker가 사라져도 자식 process group이 살아 있으면 실행 중으로 보고 새 checkout을 차단한다. stop/kill/retry/delete 명령은 제공하지 않는다.

## 5. 결과 회수

```bash
bash scripts/tsmixer/remote_fetch.sh tsmixer_v0_trial001
```

종료한 실험만 회수한다. 실패한 종료 실험도 진단용으로 받을 수 있고, 실행 중·lost 상태는 거부한다. Mac 저장 위치:

```text
reports/tsmixer/remote_results/<experiment_id>/
├── config.yaml, job.json, git.json, launcher.json, state.json
├── environment.json, summary.md
├── train.log, evaluate.log, worker.log
├── train/{run.json,epochs.jsonl,summary.json}
├── eval_dev/{metrics.json,site_and_cohort_metrics.csv,group_metrics.csv}
├── checkpoint_manifest.json
└── transfer_manifest.json
```

실패 단계에 따라 아직 생성되지 않은 파일은 없다. checkpoint manifest에는 서버의 best.pt/last.pt 경로·크기·SHA-256만 기록하며 **checkpoint 자체를 복사하지 않는다.** predictions.npy·Parquet·원본·캐시는 allowlist에서 제외한다.

서버가 새 소형 export snapshot을 만든 후 rsync로 허용 파일만 전송한다. 파일별64MiB/전체128MiB 상한이 있으며, SHA-256·크기·허용 경로·중복 파일·symlink를 검증한다. Mac의 임시 `.ID.partial_*`에서 검증을 마친 결과만 최종 ID 디렉터리로 전환한다. rsync 실패·손상 시 비정상 종료하고 partial을 보존한다. 기존 결과 디렉터리는 덮어쓰지 않는다. 기본 설정 `.tsmixer-remote.json`과 `remote_results/`는 Git 제외다.

Mac의 구형 rsync와도 맞도록 `--protect-args`에 의존하지 않고 원격 source 경로를 shell-quote한다. 공백·작은따옴표·shell 특수문자의 전달을 오프라인 테스트했지만 실제 Mac/서버 rsync 조합은 최초 통합 검사 대상이다.

## 6. 검증과 실제 서버 통합 검사

오프라인 검사 — SSH를 호출하지 않는다.

```bash
python3 -B -m unittest discover -s scripts/tsmixer/tests -v
```

이 테스트는 SSH·rsync·GPU를 모의 처리한다. Git fast-forward/분기 거부는 임시 로컬 bare 저장소에서도 확인한다. 실제 GitHub push나 네트워크 연결은 없다. 기존 TSMixer unit suite는 별도로 유지한다.

실제 접속은 사용자가 설정을 완료한 뒤 위의 기본 setup → 명시적 sync/tests/smoke → launch → status → fetch 순서로 수행한다. 검증할 항목은 SSH alias·호스트 키·공개키 인증, bootstrap Python, GitHub 읽기 권한, 경로·quota, CUDA/VRAM, 전체 캐시, SSH 종료 후 지속 동작, 자동 dev 평가, rsync 전송과 SHA-256이다. 이번 개발 작업에서는 이 실제 서버 통합 검사를 실행하지 않았다.
