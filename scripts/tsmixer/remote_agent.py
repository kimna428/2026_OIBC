"""Standard-library SSH endpoint. No model/training changes; no work on import.

The Mac streams this source through a quoted python -c command so setup also
works before the server has fetched this commit. Requests arrive as JSON stdin.
"""
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import uuid

REQUIRED = ('SSH_HOST', 'REMOTE_REPO_DIR', 'REMOTE_BRANCH', 'REMOTE_DATA_ROOT',
            'REMOTE_CACHE_ROOT', 'REMOTE_PYTHON', 'REMOTE_RUN_ROOT')
OPTIONAL = ('REMOTE_BOOTSTRAP_PYTHON', 'REMOTE_STORAGE_ROOT', 'REMOTE_QUOTA_GIB')
TERMINAL = {'completed', 'training_failed', 'evaluation_failed', 'worker_failed', 'launch_failed'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def experiment_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('Invalid experiment ID; use 1..80 ASCII letters/digits/underscore/hyphen')
    return value


def validate_config(c):
    missing = [k for k in REQUIRED if not isinstance(c.get(k), str) or not c[k]]
    if missing:
        raise ValueError('Missing SSH configuration: '+', '.join(missing))
    if set(c) - set(REQUIRED+OPTIONAL):
        raise ValueError('Unknown remote configuration keys')
    if any(not isinstance(v, str) or any(x in v for x in ('\0', '\r', '\n')) for v in c.values()):
        raise ValueError('Configuration must contain single-line strings')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@-]*', c['SSH_HOST']):
        raise ValueError('SSH_HOST must be an SSH config alias or user@host; no options/ports/shell syntax')
    branch = c['REMOTE_BRANCH']
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]*', branch) or '..' in branch or
            branch.endswith(('/', '.', '.lock')) or '//' in branch):
        raise ValueError('Invalid REMOTE_BRANCH')
    for k in ('REMOTE_REPO_DIR', 'REMOTE_DATA_ROOT', 'REMOTE_CACHE_ROOT', 'REMOTE_RUN_ROOT',
              'REMOTE_STORAGE_ROOT'):
        if k not in c:
            continue
        p = PurePosixPath(c[k])
        if not p.is_absolute() or str(p) == '/' or '..' in p.parts or str(p) != c[k]:
            raise ValueError(f'{k} must be a normalized absolute directory, not / or a parent traversal')
    run = PurePosixPath(c['REMOTE_RUN_ROOT'])
    repo = PurePosixPath(c['REMOTE_REPO_DIR'])
    if run == repo or run in repo.parents:
        raise ValueError('Run root must not equal or contain the repository')
    if repo in run.parents and repo/'runs' not in run.parents:
        raise ValueError('An in-repository run root must be below the Git-ignored runs/ directory')
    for k in ('REMOTE_DATA_ROOT', 'REMOTE_CACHE_ROOT'):
        other = PurePosixPath(c[k])
        if run == other or run in other.parents or other in run.parents:
            raise ValueError('Run root must be separate from data and cache directories')
    for k in ('REMOTE_PYTHON', 'REMOTE_BOOTSTRAP_PYTHON'):
        value = c.get(k, 'python3')
        if not value or value.startswith('-') or '..' in PurePosixPath(value).parts:
            raise ValueError(f'Invalid {k}')
    if 'REMOTE_QUOTA_GIB' in c:
        quota = float(c['REMOTE_QUOTA_GIB'])
        if not math.isfinite(quota) or quota <= 0:
            raise ValueError('REMOTE_QUOTA_GIB must be finite and positive')
    return c


def env_for(c):
    env = os.environ.copy()
    env.update(OIBC_PYTHON=c['REMOTE_PYTHON'], OIBC_DATA_ROOT=c['REMOTE_DATA_ROOT'],
               OIBC_TSMIXER_CACHE_ROOT=c['REMOTE_CACHE_ROOT'],
               OIBC_TSMIXER_RUN_ROOT=c['REMOTE_RUN_ROOT'], PYTHONDONTWRITEBYTECODE='1')
    if c.get('REMOTE_STORAGE_ROOT'):
        env['OIBC_STORAGE_ROOT'] = c['REMOTE_STORAGE_ROOT']
    if c.get('REMOTE_QUOTA_GIB'):
        env['OIBC_QUOTA_GIB'] = c['REMOTE_QUOTA_GIB']
    return env


def execute(c, args, check=True):
    p = subprocess.run(args, cwd=c['REMOTE_REPO_DIR'], env=env_for(c),
                       text=True, capture_output=True, check=False)
    result = {'command': args, 'returncode': p.returncode,
              'stdout': p.stdout[-24000:], 'stderr': p.stderr[-8000:]}
    if check and p.returncode:
        raise RuntimeError(f'Command failed ({p.returncode}): {args!r}\n{result["stderr"]}\n{result["stdout"]}')
    return result


def git(c, *args):
    return execute(c, ['git', *args])['stdout'].strip()


def metadata(c):
    return {'branch': git(c, 'branch', '--show-current'), 'commit': git(c, 'rev-parse', 'HEAD'),
            'tracked_changes': git(c, 'status', '--porcelain', '--untracked-files=no'),
            'work_tree': git(c, 'status', '--short')}


def safe_root(c, create=False):
    p = Path(c['REMOTE_RUN_ROOT'])
    if p.resolve() != p:
        raise ValueError('Run root must not contain symlinks')
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def job_path(c, eid):
    root = safe_root(c)
    p = root/experiment_id(eid)
    if p.resolve() != p:
        raise ValueError('Experiment path must not contain symlinks')
    return p


def atomic_json(file, value):
    file = Path(file)
    if file.is_symlink():
        raise ValueError('Refusing to replace a symlink')
    temp = file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(temp, file)


def read_json(file):
    if Path(file).is_symlink():
        raise ValueError('JSON state/config must not be a symlink')
    return json.loads(Path(file).read_text())


def job_state(dest):
    state = read_json(dest/'state.json')
    launcher = dest/'launcher.json'
    if not state.get('pid') and launcher.is_file():
        state.update({k: v for k, v in read_json(launcher).items() if k in ('pid', 'process_start')})
    return state


def process_identity(pid):
    try:
        # /proc stat comm can contain spaces and parentheses; split after last ).
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # field 22, starttime since boot
    except (FileNotFoundError, ProcessLookupError):
        return None


def live(state):
    pid = state.get('pid')
    if not isinstance(pid, int) or pid <= 1:
        return False
    current = process_identity(pid)
    if current and current == state.get('process_start'):
        return True
    # A dead worker can leave a still-running training child in its session.
    # Refuse a new checkout/fetch while that process group exists.
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def tail(file, count=20):
    p = Path(file)
    if not p.is_file() or p.is_symlink():
        return []
    with p.open('rb') as stream:
        stream.seek(max(0, p.stat().st_size-65536))
        return stream.read().decode('utf-8', errors='replace').splitlines()[-count:]


def status(c, eid=None):
    root = safe_root(c)
    if eid is None:
        jobs = sorted(root.glob('*/state.json'), key=lambda p: p.stat().st_mtime) if root.exists() else []
        if not jobs:
            return {'status': 'no_experiments', 'running': False}
        eid = jobs[-1].parent.name
    p = job_path(c, eid)
    state = job_state(p)
    active = live(state)
    effective = state['status']
    if not active and effective not in TERMINAL:
        effective = 'interrupted_or_lost'
    recent = tail(p/'train/epochs.jsonl', 1)
    try:
        epoch = json.loads(recent[0]) if recent else {}
    except json.JSONDecodeError:
        epoch = {'note': 'Last epoch line is still being written'}
    phase_log = 'evaluate.log' if state.get('phase') == 'evaluation' else 'train.log'
    try:
        gpu = execute(c, ['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader'], check=False)
    except OSError as exc:
        gpu = {'returncode': 127, 'error': str(exc)}
    return {**state, 'effective_status': effective, 'running': active,
            'recent_epoch': epoch.get('epoch'), 'train_loss': epoch.get('loss'),
            'validation_loss': None,
            'validation_loss_note': '기존 v0는 validation loss를 기록하지 않음; development 지표를 함께 표시',
            'development': epoch.get('development'), 'gpu': gpu,
            'recent_log': tail(p/phase_log), 'worker_log': tail(p/'worker.log', 5)}


def synchronize(c, expected):
    if not re.fullmatch(r'[0-9a-f]{40,64}', expected):
        raise ValueError('Invalid expected Git commit')
    info = metadata(c)
    if info['branch'] != c['REMOTE_BRANCH']:
        raise ValueError('Remote branch mismatch; switch branches manually after inspecting your work tree')
    if info['tracked_changes']:
        raise ValueError('Remote tracked work tree is dirty; no fetch/merge performed')
    # Respect an existing manually launched OIBC training process as well.
    # Do not create/truncate its lock, and do not update code while it is held.
    import fcntl
    gpu_lock = Path(os.environ.get('OIBC_GPU_LOCK', '/tmp/oibc-single-gpu.lock'))
    if gpu_lock.exists():
        with gpu_lock.open('rb') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('An existing OIBC GPU process holds its lock; Git update is blocked') from exc
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    git(c, 'fetch', '--no-tags', 'origin', c['REMOTE_BRANCH'])
    fetched = git(c, 'rev-parse', 'FETCH_HEAD')
    if fetched != expected:
        raise ValueError('Pushed commit mismatch: remote origin branch is not the Mac commit')
    # ff-only never resets/rebases/overwrites diverged history. Git also refuses
    # any untracked-file conflict. Existing untracked files are preserved.
    git(c, 'merge', '--ff-only', '--no-edit', expected)
    if git(c, 'rev-parse', 'HEAD') != expected:
        raise ValueError('Remote commit did not match after fast-forward')
    return metadata(c)


def preflight(c, require_cache=False):
    result = execute(c, ['bash', 'scripts/tsmixer/server_preflight.sh'])
    if require_cache:
        if not Path(c['REMOTE_CACHE_ROOT'], 'train/manifest.json').is_file():
            raise ValueError('Full train cache missing. Run remote_setup.sh --prepare-cache explicitly first')
        execute(c, [c['REMOTE_PYTHON'], '-m', 'analysis.tsmixer', 'check-data',
                    '--data-root', c['REMOTE_DATA_ROOT'], '--cache-root', c['REMOTE_CACHE_ROOT']])
        verify = ('import pandas as pd; from analysis.tsmixer.data import open_cache; '
                  'from analysis.deep_learning.features import operating_day; '
                  'import os; from pathlib import Path; '
                  'c=open_cache(Path(os.environ["OIBC_TSMIXER_CACHE_ROOT"])/"train"); '
                  'p=pd.read_csv(Path(os.environ["OIBC_DATA_ROOT"])/"train/smp.csv"); '
                  'assert c.days.equals(operating_day(pd.to_datetime(p.ts)).unique()), "Partial cache is not ready for full training"')
        execute(c, [c['REMOTE_PYTHON'], '-c', verify])
    return result


def setup(c, request):
    options = request.get('options', [])
    if set(options) - {'sync', 'tests', 'smoke', 'prepare-cache'}:
        raise ValueError('Unsupported setup mutation option')
    if 'sync' in options:
        with launch_lock(c) as root:
            if any(live(job_state(p.parent)) for p in root.glob('*/state.json')):
                raise ValueError('An experiment is active; Git synchronization is blocked')
            synchronize(c, request['expected_commit'])
        return {**setup(c, {**request, 'options': [x for x in options if x != 'sync']}),
                'git_sync_performed': True}
    info = metadata(c)
    result = {'git': info, 'mac_expected_commit': request.get('expected_commit'),
              'commit_matches': info['commit'] == request.get('expected_commit'),
              'branch_matches': info['branch'] == c['REMOTE_BRANCH'],
              'note': '기본 setup은 검사 전용. 설치·전송·캐시 생성·Git 갱신은 자동 실행하지 않습니다.'}
    # Existing environment only; no pip install, clone, data transfer or fetch.
    try:
        result['checks'] = preflight(c)
        result['checks_passed'] = True
    except (RuntimeError, OSError) as exc:
        result.update(checks_passed=False, check_error=str(exc))
        if request.get('options'):
            raise RuntimeError('Fix the existing Python/CUDA/data environment before optional setup actions: '+str(exc)) from exc
    result['cache_present'] = Path(c['REMOTE_CACHE_ROOT'], 'train/manifest.json').is_file()
    if options:
        if not result['commit_matches'] or not result['branch_matches'] or info['tracked_changes']:
            raise ValueError('Optional setup actions require the clean, pushed Mac commit on the configured branch')
        with launch_lock(c) as root:
            if any(live(job_state(p.parent)) for p in root.glob('*/state.json')):
                raise ValueError('An experiment is active; optional setup tests/cache actions are blocked')
            dest = root/('setup_'+uuid.uuid4().hex)
            dest.mkdir()
            result['setup_output'] = str(dest)
            for option in options:
                if option == 'prepare-cache':
                    args = ['bash', 'scripts/tsmixer/prepare_cache.sh']
                elif option == 'tests':
                    args = [c['REMOTE_PYTHON'], '-m', 'unittest', 'discover', '-s', 'analysis/tsmixer/tests', '-v']
                else:
                    args = ['bash', 'scripts/tsmixer/smoke.sh', '--device', 'cuda', '--steps', '5',
                            '--output', str(dest/'synthetic_smoke')]
                record = execute(c, args, check=False)
                atomic_json(dest/(option+'.json'), record)
                if record['returncode']:
                    raise RuntimeError(f'Setup {option} failed; inspect {dest}')
                if option == 'smoke' and result['cache_present']:
                    execute(c, ['bash', 'scripts/tsmixer/smoke.sh', '--device', 'cuda', '--steps', '5',
                                '--real-cache', '--output', str(dest/'real_smoke')])
                if option == 'prepare-cache':
                    result['cache_present'] = True
    result['ready_to_launch'] = bool(result['checks_passed'] and result['cache_present'] and
                                     result['commit_matches'] and result['branch_matches'])
    result['prepare_commands'] = [
        'Missing Python/CUDA: inspect nvidia-smi; explicitly run scripts/tsmixer/install_cuda.sh on the server with TORCH_VERSION and CUDA_WHEEL_INDEX.',
        'Missing data: transfer separately into a new directory and verify data_inventory.json; setup never uploads data.',
        'Missing cache: bash scripts/tsmixer/remote_setup.sh --prepare-cache',
        'Validation: bash scripts/tsmixer/remote_setup.sh --tests --smoke']
    result['next'] = '캐시가 없으면 --prepare-cache, 검증은 --tests --smoke. Python 설치는 기존 install_cuda.sh를 서버에서 명시적으로 실행.'
    return result


@contextlib.contextmanager
def launch_lock(c):
    import fcntl
    root = safe_root(c, create=True)
    lock = root/'.launch.lock'
    if lock.is_symlink():
        raise ValueError('Launch lock must not be a symlink')
    with lock.open('a') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another launch/setup action is in progress') from exc
        try:
            yield root
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def launch(c, request):
    eid = experiment_id(request['experiment_id'])
    with launch_lock(c) as root:
        for file in root.glob('*/state.json'):
            if live(job_state(file.parent)):
                raise ValueError('Another remote experiment is active; checkout/update is blocked')
        dest = job_path(c, eid)
        if dest.exists():
            raise FileExistsError('Experiment ID already exists; results will not be overwritten')
        info = synchronize(c, request['expected_commit'])
        preflight(c, require_cache=True)
        # Check the existing config via its existing parser; never alter it.
        verify = ('from analysis.tsmixer.config import load_config; '
                  'c=load_config("configs/tsmixer/base.yaml"); '
                  'assert c["training"]["seed"]==2026; '
                  'assert c["split"]=={"outer_fold":5,"dev_start":"2025-02-01","dev_end":"2025-02-28"}')
        execute(c, [c['REMOTE_PYTHON'], '-c', verify])
        dest.mkdir(exist_ok=False)
        state = {'experiment_id': eid, 'status': 'queued', 'phase': 'launch',
                 'created_at': now(), 'commit': info['commit'], 'train_exit_code': None,
                 'evaluation_exit_code': None, 'exit_code': None}
        atomic_json(dest/'job.json', {'config': c, 'experiment_id': eid, 'expected_commit': info['commit']})
        atomic_json(dest/'git.json', info)
        atomic_json(dest/'state.json', state)
        shutil.copyfile(Path(c['REMOTE_REPO_DIR'])/'configs/tsmixer/base.yaml', dest/'config.yaml')
        args = [c['REMOTE_PYTHON'], str(Path(c['REMOTE_REPO_DIR'])/'scripts/tsmixer/remote_worker.py'),
                str(dest/'job.json')]
        try:
            with (dest/'worker.log').open('xb') as log:
                child = subprocess.Popen(args, cwd=c['REMOTE_REPO_DIR'], env=env_for(c),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True)
            # Worker owns state.json; parent never races its status updates.
            atomic_json(dest/'launcher.json', {'pid': child.pid, 'process_start': process_identity(child.pid),
                                               'created_at': now(), 'detached': True})
        except OSError as exc:
            state.update(status='launch_failed', error=str(exc), exit_code=1, finished_at=now())
            atomic_json(dest/'state.json', state)
            raise
        return {'experiment_id': eid, 'status': 'submitted', 'pid': child.pid,
                'commit': info['commit'], 'output': str(dest),
                'note': 'Detached worker rechecks readiness, trains, then evaluates development only.'}


def sha256(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_summary(dest, state):
    metrics_file = dest/'eval_dev/metrics.json'
    metrics = read_json(metrics_file) if metrics_file.is_file() else {}
    model_file = dest/'train/run.json'
    run = read_json(model_file) if model_file.is_file() else {}
    text = (f'# TSMixer 원격 실험 {state["experiment_id"]}\n\n'
            f'- 상태: {state["status"]}\n- Git commit: `{state.get("commit")}`\n'
            f'- 학습 종료 코드: {state.get("train_exit_code")}\n'
            f'- 평가 종료 코드: {state.get("evaluation_exit_code")}\n\n'
            '## Development 성능\n\n```json\n'+json.dumps(metrics.get('metrics', {}), indent=2, ensure_ascii=False)+'\n```\n\n'
            '## 모델 설정\n\n```json\n'+json.dumps(run.get('config', {}).get('model', {}), indent=2)+'\n```\n\n'
            '기존 weighted L1, seed=2026, cutoff=2025-01-31 17:00 KST, '
            'dev 운영일 [2025-02-01,2025-02-28)를 유지했습니다. '
            'Holdout 평가와 공식 test 추론은 수행하지 않습니다. '
            '기존 v0는 validation loss를 기록하지 않으므로 dev CF/kWh MAE와 정산 지표를 확인하십시오. '
            '학습 성공과 평가 실패는 별도 상태로 기록합니다. '
            'checkpoint·예측 NumPy·Parquet·원본·캐시는 기본 회수 대상이 아닙니다.\n')
    (dest/'summary.md').write_text(text)


def bundle(c, eid):
    dest = job_path(c, eid)
    state = job_state(dest)
    if live(state) or state['status'] not in TERMINAL:
        raise ValueError('Fetch requires a finished experiment; running/lost jobs are not exported')
    export = dest/'exports'/uuid.uuid4().hex
    if export.resolve() != export:
        raise ValueError('Export directory must not contain symlinks')
    export.mkdir(parents=True, exist_ok=False)
    allowed = ['config.yaml', 'job.json', 'git.json', 'launcher.json', 'state.json', 'summary.md',
               'environment.json', 'worker.log', 'train.log', 'evaluate.log',
               'train/run.json', 'train/epochs.jsonl', 'train/summary.json',
               'eval_dev/metrics.json', 'eval_dev/site_and_cohort_metrics.csv', 'eval_dev/group_metrics.csv']
    checkpoints = []
    for name in ('best.pt', 'last.pt'):
        file = dest/'train'/name
        if file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(dest):
            checkpoints.append({'path': str(file), 'bytes': file.stat().st_size, 'sha256': sha256(file)})
    atomic_json(export/'checkpoint_manifest.json', {'copied': False, 'checkpoints': checkpoints})
    for name in allowed:
        src = dest/name
        if not src.exists():
            continue
        if not src.is_file() or src.is_symlink() or not src.resolve().is_relative_to(dest):
            raise ValueError('Unsafe export source: '+name)
        if src.stat().st_size > 64*1024**2:
            raise ValueError('Export text exceeds 64MiB; retrieve it separately after inspection')
        out = export/name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    files = [{'path': str(p.relative_to(export)), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(export.rglob('*')) if p.is_file()]
    if sum(x['bytes'] for x in files) > 128*1024**2:
        raise ValueError('Export exceeds 128MiB')
    manifest = {'experiment_id': eid, 'commit': state.get('commit'), 'status': state['status'],
                'artifact_root': str(export), 'files': files}
    atomic_json(export/'transfer_manifest.json', manifest)
    return manifest


def dispatch(request):
    if sys.version_info < (3, 9):
        raise ValueError('Remote bootstrap Python >=3.9 is required; configure REMOTE_BOOTSTRAP_PYTHON')
    c = validate_config(request['config'])
    if not Path(c['REMOTE_REPO_DIR'], '.git').exists():
        raise ValueError('Existing remote checkout not found; clone manually into a new directory first')
    op = request['operation']
    if op == 'setup':
        return setup(c, request)
    if op == 'launch':
        return launch(c, request)
    if op == 'status':
        return status(c, request.get('experiment_id'))
    if op == 'fetch':
        return bundle(c, request['experiment_id'])
    raise ValueError('Unknown remote operation')


if __name__ == '__main__':
    try:
        print(json.dumps({'ok': True, 'result': dispatch(json.load(sys.stdin))}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False))
        sys.exit(2)
