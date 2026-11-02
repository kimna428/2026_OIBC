"""Mac standard-library controller. Network calls happen only in explicit main()."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import uuid

if __package__:
    from . import remote_agent as agent
else:
    import remote_agent as agent

ROOT = Path(__file__).resolve().parents[2]
SSH_OPTIONS = ['-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']


def config(file=None):
    path = Path(file or os.environ.get('OIBC_TSMIXER_REMOTE_CONFIG', ROOT/'.tsmixer-remote.json'))
    value = json.loads(path.read_text()) if path.exists() else {}
    for key in agent.REQUIRED + agent.OPTIONAL:
        if key in os.environ:
            value[key] = os.environ[key]
    return agent.validate_config(value)


def local_git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError('Local Git failed: '+p.stderr.strip())
    return p.stdout.strip()


def pushed_commit(c):
    if local_git('branch', '--show-current') != c['REMOTE_BRANCH']:
        raise ValueError('Mac branch differs from REMOTE_BRANCH')
    if local_git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Commit tracked Mac changes before remote launch')
    head = local_git('rev-parse', 'HEAD')
    pushed = local_git('ls-remote', '--exit-code', 'origin', 'refs/heads/'+c['REMOTE_BRANCH'])
    if not pushed or pushed.split()[0] != head:
        raise ValueError('Mac HEAD is not the pushed origin branch commit; push explicitly first')
    for file in ('remote_control.py', 'remote_agent.py', 'remote_worker.py'):
        relative = 'scripts/tsmixer/'+file
        committed = local_git('rev-parse', head+':'+relative)
        if local_git('hash-object', relative) != committed:
            raise ValueError('Commit and push the remote controller sources before launch: '+relative)
    return head


def request(c, operation, **kwargs):
    # No shell=True, eval, interpolation of configuration into Python source,
    # remote bash source files or SSH options. Host uses a strict alias grammar.
    source = (Path(__file__).with_name('remote_agent.py')).read_text()
    command = shlex.join([c.get('REMOTE_BOOTSTRAP_PYTHON', 'python3'), '-c', source])
    payload = {'config': c, 'operation': operation, **kwargs}
    p = subprocess.run(['ssh', *SSH_OPTIONS, '--', c['SSH_HOST'], command],
                       input=json.dumps(payload), text=True, capture_output=True, check=False)
    try:
        response = json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'SSH/protocol failed ({p.returncode}): {p.stderr[-4000:]}') from exc
    if p.returncode or not response.get('ok'):
        raise RuntimeError(response.get('error') or p.stderr or 'Remote operation failed')
    return response['result']


def validate_manifest(c, eid, manifest):
    if manifest.get('experiment_id') != eid:
        raise ValueError('Result experiment ID mismatch')
    root = PurePosixPath(manifest['artifact_root'])
    expected = PurePosixPath(c['REMOTE_RUN_ROOT'])/eid/'exports'
    if (root.parent != expected or not re.fullmatch(r'[0-9a-f]{32}', root.name) or
            '..' in root.parts):
        raise ValueError('Unsafe remote result path')
    allowed = {'config.yaml', 'job.json', 'git.json', 'launcher.json', 'state.json', 'summary.md',
               'environment.json', 'worker.log', 'train.log', 'evaluate.log',
               'train/run.json', 'train/epochs.jsonl', 'train/summary.json',
               'eval_dev/metrics.json', 'eval_dev/site_and_cohort_metrics.csv',
               'eval_dev/group_metrics.csv', 'checkpoint_manifest.json'}
    seen = set()
    files = manifest.get('files', [])
    if not files or len(files) > len(allowed):
        raise ValueError('Invalid result file count')
    for item in files:
        name = item['path']
        if name not in allowed or name in seen:
            raise ValueError('Unexpected/duplicate artifact; checkpoints/raw data cannot be fetched')
        seen.add(name)
        if not isinstance(item['bytes'], int) or not 0 <= item['bytes'] <= 64*1024**2:
            raise ValueError('Invalid artifact size')
        if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid artifact SHA-256')
    if sum(x['bytes'] for x in files) > 128*1024**2:
        raise ValueError('Result export exceeds 128MiB')
    return files


def fetch(c, eid, result_root=None):
    eid = agent.experiment_id(eid)
    result_root = Path(result_root or ROOT/'reports/tsmixer/remote_results')
    if result_root.resolve() != result_root.absolute():
        raise ValueError('Local result root must not contain symlinks')
    result_root.mkdir(parents=True, exist_ok=True)
    dest = result_root/eid
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Local experiment result already exists; no overwrite')
    manifest = request(c, 'fetch', experiment_id=eid)
    files = validate_manifest(c, eid, manifest)
    stage = Path(tempfile.mkdtemp(prefix='.'+eid+'.partial_', dir=result_root))
    filelist = stage/'.filelist'
    filelist.write_text(''.join(x['path']+'\n' for x in files))
    # Legacy macOS rsync lacks --protect-args. Quote the remote path for the
    # remote shell; host is validated separately and never contains shell syntax.
    source = c['SSH_HOST']+':'+shlex.quote(manifest['artifact_root']+'/')
    args = ['rsync', '-rt', '--checksum', '--files-from='+str(filelist),
            '-e', shlex.join(['ssh', *SSH_OPTIONS]), '--', source, str(stage)+'/']
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError(f'rsync failed ({p.returncode}); partial files preserved at {stage}: {p.stderr[-4000:]}')
    for item in files:
        file = stage/item['path']
        if (file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(stage) or
                file.stat().st_size != item['bytes'] or agent.sha256(file) != item['sha256']):
            raise ValueError(f'Integrity check failed: {item["path"]}; partial files kept at {stage}')
    expected = {x['path'] for x in files} | {'.filelist'}
    if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file() or p.is_symlink()} != expected:
        raise ValueError('Unexpected local export contents; partial files preserved')
    filelist.unlink()  # our own temporary file only, never a user result
    agent.atomic_json(stage/'transfer_manifest.json', {**manifest, 'verified_at': agent.now()})
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Local result appeared during transfer; partial files preserved')
    stage.rename(dest)
    return {'experiment_id': eid, 'verified_files': len(files), 'output': str(dest),
            'status': manifest['status'], 'commit': manifest.get('commit')}


def main(argv=None):
    p = argparse.ArgumentParser(description='Explicit SSH remote control; never connects on import/help')
    p.add_argument('operation', choices=['setup', 'launch', 'status', 'fetch'])
    p.add_argument('experiment_id', nargs='?')
    p.add_argument('--config', help='Local JSON settings (never sourced as shell)')
    p.add_argument('--tests', action='store_true', help='setup only: explicitly run unit tests')
    p.add_argument('--sync', action='store_true', help='setup only: explicitly fetch/fast-forward the pushed commit')
    p.add_argument('--smoke', action='store_true', help='setup only: explicitly run CUDA smoke')
    p.add_argument('--prepare-cache', action='store_true', help='setup only: explicitly generate server cache')
    args = p.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f'Run from project root: {ROOT}')
    if args.operation != 'setup' and (args.sync or args.tests or args.smoke or args.prepare_cache):
        raise ValueError('Setup mutation options are only accepted by remote_setup.sh')
    c = config(args.config)
    if args.operation == 'launch':
        eid = args.experiment_id or ('tsmixer_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
        eid = agent.experiment_id(eid)
        head = pushed_commit(c)
        result = request(c, 'launch', expected_commit=head, experiment_id=eid)
    elif args.operation == 'setup':
        options = [name for name, flag in [('sync', args.sync), ('prepare-cache', args.prepare_cache),
                    ('tests', args.tests), ('smoke', args.smoke)] if flag]
        head = pushed_commit(c) if args.sync else local_git('rev-parse', 'HEAD')
        result = request(c, 'setup', expected_commit=head, options=options)
    elif args.operation == 'fetch':
        if not args.experiment_id:
            raise ValueError('Usage: remote_fetch.sh <experiment_id>')
        result = fetch(c, args.experiment_id)
    else:
        if args.experiment_id:
            agent.experiment_id(args.experiment_id)
        result = request(c, 'status', experiment_id=args.experiment_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if args.operation == 'setup' and not result.get('checks_passed') else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f'TSMixer remote error: {exc}', file=sys.stderr)
        sys.exit(2)
