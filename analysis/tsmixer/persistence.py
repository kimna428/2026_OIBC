"""Portable, verified generation backups; no Drive/Colab paths or API imports.

A generation becomes recoverable only after its manifest and COMMITTED marker
are complete and every file verifies. Previous generations are never deleted.
Drive/FUSE persistence is still subject to the storage provider's guarantees.
"""
import hashlib
import errno
import json
import os
import pickle
from pathlib import Path, PurePosixPath
import shutil
import time
import uuid

TOP_FILES = {'last.pt', 'best.pt', 'run.json', 'epochs.jsonl', 'summary.json',
             'console.log', 'evaluation_console.log', 'config.yaml', 'environment_freeze.txt'}
EVAL_FILES = {'metrics.json', 'predictions.npy', 'site_and_cohort_metrics.csv',
              'group_metrics.csv', 'settlement_ledger.parquet'}
CSV_FILES = {'development_site_predictions.csv', 'development_group_forecast.csv',
             'composition.csv', 'metadata.json'}


def sha256(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def allowed(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        return False
    return (name in TOP_FILES or
            len(p.parts) == 2 and p.parts[0] == 'eval_dev' and p.name in EVAL_FILES or
            len(p.parts) == 3 and p.parts[:2] == ('eval_dev', 'csv') and p.name in CSV_FILES)


def regular(root, name):
    file = root/name
    if not allowed(name) or file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe backup file: {name}')
    return file


def durable_json(file, data):
    temp = file.with_name(file.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n'); flush_file(stream)
    os.replace(temp, file)


def flush_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as exc:
        # Some FUSE providers do not implement fsync. Always retain read-back
        # checksums/commit markers; genuine I/O failures must still stop backup.
        if exc.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def copy_verified(src, dst, expected=None):
    """Stream-copy into a new destination and verify by rereading both files."""
    if src.is_symlink() or not src.is_file():
        raise ValueError(f'Not a regular source: {src}')
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f'Refusing to overwrite {dst}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(src)
    if expected is not None and digest != expected:
        raise ValueError(f'Source checksum mismatch: {src}')
    with src.open('rb') as source, dst.open('xb') as target:
        shutil.copyfileobj(source, target, length=1024*1024)
        flush_file(target)
    if sha256(dst) != digest or sha256(src) != digest:
        raise OSError(f'Copy verification failed: {dst}')
    return {'bytes': dst.stat().st_size, 'sha256': digest}


def valid_checkpoint(file):
    from .checkpoint import load
    state = load(file)
    required = {'model', 'optimizer', 'scheduler', 'scaler', 'epoch', 'global_step',
                'best', 'rng', 'loader_rng', 'normalizer', 'config', 'run_signature',
                'cache_signature', 'feature_signature', 'sources'}
    if not required <= set(state) or state['epoch'] < 1 or state['global_step'] < 1:
        raise ValueError('Checkpoint lacks complete epoch/resume state')
    if state['scheduler'] is None:
        raise ValueError('Scheduler state is required for recoverable snapshots')
    return state


def backup_run(run_root, backup_root):
    run_root, backup_root = Path(run_root).resolve(), Path(backup_root).resolve()
    if (run_root == backup_root or run_root in backup_root.parents or backup_root in run_root.parents):
        raise ValueError('Backup and local run must be separate, non-nested directories')
    state = valid_checkpoint(run_root/'last.pt')
    best_state = valid_checkpoint(run_root/'best.pt')
    if (best_state['run_signature'] != state['run_signature'] or best_state['epoch'] > state['epoch'] or
            best_state['best'] != state['best']):
        raise ValueError('Best/last checkpoint recipe or selection state differs')
    generation = f'epoch_{state["epoch"]:04d}_step_{state["global_step"]:09d}_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    target = backup_root/'snapshots'/generation
    target.mkdir(parents=True, exist_ok=False)
    names = sorted(str(p.relative_to(run_root)) for p in run_root.rglob('*')
                   if p.is_file() and allowed(str(p.relative_to(run_root))))
    if not {'last.pt', 'best.pt', 'run.json', 'epochs.jsonl'} <= set(names):
        raise ValueError('Incomplete local run; last/best/run/epoch log are required')
    records = []
    for name in names:
        records.append({'path': name, **copy_verified(regular(run_root, name), target/name)})
    # Readability of the copied checkpoint matters in addition to byte integrity.
    copied = valid_checkpoint(target/'last.pt')
    if copied['run_signature'] != state['run_signature']:
        raise ValueError('Checkpoint changed during backup')
    manifest = {'format': 1, 'created_ns': time.time_ns(), 'epoch': state['epoch'],
                'global_step': state['global_step'], 'run_signature': state['run_signature'],
                'config_fingerprint': state.get('config_fingerprint'),
                'git_commit': state.get('environment', {}).get('git_commit'), 'files': records}
    durable_json(target/'manifest.json', manifest)
    # Commit marker last: a half-written/copied generation is never selected.
    durable_json(target/'COMMITTED.json', {'manifest_sha256': sha256(target/'manifest.json')})
    verify_generation(target)
    return target


def verify_generation(root):
    if (root/'COMMITTED.json').is_symlink() or (root/'manifest.json').is_symlink():
        raise ValueError('Backup markers must not be symlinks')
    marker = json.loads((root/'COMMITTED.json').read_text())
    if sha256(root/'manifest.json') != marker['manifest_sha256']:
        raise ValueError('Backup manifest checksum mismatch')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['format'] != 1:
        raise ValueError('Unsupported backup format')
    names = set()
    for item in manifest['files']:
        name = item['path']
        if name in names:
            raise ValueError('Duplicate backup file')
        names.add(name)
        file = regular(root, name)
        if file.stat().st_size != item['bytes'] or sha256(file) != item['sha256']:
            raise ValueError(f'Backup checksum mismatch: {name}')
    if not {'last.pt', 'best.pt', 'run.json', 'epochs.jsonl'} <= names:
        raise ValueError('Backup missing required resume files')
    state = valid_checkpoint(root/'last.pt')
    best = valid_checkpoint(root/'best.pt')
    if best['run_signature'] != state['run_signature'] or best['best'] != state['best']:
        raise ValueError('Best checkpoint is inconsistent with the resume state')
    if (state['run_signature'] != manifest['run_signature'] or state['epoch'] != manifest['epoch'] or
            state['global_step'] != manifest['global_step']):
        raise ValueError('Backup manifest/checkpoint state mismatch')
    return manifest


def restore_latest(backup_root, run_root):
    """Return selected generation or None if no backup has ever been started.

    Invalid/incomplete newer generations are ignored. If all candidates are bad,
    fail rather than silently starting a fresh run. Never overwrite a local run.
    """
    backup_root, run_root = Path(backup_root).resolve(), Path(run_root).absolute()
    if run_root.resolve() != run_root or run_root.exists():
        raise FileExistsError('Restore requires a fresh local run directory without symlinks')
    snapshots = backup_root/'snapshots'
    candidates = list(snapshots.iterdir()) if snapshots.is_dir() else []
    if not candidates:
        return None
    verified = []
    for folder in candidates:
        try:
            if folder.is_symlink() or not folder.is_dir():
                continue
            manifest = verify_generation(folder)
            verified.append((manifest['epoch'], manifest['global_step'], manifest['created_ns'], folder, manifest))
        except (OSError, ValueError, KeyError, TypeError, EOFError, RuntimeError, pickle.UnpicklingError):
            continue
    if not verified:
        raise RuntimeError('No fully verified backup generation; inspect Drive without deleting older snapshots')
    for _, _, _, folder, manifest in sorted(verified, key=lambda x: x[:3], reverse=True):
        stage = run_root.parent/('.restore_'+uuid.uuid4().hex)
        stage.mkdir(parents=True, exist_ok=False)
        try:
            for item in manifest['files']:
                copy_verified(regular(folder, item['path']), stage/item['path'], item['sha256'])
            valid_checkpoint(stage/'last.pt')
            if run_root.exists():
                raise FileExistsError('Local run appeared during restore')
            stage.rename(run_root)
            return folder
        except FileExistsError:
            raise
        except (OSError, ValueError, EOFError, RuntimeError, pickle.UnpicklingError):
            # Own partial restore is preserved for diagnosis; try an older valid snapshot.
            continue
    raise RuntimeError('All verified snapshots failed during restore; partial copies preserved')
