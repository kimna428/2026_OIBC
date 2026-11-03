"""Path-agnostic notebook utilities; no google.colab dependency in model code."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import uuid
import numpy as np
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.features import aware, day_hours
from analysis.deep_learning.prepare import REQUIRED
from .persistence import copy_verified, sha256, durable_json, flush_file


def stage_dataset(source_root, local_root, reference):
    """Copy only the eight required originals to local disk, preserving existing files."""
    source_root, local_root = Path(source_root).resolve(), Path(local_root).resolve()
    if source_root == local_root or source_root in local_root.parents or local_root in source_root.parents:
        raise ValueError('Drive source and local dataset must be separate directories')
    inventory = json.loads(Path(reference).read_text())['files']
    for name in REQUIRED:
        if not (source_root/name).is_file():
            raise FileNotFoundError(f'Drive data missing: {name}')
    local_root.mkdir(parents=True, exist_ok=True)
    needed = sum((source_root/name).stat().st_size for name in REQUIRED if not (local_root/name).exists())
    if shutil.disk_usage(local_root).free < needed+512*1024**2:
        raise OSError('Insufficient local disk for data copy and reserve')
    result = []
    for name in REQUIRED:
        expected = inventory[name]['sha256']
        target = local_root/name
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file() or sha256(target) != expected:
                raise ValueError(f'Existing local dataset differs; choose a fresh path: {target}')
        else:
            temporary = target.with_name(target.name+'.'+uuid.uuid4().hex+'.copying')
            copy_verified(source_root/name, temporary, expected)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            temporary.rename(target)
        result.append({'path': name, 'bytes': target.stat().st_size, 'sha256': expected})
    return result


def bind_session(backup_root, commit, config, mode):
    """Pin source/config on Drive before training; same run ID cannot change recipe."""
    if not re.fullmatch(r'[0-9a-f]{40,64}', commit):
        raise ValueError('A full Git commit is required')
    if mode not in ('smoke', 'development'):
        raise ValueError('Only smoke/development modes are supported')
    expected = {'format': 1, 'git_commit': commit, 'config_fingerprint': fingerprint(config), 'mode': mode}
    path = Path(backup_root)/'session_ref.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Session reference cannot be a symlink')
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('This Drive run ID belongs to another commit/config/mode; keep its recipe or use a new ID')
    else:
        with path.open('x') as stream:
            json.dump(expected, stream, indent=2); stream.write('\n')
            flush_file(stream)
    return expected


def run_logged(argv, logfile):
    """Stream a subprocess into the notebook and an append-only local text log."""
    logfile = Path(logfile)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as stream:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            code = proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            raise
        finally:
            proc.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return code


def export_development(evaluation_root, cache, plan):
    """CSV from existing postprocessed dev predictions; no refit/test/holdout inference.

    These are selected development predictions, not independent OOF predictions.
    Whole-day canonical site order and existing G1/G2 composition are preserved.
    """
    root = Path(evaluation_root)
    metrics = json.loads((root/'metrics.json').read_text())
    if (metrics.get('split') != 'dev' or metrics.get('holdout_used') is not False or
            metrics['training_cutoff'] != plan['cutoff'] or cache.manifest['partition'] != 'train' or
            metrics.get('cache_signature') != cache.manifest['signature']):
        raise ValueError('Only matching development predictions can be exported')
    days = np.flatnonzero((cache.days >= aware(plan['dev_start'])) & (cache.days < aware(plan['dev_end'])))
    grid = np.load(root/'predictions.npy', mmap_mode='r', allow_pickle=False)
    if grid.shape != (len(days), cache.num_sites, 24) or not len(days):
        raise ValueError('Canonical development prediction shape mismatch')
    dest = root/'csv'
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('CSV export already exists; do not overwrite it')
    stage = root/('.csv_'+uuid.uuid4().hex)
    stage.mkdir()
    sites = cache.sites.site_id.to_numpy()
    capacity = cache.sites.capacity_kw.to_numpy()
    group = cache.composition.set_index('site_id').group.reindex(sites).to_numpy()
    with (stage/'development_site_predictions.csv').open('x', newline='') as sf, \
         (stage/'development_group_forecast.csv').open('x', newline='') as gf:
        sw, gw = csv.writer(sf), csv.writer(gf)
        sw.writerow(['site_id', 'ts', 'pred_cf', 'pred_kwh'])
        gw.writerow(['group', 'ts', 'forecast_kwh'])
        for i, d in enumerate(days):
            pred = np.asarray(grid[i])
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError('Nonfinite/negative exported forecast')
            times = [x.isoformat() for x in day_hours(cache.days[d])]
            energy = pred*capacity[:, None]
            for j, site in enumerate(sites):
                sw.writerows((site, times[h], float(pred[j,h]), float(energy[j,h])) for h in range(24))
            for label in ('G1', 'G2'):
                values = energy[group == label].sum(axis=0)
                gw.writerows((label, times[h], float(values[h])) for h in range(24))
    cache.composition.to_csv(stage/'composition.csv', index=False)
    durable_json(stage/'metadata.json', {'split': 'dev', 'independent_oof': False, 'test_submission': False,
                                        'site_rows': len(days)*cache.num_sites*24,
                                        'group_rows': len(days)*2*24,
                                        'training_cutoff': plan['cutoff'],
                                        'cache_signature': cache.manifest['signature'],
                                        'note': 'Development predictions selected on this dev window; not unbiased OOF'})
    stage.rename(dest)
    return dest
