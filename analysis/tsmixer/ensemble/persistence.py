import json
import os
import time
from pathlib import Path
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.storage import result_guard
from analysis.tsmixer.comparison.io import file_sha


def atomic_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.pending')
    if tmp.exists():tmp.rename(tmp.with_name(tmp.name+f'.interrupted_{time.time_ns()}'))
    with tmp.open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def reserve(path, recipe, resume=False):
    path=Path(path).resolve(); marker=path/'recipe.json'
    result_guard(path,512*1024**2)
    if path.exists():
        if not resume: raise FileExistsError('Choose a fresh directory or explicit --resume')
        if not marker.is_file() or json.loads(marker.read_text())!=recipe: raise ValueError('Resume recipe/source/config/data differs')
    else:
        path.mkdir(parents=True,exist_ok=False);atomic_json(marker,recipe)
    return path


def complete(path, artifacts):
    path=Path(path)
    record={name:file_sha(path/name) for name in artifacts}
    atomic_json(path/'complete.json',dict(files=record))


def is_complete(path):
    path=Path(path);marker=path/'complete.json'
    if not marker.exists(): return False
    record=json.loads(marker.read_text())
    if not all((path/n).is_file() and file_sha(path/n)==h for n,h in record['files'].items()):
        raise ValueError('Completed artifact hash mismatch; do not retrain/overwrite corrupt results')
    return True


def sources():
    from analysis.tsmixer.train import source_hashes
    return source_hashes()
