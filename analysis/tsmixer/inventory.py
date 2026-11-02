"""Bounded metadata inspection and optional streaming checksum validation."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from analysis.deep_learning.prepare import REQUIRED, inspect_inputs
from .data import open_cache


def sha256(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_data(root, cache_root=None, checksums=False, reference=None):
    root = Path(root)
    inspect_inputs(root)
    result = {'files': {}, 'root': str(root), 'read_only': True}
    files = list(REQUIRED) + ['train/observed_wa.parquet', 'train/observed_wb.parquet']
    for name in files:
        file = root/name
        if not file.is_file():
            if name in REQUIRED:
                raise FileNotFoundError(file)
            result['files'][name] = {'optional': True, 'present': False}
            continue
        item = {'bytes': file.stat().st_size, 'present': True, 'required': name in REQUIRED}
        if file.suffix == '.parquet':
            pf = pq.ParquetFile(file)
            item.update(rows=pf.metadata.num_rows,
                        schema={f.name: str(f.type) for f in pf.schema_arrow})
        else:
            d = pd.read_csv(file)
            item.update(rows=len(d), schema=d.dtypes.astype(str).to_dict())
        if checksums:
            item['sha256'] = sha256(file)
        result['files'][name] = item
    if reference:
        old = json.loads(Path(reference).read_text())['files']
        for name, expected in old.items():
            if not expected.get('present') or not expected.get('required'):
                continue
            actual = result['files'][name]
            for key in ('bytes', 'rows', 'sha256'):
                if key in expected and actual.get(key) != expected[key]:
                    raise ValueError(f'Data inventory mismatch: {name} {key}; --checksums may be required')
            # CSV string dtype labels can vary across pandas versions; compare names.
            if set(actual['schema']) != set(expected['schema']):
                raise ValueError(f'Data columns differ: {name}')
            if name.endswith('.parquet') and actual['schema'] != expected['schema']:
                raise ValueError(f'Parquet schema differs: {name}')
        result['reference_verified'] = str(reference)
    if cache_root:
        path = Path(cache_root)/'train'
        if not (path/'manifest.json').exists():
            raise FileNotFoundError(f'No compatible train cache at {path}; prepare on Linux CUDA server')
        cache = open_cache(path)
        if cache.manifest['kind'] != 'provided':
            raise ValueError('Server data check requires a provided-data cache')
        # Bind cache to actual input bytes, not just a self-consistent manifest.
        for name, expected in cache.manifest['input_hashes'].items():
            if name not in REQUIRED or sha256(root/name) != expected:
                raise ValueError(f'Cache input hash mismatch: {name}')
        result['cache'] = {k: cache.manifest.get(k) for k in
                           ('signature', 'feature_signature', 'layout', 'start', 'end_exclusive', 'shape')}
    return result
