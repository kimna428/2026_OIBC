"""Config paths are workspace-relative or explicitly supplied environment variables."""
from pathlib import Path
import copy, hashlib, json, os, re
import yaml
ROOT=Path(__file__).resolve().parents[2]

def expand(value):
    if isinstance(value,dict): return {k:expand(v) for k,v in value.items()}
    if isinstance(value,list): return [expand(v) for v in value]
    if isinstance(value,str):
        def replace(m):
            result=os.environ.get(m[1]) or m[2]
            if result is None: raise ValueError(f'Missing environment variable {m[1]}')
            return result
        return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)(?::-([^}]*))?\}',replace,value)
    return value

def path(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p

def fingerprint(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def load_config(filename):
    with path(filename).open() as f: c=expand(yaml.safe_load(f))
    for key in ['name','model','data','split','loss','training','group']:
        if key not in c: raise ValueError(f'Missing config section {key}')
    if c['model']['name'] not in ['cnn','transformer','cross_site']: raise ValueError('Unknown model')
    if not 0<=c['split']['outer_fold']<=5: raise ValueError('outer_fold must be 0..5')
    if c['training']['batch_size']<1 or c['training']['accumulation_steps']<1: raise ValueError('Invalid batch/accumulation')
    if c['group']['enabled'] and c['model']['name']!='cross_site': raise ValueError('Group loss needs full-fleet batches')
    if c['loss']['name'] not in ['huber','l1','weighted_l1']: raise ValueError('Unknown loss')
    if c['group']['alpha']<0 or c['group']['beta']<0 or c['group']['alpha']+c['group']['beta']==0: raise ValueError('Loss alpha/beta must be nonnegative with nonzero total')
    if not c['group']['enabled'] and c['group']['beta']!=0: raise ValueError('Disabled group loss must have beta=0')
    return copy.deepcopy(c)
