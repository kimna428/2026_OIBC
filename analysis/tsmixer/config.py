"""Independent strict v0 configuration; existing deep-learning registry stays untouched."""
import copy
import inspect
import math
from pathlib import Path
import yaml
from analysis.deep_learning.config import ROOT, expand, fingerprint
from .model import TSMixer


def require_root():
    if Path.cwd().resolve() != ROOT.resolve():
        raise ValueError(f'Run from project root: cd {ROOT}')


def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def validate(c):
    sections = {'name', 'model', 'data', 'split', 'loss', 'group', 'training'}
    if set(c) != sections:
        raise ValueError(f'Configuration sections must be {sorted(sections)}')
    if set(c['model']) - (set(inspect.signature(TSMixer).parameters) -
                          {'weather_dim', 'temporal_dim', 'static_dim', 'num_sites'}):
        raise ValueError('Unknown TSMixer architecture setting')
    TSMixer(**c['model'])  # validates values; caller seeds AFTER config parsing
    if c['loss'] != {'name': 'weighted_l1', 'huber_delta': 0.05}:
        raise ValueError('v0 preserves the existing weighted_l1 loss contract')
    if c['group'] != {'enabled': False, 'alpha': 1.0, 'beta': 0.0}:
        raise ValueError('v0 is site-only; group training is reserved for a later experiment')
    t = c['training']
    for k in ('batch_size', 'accumulation_steps', 'epochs', 'patience', 'cpu_threads'):
        if isinstance(t[k], bool) or not isinstance(t[k], int) or t[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if not isinstance(t['num_workers'], int) or t['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    for k in ('learning_rate', 'gradient_clip'):
        if not math.isfinite(t[k]) or t[k] <= 0:
            raise ValueError(f'{k} must be finite and positive')
    for k in ('weight_decay', 'min_delta'):
        if not math.isfinite(t[k]) or t[k] < 0:
            raise ValueError(f'{k} must be finite and nonnegative')
    if t['optimizer'] != 'AdamW' or t['device'] not in ('auto', 'cpu', 'mps', 'cuda'):
        raise ValueError('Only AdamW and auto/cpu/mps/cuda devices are supported')
    if t['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32/fp16/bf16')
    if t['selection_metric'] not in ('penalty_per_hour', 'site_cf_mae', 'site_kwh_mae'):
        raise ValueError('Unsupported selection metric')
    if t['postprocess'] not in ('none', 'night_only', 'train_calibrated'):
        raise ValueError('Unsupported postprocess policy')
    if not 0 <= c['split']['outer_fold'] <= 5:
        raise ValueError('outer_fold must be 0..5')
    return c


def load_config(file, overrides=()):
    c = expand(yaml.safe_load(resolve(file).read_text()))
    for item in overrides:
        key, sep, value = item.partition('=')
        parts = key.split('.')
        if not sep or len(parts) != 2 or parts[0] not in c or parts[1] not in c[parts[0]]:
            raise ValueError(f'Use --set existing.section_key=value, got {item}')
        c[parts[0]][parts[1]] = yaml.safe_load(value)
    return validate(copy.deepcopy(c))
