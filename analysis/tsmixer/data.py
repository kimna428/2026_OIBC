"""Existing read-only cache/data contracts, with bounded day views for smoke tests."""
import numpy as np
from analysis.deep_learning.dataset import DayCache, DayDataset
from analysis.deep_learning.features import contract, STATIC, TIME, aware
from analysis.deep_learning.inference import dimensions


def open_cache(root):
    c = DayCache(root)
    if (c.manifest['weather_features'] != contract()['weather_features'] or
            c.manifest['temporal_features'] != TIME or c.manifest['static_features'] != STATIC):
        raise ValueError('Cache feature order differs from the existing CNN/Transformer contract')
    if dimensions(c) != dict(weather_dim=90, temporal_dim=8, static_dim=146, num_sites=253):
        raise ValueError('v0 requires the existing 90/8/146/253 input contract')
    return c


class DayArrayView:
    def __init__(self, array, indices):
        self.array, self.indices = array, np.asarray(indices)
        self.shape = (len(indices),) + array.shape[1:]

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        return self.array[(int(self.indices[keys[0]]),) + keys[1:]]


class SmokeCache:
    """Index view of <=2 historical days; never copies the full cache.

    Original signatures remain intact; callers MUST tag outputs as smoke-only.
    No smoke checkpoint may be promoted to development training/evaluation.
    """
    def __init__(self, cache, cutoff, max_days=2):
        import pandas as pd
        eligible = np.flatnonzero(cache.days + pd.Timedelta(hours=1) <= aware(cutoff))
        indices = eligible[-max_days:]
        if not len(indices):
            raise ValueError('No historical days before the training cutoff')
        self.root, self.manifest = cache.root, cache.manifest
        self.sites, self.composition = cache.sites, cache.composition
        self.num_sites, self.days = cache.num_sites, cache.days[indices]
        self.static = cache.static
        self.source_day_indices = indices.tolist()
        for name in ('weather', 'temporal', 'target', 'energy', 'missing_kind', 'smp', 'k'):
            setattr(self, name, DayArrayView(getattr(cache, name), indices))
