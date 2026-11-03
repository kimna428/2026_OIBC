import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from analysis.tsmixer.year_backtest import (
    plan, PriceFreeCache, compositions, composition_hash, score, metadata_composition)
from analysis.deep_learning.features import aware, day_hours, issuance, fit_normalizer
from analysis.deep_learning.smoke import synthetic_cache
from analysis.tsmixer.data import open_cache, DayDataset
from analysis.validation import hourly_weights, validate_composition


class YearBacktestTests(unittest.TestCase):
    def test_boundaries_and_clock(self):
        for phase, cutoff, start, end, days in [
            ('selection','2024-03-31 17:00','2024-04-01','2024-06-01',61),
            ('refit','2024-05-31 17:00','2024-06-01','2025-06-01',365)]:
            p = plan(phase)
            self.assertEqual(aware(p['cutoff']), aware(cutoff))
            self.assertEqual(aware(p['cutoff']), issuance(day_hours(start))[0])
            self.assertEqual((aware(p['dev_end'])-aware(p['dev_start'])).days, days)
            self.assertEqual(day_hours(aware(end)-pd.Timedelta(days=1))[-1], aware(end))
        with self.assertRaises(ValueError):
            plan('development')

    def test_price_blindness_and_future_normalizer_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(Path(tmp))
            c = open_cache(Path(tmp)/'train')
            cutoff = str(c.days[1]+pd.Timedelta(hours=17))
            free = PriceFreeCache(c)
            self.assertTrue((free.smp == 1).all())
            before = fit_normalizer(free, cutoff)
            # Make every later day inaccessible to normalizer/night calibration.
            class Guarded:
                def __init__(self, array): self.array = array; self.shape = array.shape
                def __getitem__(self, key):
                    d = key[0] if isinstance(key, tuple) else key
                    if d > 1: raise AssertionError('future data read')
                    return self.array[key]
            for name in ('weather','temporal','target'):
                setattr(free, name, Guarded(getattr(c, name)))
            self.assertEqual(before, fit_normalizer(free, cutoff))
            blind = PriceFreeCache(c, blind=True)
            self.assertTrue(np.isnan(blind.target[0, 1]).all())
            self.assertTrue(np.isnan(blind.energy[0]).all())
            # Settlement arrays cannot influence a training batch/loss weights.
            p = dict(cutoff=cutoff, dev_start=str(c.days[2]), dev_end=str(c.days[3]))
            batch = DayDataset(PriceFreeCache(c), before, p, 'train')[0]
            self.assertTrue((batch['smp'] == 1).all())
            self.assertTrue((batch['k'] == 1).all())
            self.assertFalse(batch['loss_mask'][17:].any() if int(batch['day_index']) == 1 else False)

    def test_group_integrity_reproducibility_and_missing_targets(self):
        sites = pd.DataFrame(dict(site_id=[f'PV{i:03}' for i in range(253)], capacity_kw=np.full(253, 200.)))
        comp = metadata_composition(sites, 2026)
        pd.testing.assert_frame_equal(comp, metadata_composition(sites, 2026))
        validate_composition(comp, sites)
        cache = SimpleNamespace(sites=sites, num_sites=253, composition=comp,
                                days=pd.DatetimeIndex([aware('2024-06-01')]))
        cache.energy = np.full((1,253,24), 100.)
        cache.energy[0,0,0] = np.nan
        cache.smp = np.full((1,24), 100.)
        cache.k = np.array([hourly_weights(day_hours(cache.days[0]))])
        predictions = np.full((1,253,24), .5)
        assignments = compositions(cache, optimize=True)
        with tempfile.TemporaryDirectory() as tmp:
            first = score(cache, [0], predictions, assignments, tmp)
            a = first['assignments']['baseline/complete_fleet']
            b = first['assignments']['baseline/observed_only']
            self.assertEqual(a['contract']['valid_hours'], 23)
            self.assertEqual(b['contract']['valid_hours'], 24)
            self.assertEqual(a['contract']['observed_rows'], 23*253)
            self.assertEqual(b['contract']['observed_rows'], 24*253-1)
            self.assertEqual(a['W'], 0)
            self.assertEqual(b['W'], 0)  # missing actual did not become zero
            self.assertFalse(b['comparable_internal_score'])
            from analysis.tsmixer.year_backtest import REFERENCE
            reference = dict(penalty_reference=REFERENCE, contract=b['contract'])
            second = score(cache, [0], predictions, assignments, tmp, reference)
            self.assertTrue(second['assignments']['baseline/observed_only']['comparable_internal_score'])
            self.assertFalse(second['assignments']['baseline/complete_fleet']['comparable_internal_score'])
            # Hash is insensitive to CSV order, sensitive to membership.
            self.assertEqual(composition_hash(comp), composition_hash(comp.iloc[::-1]))
            changed = comp.copy(); changed.loc[0,'group'] = 'G2' if changed.loc[0,'group']=='G1' else 'G1'
            self.assertNotEqual(composition_hash(comp), composition_hash(changed))
        for invalid in [pd.concat([comp,comp.iloc[:1]]), comp.iloc[:-1], comp.assign(group='G1')]:
            with self.assertRaises(ValueError): validate_composition(invalid, sites)
        with self.assertRaises(ValueError): validate_composition(comp, sites.assign(capacity_kw=1.))


if __name__ == '__main__':
    unittest.main()
