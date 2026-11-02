import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from analysis.deep_learning.features import (
    aware, day_hours, issuance, operating_day, assert_asof, split_plan,
    fit_normalizer, normalize, contract, apply_postprocess)
from analysis.deep_learning.models.common import model_inputs
from analysis.deep_learning.losses import compute_loss
from analysis.deep_learning.smoke import synthetic_cache
from analysis.tsmixer.model import TSMixer, TimeMixing, FeatureMixing
from analysis.tsmixer.config import load_config
from analysis.tsmixer.runtime import device_for, portable_batch, amp_context, seed_everything
from analysis.tsmixer.data import open_cache, DayDataset, SmokeCache
from analysis.tsmixer.checkpoint import save, load, capture_rng, restore_rng
from analysis.tsmixer.evaluate import canonical_day, evaluate
from analysis.tsmixer.train import train_epoch, run


def inputs(n=2):
    return {'weather': torch.randn(n, 24, 90),
            'weather_missing': torch.zeros(n, 24, 90, dtype=torch.bool),
            'temporal': torch.randn(n, 24, 8), 'static': torch.randn(n, 146),
            'site_index': torch.arange(1, n+1), 'site_pos': torch.arange(n),
            'metadata_valid': torch.ones(n, dtype=torch.bool)}


def small_model(**kwargs):
    return TSMixer(hidden_dim=16, blocks=1, time_hidden=12, feature_hidden=24,
                   dropout=0., site_dropout=0., **kwargs)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_default_shape_and_parameters(self):
        m = TSMixer().eval()
        self.assertEqual(m(inputs()).shape, (2, 24))
        self.assertEqual(m.blocks[0][0].mlp[0].in_features, 24)
        self.assertEqual(m.blocks[0][1].mlp[0].in_features, 128)
        self.assertGreater(sum(p.numel() for p in m.parameters()), 100000)

    def test_five_feature_public_contract(self):
        b = inputs()
        del b['site_pos']; del b['metadata_valid']
        self.assertEqual(small_model()(b).shape, (2, 24))

    def test_invalid_dimensions_and_forbidden_inputs(self):
        m = small_model()
        for key in ('smp', 'k', 'energy', 'target_cf', 'loss_mask'):
            b = inputs(); b[key] = torch.zeros(2)
            with self.assertRaises(ValueError):
                m(b)
        b = inputs(); b['weather'] = b['weather'][:, :23]
        with self.assertRaises(ValueError):
            m(b)
        with self.assertRaises(ValueError):
            TSMixer(time_hidden=0)

    def test_backward_finite(self):
        m = small_model()
        m(inputs()).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(float(m.blocks[0][0].mlp[0].weight.grad.abs().sum().detach()), 0)
        self.assertGreater(float(m.blocks[0][1].mlp[0].weight.grad.abs().sum().detach()), 0)

    def test_time_mlp_axis_and_cross_hour_influence(self):
        m = TimeMixing(4, 5, 0.).eval()
        # Time norm acts on features; MLP sees precisely [..., features,24].
        seen = []
        h = m.mlp[0].register_forward_pre_hook(lambda _, x: seen.append(tuple(x[0].shape)))
        x = torch.randn(2, 24, 4)
        changed = x.clone(); changed[0, 0, 0] += 2
        y, z = m(x), m(changed)
        h.remove()
        self.assertEqual(seen[0], (2, 4, 24))
        self.assertGreater(float((z[0, 1:] - y[0, 1:]).abs().sum().detach()), 0)
        torch.testing.assert_close(y[1], z[1], rtol=0, atol=0)

    def test_feature_mlp_axis_no_cross_hour_influence(self):
        m = FeatureMixing(4, 7, 0.).eval()
        seen = []
        h = m.mlp[0].register_forward_pre_hook(lambda _, x: seen.append(tuple(x[0].shape)))
        x = torch.randn(2, 24, 4)
        changed = x.clone(); changed[0, 0, 0] += 2
        y, z = m(x), m(changed)
        h.remove()
        self.assertEqual(seen[0], (2, 24, 4))
        self.assertGreater(float((y[0, 0, 1:]-z[0, 0, 1:]).abs().sum().detach()), 0)
        torch.testing.assert_close(y[:, 1:], z[:, 1:], rtol=0, atol=0)

    def test_no_batch_mixing_and_cpu_inference_consistency(self):
        m, b = small_model().eval(), inputs(3)
        a = m(b)
        single = torch.cat([m({k: v[i:i+1] for k, v in b.items()}) for i in range(3)])
        torch.testing.assert_close(a, single, rtol=1e-5, atol=1e-6)
        changed = {k: v.clone() for k, v in b.items()}
        changed['weather'][1] += 9
        torch.testing.assert_close(m(changed)[0], a[0], rtol=0, atol=0)
        torch.testing.assert_close(m(b), m(b), rtol=0, atol=0)

    def test_static_features_affect_output(self):
        m, b = small_model().eval(), inputs()
        a = m(b)
        b['static'][0, 0] += 5
        self.assertGreater(float((a[0]-m(b)[0]).abs().sum().detach()), 1e-7)

    def test_fleet_interface_and_canonical_order(self):
        m = small_model().eval()
        b = inputs(253)
        fleet = {k: v.unsqueeze(0) for k, v in b.items()}
        torch.testing.assert_close(m.forward_fleet(fleet)[0], m(b), rtol=1e-5, atol=1e-6)
        bad = {k: v[:, :252] for k, v in fleet.items()}
        with self.assertRaises(ValueError):
            m.forward_fleet(bad)
        fleet['site_pos'] = fleet['site_pos'].flip(1)
        with self.assertRaises(ValueError):
            m.forward_fleet(fleet)

    def test_canonical_reconstruction_rejects_partial(self):
        pred = np.arange(253*24).reshape(253,24)
        perm = np.random.default_rng(2026).permutation(253)
        np.testing.assert_array_equal(canonical_day(pred[perm], perm), pred)
        with self.assertRaises(ValueError):
            canonical_day(pred[:32], np.arange(32))
        with self.assertRaises(ValueError):
            canonical_day(pred, np.zeros(253,dtype=int))

    def test_cpu_checkpoint_save_reload(self):
        m, b = small_model().eval(), inputs()
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/'cpu.pt'
            save(file, {'model': m.state_dict(), 'smoke_test': True})
            n = small_model().eval(); n.load_state_dict(load(file)['model'])
            torch.testing.assert_close(m(b), n(b), rtol=0, atol=0)

    def test_unavailable_devices_fail(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                device_for('cuda')
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'MPS unavailable'):
                device_for('mps')
        with patch('platform.system', return_value='Darwin'):
            self.assertEqual(device_for('auto').type, 'cpu')
        with self.assertRaises(ValueError):
            amp_context(torch.device('cpu'), 'fp16')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS runtime unavailable')
    def test_mps_smoke_and_cpu_parity(self):
        cpu, b = small_model().eval(), inputs()
        mps = copy.deepcopy(cpu).to('mps').eval()
        mb = portable_batch(b, torch.device('mps'))
        torch.testing.assert_close(cpu(b), mps(mb).cpu(), rtol=2e-3, atol=2e-4)
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            # MPS embedding backward does not implement deterministic mode.
            # This changes the reproducibility guarantee, not model arithmetic.
            torch.use_deterministic_algorithms(False)
            mps(mb).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in mps.parameters()))
        finally:
            torch.use_deterministic_algorithms(deterministic)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA runtime unavailable')
    def test_cuda_smoke_and_cpu_parity(self):
        cpu, b = small_model().eval(), inputs()
        cuda = copy.deepcopy(cpu).to('cuda').eval()
        cb = portable_batch(b, torch.device('cuda'))
        torch.testing.assert_close(cpu(b), cuda(cb).cpu(), rtol=2e-3, atol=2e-4)
        cuda(cb).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in cuda.parameters()))
        cuda.zero_grad(set_to_none=True)
        with amp_context(torch.device('cuda'), 'fp16'):
            mixed = cuda(cb)
        torch.testing.assert_close(cpu(b), mixed.float().cpu(), rtol=2e-2, atol=3e-3)
        scaler = torch.amp.GradScaler('cuda')
        scaler.scale(mixed.float().square().mean()).backward()
        optimizer = torch.optim.AdamW(cuda.parameters(), lr=.001)
        scaler.unscale_(optimizer)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in cuda.parameters()))
        scaler.step(optimizer); scaler.update()


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = synthetic_cache(Path(cls.tmp.name)/'cache')
        cls.cache = open_cache(cls.root/'train')
        cls.config = load_config('configs/tsmixer/base.yaml')
        cls.plan = split_plan(cls.config, cls.cache, 'development')
        cls.normalizer = fit_normalizer(cls.cache, cls.plan['cutoff'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_operating_day_and_17_cutoff(self):
        t = day_hours('2025-02-01')
        self.assertEqual(t[0], aware('2025-02-01 01:00'))
        self.assertEqual(t[-1], aware('2025-02-02 00:00'))
        self.assertTrue((operating_day(t) == aware('2025-02-01')).all())
        self.assertTrue((issuance(t) == aware('2025-01-31 17:00')).all())
        assert_asof(issuance(t), t)
        with self.assertRaises(ValueError):
            assert_asof(issuance(t)+pd.Timedelta(seconds=1), t)
        self.assertEqual(self.plan['cutoff'], str(aware('2025-01-31 17:00')))

    def test_round_contract(self):
        c = contract()
        self.assertEqual(len(c['weather_features']), 90)
        self.assertEqual(c['weather_features'][30:60], [v+'_r10' for v in c['weather_base']])
        self.assertEqual(c['weather_features'][60:], [v+'_revision' for v in c['weather_base']])

    def test_normalizer_ignores_future_labels_and_weather(self):
        other = copy.copy(self.cache)
        cutoff = aware(self.plan['cutoff'])
        class FutureChanged:
            def __init__(self, array, days): self.array, self.days, self.shape = array, days, array.shape
            def __getitem__(self, key):
                keys = key if isinstance(key, tuple) else (key,)
                result = np.array(self.array[key],copy=True)
                if self.days[int(keys[0])] > cutoff: result[:] = 999
                return result
        other._arrays = dict(self.cache._arrays)
        for k in ('target','weather'):
            other._arrays[k] = FutureChanged(getattr(self.cache,k), self.cache.days)
        self.assertEqual(self.normalizer, fit_normalizer(other, self.plan['cutoff']))

    def test_missing_and_target_mask(self):
        ds = DayDataset(self.cache, self.normalizer, self.plan, 'train')
        b = next(iter(DataLoader(ds,batch_size=2)))
        self.assertTrue(torch.isfinite(b['weather']).all())
        self.assertTrue((b['weather'][b['weather_missing']] == 0).all())
        self.assertTrue(torch.isnan(b['target_cf'][~b['target_mask']]).all())
        pred = small_model()(model_inputs(b))
        loss, _ = compute_loss(pred,b,self.config['loss'],self.config['group'],None)
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        y, mask = normalize(np.array([[np.nan,2]],dtype='float32'),
                            {'mean':[1,1],'std':[1,1]})
        self.assertEqual(y.tolist(), [[0.,1.]])
        self.assertEqual(mask.tolist(), [[True,False]])

    def test_last_day_masks_after_cutoff(self):
        ds = DayDataset(self.cache, self.normalizer, self.plan, 'train')
        idx = np.flatnonzero(ds.items[:,0] == 2)[0]
        b = ds[int(idx)]
        self.assertFalse(b['loss_mask'][17:].any())
        self.assertTrue(torch.isnan(b['target_cf'][17:]).all())

    def test_invalid_targets_excluded_only_from_training(self):
        # Use an in-memory copy of ONE day to avoid modifying the fixture/cache.
        other = copy.copy(self.cache); other._arrays = dict(self.cache._arrays)
        y = np.array(self.cache.target[0], copy=True); y[20,0]=-0.1; y[20,1]=1.2
        class Patched:
            shape = self.cache.target.shape
            def __getitem__(self, key):
                keys = key if isinstance(key,tuple) else (key,)
                a = y if keys[0] == 0 else self_cache.target[keys[0]]
                return a[keys[1:]] if len(keys)>1 else a
        self_cache = self.cache
        other._arrays['target'] = Patched()
        ds = DayDataset(other,self.normalizer,self.plan,'train')
        i = np.flatnonzero((ds.items[:,0]==0)&(ds.items[:,1]==20))[0]
        self.assertFalse(ds[int(i)]['loss_mask'][:2].any())
        self.assertTrue(ds[int(i)]['target_mask'][:2].all())

    def test_weighted_l1_uses_capacity_and_price_but_not_model_inputs(self):
        b = next(iter(DataLoader(DayDataset(self.cache,self.normalizer,self.plan,'train'),batch_size=2)))
        p = torch.ones_like(b['target_cf'])*0.3
        loss, _ = compute_loss(p,b,self.config['loss'],self.config['group'],None)
        mask = b['loss_mask']; safe = torch.where(mask,b['target_cf'],0.)
        weights = mask*b['capacity_kw'][:,None]*b['smp']*b['k']
        torch.testing.assert_close(loss, ((p-safe).abs()*weights).sum()/weights.sum())
        self.assertNotIn('smp',model_inputs(b)); self.assertNotIn('k',model_inputs(b))

    def test_bounded_smoke_view_and_loader(self):
        view = SmokeCache(self.cache,self.plan['cutoff'])
        self.assertLessEqual(len(view.days),2)
        self.assertEqual(view.manifest['signature'],self.cache.manifest['signature'])
        norm = fit_normalizer(view,self.plan['cutoff'])
        ds = DayDataset(view,norm,self.plan,'train')
        b = next(iter(DataLoader(ds,batch_size=2)))
        self.assertEqual(tuple(b['weather'].shape),(2,24,90))

    def test_accumulation_matches_combined_weighted_batch(self):
        b = next(iter(DataLoader(DayDataset(self.cache,self.normalizer,self.plan,'train'),batch_size=4)))
        a = small_model(); z = copy.deepcopy(a)
        ca = copy.deepcopy(self.config); cz = copy.deepcopy(self.config)
        ca['training']['accumulation_steps']=2; cz['training']['accumulation_steps']=1
        micro = [{k:v[:1] for k,v in b.items()}, {k:v[1:] for k,v in b.items()}]
        oa=torch.optim.SGD(a.parameters(),lr=.01); oz=torch.optim.SGD(z.parameters(),lr=.01)
        sa=torch.amp.GradScaler('cuda',enabled=False); sz=torch.amp.GradScaler('cuda',enabled=False)
        train_epoch(a,micro,oa,sa,ca,torch.device('cpu'))
        train_epoch(z,[b],oz,sz,cz,torch.device('cpu'))
        for p,q in zip(a.parameters(),z.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)

    def test_epoch_checkpoint_resume_exact(self):
        b = next(iter(DataLoader(DayDataset(self.cache,self.normalizer,self.plan,'train'),batch_size=2)))
        seed_everything(2026)
        m=TSMixer(hidden_dim=16, blocks=1, time_hidden=12, feature_hidden=24, dropout=.1, site_dropout=.1); o=torch.optim.AdamW(m.parameters(),lr=.001)
        s=torch.amp.GradScaler('cuda',enabled=False)
        train_epoch(m,[b],o,s,self.config,torch.device('cpu'))
        with tempfile.TemporaryDirectory() as d:
            save(Path(d)/'last.pt',{'model':m.state_dict(),'optimizer':o.state_dict(),'rng':capture_rng()})
            train_epoch(m,[b],o,s,self.config,torch.device('cpu'))
            ck=load(Path(d)/'last.pt'); n=TSMixer(hidden_dim=16, blocks=1, time_hidden=12, feature_hidden=24, dropout=.1, site_dropout=.1); n.load_state_dict(ck['model'])
            no=torch.optim.AdamW(n.parameters(),lr=.001); no.load_state_dict(ck['optimizer'])
            restore_rng(ck['rng']); train_epoch(n,[b],no,s,self.config,torch.device('cpu'))
            for k,v in m.state_dict().items():
                torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)

    def test_streaming_evaluation_matches_manual_canonical_predictions(self):
        plan=dict(self.plan,dev_end=str(aware('2025-02-02')))
        ds=DayDataset(self.cache,self.normalizer,plan,'dev')
        t=dict(self.config['training'],batch_size=17,postprocess='none')
        m=small_model().eval()
        from analysis.deep_learning.metrics import Evaluator
        e=Evaluator(self.cache,self.normalizer['seen_hours'])
        b=next(iter(DataLoader(ds,batch_size=253)))
        with torch.inference_mode(): pred=m(model_inputs(b)).numpy()
        e.add_day(int(ds.day_indices[0]),apply_postprocess(pred,self.cache,int(ds.day_indices[0]),self.normalizer,'none'))
        actual=evaluate(m,ds,t,torch.device('cpu'))[0]; expected=e.finish()[0]
        self.assertEqual(actual['scored_hours'],expected['scored_hours'])
        self.assertAlmostEqual(actual['site_cf_mae'],expected['site_cf_mae'],places=6)
        with self.assertRaises(ValueError):
            evaluate(m,DayDataset(self.cache,self.normalizer,self.plan,'train'),t,torch.device('cpu'))

    def test_cache_signature_tampering_rejected(self):
        from analysis.deep_learning.config import fingerprint
        self.assertEqual(self.cache.manifest['signature'],
                         fingerprint({k:v for k,v in self.cache.manifest.items() if k!='signature'}))
        bad = dict(self.normalizer, cache_signature='mismatch')
        with self.assertRaisesRegex(ValueError,'Normalizer/cache mismatch'):
            DayDataset(self.cache,bad,self.plan,'train')

    def test_spawn_worker_bounded_loader(self):
        view=SmokeCache(self.cache,self.plan['cutoff'])
        norm=fit_normalizer(view,self.plan['cutoff'])
        ds=DayDataset(view,norm,self.plan,'train')
        from torch.utils.data import Subset
        ds=Subset(ds,[0,1])
        b=next(iter(DataLoader(ds,batch_size=2,num_workers=1,multiprocessing_context='spawn')))
        self.assertEqual(tuple(b['weather'].shape),(2,24,90))

    def test_v0_rejects_group_loss_and_invalid_config(self):
        from analysis.tsmixer.config import validate
        c=copy.deepcopy(self.config); c['group']['enabled']=True
        with self.assertRaises(ValueError): validate(c)
        with self.assertRaises(ValueError):
            load_config('configs/tsmixer/base.yaml',['model.typo=1'])

    def test_mac_full_train_guard_before_data_or_writes(self):
        with patch('platform.system',return_value='Darwin'):
            with self.assertRaisesRegex(ValueError,'Full training requires'):
                run(self.config,'/invalid/no-write',requested_device='cpu',full_train=True)


if __name__ == '__main__':
    unittest.main()
