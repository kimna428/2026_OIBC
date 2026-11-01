import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from analysis.deep_learning.config import ROOT,load_config,fingerprint
from analysis.deep_learning.features import day_hours,issuance,assert_asof,assert_allowed,split_plan,fit_normalizer,aware
from analysis.deep_learning.dataset import DayCache,DayDataset
from analysis.deep_learning.smoke import synthetic_cache,smoke_config
from analysis.deep_learning.models import build_model,model_inputs
from analysis.deep_learning.losses import compute_loss
from analysis.deep_learning.checkpoint import save_checkpoint,load_checkpoint,capture_rng,restore_rng
from analysis.deep_learning.metrics import Evaluator
from analysis.deep_learning.environment import parameter_report
from analysis.deep_learning.train import loss_composition

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); cls.temp=tempfile.TemporaryDirectory(); cls.root=Path(cls.temp.name)
        synthetic_cache(cls.root/'cache'); cls.cache=DayCache(cls.root/'cache/train')
        cls.config=smoke_config('cnn',cls.root/'cache',cls.root); cls.config['_smoke_test']=True
        cls.plan=split_plan(cls.config,cls.cache,'development'); cls.stats=fit_normalizer(cls.cache,cls.plan['cutoff'])
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def batch(self,model):
        data=DayDataset(self.cache,self.stats,self.plan,'train',model=='cross_site')
        return next(iter(DataLoader(data,batch_size=1)))
    def dims(self): return {'weather_dim':90,'temporal_dim':8,'static_dim':146,'num_sites':253}
    def test_three_models_forward_backward(self):
        for name in ['cnn','transformer','cross_site']:
            config=smoke_config(name,self.root/'cache',self.root); model=build_model(config['model'],self.dims()); batch=self.batch(name)
            pred=model(model_inputs(batch)); expected=(1,253,24) if name=='cross_site' else (1,24)
            self.assertEqual(tuple(pred.shape),expected)
            loss,_=compute_loss(pred,batch,config['loss'],config['group'],torch.tensor([0]*126+[1]*127)); loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads)); self.assertTrue(any(g.abs().sum()>0 for g in grads))
    def test_real_config_small_forward_backward(self):
        for name,file in [('cnn','cnn_small'),('transformer','transformer_small'),('cross_site','cross_site_base')]:
            model=build_model(load_config(f'configs/deep_learning/{file}.yaml')['model'],self.dims()); batch=self.batch(name)
            output=model(model_inputs(batch)); output.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_target_not_a_feature(self):
        with self.assertRaises(ValueError): assert_allowed(['wa_cloud','target_cf'])
        model=build_model(smoke_config('cnn',self.root/'cache',self.root)['model'],self.dims()).eval(); batch=self.batch('cnn')
        with self.assertRaises(ValueError): model(batch)
        a=model(model_inputs(batch)); batch['target_cf'][:]=999; batch['target_mask'][:]=False; batch['smp'][:]=99999
        torch.testing.assert_close(a,model(model_inputs(batch)))
    def test_fold_contract_matches_existing_pipeline(self):
        import ast
        from analysis.deep_learning.features import FOLDS
        tree=ast.parse((ROOT/'analysis/expanding_pipeline.py').read_text())
        old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FOLDS' for t in n.targets))
        self.assertEqual(FOLDS,old)
    def test_weighted_and_normalized_huber_losses(self):
        pred=torch.full((1,24),.2,requires_grad=True)
        batch={'target_cf':torch.zeros(1,24),'loss_mask':torch.ones(1,24,dtype=torch.bool),'smp':torch.ones(1,24),'k':torch.ones(1,24),'capacity_kw':torch.tensor([100.])}
        for name,expected in [('l1',.2),('weighted_l1',.2),('huber',.175)]:
            loss,_=compute_loss(pred,batch,{'name':name,'huber_delta':.05},{'enabled':False,'alpha':1.,'beta':0.},torch.tensor([0,1]))
            self.assertAlmostEqual(float(loss),expected,places=5)
        batch['smp'][:]=0
        loss,info=compute_loss(pred,batch,{'name':'weighted_l1'},{'enabled':False,'alpha':1.,'beta':0.},torch.tensor([0,1]))
        self.assertEqual(float(loss),0); self.assertEqual(info['site_mass'],0)
    def test_nan_target_mask_and_gradient(self):
        pred=torch.ones(1,24,requires_grad=True); target=torch.zeros(1,24); target[0,0]=float('nan'); mask=torch.isfinite(target)
        b={'target_cf':target,'loss_mask':mask,'smp':torch.ones(1,24),'k':torch.ones(1,24),'capacity_kw':torch.tensor([100.])}
        loss,_=compute_loss(pred,b,{'name':'l1'},{'enabled':False,'alpha':1.,'beta':0.},torch.tensor([0,1])); loss.backward()
        self.assertEqual(pred.grad[0,0],0); self.assertTrue(torch.isnan(b['target_cf'][0,0]))
    def test_group_loss_requires_full_fleet(self):
        b=self.batch('cross_site'); p=torch.ones_like(b['target_cf'],requires_grad=True); group={'enabled':True,'alpha':0.,'beta':1.}
        b['loss_mask'][:]=False; b['loss_mask'][:,0,:]=True; b['target_cf'][:]=.2; b['group_complete'][:]=False
        loss,info=compute_loss(p,b,{'name':'l1'},group,torch.tensor([0]*126+[1]*127)); self.assertEqual(info['group_mass'],0); self.assertEqual(float(loss),0)
        b['loss_mask'][:]=True; b['group_complete'][:]=True
        loss,info=compute_loss(p,b,{'name':'l1'},group,torch.tensor([0]*126+[1]*127)); self.assertAlmostEqual(float(loss),.8,places=5)
        loss.backward(); self.assertTrue(torch.isfinite(p.grad).all())
    def test_k_weight_is_interval_end_hour(self):
        from analysis.validation import hourly_weights
        times=day_hours('2025-02-01'); weights=hourly_weights(times)
        self.assertEqual(list(times.hour[[9,10,14,15,23]]),[10,11,15,16,0]); self.assertEqual(list(weights[[9,10,14,15,23]]),[1,3,3,1,1])
    def test_midnight_and_future(self):
        times=day_hours('2025-03-01'); self.assertEqual(times[0],aware('2025-03-01 01:00')); self.assertEqual(times[-1],aware('2025-03-02 00:00'))
        self.assertEqual(issuance(times)[-1],aware('2025-02-28 17:00'))
        with self.assertRaises(ValueError): assert_asof([aware('2025-02-28 18:00')],[times[0]])
    def test_normalization_and_embedding_do_not_see_future_labels(self):
        cache=DayCache(self.root/'cache/train'); original=np.array(cache.target)
        writable=np.load(cache.root/'target.npy',mmap_mode='r+'); writable[3:]=999; writable[2,:,17:]=999; writable.flush()
        try:
            again=fit_normalizer(cache,self.plan['cutoff']); self.assertEqual(self.stats,again)
        finally: writable[:]=original; writable.flush()
    def test_training_masks_partial_last_day(self):
        ds=DayDataset(self.cache,self.stats,self.plan,'train',True)
        b=ds[-1]; self.assertTrue(b['loss_mask'][:,:17].any()); self.assertFalse(b['loss_mask'][:,17:].any()); self.assertFalse(b['group_complete'][17:].any()); self.assertTrue(torch.isnan(b['target_cf'][:,17:]).all()); self.assertFalse(b['target_mask'][:,17:].any())
    def test_cross_site_order_and_no_target_attention_mask(self):
        b=self.batch('cross_site'); model=build_model(smoke_config('cross_site',self.root/'cache',self.root)['model'],self.dims()).eval()
        before=model(model_inputs(b)); b['target_mask'][:]=False; b['group_complete'][:]=False
        torch.testing.assert_close(before,model(model_inputs(b)))
        b['site_pos']=b['site_pos'].flip(1)
        with self.assertRaises(ValueError): model(model_inputs(b))
    def test_all_missing_metadata_attention_is_finite(self):
        b=self.batch('cross_site'); b['metadata_valid'][:]=False
        model=build_model(smoke_config('cross_site',self.root/'cache',self.root)['model'],self.dims()); self.assertTrue(torch.isfinite(model(model_inputs(b))).all())
    def test_unknown_site_embedding_is_zero_and_fallback_works(self):
        b=self.batch('cnn'); model=build_model(smoke_config('cnn',self.root/'cache',self.root)['model'],self.dims()).eval(); b['site_index'][:]=0
        before=model(model_inputs(b))
        with torch.no_grad(): model.encoder.embedding.weight[1:]=999
        torch.testing.assert_close(before,model(model_inputs(b))); self.assertEqual(float(model.encoder.embedding.weight[0].abs().sum()),0)
    def test_missing_weather_and_missing_targets_are_distinct(self):
        b=self.batch('cnn'); self.assertEqual(b['weather_missing'].dtype,torch.bool); self.assertEqual(b['target_mask'].dtype,torch.bool); self.assertTrue(torch.isfinite(b['weather']).all())
        y=self.cache.target[0,0]; self.assertTrue(np.isnan(y).all()); self.assertTrue((self.cache.missing_kind[0,0]==1).all())
    def test_checkpoint_and_rng_roundtrip(self):
        model=build_model(smoke_config('cnn',self.root/'cache',self.root)['model'],self.dims()); p=self.root/'unit.pt'; rng=capture_rng()
        save_checkpoint(p,{'format_version':1,'model':model.state_dict(),'rng':rng}); state=load_checkpoint(p); clone=build_model(smoke_config('cnn',self.root/'cache',self.root)['model'],self.dims()); clone.load_state_dict(state['model'])
        for a,b in zip(model.parameters(),clone.parameters()): torch.testing.assert_close(a,b)
        restore_rng(rng); a=torch.rand(3); restore_rng(state['rng']); torch.testing.assert_close(a,torch.rand(3))
    def test_config_and_parameter_sizes(self):
        report=parameter_report(); self.assertEqual(len(report),9)
        for file in report: load_config(f'configs/deep_learning/{file}.yaml')
        self.assertTrue(1.8e6<report['cnn_small']['parameters']<2.3e6); self.assertTrue(28e6<report['transformer_large']['parameters']<32e6)
    def test_official_scorer_bridge_and_incomplete_hours(self):
        e=Evaluator(self.cache,self.stats['seen_hours']); day=3; pred=np.nan_to_num(np.asarray(self.cache.target[day]),nan=.3)
        e.add_day(day,pred); metrics,detail,groups,ledger=e.finish(); self.assertTrue(metrics['score_available']); self.assertAlmostEqual(metrics['penalty_krw'],0.,places=6)
        self.assertEqual(metrics['scored_hours'],int(np.isfinite(self.cache.target[day]).all(0).sum()))
        e=Evaluator(self.cache,self.stats['seen_hours']); e.add_day(0,np.zeros((253,24))); m,*_=e.finish(); self.assertFalse(m['score_available']); self.assertIsNone(m['penalty_krw'])
    def test_future_group_assignment_rejected(self):
        comp=self.root/'custom.csv'; self.cache.composition.to_csv(comp,index=False); c=copy.deepcopy(self.config); c['group'].update(composition_path=str(comp),available_at='2025-02-28 17:00:00+09:00')
        with self.assertRaises(ValueError): loss_composition(c,self.cache,self.plan['cutoff'])
    def test_future_development_window_rejected(self):
        c=copy.deepcopy(self.config); c['split']['dev_end']='2025-03-05'
        with self.assertRaises(ValueError): split_plan(c,self.cache,'development')
    def test_ensemble_simplex_and_original_complete_mask(self):
        from analysis.deep_learning.ensemble import simplex_weights,objective
        weights=list(simplex_weights(3,2)); self.assertEqual(len(weights),6)
        self.assertTrue(all(min(w)>=0 and abs(sum(w)-1)<1e-12 for w in weights))
        with self.assertRaises(ValueError): objective([np.zeros((1,253,24))],self.cache,[0],[1.])
        self.assertAlmostEqual(objective([np.zeros((1,253,24))],self.cache,[3],[1.]),242880.,places=5)
    def test_full_cpu_training_is_blocked(self):
        from types import SimpleNamespace
        from analysis.deep_learning.train import train
        args=SimpleNamespace(config='configs/deep_learning/cnn_small.yaml',resume=None,phase=None,fold=None,device='cpu',smoke_test=False,max_steps=None)
        with self.assertRaisesRegex(ValueError,'Full local CPU training is disabled'): train(args)
    def test_gpu_process_lock_without_cuda_allocation(self):
        from unittest.mock import patch
        from analysis.deep_learning.runtime import GPULock
        with patch.dict('os.environ',{'OIBC_GPU_LOCK':str(self.root/'gpu.lock')}):
            with GPULock(torch.device('cuda')):
                with self.assertRaises(RuntimeError):
                    with GPULock(torch.device('cuda')): pass
    def test_missing_data_does_not_create_synthetic_fallback(self):
        from analysis.deep_learning.prepare import inspect_inputs
        with self.assertRaisesRegex(FileNotFoundError,'no data will be synthesized'): inspect_inputs(self.root/'absent')
    def test_spawn_workers_read_mmap_batches(self):
        from analysis.deep_learning.runtime import loader
        ds=DayDataset(self.cache,self.stats,self.plan,'train',False)
        settings={'batch_size':2,'num_workers':2,'persistent_workers':False,'pin_memory':False}
        batch=next(iter(loader(ds,False,settings,torch.Generator().manual_seed(1))))
        self.assertEqual(tuple(batch['weather'].shape),(2,24,90))
    def test_mmap_can_be_reopened_after_pickle(self):
        import pickle
        clone=pickle.loads(pickle.dumps(self.cache)); self.assertIsInstance(clone.weather,np.memmap)
        self.assertEqual(clone.weather.shape,self.cache.weather.shape)

if __name__=='__main__': unittest.main()
