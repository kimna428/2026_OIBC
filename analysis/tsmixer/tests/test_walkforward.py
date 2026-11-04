import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from analysis.deep_learning.smoke import synthetic_cache
from analysis.deep_learning.features import aware,day_hours,fit_normalizer
from analysis.tsmixer.data import open_cache
from analysis.tsmixer.config import load_config
from analysis.tsmixer.comparison.io import file_sha
from analysis.tsmixer.tests.test_comparison import fixture
from analysis.tsmixer.ensemble.folds import FOLDS,validate_folds
from analysis.tsmixer.ensemble.data import History
from analysis.tsmixer.ensemble.features import Features,observed_targets,outages,FC
from analysis.tsmixer.ensemble.datasets import fit_stats,NeuralDays
from analysis.tsmixer.ensemble.models import SJGRU,StateGRU,state_loss
from analysis.tsmixer.ensemble.trainer import fit_component,neural_predict
from analysis.tsmixer.ensemble.predictions import validate_provenance,save_prediction,load_prediction
from analysis.tsmixer.ensemble.weight_selection import select,freeze,read_frozen
from analysis.tsmixer.ensemble.persistence import reserve,complete,is_complete
from analysis.tsmixer.ensemble.blend import mix,candidates
from analysis.tsmixer.ensemble.evaluation import site_metrics


class WalkforwardTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_python310_hash_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bytes';p.write_bytes(b'past-only')
            expected=file_sha(p)
            with patch('analysis.tsmixer.comparison.io.hashlib.file_digest',None):self.assertEqual(file_sha(p),expected)
    def test_folds_and_leap_year_and_inner_bound(self):
        self.assertTrue(validate_folds());self.assertEqual([len(f.times) for f in FOLDS.values()],[8760,8784,8760])
        self.assertEqual(FOLDS['F1'].end,FOLDS['F2'].start)
        for f in FOLDS.values():
            self.assertEqual(f.start-pd.Timedelta(hours=8),f.cutoff)
            self.assertLess(pd.Timestamp(f.plan('selection')['cutoff']),pd.Timestamp(f.plan('selection')['dev_start']))
            self.assertEqual(pd.Timestamp(f.plan('selection')['selection_target_end']),f.cutoff)
    def test_history_blocks_future_labels_stats_and_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp);c=open_cache(Path(tmp)/'train');cutoff=c.days[1]+pd.Timedelta(hours=17)
            h=History(c,cutoff)
            self.assertEqual(len(h.days),2)
            self.assertTrue(np.isnan(h.target[1][:,17:]).all());self.assertTrue((h.smp[1][17:]==0).all())
            before=fit_normalizer(h,str(cutoff))
            # Poison the entire later axis without touching on-disk cache.
            class Block:
                def __init__(self,a):self.a=a;self.shape=a.shape
                def __getitem__(self,k):
                    d=k[0] if isinstance(k,tuple) else k
                    if d>1:raise AssertionError('future array read')
                    return self.a[k]
            for name in ('target','energy','weather','temporal'):c._arrays[name]=Block(getattr(c,name))
            self.assertEqual(before,fit_normalizer(History(c,cutoff),str(cutoff)))
    def test_prediction_provenance_shape_alignment_and_units(self):
        grid,_=fixture();f=FOLDS['F3'];value=np.full_like(grid.actual,100.)
        meta=dict(model='tsmixer',fold=f.name,start=str(f.start),end_exclusive=str(f.end),smoke_only=False,
            protocol='past_only_walkforward_v1',units='kWh',timestamp_convention='interval_end',
            inference_inputs=['forecasts'],**{k+'_cutoff':str(f.cutoff) for k in ('training','normalization','selection')},
            auxiliary_cutoff=None,outage_cutoff=None,calibration_cutoff=None)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.parquet';save_prediction(path,value,grid.times,grid.sites,meta)
            got,m=load_prediction(path,path.with_suffix('.json'),grid,f,'tsmixer')
            np.testing.assert_array_equal(got,value)
            m['units']='capacity_factor';Path(tmp,'cf.json').write_text(json.dumps(m))
            cf,_=load_prediction(path,Path(tmp,'cf.json'),grid,f,'tsmixer');self.assertTrue((cf==20000).all())
            partial=value.copy();partial[0,0]=np.nan
            imported=Path(tmp)/'observed_only.parquet'
            save_prediction(imported,partial,grid.times,grid.sites,meta,observed=grid.observed)
            imported.with_suffix('.json').unlink()
            save_prediction(imported,partial,grid.times,grid.sites,meta,observed=grid.observed,resume=True)
            restored,_=load_prediction(imported,imported.with_suffix('.json'),grid,f,'tsmixer')
            self.assertTrue(np.isnan(restored[0,0]))
            with self.assertRaises(FileExistsError):save_prediction(path,value,grid.times,grid.sites,meta)
            for change in [dict(selection_cutoff='2024-06-01 00:00:00+09:00'),dict(training_cutoff='2025-01-01 00:00:00+09:00'),dict(protocol='original_state01'),dict(smoke_only=True),dict(calibration_factor=.96)]:
                with self.assertRaises(ValueError):validate_provenance({**meta,**change},f,'tsmixer')
    def test_selection_cannot_see_F3_and_frozen_hash(self):
        a=dict(weights=dict(tsmixer=1.,sj_ensemble=0.,state01=0.),target_end=str(FOLDS['F2'].cutoff),site_nmae_pct=3.,W=200.,valid_hours=100)
        b=dict(weights=dict(tsmixer=0.,sj_ensemble=1.,state01=0.),target_end=str(FOLDS['F2'].cutoff),site_nmae_pct=4.,W=100.,valid_hours=100)
        one=select({'F1':[a,b]},['F1'],'site');self.assertEqual(one['weights']['tsmixer'],1)
        self.assertEqual(select({'F1':[a,b]},['F1'],'penalty')['weights']['sj_ensemble'],1)
        self.assertEqual(select({'F1':[a,b],'F2':[a,b]},['F1','F2'])['selection_folds'],['F1','F2'])
        with self.assertRaises(ValueError):select({'F3':[a,b]},['F3'])
        late={**a,'target_end':str(FOLDS['F2'].start-pd.Timedelta(hours=1))}
        with self.assertRaises(ValueError):select({'F1':[late,b]},['F1'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json';freeze(path,one,{'F1':'actual_input_sha'})
            self.assertEqual(read_frozen(path)['selection']['weights'],one['weights'])
            with self.assertRaises(FileExistsError):freeze(path,one,{})
            r=json.loads(path.read_text());r['selection']['weights']['tsmixer']=0;path.write_text(json.dumps(r))
            with self.assertRaises(ValueError):read_frozen(path)
    def test_completed_job_resume_and_corruption_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp,patch('analysis.tsmixer.ensemble.persistence.result_guard'):
            path=Path(tmp)/'run';reserve(path,{'recipe':1});(path/'a').write_text('model')
            complete(path,['a']);self.assertTrue(is_complete(path))
            reserve(path,{'recipe':1},True)
            with self.assertRaises(ValueError):reserve(path,{'recipe':2},True)
            with self.assertRaises(FileExistsError):reserve(path,{'recipe':1})
            (path/'a').write_text('corrupt')
            with self.assertRaises(ValueError):is_complete(path)
    def test_site_metric_extremes_and_weighted_group_loss(self):
        grid,comp=fixture();p=np.full_like(grid.actual,100.)
        m=site_metrics(grid,p,np.zeros(253));self.assertEqual(m['all']['site_cf_mae'],0)
        self.assertGreater(m['day']['observed_rows'],0)
        for w in candidates():
            if w['tsmixer']==1:np.testing.assert_array_equal(mix({'tsmixer':p},w),p)
        b,s,h=1,2,24;y=torch.ones(b,s,h)*.5;pred=y.clone();pred[:,0]+=.1;pred[:,1]-=.1
        loss=state_loss(pred,y,torch.ones_like(y,dtype=torch.bool),torch.ones(s),torch.ones(b,h),torch.zeros(s,dtype=torch.bool),
              torch.zeros(b,s,h,2),torch.zeros(b,s,h,2),torch.zeros(b,s,h,2,dtype=torch.bool),.5)
        self.assertAlmostEqual(float(loss),.05,places=6)
    def test_lazy_feature_contract_and_train_only_sj_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_cache(tmp);cache=open_cache(Path(tmp)/'train');cache.sites['current_group']=np.where(np.arange(253)<127,'A','B')
            spec=json.loads(Path('configs/tsmixer/ensemble_sj_features.json').read_text());features=Features(cache,spec)
            f=features.sj(0);self.assertEqual(f['raw'].shape,(253,24,86));self.assertEqual(f['xgb'].shape,(253,24,101))
            self.assertEqual(features.state(0)['raw'].shape,(253,24,45))
            h=History(cache,cache.days[1]+pd.Timedelta(hours=17))
            stats=fit_stats(h,features,'sj_ensemble')
            self.assertEqual(stats['fit_cutoff'],str(h.cutoff));self.assertEqual(len(stats['numeric']),80)
            ds=NeuralDays(h,features,stats,'sj_ensemble',range(2),True)
            b=ds[0];self.assertTrue(torch.isfinite(b['numeric']).all());self.assertEqual(b['categorical'].shape,(24,6))
            model=SJGRU(80,[len(v)+1 for v in stats['levels']],[8,4,4,2,2,2])
            out=model({k:b[k][None] for k in ('numeric','missing','categorical')});out.sum().backward()
            self.assertEqual(out.shape,(1,24))
    def test_epoch_checkpoint_resume_bounded_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as tmp,patch('analysis.tsmixer.ensemble.trainer.result_guard'):
            root=Path(tmp);synthetic_cache(root/'cache');cache=open_cache(root/'cache/train');cache.sites['current_group']=np.where(np.arange(253)<127,'A','B')
            h=History(cache,cache.days[1]+pd.Timedelta(hours=17));config=load_config('configs/tsmixer/year_backtest.yaml')
            config['model'].update(hidden_dim=16,blocks=1,time_hidden=8,feature_hidden=24,dropout=0)
            config['training'].update(batch_size=4,epochs=2,cpu_threads=2)
            plan=dict(cutoff=str(h.cutoff),dev_start=str(cache.days[2]),dev_end=str(cache.days[3]))
            model,stats,_=fit_component('tsmixer',config,cache,h,None,None,plan,root/'stage',torch.device('cpu'),epochs=1)
            first={k:v.clone() for k,v in model.state_dict().items()}
            self.assertTrue(is_complete(root/'stage'))
            restored,_,_=fit_component('tsmixer',config,cache,h,None,None,plan,root/'stage',torch.device('cpu'),epochs=1)
            for k,v in first.items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
    def test_corrected_state_aux_fit_and_both_neural_training_paths(self):
        with tempfile.TemporaryDirectory() as tmp,patch('analysis.tsmixer.ensemble.trainer.result_guard'):
            root=Path(tmp);synthetic_cache(root/'cache');cache=open_cache(root/'cache/train')
            cache.sites['current_group']=np.where(np.arange(253)<127,'A','B')
            spec=json.loads(Path('configs/tsmixer/ensemble_sj_features.json').read_text());features=Features(cache,spec)
            h=History(cache,cache.days[1]+pd.Timedelta(hours=17))
            times=pd.DatetimeIndex([t for d in cache.days for t in day_hours(d)])
            obs={}
            for provider in ('wa','wb'):
                stations=sorted(cache.sites[provider+'_station'].unique())
                index=pd.MultiIndex.from_product([stations,times],names=[provider+'_station','ts'])
                obs[provider]=pd.DataFrame({n:np.full(len(index),.1,dtype='float32') for n in FC if n.startswith(provider+'_') and n!='wb_precip_prob'},index=index)
            before=fit_stats(h,features,'state01',obs=obs)
            modified={p:f.copy() for p,f in obs.items()}
            for p in modified:modified[p].loc[modified[p].index.get_level_values('ts')>h.cutoff]=.3
            self.assertEqual(before,fit_stats(h,features,'state01',obs=modified))
            ds=NeuralDays(h,features,before,'state01',range(2),True,obs=obs)
            self.assertGreater(int(ds[0]['aux_valid'].sum()),0)
            # Finite standardized zero remains a valid auxiliary observation.
            self.assertTrue(torch.isfinite(ds[0]['x']).all())
            cfg=load_config('configs/tsmixer/year_backtest.yaml');cfg['state_width']=8
            cfg['training'].update(batch_size=128,state_batch_days=1,epochs=2)
            plan=dict(cutoff=str(h.cutoff),dev_start=str(cache.days[2]),dev_end=str(cache.days[3]))
            for component in ('sj_bigru','cloud','all'):
                model,stats,epoch=fit_component(component,cfg,cache,h,None,features,plan,root/component,torch.device('cpu'),obs=obs,epochs=1)
                self.assertEqual(epoch,1);self.assertTrue(is_complete(root/component))
            # Teacher feature builder consumes forecasts only and has the pinned schema.
            from analysis.tsmixer.ensemble.teacher import daily_inputs
            daily=daily_inputs(features)
            self.assertTrue(set(spec['weather_columns'])<=set(daily))

    def test_temporal_import_score_freeze_apply_fixture(self):
        from analysis.tsmixer.ensemble.run import import_prediction,score_fold,select_weights,apply_weights
        from analysis.tsmixer.comparison.core import Grid
        from analysis.validation import hourly_weights
        base,_=fixture()
        grids={}
        for name,f in FOLDS.items():
            times=f.times[9:12]
            grids[name]=Grid(base.sites,times,base.actual.copy(),base.smp.copy(),hourly_weights(times))
        with tempfile.TemporaryDirectory() as tmp,patch('analysis.tsmixer.ensemble.persistence.result_guard'),patch('analysis.tsmixer.ensemble.run.result_guard'):
            root=Path(tmp);c=dict(run_root=str(root/'run'),data_root='fixture',sj_group=str(root/'missing.csv'),grid_step=.1,joint_site_weight=.8,joint_penalty_weight=.2)
            def data(_,fold):return grids[fold.name],np.ones(253)*100,{}
            with patch('analysis.tsmixer.ensemble.run.actual_grid',side_effect=data):
                for name,f in FOLDS.items():
                    for model,value in [('tsmixer',90.),('sj_ensemble',110.),('state01',120.)]:
                        incoming=root/(name+'_'+model+'.parquet')
                        meta=dict(model=model,fold=name,start=str(f.start),end_exclusive=str(f.end),smoke_only=False,protocol='past_only_walkforward_v1',inference_inputs=['forecasts'],
                            **{stage+'_cutoff':str(f.cutoff) for stage in ('training','normalization','selection')},auxiliary_cutoff=None,outage_cutoff=None,calibration_cutoff=None)
                        g=grids[name];save_prediction(incoming,np.full_like(g.actual,value),g.times,g.sites,meta)
                        import_prediction(c,f,model,incoming,incoming.with_suffix('.json'))
                        import_prediction(c,f,model,incoming,incoming.with_suffix('.json'),True)
                one=score_fold(c,FOLDS['F1'],root/'score1')
                two=score_fold(c,FOLDS['F2'],root/'score2')
                self.assertEqual(len(one['trials']),67)
                f1=root/'f1.json';select_weights(c,[root/'score1/metrics.json'],f1,'site')
                fixed_hash=file_sha(f1)
                result=apply_weights(c,FOLDS['F2'],f1,root/'apply2')
                self.assertEqual(result['status'],'computed');self.assertEqual(file_sha(f1),fixed_hash)
                self.assertEqual(result['result']['site']['all']['site_nmae_pct'],0)
                with self.assertRaises(ValueError):apply_weights(c,FOLDS['F1'],f1,root/'bad')
                final=root/'f12.json';select_weights(c,[root/'score1/metrics.json',root/'score2/metrics.json'],final,'site')
                self.assertEqual(apply_weights(c,FOLDS['F3'],final,root/'apply3')['status'],'computed')
                # F3 never writes a candidate grid for fitting.
                f3=score_fold(c,FOLDS['F3'],root/'score3');self.assertEqual(f3['trials'],[])
                with self.assertRaises(ValueError):select_weights(c,[root/'score3/metrics.json'],root/'badweights.json','site')

    def test_mac_real_training_guard(self):
        from analysis.tsmixer.ensemble.walkforward import train
        with patch('analysis.tsmixer.ensemble.walkforward.platform.system',return_value='Darwin'),patch('analysis.tsmixer.ensemble.walkforward.open_cache') as cache:
            with self.assertRaises(ValueError):train({},FOLDS['F1'],'state01')
            cache.assert_not_called()

if __name__=='__main__':unittest.main()
