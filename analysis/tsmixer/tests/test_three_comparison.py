"""Fixture-only protocol/adapter/convex-ensemble checks. No teammate execution."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.tsmixer.tests.test_comparison import fixture
from analysis.tsmixer.comparison.core import START,END,CUTOFF,evaluate,mapping_hash
from analysis.tsmixer.comparison.io import file_sha
from analysis.tsmixer.comparison.providers import (
    load_state01_table,load_state01_run,load_sj,validate_protocol,STATE_PARTS,FIT_STAGES)
from analysis.tsmixer.comparison.multi_ensemble import weight_grid,mix,sweep,select_oof,MODELS,validate_oof_group
from analysis.tsmixer.comparison.state01_audit import parameter_counts,inventory
from analysis.tsmixer.comparison.three_diagnostics import complementarity


def metadata(path=None):
    result=dict(profile='year_validation',protocol='cutoff_safe_v1',timezone='Asia/Seoul',
        timestamp_convention='interval_end',units='kWh',prediction_column='kwh',
        prediction_start=str(START),prediction_end_exclusive=str(END),
        inference_inputs=['forecasts','solar_geometry','calendar','site_metadata'],
        leakage_controls=dict(outages_fit_before_cutoff=True,aux_bounds_fit_before_cutoff=True,no_target_inputs=True),
        **{k:str(CUTOFF) for k in FIT_STAGES})
    if path:result['prediction_sha256']=file_sha(path)
    return result


def frame_for(grid,value=110.):
    return pd.DataFrame(dict(site_id=np.tile(grid.sites.site_id,len(grid.times)),
        ts=grid.times.repeat(253).tz_convert('UTC'),kwh=value))


class ThreeComparisonTests(unittest.TestCase):
    def test_state_adapter_units_order_mask_and_hash(self):
        grid,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'pred.parquet';meta=root/'meta.json'
            frame=frame_for(grid).sample(frac=1,random_state=1)
            frame.to_parquet(path,index=False);record=metadata(path);meta.write_text(json.dumps(record))
            values,evidence=load_state01_table(path,grid,meta)
            self.assertTrue((values['state01']==110.).all())
            self.assertFalse(evidence['protocol']['independently_audited'])
            record['units']='capacity_factor';meta.write_text(json.dumps(record))
            values,_=load_state01_table(path,grid,meta)
            self.assertTrue((values['state01']==22000.).all())
            # Dropping even one observed target is forbidden, never inner-join away.
            frame=frame_for(grid).iloc[1:];frame.to_parquet(path,index=False)
            record=metadata(path);meta.write_text(json.dumps(record))
            # First target is missing, so that one forecast may be absent.
            self.assertTrue(np.isnan(load_state01_table(path,grid,meta)[0]['state01'][0,0]))
            frame=frame.iloc[1:];frame.to_parquet(path,index=False);record=metadata(path);meta.write_text(json.dumps(record))
            with self.assertRaises(ValueError):load_state01_table(path,grid,meta)
            frame_for(grid).to_parquet(path,index=False)
            # Stale prediction binding fails before any scoring.
            with self.assertRaises(ValueError):load_state01_table(path,grid,meta)

    def test_reject_original_late_cutoff_final_oracle_and_global_stats(self):
        record=metadata()
        for key in FIT_STAGES:
            invalid={**record,key:'2024-06-01 00:00:00+09:00'}
            with self.assertRaisesRegex(ValueError,key):validate_protocol(invalid)
        for change in [dict(profile='final'),dict(training_cutoff='2025-05-31 17:00:00+09:00'),
                       dict(protocol='original_main_all'),dict(inference_inputs=['observed_weather']),
                       dict(inference_inputs=['past_generation']),dict(timezone='UTC'),
                       dict(timestamp_convention='interval_start'),dict(units='MW'),
                       dict(prediction_start='2024-06-01 00:00:00+09:00')]:
            with self.assertRaises(ValueError):validate_protocol({**record,**change})
        for key in record['leakage_controls']:
            with self.assertRaises(ValueError):validate_protocol({**record,'leakage_controls':{**record['leakage_controls'],key:False}})
        self.assertEqual(validate_protocol(record)['status'],'declared_protocol_bounds_checked')

    def test_native_components_equal_weight_seeds_no_fill(self):
        grid,_=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=metadata();record['seeds']=[0,1];record['prediction_files']={}
            expected={}
            for j,(name,part) in enumerate(STATE_PARTS.items()):
                directory='gbdt' if name in ('state01_lgb','state01_xgb') else 'gru'
                expected[name]=100+j+1 # mean of 100+j and 102+j
                for seed in (0,1):
                    relative=f'main_all/{directory}/pred_{part}_s{seed}.parquet'
                    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    frame_for(grid,100+j+2*seed).to_parquet(path,index=False)
                    record['prediction_files'][relative]=file_sha(path)
            meta=root/'metadata.json';meta.write_text(json.dumps(record))
            values,_=load_state01_run(root,grid,meta)
            for name,value in expected.items():self.assertTrue((values[name]==value).all())
            self.assertTrue(np.allclose(values['state01'],.1*expected['state01_lgb']+.1*expected['state01_xgb']+.4*expected['state01_cloud']+.4*expected['state01_all']))
            missing=next(iter(record['prediction_files']));(root/missing).unlink()
            with self.assertRaises(ValueError):load_state01_run(root,grid,meta)

    def test_sj_names_validation_factor_and_standalone_table(self):
        grid,_=fixture()
        cfg=dict(profiles=dict(validation=dict(cutoff=str(CUTOFF),start=str(START),end_exclusive=str(END),factor=.96)),
                 weights=dict(lightgbm=.25,xgboost=.25,bigru=.5))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sj.parquet';frame=frame_for(grid)
            frame['lightgbm']=100.;frame['xgboost']=110.;frame['bigru']=120.;frame['forecast_kwh']=108.
            frame.to_parquet(path,index=False)
            values,_=load_sj(path,grid,cfg)
            self.assertEqual(set(values),{'sj_lightgbm','sj_xgboost','sj_bigru','sj_ensemble'})
            self.assertTrue((values['sj_ensemble']==108.).all())
            frame['forecast_kwh']=109.;frame.to_parquet(path,index=False)
            with self.assertRaises(ValueError):load_sj(path,grid,cfg)
            frame=frame_for(grid);frame['forecast_kwh']=105.;frame=frame.drop(columns='kwh');frame.to_parquet(path,index=False)
            self.assertEqual(set(load_sj(path,grid,cfg)[0]),{'sj_ensemble'})

    def test_grid_vertices_edges_interior_and_invalid_weights(self):
        plans=weight_grid(.1)
        self.assertEqual(len(plans),66);self.assertEqual(len(weight_grid(.05)),231)
        self.assertEqual(sum(sum(w>0 for w in weights.values())==1 for weights in plans),3)
        self.assertEqual(sum(sum(w>0 for w in weights.values())==2 for weights in plans),27)
        self.assertEqual(sum(sum(w>0 for w in weights.values())==3 for weights in plans),36)
        self.assertTrue(all(np.isclose(sum(w.values()),1) and min(w.values())>=0 for w in plans))
        predictions=dict(tsmixer=np.ones((2,253)))
        np.testing.assert_array_equal(mix(predictions,dict(tsmixer=1,sj_ensemble=0,state01=0)),predictions['tsmixer'])
        for weights in [dict(tsmixer=.7,state01=.3),dict(tsmixer=1.1,state01=-.1),dict(tsmixer=.9),dict(tsmixer=float('nan'))]:
            with self.assertRaises(ValueError):mix(predictions,weights)
        with self.assertRaises(ValueError):weight_grid(.2)

    def test_sweep_same_contract_and_oof_future_guard(self):
        grid,comp=fixture()
        predictions={name:np.full_like(grid.actual,value) for name,value in zip(MODELS,[90.,110.,120.])}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            ledgers={name:evaluate(grid,p,comp)[1] for name,p in predictions.items()}
            detail=complementarity(grid,predictions,dict(sj_final=ledgers),root)
            self.assertEqual(detail['site_pair_rows'],3*253)
            self.assertEqual(detail['group_pair_rows'],6)
            g=pd.read_csv(root/'group_complementarity.csv')
            pair=g[(g.model_a=='tsmixer')&(g.model_b=='sj_ensemble')]
            self.assertTrue((pair.W_half_exploratory==0).all())
            self.assertTrue((pair.opposite_sign_fraction==1).all())
        result=sweep(grid,predictions,dict(sj_final=comp))
        self.assertEqual(len(result['trials']),66);self.assertFalse(result['weight_selection_performed'])
        for trial in result['trials']:
            if trial['kind']=='single':
                name=next(n for n,w in trial['weights'].items() if w>0)
                direct,_=evaluate(grid,predictions[name],comp)
                self.assertEqual(trial['W'],direct['W'])
            self.assertEqual(trial['observed_rows'],grid.observed.sum())
        partial=sweep(grid,{'tsmixer':predictions['tsmixer']},dict(sj_final=comp))
        self.assertEqual(len(partial['trials']),1);self.assertEqual(len(partial['unavailable']),65)
        frame=pd.DataFrame(dict(site_id=grid.sites.site_id,ts=pd.Timestamp('2024-03-01 11:00',tz='Asia/Seoul'),
                  energy=100.,smp=100.,k=3,tsmixer=100.,sj_ensemble=120.,state01=130.,
                  **{f'{name}_trained_until':'2024-02-28 17:00:00+09:00' for name in MODELS}))
        self.assertIn('status',validate_oof_group(frame,comp,dict(basis='metadata_only',composition_sha256=mapping_hash(comp))))
        with self.assertRaises(ValueError):validate_oof_group(frame,comp,dict(basis='historical_residuals',cutoff=str(CUTOFF),composition_sha256=mapping_hash(comp)))
        selected=select_oof(frame,grid.sites,comp)
        self.assertEqual(selected['selected_weights']['tsmixer'],1.)
        for name in MODELS:
            with self.assertRaises(ValueError):select_oof(frame.assign(**{f'{name}_trained_until':'2024-03-01 11:00:00+09:00'}),grid.sites,comp)
        with self.assertRaises(ValueError):select_oof(frame.assign(ts=pd.Timestamp('2024-06-01 11:00',tz='Asia/Seoul')),grid.sites,comp)

    def test_analytic_parameter_count_matches_independent_torch_modules(self):
        import torch.nn as nn
        for n in (2,14):
            layers=nn.ModuleList([nn.Linear(45,192),nn.GRU(192,192,num_layers=2,bidirectional=True,dropout=.1),
                       nn.LayerNorm(384),nn.Linear(384,1),nn.LayerNorm(384),nn.Linear(384,n)])
            count=sum(p.numel() for p in layers.parameters())
            self.assertEqual(count,parameter_counts()[f'aux_{n}']['parameters'])

    def test_three_cli_fixture_and_no_overwrite(self):
        from analysis.tsmixer.comparison.three import main
        grid,comp=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state';state.mkdir();sj=root/'sj';sj.mkdir();ts=root/'ts';ts.mkdir();inbox=root/'inbox';inbox.mkdir()
            cfg=dict(profiles=dict(validation=dict(cutoff=str(CUTOFF),start=str(START),end_exclusive=str(END),factor=1.)),weights=dict(lightgbm=.25,xgboost=.25,bigru=.5))
            (sj/'config.json').write_text(json.dumps(cfg));composition=root/'comp.csv';comp.to_csv(composition,index=False)
            # Mark presence; loading is replaced by a bounded fixture adapter.
            (ts/'predictions.npy').write_bytes(b'fixture');(ts/'prediction_axis.json').write_text('{}')
            frame=frame_for(grid);frame['forecast_kwh']=105.;frame.to_parquet(inbox/'sj_validation.parquet',index=False)
            frame_for(grid).to_parquet(inbox/'state01_validation.parquet',index=False)
            (inbox/'state01_validation.json').write_text(json.dumps(metadata(inbox/'state01_validation.parquet')))
            argv=['compare','--state01-root',str(state),'--sj-package',str(sj),'--sj-composition',str(composition),
                  '--tsmixer-run',str(ts),'--inbox',str(inbox),'--output','runs/tsmixer/comparison/fixture']
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch('sys.argv',argv),patch('analysis.tsmixer.comparison.three.result_guard'),\
                     patch('analysis.tsmixer.comparison.three.load_actual',return_value=(grid,np.zeros(253),{})),\
                     patch('analysis.tsmixer.comparison.three.audit_state01',return_value=inventory(state)),\
                     patch('analysis.tsmixer.comparison.three.load_tsmixer',return_value=(np.full_like(grid.actual,100.),dict(cutoff_verified=True))),contextlib.redirect_stdout(io.StringIO()):
                    main()
                    with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):main()
                result=json.loads(Path('runs/tsmixer/comparison/fixture/metrics.json').read_text())
                self.assertTrue(all(result['availability'][name]['status']=='available' for name in MODELS))
                self.assertFalse(result['reference']['verified'])
                self.assertEqual(result['results']['tsmixer']['sj_final/observed_only']['observed_rows'],grid.observed.sum())
                # Explicit nonexistent provenance must not silently downgrade to config-only.
                missing_argv=argv[:-1]+['runs/tsmixer/comparison/missing_meta','--sj-metadata',str(root/'missing.json')]
                with patch('sys.argv',missing_argv),patch('analysis.tsmixer.comparison.three.result_guard'),\
                     patch('analysis.tsmixer.comparison.three.load_actual',return_value=(grid,np.zeros(253),{})),\
                     patch('analysis.tsmixer.comparison.three.audit_state01',return_value=inventory(state)),\
                     patch('analysis.tsmixer.comparison.three.load_tsmixer',return_value=(np.full_like(grid.actual,100.),dict(cutoff_verified=True))),contextlib.redirect_stdout(io.StringIO()):
                    main()
                missing=json.loads(Path('runs/tsmixer/comparison/missing_meta/metrics.json').read_text())
                self.assertEqual(missing['availability']['sj_ensemble']['status'],'ineligible_or_invalid')
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()
