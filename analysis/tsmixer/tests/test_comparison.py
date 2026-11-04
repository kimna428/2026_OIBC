import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from analysis.tsmixer.comparison.core import Grid,evaluate,verify_reference,REFERENCE,CUTOFF,select_oof_alpha
from analysis.tsmixer.comparison.io import align_frame,load_tsmixer,load_actual
from analysis.tsmixer.comparison.audit import recover_composition
from analysis.tsmixer.comparison.diagnostics import ensemble_sweep,residual_analysis,group_pair_analysis
from analysis.validation import hourly_weights,validate_composition


def fixture(hours=3):
    sites=pd.DataFrame(dict(site_id=[f'PV{i:03}' for i in range(1,254)],capacity_kw=np.full(253,200.),
                     lat=np.full(253,35.),lon=np.full(253,127.),wa_station='WA001',wb_station='WB001',current_group=['A']*127+['B']*126))
    times=pd.date_range('2024-06-01 10:00',periods=hours,freq='h',tz='Asia/Seoul')
    actual=np.full((hours,253),100.); actual[0,0]=np.nan
    grid=Grid(sites,times,actual,np.full(hours,100.),hourly_weights(times))
    comp=sites[['site_id']].assign(group=['G1']*127+['G2']*126)
    return grid,comp


class ComparisonTests(unittest.TestCase):
    def test_keyed_alignment_order_utc_and_unknown_ids(self):
        grid,comp=fixture()
        frame=pd.DataFrame(dict(site_id=np.tile(grid.sites.site_id,len(grid.times)),ts=grid.times.repeat(253).tz_convert('UTC'),forecast_kwh=100.))
        got=align_frame(frame.sample(frac=1,random_state=42),grid,['forecast_kwh'])['forecast_kwh']
        self.assertTrue((got==100).all())
        for bad in [pd.concat([frame,frame.iloc[:1]]),frame.assign(site_id='UNKNOWN'),frame.assign(ts=frame.ts.dt.tz_localize(None))]:
            with self.assertRaises(ValueError):align_frame(bad,grid,['forecast_kwh'])
        missing=frame.loc[~((frame.site_id=='PV002')&(frame.ts==frame.ts.iloc[0]))]
        with self.assertRaises(ValueError):align_frame(missing,grid,['forecast_kwh'])
        # Missing forecast at a missing target is allowed; never changes actual mask.
        missing=frame.loc[~((frame.site_id=='PV001')&(frame.ts==frame.ts.iloc[0]))]
        self.assertTrue(np.isnan(align_frame(missing,grid,['forecast_kwh'])['forecast_kwh'][0,0]))

    def test_missing_targets_penalty_money_and_group_invariance(self):
        grid,comp=fixture()
        prediction=np.full_like(grid.actual,110.)
        result,ledger=evaluate(grid,prediction,comp)
        self.assertEqual(result['observed_rows'],3*253-1)
        self.assertEqual(result['valid_hours'],3)
        self.assertAlmostEqual(result['site_nmae_pct'],5.)
        expected=float((.1*100*grid.k*grid.observed.sum(axis=1)*10).sum())
        self.assertAlmostEqual(result['W'],expected)
        revenue=float(np.nansum(grid.actual*100))
        self.assertAlmostEqual(result['S'],revenue-expected)
        self.assertAlmostEqual(result['internal_score'],100*(1-expected/REFERENCE))
        self.assertEqual(result['internal_score_status'],'provisional')
        changed=comp.copy();changed.loc[[0,127],'group']=['G2','G1']
        r2,_=evaluate(grid,prediction,changed)
        self.assertEqual(result['site_nmae_pct'],r2['site_nmae_pct'])
        self.assertNotEqual(result['contract']['composition_sha256'],r2['contract']['composition_sha256'])
        complete,_=evaluate(grid,prediction,comp,'complete_fleet')
        self.assertEqual(complete['valid_hours'],2)
        self.assertEqual(complete['observed_rows'],506)
        with self.assertRaises(ValueError):Grid(grid.sites,grid.times,grid.actual,grid.smp,np.ones(3))
        for invalid in [comp.iloc[:-1],pd.concat([comp,comp.iloc[:1]]),comp.assign(group='G1')]:
            with self.assertRaises(ValueError):evaluate(grid,prediction,invalid)
        with self.assertRaises(ValueError):validate_composition(comp,grid.sites.assign(capacity_kw=1.))

    def test_group_ratio_of_sums_is_not_macro_mean_and_cancellation(self):
        grid,comp=fixture(1)
        grid.actual[0,3]=np.nan
        prediction=np.full_like(grid.actual,100.)
        prediction[0,1]=120.; prediction[0,2]=80.
        result,_=evaluate(grid,prediction,comp)
        self.assertAlmostEqual(result['W'],0.)
        self.assertAlmostEqual(result['group_cancellation_fraction'],1.)
        prediction[0,2]=100.
        result,_=evaluate(grid,prediction,comp)
        self.assertNotEqual(result['group_nmae_pct'],result['group_nmae_macro_pct'])
        # Empty targets produce unavailable settlement, not a perfect score.
        empty=Grid(grid.sites,grid.times,np.full_like(grid.actual,np.nan),grid.smp,grid.k)
        r,_=evaluate(empty,prediction,comp)
        self.assertIsNone(r['W']);self.assertIsNone(r['internal_score'])

    def test_reference_recomputation_and_contract_prices(self):
        grid,comp=fixture()
        baseline=grid.actual.copy()
        n=grid.observed.sum(axis=1)
        delta=REFERENCE/float((.1*grid.smp*grid.k*n).sum())
        baseline+=delta
        self.assertTrue(verify_reference(grid,baseline,comp)['verified'])
        changed=Grid(grid.sites,grid.times,grid.actual,grid.smp*2,grid.k)
        self.assertFalse(verify_reference(changed,baseline,comp)['verified'])
        self.assertNotEqual(grid.contract(comp,'observed_only')['smp_sha256'],changed.contract(comp,'observed_only')['smp_sha256'])
        self.assertEqual(evaluate(grid,baseline,comp)[0],evaluate(grid,baseline,comp)[0])

    def test_tsmixer_axis_and_capacity_mapping(self):
        grid,_=fixture(24)
        grid.times=pd.date_range('2024-06-01 01:00',periods=24,freq='h',tz='Asia/Seoul')
        grid.k=hourly_weights(grid.times)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=grid.sites.site_id.tolist()[::-1]
            axis=dict(site_ids=ids,operating_days=['2024-06-01 00:00:00+09:00'],units='capacity_factor',hours='01:00 through next-day 00:00 KST')
            (root/'prediction_axis.json').write_text(json.dumps(axis))
            raw=np.broadcast_to(np.arange(253,dtype='float32')[None,:,None]/1000,(1,253,24)).copy()
            np.save(root/'predictions.npy',raw)
            got,meta=load_tsmixer(root,grid)
            np.testing.assert_allclose(got[0],np.arange(253)[::-1]*.2,rtol=1e-6)
            self.assertFalse(meta['cutoff_verified'])
            (root/'run.json').write_text(json.dumps(dict(plans=dict(refit=dict(cutoff=str(CUTOFF))))))
            self.assertTrue(load_tsmixer(root,grid)[1]['cutoff_verified'])
            axis['operating_days']=['2024-06-01 01:00:00+09:00']
            (root/'prediction_axis.json').write_text(json.dumps(axis))
            with self.assertRaises(ValueError):load_tsmixer(root,grid)

    def test_oof_rejects_evaluation_targets_and_late_training(self):
        grid,comp=fixture(1)
        frame=pd.DataFrame(dict(site_id=grid.sites.site_id,ts=pd.Timestamp('2024-03-01 11:00',tz='Asia/Seoul'),energy=100.,
                     teammate=120.,tsmixer=100.,teammate_trained_until='2024-02-28 17:00:00+09:00',tsmixer_trained_until='2024-02-28 17:00:00+09:00',smp=100.,k=3))
        selection=select_oof_alpha(frame,grid.sites,comp)
        self.assertEqual(selection['selected_alpha'],.5)
        for bad in [frame.assign(ts=pd.Timestamp('2024-06-01 11:00',tz='Asia/Seoul')),
                    frame.assign(teammate_trained_until='2024-03-01 10:00:00+09:00'),
                    frame.assign(tsmixer_trained_until='2024-03-01 10:00:00+09:00')]:
            with self.assertRaises(ValueError):select_oof_alpha(bad,grid.sites,comp)
        sweep=ensemble_sweep(grid,dict(tsmixer=np.full_like(grid.actual,100.),teammate_ensemble=np.full_like(grid.actual,120.)),dict(teammate=comp),oof=selection)
        self.assertEqual(len(sweep['trials']),8)
        self.assertFalse(sweep['independent_final_evaluation'])
        self.assertEqual(sweep['oof_selected_evaluation'][0]['alpha'],.5)

    def test_recover_sources_must_agree(self):
        grid,comp=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repro=root/'repro';train=root/'train';repro.mkdir();train.mkdir()
            (train/'config.json').write_text(json.dumps(dict(composition=dict(zip(comp.site_id,comp.group)))))
            md=root/'Clustering_sj.md'
            md.write_text('\n'.join(['## G1 전체 발전소 목록']+[f'| {sid} |' for sid in comp.loc[comp.group=='G1','site_id']]+['## G2 전체 발전소 목록']+[f'| {sid} |' for sid in comp.loc[comp.group=='G2','site_id']]))
            restored,evidence=recover_composition(repro,train,md,grid.sites)
            self.assertTrue(evidence['all_memberships_match'])
            self.assertEqual(len(restored),253)
            (repro/'reference').mkdir();bad=comp.copy();bad.loc[[0,127],'group']=['G2','G1'];bad.to_csv(repro/'reference/composition.csv',index=False)
            with self.assertRaises(ValueError):recover_composition(repro,train,md,grid.sites)

    def test_history_cohort_excludes_future_targets(self):
        grid,_=fixture(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train').mkdir();grid.sites.to_csv(root/'sites.csv',index=False)
            # One old target, one new target: future target cannot make PV002 seen.
            gen=pd.DataFrame(dict(site_id=['PV001','PV002'],ts=pd.to_datetime(['2024-05-31 17:00:00+09:00','2024-06-01 10:00:00+09:00']),energy=[100.,100.]))
            gen.to_parquet(root/'train/generation.parquet',index=False)
            times=pd.date_range('2024-06-01 01:00','2025-06-01 00:00',freq='h',tz='Asia/Seoul')
            pd.DataFrame(dict(ts=times,smp=100.)).to_csv(root/'train/smp.csv',index=False)
            _,seen,_=load_actual(root)
            self.assertEqual(seen[0],1);self.assertEqual(seen[1],0)

    def test_diagnostics_same_rows_and_cases(self):
        grid,comp=fixture()
        models=dict(tsmixer=np.full_like(grid.actual,101.),teammate_ensemble=np.full_like(grid.actual,110.))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=residual_analysis(grid,models,np.zeros(253),root)
            self.assertEqual(info['pairs'][0]['observed_rows'],3*253-1)
            ledgers={name:evaluate(grid,pred,comp)[1] for name,pred in models.items()}
            group_pair_analysis(dict(teammate=ledgers),root)
            self.assertTrue((root/'site_errors.csv').is_file())
            self.assertEqual(len(list(root.glob('cases_*.csv'))),2)


if __name__=='__main__':unittest.main()
