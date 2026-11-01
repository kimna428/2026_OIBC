import unittest
import json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from third_experiment import validation_mask,stamp,split_indices,fit_indices,select_development,interval_elevation,postprocess,fit_night_thresholds,optimize_groups,latest_history_source,parameter_schedule,freeze_preferred_candidate,frozen_history_features
from expanding_pipeline import issuance

class ThirdExperimentTests(unittest.TestCase):
    def test_parameter_selection_respects_issue_clock(self):
        rows=[]
        losses=[[.1,.2,.3,.4],[.05,.1,.2,.3],[.2,.01,.01,.001]]
        for family in ['B','C']:
            for grid in range(3):
                for fold in range(4): rows.append(dict(family=family,fold=fold,candidate=f'tune_{family}_{grid}',cf_mae=losses[grid][fold]))
        d=pd.DataFrame(rows); schedule=parameter_schedule(d)
        self.assertEqual(schedule['B']['0']['completed_tuning_folds'],[])
        self.assertEqual(schedule['B']['1']['params']['n_estimators'],120)
        self.assertEqual(schedule['B']['4']['completed_tuning_folds'],[0,1,2])
        self.assertEqual(schedule['B']['4']['parameter_source_max'],str(stamp('2024-09-01')))
        d.loc[d.fold==3,'cf_mae']=999
        self.assertEqual(parameter_schedule(d)['B']['4'],schedule['B']['4'])
    def test_parameter_schedule_rejects_holdout(self):
        with self.assertRaises(ValueError): parameter_schedule(pd.DataFrame({'fold':[5]}))
    def test_preferred_candidate_is_locked_before_holdout(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder); (out/'predictions').mkdir()
            lock={'chosen':{'B':{'candidate':'B_x','policy':'raw'},'C':{'candidate':'C_x','policy':'raw'}},'selection_source_max':'2025-02-28 17:00:00+09:00'}
            (out/'selection_lock.json').write_text(json.dumps(lock))
            pd.DataFrame([dict(fold=4,candidate=f+'_x',policy='raw',group='total',score_available=True,penalty_krw=v) for f,v in [('B',10.),('C',20.)]]).to_csv(out/'scores_selection.csv',index=False)
            result=freeze_preferred_candidate(out)
            self.assertEqual(result['preferred']['family'],'B')
            (out/'preferred_candidate_lock.json').unlink()
            (out/'predictions'/'B_x_fold5.parquet').touch()
            with self.assertRaises(ValueError): freeze_preferred_candidate(out)
    def test_frozen_history_cannot_read_later_arrivals(self):
        g=pd.DataFrame({'site_id':['PV001']*3,'ts':pd.DatetimeIndex(['2025-02-20 12:00','2025-02-28 12:00','2025-03-01 12:00'],tz='Asia/Seoul'),'cf':[.1,.2,.9]})
        target=pd.DataFrame({'site_id':['PV001'],'ts':[stamp('2025-03-02 12:00')]})
        h=frozen_history_features(g,target,stamp('2025-02-20 17:00'))
        self.assertTrue(h.lag48_cf.isna().all()); self.assertTrue(h.history7d_cf.isna().all())
        self.assertEqual(h.history7d_count.iloc[0],0)
        g.loc[g.ts>stamp('2025-02-20 17:00'),'cf']=999
        pd.testing.assert_frame_equal(h,frozen_history_features(g,target,stamp('2025-02-20 17:00')))
    def test_history_mixed_timestamp_resolution(self):
        g=pd.DataFrame({'site_id':['PV001']*3,'ts':pd.DatetimeIndex(['2025-02-20 12:00','2025-02-28 12:00','2025-03-01 12:00'],tz='Asia/Seoul').as_unit('us'),'cf':[.1,.2,.9]})
        target=pd.DataFrame({'site_id':['PV001'],'ts':[stamp('2025-03-02 12:00')]})
        h=frozen_history_features(g,target,stamp('2025-03-01 17:00'))
        self.assertAlmostEqual(h.lag48_cf.iloc[0],.2)
        self.assertAlmostEqual(h.history7d_cf.iloc[0],.55)
        self.assertEqual(h.history7d_count.iloc[0],2)
    def test_operating_boundaries(self):
        t=pd.DatetimeIndex(['2025-03-01 00:00','2025-03-01 01:00','2025-06-01 00:00','2025-06-01 01:00'],tz='Asia/Seoul')
        np.testing.assert_array_equal(validation_mask(t,'2025-03-01','2025-06-01'),[False,True,True,False])
    def test_midnight_issue_is_previous_operating_day(self):
        t=pd.DatetimeIndex(['2025-03-02 00:00'],tz='Asia/Seoul')
        self.assertEqual(issuance(t)[0],stamp('2025-02-28 17:00'))
    def test_split_strict_issuance(self):
        g=pd.DataFrame({'ts':pd.DatetimeIndex(['2025-02-28 17:00','2025-02-28 18:00','2025-03-01 00:00','2025-03-01 01:00','2025-06-01 00:00'],tz='Asia/Seoul')})
        tr,va,cut=split_indices(g,5)
        np.testing.assert_array_equal(tr,[0]); np.testing.assert_array_equal(va,[3,4])
        self.assertEqual(cut,stamp('2025-02-28 17:00'))
    def test_sample_is_train_only_and_excludes_anomalies(self):
        g=pd.DataFrame({'cf':[0.,-.1,.2,1.1,.3,.4]})
        np.testing.assert_array_equal(fit_indices(g,np.array([0,1,2,3]),0),[0,2])
    def test_holdout_selection_rejected(self):
        d=pd.DataFrame({'fold':[0,5],'candidate':['a','b'],'cf_mae':[.1,0.]})
        with self.assertRaises(ValueError): select_development(d)
    def test_selection_does_not_need_holdout(self):
        d=pd.DataFrame({'fold':[0,1,0,1],'candidate':['a','a','b','b'],'cf_mae':[.2,.1,.5,.7]})
        self.assertEqual(select_development(d),'a')
    def test_twilight_not_clipped_by_night_only(self):
        np.testing.assert_array_equal(postprocess([.2,.2,.2],[-7,-3,4],[-6,-6,-6],'night_only'),[0,.2,.2])
    def test_calibration_policy_uses_supplied_thresholds(self):
        np.testing.assert_array_equal(postprocess([.1,.1],[-5,-5],[-6,-3],'train_calibrated'),[.1,0])
    def test_partial_sun_interval_preserved(self):
        t=pd.DatetimeIndex(['2025-03-20 07:00'],tz='Asia/Seoul')
        a=interval_elevation(t,[36],[128]); self.assertGreater(a[0],0)
        self.assertEqual(postprocess([.1],a,[-6],'night_only')[0],.1)
    def test_calibration_future_target_rejected(self):
        train=pd.DataFrame({'site_id':['PV001'],'ts':[stamp('2025-03-01 18:00')],'cf':[0.]})
        with self.assertRaises(ValueError): fit_night_thresholds(train,[-10],stamp('2025-03-01 17:00'))
    def test_group_capacity_and_monotonic_penalty(self):
        residual=np.array([[1.,1.,-1.,-1.],[2.,2.,-2.,-2.]])
        mask,trace=optimize_groups(residual,np.ones(2),np.repeat(11000.,4),[True,True,False,False],200)
        self.assertGreaterEqual(11000*sum(mask),20000); self.assertGreaterEqual(11000*sum(~mask),20000)
        self.assertTrue(all(b['penalty_krw']<=a['penalty_krw'] for a,b in zip(trace,trace[1:])))
    def test_missing_group_target_rejected(self):
        with self.assertRaises(ValueError): optimize_groups(np.array([[np.nan,0,1,2]]),[1],np.repeat(11000.,4),[True,True,False,False])
    def test_latest_history_is_not_future(self):
        g=pd.DataFrame({'site_id':['PV001']*3,'ts':pd.DatetimeIndex(['2025-03-01 16:00','2025-03-01 18:00','2025-03-02 12:00'],tz='Asia/Seoul')})
        target=pd.DataFrame({'site_id':['PV001'],'ts':[stamp('2025-03-02 12:00')]})
        self.assertEqual(latest_history_source(g,target)[0],stamp('2025-03-01 16:00'))
if __name__=='__main__': unittest.main()
