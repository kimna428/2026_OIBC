"""Synthetic fixtures test formulas only; never reported as competition scores."""
import unittest
import numpy as np
import pandas as pd
from validation import (TZ, TRAIN_END, DEV_END, HOLDOUT_END, local_hours, split_labels,
    hourly_weights, validate_composition, build_windows, EvaluationWindow,
    score_groups, aggregate_site_predictions, evaluate_sites)

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.sites=pd.DataFrame({'site_id':[f'PV{i:03}' for i in range(1,254)],'capacity_kw':200.})
        self.comp=self.sites[['site_id']].assign(group=['G1']*126+['G2']*127)
        self.ts=pd.date_range('2025-01-01 10:00',periods=7,freq='h',tz=TZ)
        w=pd.DataFrame(1.,index=self.ts,columns=self.sites.site_id)
        detail=pd.DataFrame({'included':True},index=self.ts)
        self.window=EvaluationWindow('dev',w,pd.Series(100.,index=self.ts),detail)
        self.p=w.rename_axis(index='ts',columns='site_id').stack().rename('pred_kwh').reset_index()
    def group_pred(self):
        return aggregate_site_predictions(self.p,self.comp,self.sites,self.window)
    def test_perfect_and_units(self):
        m,total,ledger=score_groups(self.group_pred(),self.comp,self.sites,self.window)
        self.assertEqual(total['penalty_krw'],0)
        self.assertEqual(total['mae_kwh'],0)
        self.assertEqual(total['score_krw'],253*7*100)
        self.assertEqual(len(ledger),14)
    def test_hand_calculated_penalty(self):
        p=self.group_pred(); p.pred_kwh+=2
        _,t,_=score_groups(p,self.comp,self.sites,self.window)
        self.assertEqual(t['penalty_krw'],.1*100*2*(1+3*5+1)*2)
        self.assertEqual(t['mae_kwh'],2); self.assertEqual(t['rmse_kwh'],2)
    def test_partition_invariant_revenue(self):
        p=self.p.copy(); p.pred_kwh*=1.1
        a=score_groups(aggregate_site_predictions(p,self.comp,self.sites,self.window),self.comp,self.sites,self.window)[1]
        alt=self.comp.copy(); alt['group']=np.where(np.arange(253)%2,'G1','G2')
        b=score_groups(aggregate_site_predictions(p,alt,self.sites,self.window),alt,self.sites,self.window)[1]
        self.assertAlmostEqual(a['revenue_krw'],b['revenue_krw'])
    def test_partition_failures(self):
        for c in [self.comp.iloc[:-1],pd.concat([self.comp,self.comp.iloc[:1]]),self.comp.assign(group='G1'),self.comp.assign(group=['G1']+['G2']*252)]:
            with self.assertRaises(ValueError): validate_composition(c,self.sites)
        s=self.sites.copy(); s.capacity_kw=100
        with self.assertRaises(ValueError): validate_composition(self.comp,s)
    def test_duplicate_group(self):
        p=self.group_pred()
        with self.assertRaises(ValueError): score_groups(pd.concat([p,p.iloc[:1]]),self.comp,self.sites,self.window)
    def test_duplicate_site(self):
        with self.assertRaises(ValueError): aggregate_site_predictions(pd.concat([self.p,self.p.iloc[:1]]),self.comp,self.sites,self.window)
        with self.assertRaises(ValueError): evaluate_sites(pd.concat([self.p,self.p.iloc[:1]]),self.window)
    def test_missing_forecast_fails(self):
        with self.assertRaises(ValueError): score_groups(self.group_pred().iloc[1:],self.comp,self.sites,self.window)
        with self.assertRaises(ValueError): aggregate_site_predictions(self.p.iloc[1:],self.comp,self.sites,self.window)
        with self.assertRaises(ValueError): evaluate_sites(self.p.iloc[1:],self.window)
    def test_missing_actual_fixed_mask_and_site_evaluation(self):
        self.window.actual.iloc[0,0]=np.nan
        self.window.completeness.loc[self.ts[0],'included']=False
        gp=self.group_pred()
        _,t,_=score_groups(gp,self.comp,self.sites,self.window)
        self.assertEqual(t['used_hours'],6); self.assertEqual(t['excluded_hours'],1)
        metrics,site=evaluate_sites(self.p,self.window)
        self.assertEqual(site['observed_rows'],253*7-1)
        self.assertEqual(site['unobserved_rows'],1)
        self.assertEqual(site['mae_kwh'],0)
    def test_boundaries_and_utc(self):
        ts=pd.DatetimeIndex([TRAIN_END,TRAIN_END+pd.Timedelta(hours=1),DEV_END,DEV_END+pd.Timedelta(hours=1),HOLDOUT_END])
        self.assertEqual(list(split_labels(ts)),['train','dev','dev','holdout','holdout'])
        self.assertEqual(list(split_labels(ts.tz_convert('UTC'))),list(split_labels(ts)))
        self.assertEqual(list(hourly_weights(self.ts)),[1,3,3,3,3,3,1])
        self.assertEqual(list(hourly_weights(self.ts.tz_convert('UTC'))),[1,3,3,3,3,3,1])
        for invalid in [ts.tz_localize(None),ts+pd.Timedelta(minutes=1)]:
            with self.assertRaises(ValueError): local_hours(invalid)
        with self.assertRaises(ValueError): split_labels([HOLDOUT_END+pd.Timedelta(hours=1)])
    def test_build_windows_missing_nonfinite_and_duplicates(self):
        ts=pd.date_range(TRAIN_END,HOLDOUT_END,freq='h')
        prices=pd.DataFrame({'ts':ts,'smp':100.})
        gen=pd.DataFrame({'ts':[ts[1],ts[2]],'site_id':['PV001','PV001'],'energy':[np.nan,1.]})
        win=build_windows(gen,prices,self.sites)
        self.assertEqual(len(win['dev'].actual),2160)
        self.assertEqual(len(win['holdout'].actual),1464)
        self.assertEqual(len(win['dev'].times),0)
        self.assertEqual(win['dev'].completeness.observed_sites.iloc[0],0)
        with self.assertRaises(ValueError): score_groups(self.group_pred(),self.comp,self.sites,win['dev'])
        with self.assertRaises(ValueError): build_windows(pd.concat([gen,gen.iloc[:1]]),prices,self.sites)
        with self.assertRaises(ValueError): build_windows(gen,pd.concat([prices,prices.iloc[:1]]),self.sites)
        with self.assertRaises(ValueError): build_windows(gen,prices.iloc[:-1],self.sites)
    def test_invalid_predictions(self):
        for value in [np.nan,np.inf,-1.]:
            p=self.group_pred(); p.loc[0,'pred_kwh']=value
            with self.assertRaises(ValueError): score_groups(p,self.comp,self.sites,self.window)

if __name__=='__main__': unittest.main(verbosity=2)
