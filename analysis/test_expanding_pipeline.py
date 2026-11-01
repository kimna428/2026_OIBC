import unittest
import pandas as pd
import numpy as np
from expanding_pipeline import issuance,check_asof,check_split,interpolate_single,history_features,validate_generation
TZ='Asia/Seoul'
def ts(x): return pd.DatetimeIndex(x,tz=TZ)
class DayAheadTests(unittest.TestCase):
    def test_midnight_belongs_to_previous_day(self):
        self.assertEqual(issuance(ts(['2025-03-02 00:00']))[0],pd.Timestamp('2025-02-28 17:00',tz=TZ))
    def test_future_source_rejected(self):
        with self.assertRaises(ValueError): check_asof(ts(['2025-03-01 18:00']),ts(['2025-03-02 12:00']))
        check_asof(ts(['2025-03-01 17:00']),ts(['2025-03-02 12:00']))
    def test_training_cutoff(self):
        with self.assertRaises(ValueError): check_split(ts(['2025-03-01 18:00']),ts(['2025-03-02 01:00']))
        check_split(ts(['2025-03-01 17:00']),ts(['2025-03-02 01:00','2025-03-03 00:00']))
    def test_naive_rejected(self):
        with self.assertRaises(ValueError): issuance(pd.DatetimeIndex(['2025-03-02']))
    def test_no_extrapolation_or_long_gap_interpolation(self):
        g=pd.DataFrame({'site_id':['PV001']*3,'ts':ts(['2025-03-01 01:00','2025-03-01 03:00','2025-03-01 11:00']),'energy':[1.,3.,9.]})
        e=interpolate_single(g)
        self.assertEqual(len(e),1); self.assertEqual(e.energy.iloc[0],2.)
        self.assertEqual(e.available_at.iloc[0],g.ts.iloc[1])
    def test_interpolation_rejects_anomalous_endpoints(self):
        g=pd.DataFrame({'site_id':['PV001']*2,'ts':ts(['2025-03-01 01:00','2025-03-01 03:00']),'energy':[-1.,3.],'cf':[-.1,.3]})
        self.assertTrue(interpolate_single(g).empty)
    def test_interpolation_endpoint_must_be_available(self):
        g=pd.DataFrame({'site_id':['PV001']*2,'ts':ts(['2025-03-01 16:00','2025-03-01 18:00']),'energy':[1.,3.]})
        e=interpolate_single(g)
        cutoff=pd.Timestamp('2025-03-01 17:00',tz=TZ)
        self.assertTrue(e[e.available_at<=cutoff].empty)
    def test_future_perturbation_does_not_change_history(self):
        times=pd.date_range('2025-02-20',periods=24*12,freq='h',tz=TZ)
        g=pd.DataFrame({'site_id':'PV001','ts':times,'cf':.2})
        target=pd.DataFrame({'site_id':['PV001'],'ts':ts(['2025-03-02 12:00'])})
        before=history_features(g,target)
        g.loc[g.ts>pd.Timestamp('2025-03-01 17:00',tz=TZ),'cf']=999
        pd.testing.assert_frame_equal(before,history_features(g,target))
    def test_duplicate_generation_key_rejected(self):
        sites=pd.DataFrame({'site_id':[f'PV{i:03d}' for i in range(1,254)],'capacity_kw':1.})
        g=pd.DataFrame({'site_id':['PV001']*2,'ts':ts(['2025-03-01','2025-03-01']),'energy':[0.,1.]})
        with self.assertRaises(ValueError): validate_generation(g,sites)
if __name__=='__main__': unittest.main()
