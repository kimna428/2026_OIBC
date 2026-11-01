import unittest
import numpy as np
import pandas as pd
from lightgbm_baseline import Climatology,postprocess,prepare_features,metrics
class BaselineTests(unittest.TestCase):
    def training(self):
        return pd.DataFrame({'site_id':['A']*24+['B']*2,'month':[1]*26,'operating_hour':[12]*26,'target_cf':[.5]*24+[1.]*2,'ts':pd.date_range('2024-01-01',periods=26,freq='h',tz='Asia/Seoul')})
    def test_train_only_fallback(self):
        d=self.training();m=Climatology().fit(d)
        x=pd.DataFrame({'site_id':['A','B','C','C','C'],'month':[1,1,1,2,2],'operating_hour':[12,12,12,12,13]})
        p,s=m.predict(x)
        self.assertAlmostEqual(p[0],.5)
        np.testing.assert_allclose(p[1:],[14/26]*4)
        self.assertEqual(list(s),['site_month_hour','global_month_hour','global_month_hour','global_hour','global_mean'])
        d.ts=d.ts+pd.DateOffset(years=1)
        with self.assertRaises(ValueError):Climatology().fit(d)
    def test_postprocessing(self):
        cf,energy=postprocess([-1,.2,1.2],[100,200,100])
        np.testing.assert_allclose(cf,[0,.2,1.2]);np.testing.assert_allclose(energy,[0,40,120])
        _,energy=postprocess([-1,.2,1.2],[100,200,100],True)
        np.testing.assert_allclose(energy,[0,40,100])
    def test_categories_and_nan_preserved(self):
        d=pd.DataFrame({'site_id':['B'],'weather':[np.nan],'energy':[100.]})
        x=prepare_features(d,['site_id','weather'],['A','B'])
        self.assertEqual(list(x.site_id.cat.categories),['A','B']);self.assertTrue(x.weather.isna().all());self.assertNotIn('energy',x)
    def test_metrics_units_and_peak(self):
        d=pd.DataFrame({'ts':pd.date_range('2025-01-01 10:00',periods=2,freq='h',tz='Asia/Seoul'),'actual_kwh':[0.,10.],'pred_kwh':[2.,14.],'capacity_kw':[100.,200.],'smp':[100.,100.],'k':[1,3]})
        m=metrics(d)
        self.assertEqual(m['mae_kwh'],3);self.assertAlmostEqual(m['rmse_kwh'],np.sqrt(10));self.assertAlmostEqual(m['cf_mae'],.02)
        self.assertEqual(m['weighted_error_krw'],1400);self.assertEqual(m['peak_weighted_error_krw'],1200);self.assertEqual(m['peak_weighted_mae_kwh'],4)
if __name__=='__main__':unittest.main(verbosity=2)
