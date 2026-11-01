import unittest
import pandas as pd
import numpy as np
from feature_pipeline import build_chunk, feature_columns
from validation import TZ

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.sites=pd.DataFrame({'site_id':[f'PV{i:03}' for i in range(1,254)],'capacity_kw':100.,'lat':36.,'lon':128.,'wa_dist':1000.,'wb_dist':2000.,'wa_station':'WA001','wb_station':'WB001'})
        self.times=pd.date_range('2025-01-01 00:00',periods=2,freq='h',tz=TZ)
        self.gen=pd.DataFrame({'site_id':['PV001','PV002'],'ts':self.times,'energy':[-1.,101.]})
        self.prices=pd.DataFrame({'ts':self.times,'smp':100.})
        self.cols={'wa':['wa_wind_speed'],'wb':['wb_wind_speed']}
        self.weather={'wa':pd.DataFrame({'wa_station':'WA001','round':17,'ts':self.times,'wa_wind_speed':[1.,np.nan]}),'wb':pd.DataFrame({'wb_station':['WB001'],'round':[17],'ts':self.times[:1],'wb_wind_speed':[3.6]})}
    def build(self):return build_chunk(self.times,self.sites,self.gen,self.prices,self.weather,self.cols)
    def test_cardinality_targets_flags_and_raw_units(self):
        d=self.build(); self.assertEqual(len(d),506);self.assertEqual(d.site_id.nunique(),253)
        self.assertEqual(int(d.target_present.sum()),2)
        self.assertEqual(d.energy.dropna().tolist(),[-1.,101.])
        np.testing.assert_allclose(d.loc[d.target_present,'target_cf']*d.loc[d.target_present,'capacity_kw'],[-1,101])
        self.assertEqual(int(d.negative_energy.sum()),1);self.assertEqual(int(d.cf_gt_1.sum()),1)
        self.assertEqual(d.wa_wind_speed.iloc[0],1.);self.assertEqual(d.wb_wind_speed.iloc[0],3.6)
        self.assertEqual(int(d.wa_row_missing.sum()),0);self.assertEqual(int(d.wa_any_missing.sum()),253)
        self.assertEqual(int(d.wb_row_missing.sum()),253)
    def test_operating_day_and_schema(self):
        d=self.build();self.assertEqual(d.operating_hour.iloc[0],24);self.assertEqual(d.month.iloc[0],12)
        self.assertEqual(d.dayofyear.iloc[0],366);self.assertEqual(d.operating_hour.iloc[-1],1)
        features=feature_columns(self.cols)
        self.assertFalse(set(features)&{'energy','target_cf','target_present','negative_energy','cf_gt_1','smp','k','ts'})
        train=d[features].dtypes
        self.gen=self.gen.iloc[:0]
        test=self.build()
        self.assertTrue(train.equals(test[features].dtypes));self.assertEqual(int(test.target_present.sum()),0)
    def test_duplicate_and_future_round_rejected(self):
        self.weather['wa']=pd.concat([self.weather['wa'],self.weather['wa'].iloc[:1]])
        with self.assertRaises(ValueError):self.build()
        self.weather['wa']=self.weather['wa'].iloc[:2].copy();self.weather['wa']['round']=10
        with self.assertRaises(ValueError):self.build()
    def test_off_hour_forecast_is_not_rounded(self):
        self.weather['wb']['ts']=self.weather['wb'].ts+pd.Timedelta(minutes=55)
        d=self.build()
        self.assertEqual(int(d.wb_row_missing.sum()),506)
        self.assertTrue(d.wb_wind_speed.isna().all())
    def test_duplicate_target_rejected(self):
        self.gen=pd.concat([self.gen,self.gen.iloc[:1]])
        with self.assertRaises(ValueError):self.build()
if __name__=='__main__': unittest.main(verbosity=2)
