"""Regression checks for forecast availability, joins and scoring boundaries."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
import forecast_baseline as pipeline


class ForecastBaselineTests(unittest.TestCase):
    def test_only_round17_and_site_station_join(self):
        with tempfile.TemporaryDirectory() as temp:
            data=Path(temp)
            ts=pd.date_range('2024-06-01 00:00',periods=2,freq='h',tz='Asia/Seoul')
            for folder in ['train','test']:
                (data/folder).mkdir()
                for provider in ['wa','wb']:
                    weather=pd.DataFrame({provider+'_station':['S2','S1','S2','S1'],
                        'ts':[ts[0],ts[0],ts[0],ts[0]],'round':[10,10,17,17],
                        provider+'_cloud':[999.,999.,20.,10.]})
                    weather.to_parquet(data/folder/f'forecast_{provider}.parquet',index=False)
            sites=pd.DataFrame({'site_id':['PV1','PV2'],'wa_station':['S1','S2'],
                'wb_station':['S2','S1'],'capacity_kw':[10.,20.],'wa_dist':[100.,200.],
                'wb_dist':[300.,400.],'lat':[35.,36.],'lon':[127.,128.],
                'current_group':['A','B']}).set_index('site_id')
            base=pd.DataFrame({'site_id':['PV2','PV1','PV1'],'ts':[ts[0],ts[0],ts[1]],
                               'energy':[777.,888.,999.]})
            with patch.object(pipeline,'DATA',data):
                features,quality=pipeline.build_features(base,sites)
            self.assertEqual(features.wa_cloud.iloc[:2].tolist(),[20.,10.])
            self.assertEqual(features.wb_cloud.iloc[:2].tolist(),[10.,20.])
            self.assertTrue(pd.isna(features.wa_cloud.iloc[2]))
            self.assertEqual(features.hour.tolist(),[0,0,1])
            self.assertEqual(features.capacity_kw.tolist(),[20.,10.,10.])
            self.assertFalse(set(features)&{'energy','observed_wa','smp','round','current_group'})
            self.assertEqual(quality.loc[quality.feature=='wa_forecast_row','missing_rows'].iloc[0],1)

    def test_peak_boundaries_and_residual_sign(self):
        ts=pd.DatetimeIndex([pd.Timestamp(f'2024-06-01 {h:02d}:00',tz='Asia/Seoul') for h in [10,11,15,16]])
        pred=pd.DataFrame({'site_id':['PV1']*4,'ts':ts,'actual_cf':[.1,.2,.3,.4],
            'pred_cf':[0.]*4,'actual_kwh':[1.,2.,3.,4.],'pred_kwh':[0.]*4,
            'residual_kwh':[1.,2.,3.,4.]})
        smp=pd.DataFrame({'ts':ts,'smp':[10.]*4})
        result=pipeline.metrics_frame(pred,smp,None)
        self.assertEqual(result.k.tolist(),[1.,3.,3.,1.])
        self.assertEqual(result.weighted_error.sum(),200.)
        result['all']='all'
        summary=pipeline.summarize(result,['all']).iloc[0]
        self.assertAlmostEqual(summary.energy_mae_kwh,2.5)
        self.assertAlmostEqual(summary.bias_kwh,2.5)
        self.assertAlmostEqual(summary.peak_energy_mae_kwh,2.5)
        self.assertAlmostEqual(summary.weighted_mae_kwh,2.5)
        self.assertAlmostEqual(summary.rmse_kwh,np.sqrt(7.5))


if __name__=='__main__':unittest.main()
