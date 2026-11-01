"""Independently check saved predictions, raw targets, score ledgers and model replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm_baseline import OUT, DATASETS, DATA, ROOT, prepare_features, postprocess, metrics
from validation import local_hours
from first_eda import sha

def main():
    run=json.loads((OUT/'run_manifest.json').read_text())
    model=lgb.Booster(model_file=str(OUT/'model.txt'))
    checks={}
    for split in ('dev','holdout'):
        actual=pd.read_parquet(DATASETS/f'{split}.parquet',columns=['site_id','ts','energy','capacity_kw'])
        raw=pd.read_parquet(DATA/'train/generation.parquet',filters=[('ts','>=',actual.ts.min()),('ts','<=',actual.ts.max())])
        truth=actual.merge(raw,on=['site_id','ts'],validate='one_to_one',suffixes=('_dataset','_raw'))
        assert len(truth)==len(actual)
        np.testing.assert_allclose(truth.energy_dataset,truth.energy_raw,rtol=0,atol=0)
        for name in ('lightgbm','climatology'):
            p=pd.read_csv(OUT/f'{split}_{name}_predictions.csv');p['ts']=local_hours(pd.to_datetime(p.ts))
            assert not p.duplicated(['site_id','ts']).any()
            joined=p.merge(actual,on=['site_id','ts'],validate='one_to_one',suffixes=('','_source'))
            assert len(joined)==len(actual)==len(p)
            np.testing.assert_allclose(joined.actual_kwh,joined.energy,rtol=0,atol=1e-12)
            np.testing.assert_allclose(joined.capacity_kw,joined.capacity_kw_source,rtol=0,atol=1e-12)
            np.testing.assert_allclose(p.pred_kwh,p.pred_cf*p.capacity_kw,rtol=1e-12,atol=1e-12)
            assert (p.pred_kwh>=0).all()
            l=pd.read_csv(OUT/f'{split}_{name}_settlement_ledger.csv');l['ts']=local_hours(pd.to_datetime(l.ts))
            assert not l.duplicated(['group','ts']).any()
            fixed=local_hours(pd.to_datetime(pd.read_csv(ROOT/f'reports/validation/{split}_settlement_times.csv').ts))
            for group in ('G1','G2'):
                assert l.loc[l.group==group,'ts'].sort_values().tolist()==fixed.tolist()
            price=pd.read_csv(DATA/'train/smp.csv');price['ts']=local_hours(pd.to_datetime(price.ts))
            price=price.set_index('ts').smp.reindex(l.ts).to_numpy()
            np.testing.assert_allclose(l.smp,price,rtol=0,atol=0)
            expected_k=np.where(l.ts.dt.hour.between(11,15),3,1)
            np.testing.assert_array_equal(l.k,expected_k)
            revenue=price*l.actual_kwh;penalty=.1*price*expected_k*(l.pred_kwh-l.actual_kwh).abs()
            np.testing.assert_allclose(l.revenue_krw,revenue,rtol=1e-12,atol=1e-7)
            np.testing.assert_allclose(l.penalty_krw,penalty,rtol=1e-12,atol=1e-7)
            comparison=pd.read_csv(OUT/'comparison.csv');row=comparison[(comparison.split==split)&(comparison.model==name)].iloc[0]
            np.testing.assert_allclose([row.revenue_krw,row.penalty_krw,row.score_krw],[revenue.sum(),penalty.sum(),(revenue-penalty).sum()],rtol=1e-12)
            for key,val in metrics(p).items():np.testing.assert_allclose(row[key],val,rtol=1e-12,atol=1e-9)
            checks[f'{split}_{name}']='targets, units, errors, fixed times, SMP, k and settlement verified'
        # Predicate reads select a real contiguous first-day sample for independent inference.
        sample=pd.read_parquet(DATASETS/f'{split}.parquet',filters=[('ts','<=',actual.ts.min()+pd.Timedelta(hours=3))])
        pred=postprocess(model.predict(prepare_features(sample,run['feature_columns'],run['site_categories'])),sample.capacity_kw,run['clip_capacity'])[1]
        saved=pd.read_csv(OUT/f'{split}_lightgbm_predictions.csv',nrows=len(sample))
        np.testing.assert_allclose(pred,saved.pred_kwh,rtol=1e-12,atol=1e-12)
    for p,h in run['input_sha256'].items():assert sha(ROOT/p)==h
    assert sha(OUT/'model.txt')==run['model_sha256']
    checks['model_replay']='both validation samples reproduced'
    checks['input_hashes']='all unchanged'
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
