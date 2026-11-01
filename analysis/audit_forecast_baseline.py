"""Check saved OOS artifacts against source data and summarize residual structure."""
import json
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import forecast_baseline as f


def main():
    out=f.DEFAULT_OUT
    manifest=json.loads((out/'feature_manifest.json').read_text())
    pred=pd.read_parquet(out/'validation_predictions.parquet')
    expected=['site_id','ts','actual_cf','pred_cf','actual_kwh','pred_kwh','residual_kwh']
    assert pred.columns.tolist()==expected
    assert not pred.duplicated(['site_id','ts']).any()
    assert np.isfinite(pred[expected[2:]].to_numpy()).all()
    assert (pred.pred_cf>=0).all()
    sites=pd.read_csv(f.DATA/'sites.csv').set_index('site_id')
    generation=pd.read_parquet(f.DATA/'train/generation.parquet')
    cutoff=pd.Timestamp(manifest['validation_start'])
    holdout=generation.loc[(generation.ts>=cutoff)&np.isfinite(generation.energy)].sort_values(['ts','site_id']).reset_index(drop=True)
    holdout['ts']=f.local_ts(holdout.ts)
    assert pred[['site_id','ts']].reset_index(drop=True).equals(holdout[['site_id','ts']])
    np.testing.assert_array_equal(pred.actual_kwh.to_numpy(),holdout.energy.to_numpy())
    cap=pred.site_id.map(sites.capacity_kw).to_numpy()
    np.testing.assert_allclose(pred.actual_cf,pred.actual_kwh/cap,rtol=0,atol=1e-12)
    np.testing.assert_allclose(pred.pred_kwh,pred.pred_cf*cap,rtol=0,atol=1e-12)
    np.testing.assert_allclose(pred.residual_kwh,pred.actual_kwh-pred.pred_kwh,rtol=0,atol=1e-12)
    assert pd.Timestamp(manifest['train_end'])<cutoff
    assert manifest['round']==17 and manifest['parameters']['has_time']
    assert manifest['features']==f.METADATA+f.TIME_FEATURES+f.forecast_columns('wa')+f.forecast_columns('wb')
    smp_table=pd.read_csv(f.DATA/'train/smp.csv')
    smp_table['ts']=f.local_ts(smp_table.ts)
    smp=smp_table.set_index('ts').smp
    prices=smp.reindex(pred.ts).to_numpy()
    assert np.isfinite(prices).all()
    peak=pred.ts.dt.hour.between(11,15).to_numpy()
    weighted_error=float(np.sum(prices*np.where(peak,3,1)*np.abs(pred.residual_kwh)))
    metrics=pd.read_csv(out/'validation_metrics.csv').iloc[0]
    np.testing.assert_allclose(weighted_error,metrics.weighted_error,rtol=1e-12)
    stats=pd.read_csv(out/'site_metrics.csv').set_index('site_id')
    assert stats.observed_hours.sum()==len(pred)
    assert stats.cold_start.sum()==manifest['cold_start_sites']
    for scope,counts in [('all',pred.groupby('site_id').size()),('peak',pred.loc[peak].groupby('site_id').size())]:
        overlap=pd.read_csv(out/f'residual_overlap_{scope}.csv',index_col=0)
        np.testing.assert_array_equal(np.diag(overlap),counts.reindex(sites.index).to_numpy())
    sample=pred.iloc[np.linspace(0,len(pred)-1,256,dtype=int)]
    features,_=f.build_features(sample[['site_id','ts']],sites)
    numerical=[c for c in features if c not in f.CATEGORICAL]
    features[numerical]=features[numerical].fillna(pd.Series(manifest['numeric_train_medians']))
    model=CatBoostRegressor(); model.load_model(str(out/'model.cbm'))
    replay=np.maximum(model.predict(Pool(features,cat_features=f.CATEGORICAL)),0.)
    np.testing.assert_allclose(replay,sample.pred_cf,rtol=0,atol=1e-12)
    assert all(f.sha256(f.ROOT/path)==value for path,value in manifest['input_sha256'].items())
    class LinkChecker(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src']: assert (out/value).is_file(),value
    LinkChecker().feed((out/'report.html').read_text())
    corr=pd.read_csv(out/'residual_correlation_peak.csv',index_col=0).reindex(index=sites.index,columns=sites.index).to_numpy()
    a=sites.wa_station.to_numpy(); b=sites.wb_station.to_numpy()
    same=(a[:,None]==a[None,:])|(b[:,None]==b[None,:])
    pairs=np.triu(np.ones(corr.shape,bool),1)&np.isfinite(corr)
    def pair_stats(mask):
        values=corr[mask]
        return dict(pairs=len(values),mean=float(values.mean()),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    residual_summary=dict(residual_sign='actual_kwh - pred_kwh',
        residual_kwh_quantiles={str(q):float(pred.residual_kwh.quantile(q)) for q in [0,.005,.01,.05,.5,.95,.99,.995,1]},
        peak_residual_correlation=pair_stats(pairs),
        peak_same_station_correlation=pair_stats(pairs&same),
        peak_different_station_correlation=pair_stats(pairs&~same),
        site_bias_kwh_min=float(stats.bias_kwh.min()),site_bias_kwh_max=float(stats.bias_kwh.max()),
        site_positive_bias_count=int((stats.bias_kwh>0).sum()),site_negative_bias_count=int((stats.bias_kwh<0).sum()))
    (out/'residual_summary.json').write_text(json.dumps(residual_summary,indent=2),encoding='utf-8')
    verification=dict(status='passed',validation_rows=len(pred),source_rows_and_actuals_exact_match=True,
        prediction_replay_rows=len(sample),saved_model_predictions_match=True,source_sha256_unchanged=True,
        temporal_split_verified=True,capacity_conversion_verified=True,residual_sign_verified=True,
        weighted_error_verified=True,site_and_overlap_counts_verified=True,report_links_verified=True)
    (out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    f.render_report(out,manifest,pd.read_csv(out/'validation_metrics.csv'),
        pd.read_csv(out/'cohort_metrics.csv').set_index('cohort'),stats,
        pd.read_csv(out/'feature_importance.csv'),f.REPORT_PLOTS,residual_summary)
    LinkChecker().feed((out/'report.html').read_text())
    print(json.dumps(verification,indent=2))
    print(json.dumps(residual_summary,indent=2))


if __name__=='__main__':main()
