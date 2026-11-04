"""Descriptive error analysis; evaluation-year alpha sweeps are exploratory."""
import itertools
import numpy as np
import pandas as pd
from analysis.deep_learning.features import point_elevation
from .core import ALPHAS,evaluate


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def residual_analysis(grid,predictions,seen,output):
    caps=grid.sites.capacity_kw.to_numpy();mask=grid.observed
    residual={name:(grid.validate_prediction(p)-grid.actual) for name,p in predictions.items()}
    pairs=[]
    for a,b in itertools.combinations(residual,2):
        ra,rb=residual[a],residual[b]
        pairs.append(dict(model_a=a,model_b=b,observed_rows=int(mask.sum()),
                     residual_kwh_correlation=correlation(ra[mask],rb[mask]),
                     residual_cf_correlation=correlation((ra/caps)[mask],(rb/caps)[mask])))
    pd.DataFrame(pairs,columns=['model_a','model_b','observed_rows','residual_kwh_correlation','residual_cf_correlation']).to_csv(output/'residual_correlations.csv',index=False)
    # Geometry is a proxy for solar opportunity, NOT measured irradiance.
    elevation=point_elevation((grid.times-pd.Timedelta(minutes=30)).repeat(len(caps)),
                             np.tile(grid.sites.lat,len(grid.times)),np.tile(grid.sites.lon,len(grid.times))).reshape(grid.actual.shape)
    seasons=np.array(['winter','spring','summer','autumn'])[(grid.times.month%12)//3]
    scope=[]
    for hour in range(24):scope.append(('hour',str(hour),(grid.times.hour==hour)[:,None]))
    for month in sorted(set(grid.times.strftime('%Y-%m'))):scope.append(('month',month,(grid.times.strftime('%Y-%m')==month)[:,None]))
    for season in np.unique(seasons):scope.append(('season',season,(seasons==season)[:,None]))
    for name,m in [('small_under100kw',caps<100),('medium_100to500kw',(caps>=100)&(caps<500)),('large_500kw',caps>=500),
                   ('no_training_history',seen==0),('under90days',(seen>0)&(seen<2160)),('at_least90days',seen>=2160)]:
        scope.append(('site_cohort',name,m[None,:]))
    scope.extend([('solar_geometry','elevation_over45_proxy',elevation>45),('solar_geometry','day_elevation_over0',elevation>0),
                  ('solar_geometry','night_elevation_atmost0',elevation<=0)])
    rows=[];sites=[]
    for model,r in residual.items():
        for dimension,label,selection in scope:
            used=mask&selection; n=int(used.sum())
            rows.append(dict(model=model,dimension=dimension,label=label,observed_rows=n,
                         site_nmae_pct=float(100*np.abs(r/caps)[used].mean()) if n else None,
                         mae_kwh=float(np.abs(r[used]).mean()) if n else None,
                         bias_cf_pct=float(100*(r/caps)[used].mean()) if n else None,
                         bias_kwh=float(r[used].mean()) if n else None,
                         overprediction_rows=int((r[used]>0).sum()),underprediction_rows=int((r[used]<0).sum())))
        for s,sid in enumerate(grid.sites.site_id):
            used=mask[:,s];n=int(used.sum())
            sites.append(dict(model=model,site_id=sid,capacity_kw=caps[s],train_valid_hours=int(seen[s]),observed_rows=n,
                              nmae_pct=float(100*np.abs(r[:,s][used]/caps[s]).mean()) if n else None,
                              bias_cf_pct=float(100*(r[:,s][used]/caps[s]).mean()) if n else None))
    pd.DataFrame(rows).to_csv(output/'error_slices.csv',index=False)
    pd.DataFrame(sites).to_csv(output/'site_errors.csv',index=False)
    return dict(status='computed_on_identical_observed_rows',irradiance='unavailable; solar elevation >45 is a geometry proxy only',
                cohort='finite CF [0,1] target hours through fixed training cutoff',pairs=pairs)


def group_pair_analysis(ledgers,output):
    """Compare settlement cancellation, residual correlation, opposite successes."""
    records=[]
    for assignment,models in ledgers.items():
        for (a,la),(b,lb) in itertools.combinations(models.items(),2):
            d=la.merge(lb,on=['group','ts'],validate='one_to_one',suffixes=('_a','_b'))
            if len(d)!=len(la) or len(d)!=len(lb):
                raise ValueError('Group diagnostic row masks differ')
            records.append(dict(assignment=assignment,model_a=a,model_b=b,
                                group_residual_correlation=correlation(d.residual_kwh_a,d.residual_kwh_b)))
            d['penalty_a_minus_b']=d.penalty_krw_a-d.penalty_krw_b
            for direction,z in [('a_wins',d.nsmallest(25,'penalty_a_minus_b')),('b_wins',d.nlargest(25,'penalty_a_minus_b'))]:
                z.to_csv(output/f'cases_{assignment}_{a}_{b}_{direction}.csv',index=False)
    pd.DataFrame(records,columns=['assignment','model_a','model_b','group_residual_correlation']).to_csv(output/'group_residual_correlations.csv',index=False)


def ensemble_sweep(grid,predictions,assignments,reference_verified=False,oof=None):
    if 'tsmixer' not in predictions or 'teammate_ensemble' not in predictions:
        return dict(status='unavailable',reason='Need both aligned TSMixer and teammate ensemble predictions',trials=[])
    a,b=predictions['teammate_ensemble'],predictions['tsmixer']
    rows=[]
    for alpha in ALPHAS:
        blend=(1-alpha)*a+alpha*b
        for label,composition in assignments.items():
            result,_=evaluate(grid,blend,composition,reference_verified=reference_verified)
            rows.append(dict(alpha=alpha,assignment=label,kind='exploratory_on_evaluation_year',**result))
    selected=[]
    if oof:
        blend=(1-oof['selected_alpha'])*a+oof['selected_alpha']*b
        for label,composition in assignments.items():
            result,_=evaluate(grid,blend,composition,reference_verified=reference_verified)
            selected.append(dict(alpha=oof['selected_alpha'],assignment=label,kind='pre_cutoff_oof_selected_year_evaluation',**result))
    return dict(status='computed',trials=rows,oof_selection=oof,oof_selected_evaluation=selected,
                independent_final_evaluation=False,
                warning='No evaluation-year best alpha is promoted; year was repeatedly explored by teammate')
