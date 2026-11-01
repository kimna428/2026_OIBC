"""Independent full-target, clock, selection, composition and model-replay audit."""
from pathlib import Path
import argparse, json, hashlib, importlib.metadata, sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from third_experiment import OUT, PREVIOUS, DATA, Experiment, FOLDS, POLICIES, SEED, stamp, validation_mask, postprocess, select_development, join_forecast, render_report, parameter_schedule, fit_indices, split_indices, SCALES, frozen_history_features
from expanding_pipeline import issuance, operating_day, sha256, history_features, solar
from validation import validate_composition

def audit(out):
    out=Path(out)
    (out/'environment.json').write_text(json.dumps({'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','pyarrow','lightgbm','scikit-learn','scipy','matplotlib']},'threads_per_model':4,'seed':SEED},indent=2))
    config=json.loads((out/'run_spec.json').read_text())
    assert all(sha256(DATA/p)==v for p,v in config['input_hashes'].items())
    raw=pd.read_parquet(DATA/'train/generation.parquet').sort_values(['ts','site_id']).reset_index(drop=True)
    raw.ts=pd.DatetimeIndex(raw.ts).tz_convert('Asia/Seoul').as_unit('ns')
    raw['capacity_kw']=raw.site_id.map(pd.read_csv(DATA/'sites.csv').set_index('site_id').capacity_kw); raw['cf']=raw.energy/raw.capacity_kw
    keys=pd.MultiIndex.from_frame(raw[['site_id','ts']]); actual=raw.set_index(['site_id','ts']).energy
    sites=pd.read_csv(DATA/'sites.csv')
    historical_prices=pd.read_csv(DATA/'train/smp.csv'); historical_prices.ts=pd.to_datetime(historical_prices.ts); historical_prices=historical_prices.set_index('ts').smp
    cancellation_bounds=[]
    tuning=pd.read_csv(out/'tuning_results.csv')
    assert tuning.fold.max()==3 and len(tuning)==24
    assert json.loads((out/'parameter_schedule.json').read_text())==parameter_schedule(tuning)
    tuned=json.loads((out/'tuned_parameters.json').read_text())
    for f in ['B','C']:
        winner=select_development(tuning[tuning.family==f]); idx=int(winner.rsplit('_',1)[1])
        assert tuned[f]==config['tuning_grid'][idx]
    lock=json.loads((out/'selection_lock.json').read_text())
    devscores=pd.read_csv(out/'scores_selection.csv')
    assert pd.to_datetime(devscores.selection_source_max,utc=True).max()==stamp('2025-02-28 17:00').tz_convert('UTC')
    assert devscores.fold.max()==4
    for f,c in lock['chosen'].items():
        s=devscores[(devscores.group=='total')&devscores.score_available&devscores.candidate.str.startswith(f+'_')]
        assert s.groupby(['candidate','policy']).penalty_krw.sum().sort_values().index[0]==(c['candidate'],c['policy'])
    preferred=json.loads((out/'preferred_candidate_lock.json').read_text())
    assert preferred['selection_lock_sha256']==sha256(out/'selection_lock.json') and not preferred['holdout_used']
    vals=[]
    for family,c in lock['chosen'].items():
        z=devscores[(devscores.candidate==c['candidate'])&(devscores.policy==c['policy'])&(devscores.group=='total')&devscores.score_available]
        vals.append((float(z.penalty_krw.sum()),family))
    assert preferred['preferred']['family']==min(vals)[1]
    comp_lock=json.loads((out/'composition_lock.json').read_text())
    assert comp_lock['selection_lock_hash']==sha256(out/'selection_lock.json')
    for f in ['B','C']:
        assert comp_lock['composition_hashes'][f]==sha256(out/f'composition_{f}.csv')
        validate_composition(pd.read_csv(out/f'composition_{f}.csv'),sites)
        oof=pd.read_parquet(out/f'OOF_{f}.parquet')
        assert oof.fold.max()==4 and oof.ts.max()<=stamp('2025-02-28 17:00')
        assert (oof.training_cutoff<=oof.issued_at).all() and not oof.duplicated(['site_id','ts']).any()
        peak=oof[oof.ts.dt.hour.between(11,15)].pivot(index='ts',columns='site_id',values='residual_kwh').reindex(columns=sites.site_id)
        peak.corr(min_periods=168).to_csv(out/f'oof_{f}_peak_residual_correlation.csv')
        valid=peak.notna().astype('int32'); overlap=valid.T@valid
        overlap.to_csv(out/f'oof_{f}_peak_residual_overlap.csv')
        plus=(peak>0).astype('int32'); minus=(peak<0).astype('int32')
        ((plus.T@minus+minus.T@plus)/overlap).to_csv(out/f'oof_{f}_peak_opposite_sign_rate.csv')
        complete=oof.pivot(index='ts',columns='site_id',values='residual_kwh').reindex(columns=sites.site_id).dropna(how='any')
        weight=.1*historical_prices.reindex(complete.index).to_numpy()*np.where(complete.index.hour.isin(range(11,16)),3,1)
        total=complete.sum(axis=1).to_numpy(); floor=float(np.dot(weight,np.abs(total)))
        for name,path in [('balanced',PREVIOUS/'composition.csv'),('optimized',out/f'composition_{f}.csv')]:
            comp=pd.read_csv(path).set_index('site_id').group.reindex(complete.columns)
            a=complete.loc[:,comp.eq('G1')].sum(axis=1).to_numpy()
            penalty=float(np.dot(weight,np.abs(a)+np.abs(total-a)))
            cancellation_bounds.append(dict(family=f,composition=name,training_oof_hours=len(complete),penalty_krw=penalty,fleet_error_lower_bound_krw=floor,gap_to_bound_krw=penalty-floor))
    pd.DataFrame(cancellation_bounds).to_csv(out/'optimization_cancellation_bounds.csv',index=False)
    boundaries=[]
    for fold,(start,end) in enumerate(FOLDS):
        p=pd.read_parquet(out/'provenance'/f'fold{fold}.parquet')
        np.testing.assert_array_equal(p.issued_at,issuance(p.ts))
        assert validation_mask(p.ts,start,end).all()
        sourcecols=['training_target_max','model_training_data_cutoff','r10_issued_at','r17_issued_at','lag48_source_ts','rolling_latest_actual_ts']
        for c in sourcecols: assert (p[c].isna()|(p[c]<=p.issued_at)).all()
        boundaries.append({'fold':fold,'rows':len(p),'future_violations':0})
    x=pd.read_parquet(out/'features.parquet')
    samples=[]; checks=[]
    predictions=sorted((out/'predictions').glob('*.parquet')); assert len(predictions)==96
    for path in predictions:
        tag,foldpart=path.stem.rsplit('_fold',1); fold=int(foldpart)
        d=pd.read_parquet(path)
        assert len(d) and not d.duplicated(['site_id','ts']).any()
        np.testing.assert_array_equal(d.energy,actual.reindex(pd.MultiIndex.from_frame(d[['site_id','ts']])).to_numpy())
        np.testing.assert_allclose(d.cf,d.energy/d.capacity_kw,rtol=0,atol=1e-12)
        assert validation_mask(d.ts,*FOLDS[fold]).all()
        assert np.isfinite(d.pred_cf).all() and (d.pred_cf>=0).all()
        np.testing.assert_array_equal(d.issued_at,issuance(d.ts))
        assert (d.training_cutoff<=issuance(d.ts)).all()
        checkpoint=json.loads((out/'checkpoints'/f'{tag}_fold{fold}.json').read_text())
        tr,_,cutoff=split_indices(raw,fold); train=fit_indices(raw,tr,SCALES[checkpoint['scale']])
        assert checkpoint['fit_rows']==len(train) and raw.ts.iloc[train].max()<=issuance(d.ts).min()
        present=d.parameter_source_max.notna()
        if present.any(): assert (pd.to_datetime(d.loc[present,'parameter_source_max'],utc=True)<=pd.to_datetime(d.loc[present,'issued_at'],utc=True)).all()
        if not tag.startswith('tune_'):
            family=tag.split('_')[0]
            schedule=json.loads((out/'parameter_schedule.json').read_text())[family][str(fold)]
            checkpoint=json.loads((out/'checkpoints'/f'{tag}_fold{fold}.json').read_text())
            assert checkpoint['params']==schedule['params']
        expected=raw.loc[validation_mask(raw.ts,*FOLDS[fold]),['site_id','ts']].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,d[['site_id','ts']],check_dtype=False)
        seed=SEED+int.from_bytes(hashlib.sha256(path.name.encode()).digest()[:4],'little')
        sample_pos=np.unique(np.r_[np.linspace(0,len(d)-1,16,dtype=int),np.random.default_rng(seed).choice(len(d),48,replace=False)])
        sample=d.iloc[sample_pos]
        pos=keys.get_indexer(pd.MultiIndex.from_frame(sample[['site_id','ts']]))
        model=lgb.Booster(model_file=str(out/'models'/f'{tag}_fold{fold}.txt'))
        pred=np.maximum(model.predict(x.iloc[pos][model.feature_name()],num_threads=4),0)
        np.testing.assert_allclose(pred,sample.pred_cf,rtol=1e-10,atol=1e-12)
        samples.extend(pos.tolist())
        checks.append({'file':path.name,'fold':fold,'rows':len(d),'raw_targets_match':True,'sample_model_replay':True,'actual_training_target_max':str(raw.ts.iloc[train].max())})
    # Independent source reconstruction at every unique replay sample (not from cached sources).
    pos=np.unique(samples); sample_feature_rows=len(pos); target=raw.iloc[pos].reset_index(drop=True); cached=x.iloc[pos].reset_index(drop=True)
    h=history_features(raw,target)
    for c in h: np.testing.assert_allclose(cached[c],h[c],equal_nan=True,rtol=0,atol=1e-7)
    for provider in ['wa','wb']:
        cols=[c for c in config['features']['B'] if c.startswith(provider+'_') and c not in [provider+'_station',provider+'_dist']]
        for round_number in [10,17]:
            weather=join_forecast(target,sites,provider,round_number,cols)
            for c in cols:
                col=c if round_number==17 else c+'_r10'
                np.testing.assert_allclose(cached[col],weather[c],equal_nan=True,rtol=0,atol=0)
        for c in cols: np.testing.assert_allclose(cached[c+'_revision'],cached[c]-cached[c+'_r10'],equal_nan=True,rtol=0,atol=0)
    np.testing.assert_allclose(cached.solar_elevation,solar(target.ts,cached.lat,cached.lon),atol=1e-5)
    # Recompute all experiment scores independently from site residual sums; no imputation.
    allscores=pd.concat([pd.read_csv(out/'scores_development.csv'),pd.read_csv(out/'scores_holdout.csv')])
    prices=pd.read_csv(DATA/'train/smp.csv'); prices.ts=pd.to_datetime(prices.ts); prices=prices.set_index('ts').smp
    groups=pd.read_csv(PREVIOUS/'composition.csv').set_index('site_id').group
    scorechecks=0
    for path in predictions:
        if path.name.startswith('tune_'): continue
        tag,f=path.stem.rsplit('_fold',1); fold=int(f); d=pd.read_parquet(path)
        d['group']=d.site_id.map(groups)
        counts=d.groupby('ts').site_id.nunique(); times=counts.index[counts==len(sites)]
        for policy in POLICIES:
            d['prediction']=postprocess(d.pred_cf,d.max_interval_elevation,d.learned_night_threshold,policy)*d.capacity_kw
            q=d[d.ts.isin(times)].groupby(['group','ts'])[['energy','prediction']].sum().reset_index()
            weight=.1*q.ts.map(prices)*np.where(q.ts.dt.hour.between(11,15),3,1)
            penalty=float((weight*(q.prediction-q.energy).abs()).sum())
            expected=allscores[(allscores.candidate==tag)&(allscores.fold==fold)&(allscores.policy==policy)&(allscores.group=='total')].iloc[0]
            if not len(times): assert not expected.score_available and pd.isna(expected.penalty_krw)
            else:
                np.testing.assert_allclose(penalty,expected.penalty_krw,rtol=1e-12,atol=1e-5)
                revenue=float((q.ts.map(prices)*q.energy).sum())
                np.testing.assert_allclose(revenue-penalty,expected.score_krw,rtol=1e-12,atol=1e-5)
            scorechecks+=1
            if fold==4:
                safe=q[q.ts<=stamp('2025-02-28 17:00')]
                w=.1*safe.ts.map(prices)*np.where(safe.ts.dt.hour.between(11,15),3,1)
                value=float((w*(safe.prediction-safe.energy).abs()).sum())
                selection=devscores[(devscores.candidate==tag)&(devscores.fold==4)&(devscores.policy==policy)&(devscores.group=='total')].iloc[0]
                np.testing.assert_allclose(value,selection.penalty_krw,rtol=1e-12,atol=1e-5)
    calibration=json.loads((out/'march_control_model.json').read_text())
    assert pd.Timestamp(calibration['fit_max_ts'])<=stamp('2025-02-28 17:00')
    # Offline sensitivity: preserve selected recipes, remove all post-fit generation arrivals.
    evaluator=Experiment.__new__(Experiment)
    evaluator.sites=sites; evaluator.prices=prices
    evaluator.composition=pd.read_csv(PREVIOUS/'composition.csv')
    offline_metrics=[]; offline_scores=[]
    choice=lock['chosen']['C']
    for fold in range(6):
        d=pd.read_parquet(out/'predictions'/f"{choice['candidate']}_fold{fold}.parquet")
        cutoff=stamp(FOLDS[fold][0])-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
        features=frozen_history_features(raw,d,cutoff)
        pos=keys.get_indexer(pd.MultiIndex.from_frame(d[['site_id','ts']]))
        model=lgb.Booster(model_file=str(out/'models'/f"{choice['candidate']}_fold{fold}.txt"))
        xx=x.iloc[pos][model.feature_name()].copy()
        for c in features: xx[c]=features[c].to_numpy()
        d['pred_cf']=np.maximum(model.predict(xx,num_threads=4),0); d['pred_kwh']=d.pred_cf*d.capacity_kw
        for policy in POLICIES:
            mm,ss,_=evaluator.evaluate(d,'C_frozen_history',policy)
            offline_metrics.append(mm); offline_scores.append(ss)
        if fold==5:
            d['pred_cf']=postprocess(d.pred_cf,d.max_interval_elevation,d.learned_night_threshold,choice['policy']); d['pred_kwh']=d.pred_cf*d.capacity_kw
            d['history_observation_cap']=cutoff; d['candidate']=choice['candidate']; d['policy']=choice['policy']; d['role']='offline_sensitivity'; d.to_parquet(out/'offline_C_selected_holdout.parquet',index=False)
    pd.concat(offline_metrics).to_csv(out/'offline_C_metrics.csv',index=False)
    pd.concat(offline_scores).to_csv(out/'offline_C_scores.csv',index=False)
    render_report(out)
    verification=json.loads((out/'verification.json').read_text())
    verification.update(independent_audit={'prediction_files':checks,'per_fold_provenance':boundaries,'source_reconstruction_rows':sample_feature_rows,'score_cases_recomputed':scorechecks,'offline_history_sensitivity_no_reselection':True,'tuning_reproduced':True,'development_selection_reproduced':True,'OOF_has_no_holdout':True},feature_cache_sha256=sha256(out/'features.parquet'),source_hashes={p.name:sha256(p) for p in Path(__file__).parent.glob('*third_experiment*.py')})
    assert all(sha256(DATA/p)==v for p,v in config['input_hashes'].items())
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    print(f'Audited {len(checks)} predictions, {sample_feature_rows} raw feature reconstructions, {scorechecks} scores; no violations.')
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(OUT)); audit(p.parse_args().out)
