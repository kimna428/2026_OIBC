"""Stage-three OIBC experiments. Only provided data; prospective selection lock.
Run stages in order; checkpoints contain input/settings fingerprints.
"""
from __future__ import annotations
import argparse, gc, hashlib, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from expanding_pipeline import DATA, OUT as PREVIOUS, FOLDS, SEED, issuance, operating_day, history_features, markdown, sha256
from validation import validate_composition, local_hours

OUT=PREVIOUS.parent/'third_experiment'
HISTORY=['lag48_cf','history7d_cf','history7d_count']
TUNING=[dict(n_estimators=60,learning_rate=.08,num_leaves=31,min_child_samples=20),
        dict(n_estimators=120,learning_rate=.05,num_leaves=31,min_child_samples=80),
        dict(n_estimators=180,learning_rate=.04,num_leaves=63,min_child_samples=160)]
SCALES={'200k':200000,'1m':1000000,'full':0}
POLICIES=['raw','night_only','train_calibrated']
T0=time.monotonic()

def log(*x): print(f'[{time.monotonic()-T0:.1f}s]',*x,flush=True)
def write_json(path,value): path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
def stamp(value): return pd.Timestamp(value,tz='Asia/Seoul')

def validation_mask(times, start, end):
    day=operating_day(times)
    return (day>=stamp(start))&(day<stamp(end))

def split_indices(g, fold):
    start,end=FOLDS[fold]; cutoff=stamp(start)-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    tr=np.flatnonzero((g.ts<=cutoff).to_numpy()); va=np.flatnonzero(validation_mask(g.ts,start,end))
    if not len(tr) or not len(va) or g.ts.iloc[tr].max()>issuance(g.ts.iloc[va]).min():
        raise ValueError('training target exceeds a validation issuance')
    return tr,va,cutoff

def fit_indices(g, tr, limit):
    idx=tr if not limit or len(tr)<=limit else tr[np.linspace(0,len(tr)-1,limit,dtype=int)]
    return idx[g.cf.iloc[idx].between(0,1).to_numpy()]

def select_development(table, value='cf_mae'):
    """No holdout rows may enter any selector, including accidental caller misuse."""
    if (table.fold>=5).any(): raise ValueError('holdout entered model selection')
    return table.groupby('candidate',sort=True)[value].mean().idxmin()

def parameter_schedule(table):
    if (table.fold>=4).any(): raise ValueError('tuning table contains unauthorized folds')
    schedule={}
    for family in ['B','C']:
        schedule[family]={}
        for fold,(start,_) in enumerate(FOLDS):
            cutoff=stamp(start)-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
            allowed=[i for i in range(4) if stamp(FOLDS[i][1])<=cutoff]
            if allowed:
                winner=select_development(table[(table.family==family)&table.fold.isin(allowed)])
                grid_index=int(winner.rsplit('_',1)[1])
                source_max=str(max(stamp(FOLDS[i][1]) for i in allowed))
            else: grid_index=0; source_max=None
            schedule[family][str(fold)]={'params':TUNING[grid_index],'completed_tuning_folds':allowed,'parameter_source_max':source_max,'prediction_issue_min':str(cutoff)}
    return schedule

def freeze_preferred_candidate(out):
    out=Path(out)
    lock=json.loads((out/'selection_lock.json').read_text())
    scores=pd.read_csv(out/'scores_selection.csv')
    if (scores.fold>=5).any(): raise ValueError('holdout in preferred candidate selection')
    choices=[]
    for family,c in lock['chosen'].items():
        z=scores[(scores.candidate==c['candidate'])&(scores.policy==c['policy'])&(scores.group=='total')&scores.score_available]
        choices.append({'family':family,**c,'development_penalty_krw':float(z.penalty_krw.sum())})
    preferred=min(choices,key=lambda c:(c['development_penalty_krw'],c['family']))
    value={'preferred':preferred,'holdout_used':False,'source':'scores_selection.csv','selection_source_max':lock['selection_source_max'],'selection_lock_sha256':sha256(out/'selection_lock.json')}
    path=out/'preferred_candidate_lock.json'
    if path.exists() and json.loads(path.read_text())!=value: raise ValueError('preferred candidate lock changed')
    if not path.exists():
        if list((out/'predictions').glob('*_fold5.parquet')): raise ValueError('preferred lock must precede holdout evaluation')
        write_json(path,value)
    return value

def frozen_history_features(g, target, observation_cap):
    """Sensitivity for an offline test horizon with no incoming generation files."""
    source=g[g.ts<=observation_cap]
    times={site:d for site,d in source.groupby('site_id',observed=True)}
    cut=issuance(target.ts).as_unit('ns').asi8
    lag=(pd.DatetimeIndex(target.ts)-pd.Timedelta(hours=48)).as_unit('ns').asi8
    result={c:np.full(len(target),np.nan,dtype='float32') for c in HISTORY}
    result['history7d_count'][:]=0
    for site,pos in target.groupby('site_id',observed=True).indices.items():
        if site not in times: continue
        d=times[site]; t=pd.DatetimeIndex(d.ts).as_unit('ns').asi8; cf=d.cf.to_numpy()
        right=np.searchsorted(t,cut[pos],side='right')
        left=np.searchsorted(t,cut[pos]-pd.Timedelta(hours=167).value,side='left')
        count=right-left; total=np.r_[0.,np.cumsum(cf)]
        good=count>0
        result['history7d_cf'][np.asarray(pos)[good]]=((total[right[good]]-total[left[good]])/count[good]).astype('float32')
        result['history7d_count'][pos]=count
        ix=np.searchsorted(t,lag[pos]); safe=np.minimum(ix,len(t)-1); found=(ix<len(t))&(t[safe]==lag[pos])
        result['lag48_cf'][np.asarray(pos)[found]]=cf[safe[found]]
    return pd.DataFrame(result)

def elevation(times,lat,lon):
    t=pd.DatetimeIndex(times)
    if t.tz is None: raise ValueError('solar timestamps must be aware')
    t=t.tz_convert('Asia/Seoul')
    gamma=2*np.pi*(t.dayofyear.to_numpy()-1)/365.25
    dec=np.deg2rad(23.44)*np.sin(gamma-1.39)
    eq=229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)-.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    hour=t.hour.to_numpy()+t.minute.to_numpy()/60+(np.asarray(lon)-135)/15+eq/60
    h=np.deg2rad(15*(hour-12)); p=np.deg2rad(np.asarray(lat))
    return np.rad2deg(np.arcsin(np.clip(np.sin(p)*np.sin(dec)+np.cos(p)*np.cos(dec)*np.cos(h),-1,1))).astype('float32')

def interval_elevation(times,lat,lon):
    t=pd.DatetimeIndex(times)
    return np.maximum.reduce([elevation(t-pd.Timedelta(hours=1),lat,lon),elevation(t-pd.Timedelta(minutes=30),lat,lon),elevation(t,lat,lon)])

def fit_night_thresholds(train, altitude, cutoff):
    if (train.ts>cutoff).any(): raise ValueError('night calibration uses future targets')
    rows=[]
    z=train[['site_id','cf']].copy(); z['altitude']=altitude
    for site,d in z.groupby('site_id',observed=True):
        threshold=-6.; support=0; max_cf=np.nan
        for t in [-12.,-9.,-6.,-3.,0.]:
            values=d.loc[d.altitude<=t,'cf']
            if len(values)>=168 and values.max()<=.001:
                threshold=t; support=len(values); max_cf=values.max()
        rows.append(dict(site_id=site,threshold=threshold,support=support,max_training_cf=max_cf))
    return pd.DataFrame(rows)

def postprocess(pred, max_altitude, thresholds, policy):
    pred=np.maximum(np.asarray(pred,dtype=float),0).copy()
    if policy=='raw': return pred
    cut=-6. if policy=='night_only' else np.asarray(thresholds)
    pred[np.asarray(max_altitude)<=cut]=0
    return pred

def latest_history_source(g,target):
    """Latest actual raw source in the trailing window, not the theoretical cutoff."""
    result=np.full(len(target),np.datetime64('NaT','ns'),dtype='datetime64[ns]')
    cut=issuance(target.ts).tz_convert('UTC').tz_localize(None).to_numpy(dtype='datetime64[ns]')
    source={site:d.ts.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(dtype='datetime64[ns]') for site,d in g.groupby('site_id',observed=True)}
    for site,pos in target.groupby('site_id',observed=True).indices.items():
        times=source[site]
        ix=np.searchsorted(times,cut[pos],side='right')-1
        good=ix>=0; safe=np.maximum(ix,0)
        good&=times[safe]>=cut[pos]-np.timedelta64(167,'h')
        result[np.asarray(pos)[good]]=times[safe[good]]
    return pd.to_datetime(result,utc=True).tz_convert('Asia/Seoul')

def join_forecast(g,sites,provider,round_number,columns):
    station=provider+'_station'; path=DATA/'train'/f'forecast_{provider}.parquet'
    d=pd.read_parquet(path,columns=[station,'ts','round']+columns)
    if not d['round'].isin([10,17]).all(): raise ValueError('unknown forecast issuance')
    d=d[d['round']==round_number].copy(); d.ts=pd.to_datetime(d.ts)
    if d.duplicated([station,'ts']).any(): raise ValueError('duplicate forecast keys')
    keys=pd.MultiIndex.from_frame(d[[station,'ts']])
    requested=pd.MultiIndex.from_arrays([g.site_id.map(sites.set_index('site_id')[station]),g.ts],names=[station,'ts'])
    pos=keys.get_indexer(requested); found=pos>=0
    result={}
    for c in columns:
        a=np.full(len(g),np.nan,dtype='float32'); a[found]=d[c].to_numpy(dtype='float32')[pos[found]]
        result[c]=a
    return result

class Experiment:
    def __init__(self,out=OUT):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        for name in ['models','predictions','provenance','checkpoints']: (self.out/name).mkdir(exist_ok=True)
        self.hashes=json.loads((PREVIOUS/'config.json').read_text())['input_hashes']
        if not all(sha256(DATA/p)==v for p,v in self.hashes.items()): raise ValueError('raw input differs from previous audit')
        self.sites=pd.read_csv(DATA/'sites.csv')
        self.g=pd.read_parquet(DATA/'train/generation.parquet').sort_values(['ts','site_id']).reset_index(drop=True)
        self.g.ts=local_hours(self.g.ts)
        self.g['capacity_kw']=self.g.site_id.map(self.sites.set_index('site_id').capacity_kw)
        self.g['cf']=self.g.energy/self.g.capacity_kw
        self.composition=pd.read_csv(PREVIOUS/'composition.csv'); validate_composition(self.composition,self.sites)
        prices=pd.read_csv(DATA/'train/smp.csv'); prices.ts=pd.to_datetime(prices.ts); self.prices=prices.set_index('ts').smp
        fits=json.loads((PREVIOUS/'fits.json').read_text())
        self.features={f:next(d['features'] for d in fits if d['experiment']=='V3_'+f) for f in ['B','C']}
        self.weather=[c for c in self.features['B'] if c.startswith(('wa_','wb_')) and c not in ['wa_station','wb_station','wa_dist','wb_dist']]
        cache=self.out/'features.parquet'
        feature_spec={'input_hashes':self.hashes,'previous_dataset_hash':sha256(PREVIOUS/'datasets/V3.parquet'),'feature_version':1}
        if cache.exists():
            if json.loads((self.out/'feature_spec.json').read_text())!=feature_spec: raise ValueError('feature cache fingerprint mismatch')
            self.x=pd.read_parquet(cache)
        else:
            d=pd.read_parquet(PREVIOUS/'datasets/V3.parquet')
            if not d[['site_id','ts']].equals(self.g[['site_id','ts']]): raise ValueError('previous feature alignment differs')
            self.x=d[self.features['C']].copy(); del d
            self.x.site_id=pd.Categorical(self.x.site_id,categories=list(self.sites.site_id)).codes.astype('int16')
            self.x.capacity_kw=self.x.capacity_kw.astype('float32')
            # Audit all cached history against raw inputs, before adding new forecast rounds.
            h=history_features(self.g,self.g)
            for c in HISTORY: np.testing.assert_allclose(self.x[c],h[c],equal_nan=True,atol=1e-7)
            del h
            for provider in ['wa','wb']:
                cols=[c for c in self.weather if c.startswith(provider+'_')]
                fresh=join_forecast(self.g,self.sites,provider,17,cols)
                for c in cols: np.testing.assert_allclose(self.x[c],fresh[c],equal_nan=True,rtol=0,atol=0)
                del fresh
                fresh=join_forecast(self.g,self.sites,provider,10,cols)
                for c in cols:
                    self.x[c+'_r10']=fresh[c]
                    self.x[c+'_revision']=self.x[c]-fresh[c]
                del fresh; gc.collect()
            self.x.to_parquet(cache,index=False); write_json(self.out/'feature_spec.json',feature_spec)
        self.alt=interval_elevation(self.g.ts,self.g.site_id.map(self.sites.set_index('site_id').lat),self.g.site_id.map(self.sites.set_index('site_id').lon))
        self.splits=[split_indices(self.g,i) for i in range(6)]
        self.thresholds={}
        for fold,(tr,va,cut) in enumerate(self.splits):
            path=self.out/f'night_thresholds_fold{fold}.csv'
            threshold=fit_night_thresholds(self.g.iloc[tr],self.alt[tr],cut)
            threshold.to_csv(path,index=False)
            self.thresholds[fold]=self.g.site_id.iloc[va].map(threshold.set_index('site_id').threshold).fillna(-6.).to_numpy()
        self.run_spec={'seed':SEED,'tuning_grid':TUNING,'scales':SCALES,'tuning_folds':[0,1,2,3],'selection_folds':[0,1,2,3,4],'holdout_fold':5,'weather_scale':'1m','selection_source_max':str(stamp(FOLDS[5][0])-pd.Timedelta(days=1)+pd.Timedelta(hours=17)),'input_hashes':self.hashes,'features':self.features,'lightgbm_version':importlib.metadata.version('lightgbm'),'receipt_assumption':'generation immediately available at its right-closed timestamp; actual receipt latency unknown','previous_holdout_already_observed':True,'pipeline_version':2}
        specpath=self.out/'run_spec.json'
        if specpath.exists() and json.loads(specpath.read_text())!=self.run_spec: raise ValueError('experiment settings changed; use a new output directory')
        write_json(specpath,self.run_spec)
        log('prepared',len(self.g),'rows',len(self.x.columns),'features')

    def columns(self,family,rounds):
        base=[c for c in self.features[family] if c not in self.weather]
        if rounds=='r17': return self.features[family]
        if rounds=='r10': return base+[c+'_r10' for c in self.weather]
        cols=self.features[family]+[c+'_r10' for c in self.weather]
        return cols if rounds=='both' else cols+[c+'_revision' for c in self.weather]

    def boundary_audit(self):
        rows=[]; differences=[]
        for fold,(tr,va,cut) in enumerate(self.splits):
            start,end=FOLDS[fold]; d=self.g.iloc[va].reset_index(drop=True)
            oldmask=(self.g.ts>stamp(start))&(self.g.ts<=stamp(end))
            np.testing.assert_array_equal(np.flatnonzero(oldmask),va)
            issued=issuance(d.ts)
            source=latest_history_source(self.g,d)
            lag=local_hours(d.ts)-pd.Timedelta(hours=48)
            prov=d[['site_id','ts']].copy()
            prov['operating_day']=operating_day(d.ts); prov['issued_at']=issued
            prov['training_target_max']=self.g.ts.iloc[tr].max(); prov['model_training_data_cutoff']=cut
            prov['r10_issued_at']=issued-pd.Timedelta(hours=7); prov['r17_issued_at']=issued
            prov['lag48_source_ts']=lag; prov['rolling_latest_actual_ts']=source
            for c in ['training_target_max','model_training_data_cutoff','r10_issued_at','r17_issued_at','lag48_source_ts','rolling_latest_actual_ts']:
                if (prov[c].notna()&(prov[c]>prov.issued_at)).any(): raise ValueError(f'as-of violation {c}')
            if not prov.operating_day.between(stamp(start),stamp(end)-pd.Timedelta(days=1)).all(): raise ValueError('operating date mismatch')
            prov.to_parquet(self.out/'provenance'/f'fold{fold}.parquet',index=False)
            rows.append(dict(fold=fold,operating_start=start,operating_end_exclusive=end,actual_first=str(d.ts.min()),actual_last=str(d.ts.max()),model_training_cutoff=str(cut),training_actual_max=str(self.g.ts.iloc[tr].max()),first_issue=str(issued.min()),last_issue=str(issued.max()),validation_rows=len(d),changed_rows=0,future_source_violations=0))
            for family in ['B','C']:
                old=pd.read_parquet(PREVIOUS/f'fold{fold}_V3_{family}_predictions.parquet')
                assert old[['site_id','ts']].reset_index(drop=True).equals(d[['site_id','ts']])
                expected=(old.pred_cf-old.cf).abs().mean()
                differences.append(dict(fold=fold,family=family,old_cf_mae=expected,explicit_operating_day_cf_mae=expected,delta=0.))
        pd.DataFrame(rows).to_csv(self.out/'boundary_audit.csv',index=False)
        pd.DataFrame(differences).to_csv(self.out/'boundary_metric_comparison.csv',index=False)
        log('boundary and per-sample as-of audit passed')

    def fit(self,fold,family,scale,rounds,params,tag):
        path=self.out/'predictions'/f'{tag}_fold{fold}.parquet'
        spec={'protocol_version':2,'fold':fold,'family':family,'scale':scale,'rounds':rounds,'params':params,'features':self.columns(family,rounds),'seed':SEED}
        fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        checkpoint=self.out/'checkpoints'/f'{tag}_fold{fold}.json'
        if path.exists() and checkpoint.exists():
            info=json.loads(checkpoint.read_text())
            if info['fingerprint']==fingerprint: return pd.read_parquet(path)
            log('refit changed protocol/settings',tag,fold)
        tr,va,cut=self.splits[fold]; idx=fit_indices(self.g,tr,SCALES[scale]); cols=spec['features']
        sample=self.g.iloc[idx][['site_id','ts']].copy()
        sample['season']=((operating_day(sample.ts).month%12)//3).astype(int)
        sample.groupby(['site_id','season'],observed=True).size().rename('rows').to_csv(self.out/f'sample_{tag}_fold{fold}.csv')
        model=lgb.LGBMRegressor(**params,random_state=SEED,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(self.x.iloc[idx][cols],self.g.cf.iloc[idx])
        model.booster_.save_model(str(self.out/'models'/f'{tag}_fold{fold}.txt'))
        pd.DataFrame({'feature':cols,'gain':model.booster_.feature_importance('gain'),'splits':model.booster_.feature_importance('split')}).to_csv(self.out/f'importance_{tag}_fold{fold}.csv',index=False)
        pd.DataFrame({'feature':cols,'train_missing_rate':self.x.iloc[idx][cols].isna().mean().to_numpy(),'validation_missing_rate':self.x.iloc[va][cols].isna().mean().to_numpy()}).to_csv(self.out/f'missingness_{tag}_fold{fold}.csv',index=False)
        d=self.g.iloc[va][['site_id','ts','energy','cf','capacity_kw']].copy().reset_index(drop=True)
        d['pred_cf']=np.maximum(model.predict(self.x.iloc[va][cols]),0)
        d['pred_kwh']=d.pred_cf*d.capacity_kw
        d['max_interval_elevation']=self.alt[va]
        d['learned_night_threshold']=self.thresholds[fold]
        counts=self.g.iloc[tr].groupby('site_id').size()
        d['cohort']=np.where(d.site_id.map(counts).fillna(0)>=2160,'established','new_under90d')
        d['missing_weather']=self.x.iloc[va][[c for c in cols if c in self.weather or c.endswith(('_r10','_revision'))]].isna().any(axis=1).to_numpy()
        d['fold']=fold; d['issued_at']=issuance(d.ts); d['training_cutoff']=cut
        selection_max=None
        if not tag.startswith('tune_'):
            schedule=json.loads((self.out/'parameter_schedule.json').read_text())
            selection_max=schedule[family][str(fold)]['parameter_source_max']
            if selection_max is not None and pd.Timestamp(selection_max)>issuance(d.ts).min(): raise ValueError('future tuning labels used')
        d['parameter_source_max']=pd.Timestamp(selection_max) if selection_max is not None else pd.NaT
        d.to_parquet(path,index=False)
        write_json(checkpoint,spec|{'fingerprint':fingerprint,'fit_rows':len(idx),'prediction_rows':len(d)})
        log('fit',tag,'fold',fold,'rows',len(idx))
        return d

    def evaluate(self,d,tag,policy,composition=None,composition_name='balanced'):
        z=d.copy(); z['pred_cf']=postprocess(z.pred_cf,z.max_interval_elevation,z.learned_night_threshold,policy)
        z['pred_kwh']=z.pred_cf*z.capacity_kw
        z['abs_cf']=(z.pred_cf-z.cf).abs(); z['abs_kwh']=(z.pred_kwh-z.energy).abs()
        z['hour']=local_hours(z.ts).hour
        z['period']=np.where(z.max_interval_elevation<=-6,'complete_night',np.where(z.max_interval_elevation<=6,'twilight','day'))
        days=operating_day(z.ts); z['month']=days.year*100+days.month
        removed=np.asarray(d.pred_cf)-z.pred_cf.to_numpy()
        z['removed_actual_kwh']=np.where(removed>0,z.energy.clip(lower=0),0)
        z['removed_prediction_kwh']=removed*z.capacity_kw
        rows=[]
        for key in [None,'site_id','hour','period','cohort','month','missing_weather']:
            if key is None: s=pd.DataFrame([{'label':'all','rows':len(z),'cf_mae':z.abs_cf.mean(),'mae_kwh':z.abs_kwh.mean(),'removed_actual_kwh':z.removed_actual_kwh.sum(),'removed_prediction_kwh':z.removed_prediction_kwh.sum()}])
            else:
                s=z.groupby(key,observed=True).agg(rows=('cf','size'),cf_mae=('abs_cf','mean'),mae_kwh=('abs_kwh','mean'),removed_actual_kwh=('removed_actual_kwh','sum'),removed_prediction_kwh=('removed_prediction_kwh','sum')).reset_index().rename(columns={key:'label'})
            s['dimension']=key or 'overall'; rows.append(s)
        metrics=pd.concat(rows,ignore_index=True); metrics['candidate']=tag; metrics['policy']=policy; metrics['fold']=int(z.fold.iloc[0])
        composition=self.composition if composition is None else composition
        validate_composition(composition,self.sites)
        complete=z.groupby('ts').site_id.nunique(); times=complete.index[complete==len(self.sites)]
        z['group']=z.site_id.map(composition.set_index('site_id').group)
        complete_z=z[z.ts.isin(times)]
        ledger=complete_z.groupby(['group','ts'],observed=True)[['energy','pred_kwh']].sum().reset_index()
        ledger['smp']=ledger.ts.map(self.prices); assert ledger.smp.notna().all()
        ledger['k']=np.where(ledger.ts.dt.hour.between(11,15),3,1)
        ledger['abs_error_kwh']=(ledger.pred_kwh-ledger.energy).abs()
        ledger['revenue_krw']=ledger.smp*ledger.energy; ledger['penalty_krw']=.1*ledger.smp*ledger.k*ledger.abs_error_kwh
        ledger['score_krw']=ledger.revenue_krw-ledger.penalty_krw
        groups=[]
        for group in ['G1','G2','total']:
            a=ledger if group=='total' else ledger[ledger.group==group]
            groups.append(dict(candidate=tag,policy=policy,fold=int(z.fold.iloc[0]),composition=composition_name,group=group,scored_hours=len(times),mae_kwh=a.abs_error_kwh.mean() if len(a) else np.nan,penalty_krw=a.penalty_krw.sum() if len(a) else np.nan,score_krw=a.score_krw.sum() if len(a) else np.nan,score_available=bool(len(a))))
        # Night/day settlement is classified by KST hour, separately from site-specific solar bins.
        ledger['period']=np.where(ledger.ts.dt.hour.between(7,18),'clock_day_07_18','clock_other')
        period=ledger.groupby(['group','period'],observed=True).agg(scored_hours=('ts','size'),mae_kwh=('abs_error_kwh','mean'),penalty_krw=('penalty_krw','sum'),score_krw=('score_krw','sum')).reset_index()
        period['candidate']=tag; period['policy']=policy; period['fold']=int(z.fold.iloc[0]); period['composition']=composition_name
        return metrics,pd.DataFrame(groups),period

    def tune(self):
        rows=[]
        for family in ['B','C']:
            for i,params in enumerate(TUNING):
                tag=f'tune_{family}_{i}'
                for fold in range(4):
                    d=self.fit(fold,family,'200k','r17',params,tag)
                    rows.append(dict(candidate=tag,family=family,grid_index=i,fold=fold,cf_mae=(d.pred_cf-d.cf).abs().mean()))
            pd.DataFrame(rows).to_csv(self.out/'tuning_results.csv',index=False)
        table=pd.DataFrame(rows); chosen={}
        for family in ['B','C']:
            tag=select_development(table[table.family==family]); i=int(tag.rsplit('_',1)[1]); chosen[family]=TUNING[i]
        write_json(self.out/'tuned_parameters.json',chosen)
        write_json(self.out/'parameter_schedule.json',parameter_schedule(table))
        return chosen

    def experiments(self,params,folds):
        metrics=[]; scores=[]; periods=[]; selection_scores=[]
        final_issue=self.splits[5][2]
        for family in ['B','C']:
            settings=[(scale,'r17') for scale in SCALES]+[('1m',r) for r in ['r10','both','revision']]
            for scale,rounds in settings:
                tag=f'{family}_{scale}_{rounds}'
                for fold in folds:
                    schedule=json.loads((self.out/'parameter_schedule.json').read_text())
                    effective=schedule[family][str(fold)]['params']
                    d=self.fit(fold,family,scale,rounds,effective,tag)
                    for policy in POLICIES:
                        m,s,p=self.evaluate(d,tag,policy); metrics.append(m); scores.append(s); periods.append(p)
                        if max(folds)<5:
                            eligible=d[d.ts<=final_issue].copy()
                            if len(eligible):
                                _,ss,_=self.evaluate(eligible,tag,policy)
                                ss['selection_source_max']=str(final_issue); selection_scores.append(ss)
                suffix='development' if max(folds)<5 else 'holdout'
                pd.concat(metrics).to_csv(self.out/f'metrics_{suffix}.csv',index=False)
                pd.concat(scores).to_csv(self.out/f'scores_{suffix}.csv',index=False)
                pd.concat(periods).to_csv(self.out/f'settlement_periods_{suffix}.csv',index=False)
        if selection_scores: pd.concat(selection_scores).to_csv(self.out/'scores_selection.csv',index=False)
        return pd.concat(metrics),pd.concat(selection_scores) if selection_scores else pd.concat(scores)

    def freeze_selection(self,metrics,scores):
        if (metrics.fold>=5).any() or (scores.fold>=5).any(): raise ValueError('selection touched holdout')
        chosen={}
        for family in ['B','C']:
            s=scores[(scores.group=='total')&scores.score_available&scores.candidate.str.startswith(family+'_')].copy()
            # Only fold 4 has complete fleet; compare exactly the same hours.
            if len(s):
                key=s.groupby(['candidate','policy']).penalty_krw.sum().sort_values().index[0]
                selection_metric='sum complete-fleet development penalty (fold4 only)'
            else:
                t=metrics[(metrics.dimension=='overall')&metrics.candidate.str.startswith(family+'_')]
                key=t.groupby(['candidate','policy']).cf_mae.mean().sort_values().index[0]; selection_metric='development macro CF MAE (fallback)'
            chosen[family]={'candidate':key[0],'policy':key[1],'criterion':selection_metric}
        lock={'chosen':chosen,'parameters':json.loads((self.out/'tuned_parameters.json').read_text()),'selection_used_folds':[0,1,2,3,4],'holdout_used_for_selection':False,'selection_source_max':str(self.splits[5][2]),'holdout_previously_explored_in_stage2':True}
        path=self.out/'selection_lock.json'
        if path.exists() and json.loads(path.read_text())!=lock: raise ValueError('frozen selection changed')
        write_json(path,lock)
        freeze_preferred_candidate(self.out)
        return lock

    def oof_and_groups(self,lock):
        result={}; traces=[]; comparisons=[]
        for family,chosen in lock['chosen'].items():
            ds=[]
            for fold in range(5):
                d=pd.read_parquet(self.out/'predictions'/f"{chosen['candidate']}_fold{fold}.parquet")
                d['pred_cf']=postprocess(d.pred_cf,d.max_interval_elevation,d.learned_night_threshold,chosen['policy']); d['pred_kwh']=d.pred_cf*d.capacity_kw
                ds.append(d)
            d=pd.concat(ds,ignore_index=True)
            d=d[d.ts<=self.splits[5][2]].copy()
            if (d.fold>=5).any() or (d.ts>stamp('2025-03-01')).any(): raise ValueError('holdout in OOF optimization')
            d['candidate']=chosen['candidate']; d['policy']=chosen['policy']; d['role']='training_oof'
            d['residual_kwh']=d.energy-d.pred_kwh; d['residual_cf']=d.cf-d.pred_cf
            d.to_parquet(self.out/f'OOF_{family}.parquet',index=False)
            residual=d.pivot(index='ts',columns='site_id',values='residual_kwh').reindex(columns=self.sites.site_id)
            residual.corr(min_periods=168).to_csv(self.out/f'oof_{family}_residual_correlation.csv')
            valid=residual.notna().astype('int32'); (valid.T@valid).to_csv(self.out/f'oof_{family}_residual_overlap.csv')
            # Pairwise same-hour opposing signs, using only joint observations.
            positive=(residual>0).astype('int32'); negative=(residual<0).astype('int32')
            opposite=positive.T@negative+negative.T@positive
            opposite.divide(valid.T@valid).to_csv(self.out/f'oof_{family}_opposite_sign_rate.csv')
            complete=residual.dropna(how='any')
            if not len(complete): raise ValueError('no complete OOF fleet for composition optimization')
            r=complete.to_numpy(); weight=.1*self.prices.reindex(complete.index).to_numpy()*np.where(complete.index.hour.isin(range(11,16)),3,1)
            start=self.composition.set_index('site_id').group.reindex(complete.columns).eq('G1').to_numpy()
            capacity=self.sites.set_index('site_id').capacity_kw.reindex(complete.columns).to_numpy()
            best,trace=optimize_groups(r,weight,capacity,start,iterations=5000)
            comp=pd.DataFrame({'site_id':complete.columns,'group':np.where(best,'G1','G2')})
            caps=validate_composition(comp,self.sites); comp.to_csv(self.out/f'composition_{family}.csv',index=False)
            result[family]=comp
            for t in trace: traces.append(t|{'family':family})
            for name,mask in [('balanced',start),('optimized',best)]:
                err1=r[:,mask].sum(axis=1); total=r.sum(axis=1); penalty=weight*(np.abs(err1)+np.abs(total-err1))
                independent=weight*np.abs(r).sum(axis=1)
                comparisons.append(dict(family=family,composition=name,training_oof_hours=len(r),penalty_krw=penalty.sum(),sum_individual_penalty_krw=independent.sum(),cancellation_fraction=1-penalty.sum()/independent.sum(),G1_kw=capacity[mask].sum(),G2_kw=capacity[~mask].sum()))
        pd.DataFrame(traces).to_csv(self.out/'optimization_trace.csv',index=False)
        pd.DataFrame(comparisons).to_csv(self.out/'optimization_development.csv',index=False)
        write_json(self.out/'composition_lock.json',{'used_folds':[0,1,2,3,4],'source_max':str(self.splits[5][2]),'holdout_used':False,'composition_hashes':{f:sha256(self.out/f'composition_{f}.csv') for f in result},'selection_lock_hash':sha256(self.out/'selection_lock.json')})
        return result

    def evaluate_selected_groups(self,lock,compositions):
        rows=[]; periods=[]
        for family,choice in lock['chosen'].items():
            for fold in range(6):
                d=pd.read_parquet(self.out/'predictions'/f"{choice['candidate']}_fold{fold}.parquet")
                for name,comp in [('balanced',self.composition),('optimized',compositions[family])]:
                    _,s,p=self.evaluate(d,choice['candidate'],choice['policy'],comp,name); s['family']=family; rows.append(s); p['family']=family; periods.append(p)
            d=pd.read_parquet(self.out/'predictions'/f"{choice['candidate']}_fold5.parquet")
            d['pred_cf']=postprocess(d.pred_cf,d.max_interval_elevation,d.learned_night_threshold,choice['policy']); d['pred_kwh']=d.pred_cf*d.capacity_kw
            d['candidate']=choice['candidate']; d['policy']=choice['policy']; d['role']='holdout'
            d.to_parquet(self.out/f'selected_{family}_holdout.parquet',index=False)
        pd.concat(rows).to_csv(self.out/'selected_composition_metrics.csv',index=False)
        pd.concat(periods).to_csv(self.out/'selected_composition_periods.csv',index=False)

    def march_analysis(self,lock):
        """Forecast-conditioned diagnostic, with calibration fitted on pre-March OOF only."""
        from sklearn.linear_model import Ridge
        choice=lock['chosen']['B']; tag=choice['candidate']
        controls=['solar_elevation','wa_cloud','wa_temp','wb_cloud','wb_uv_idx','wb_temp']
        keys=pd.MultiIndex.from_frame(self.g[['site_id','ts']])
        old=[]
        for fold in range(5):
            d=pd.read_parquet(self.out/'predictions'/f'{tag}_fold{fold}.parquet'); old.append(d)
        pre=pd.concat(old,ignore_index=True)
        pre=pre[pre.ts<=self.splits[5][2]].copy(); march=pd.read_parquet(self.out/'predictions'/f'{tag}_fold5.parquet')
        march=march[operating_day(march.ts).month==3].copy()
        def matrix(d):
            pos=keys.get_indexer(pd.MultiIndex.from_frame(d[['site_id','ts']]))
            z=self.x.iloc[pos][controls].copy().reset_index(drop=True); z['forecast_cf']=d.pred_cf.to_numpy()
            return z
        a=matrix(pre); b=matrix(march); med=a.median(); a=a.fillna(med); b=b.fillna(med)
        mean=a.mean(); std=a.std().replace(0,1)
        cal=Ridge(alpha=100).fit((a-mean)/std,pre.cf-pre.pred_cf)
        for d,x in [(pre,a),(march,b)]:
            d['residual_cf']=d.cf-d.pred_cf; d['conditional_residual_cf']=d.residual_cf-cal.predict((x-mean)/std)
            d['day']=operating_day(d.ts); d['daylight']=d.max_interval_elevation>6
        write_json(self.out/'march_control_model.json',{'controls':list(a),'median':med.to_dict(),'mean':mean.to_dict(),'std':std.to_dict(),'coefficient':cal.coef_.tolist(),'intercept':cal.intercept_,'fit_max_ts':str(pre.ts.max()),'holdout_labels_used_to_fit':False})
        def daily(d):
            return d[d.daylight].groupby('day').agg(rows=('cf','size'),actual_cf=('cf','mean'),pred_cf=('pred_cf','mean'),residual_cf=('residual_cf','mean'),conditional_residual_cf=('conditional_residual_cf','mean'))
        reference=daily(pre); day=daily(march)
        day['empirical_lower_tail_p']=[(1+(reference.conditional_residual_cf<=v).sum())/(len(reference)+1) for v in day.conditional_residual_cf]
        p=day.empirical_lower_tail_p.to_numpy(); order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]; q=np.empty(len(p)); q[order]=np.minimum(adjusted,1); day['BH_q']=q
        day.to_csv(self.out/'march_daily_cf_residual.csv')
        regions=self.sites.set_index('site_id').copy()
        regions['anonymous_grid']=pd.qcut(regions.lat,4,labels=False).astype(str)+'_'+pd.qcut(regions.lon,4,labels=False).astype(str)
        for name in ['wa_station','anonymous_grid']:
            march[name]=march.site_id.map(regions[name])
            regional=march[march.daylight].groupby(['day',name]).agg(sites=('site_id','nunique'),rows=('cf','size'),residual_cf=('residual_cf','mean'),conditional_residual_cf=('conditional_residual_cf','mean'))
            regional.to_csv(self.out/f'march_regional_{name}.csv')
        march[['site_id','ts','cf','pred_cf','residual_cf','conditional_residual_cf','daylight','wa_station','anonymous_grid']].to_parquet(self.out/'march_residual_diagnostics.parquet',index=False)
        # Identify contiguous negative days descriptively, without assigning a cause.
        neg=day.conditional_residual_cf<0; blocks=(neg!=neg.shift()).cumsum()
        runs=[]
        for _,z in day[neg].groupby(blocks[neg]): runs.append({'start':str(z.index.min()),'end':str(z.index.max()),'days':len(z),'mean_conditional_residual_cf':z.conditional_residual_cf.mean()})
        pd.DataFrame(runs,columns=['start','end','days','mean_conditional_residual_cf']).to_csv(self.out/'march_negative_runs.csv',index=False)

    def finalize(self):
        if not all(sha256(DATA/p)==v for p,v in self.hashes.items()): raise ValueError('raw inputs modified')
        lock=json.loads((self.out/'selection_lock.json').read_text()); comp_lock=json.loads((self.out/'composition_lock.json').read_text())
        assert comp_lock['selection_lock_hash']==sha256(self.out/'selection_lock.json')
        for f,h in comp_lock['composition_hashes'].items(): assert h==sha256(self.out/f'composition_{f}.csv')
        write_json(self.out/'verification.json',{'raw_hashes_unchanged':True,'input_hashes':self.hashes,'asof_per_sample_files':6,'operating_day_boundary_changes':0,'holdout_excluded_from_tuning_and_selection':True,'holdout_prior_exposure':True,'oof_max_fold':4,'feature_receipt_assumption':self.run_spec['receipt_assumption'],'source_hash':sha256(Path(__file__))})
        render_report(self.out)

def optimize_groups(residual,weight,capacity,start,iterations=5000):
    capacity=np.asarray(capacity); mask=np.asarray(start,dtype=bool).copy(); r=np.asarray(residual); weight=np.asarray(weight)
    if not np.isfinite(r).all(): raise ValueError('missing target cannot be filled for group optimization')
    total_cap=capacity.sum()
    if capacity[mask].sum()<20000 or capacity[~mask].sum()<20000: raise ValueError('invalid initial capacity')
    total=r.sum(axis=1); a=r[:,mask].sum(axis=1)
    def cost(v): return float(np.dot(weight,np.abs(v)+np.abs(total-v)))
    best=cost(a); cap=capacity[mask].sum(); trace=[dict(iteration=0,penalty_krw=best,G1_kw=cap)]
    rng=np.random.default_rng(SEED)
    for i in range(iterations):
        ix=int(rng.integers(len(mask))); proposed=[ix]
        if rng.random()<.7:
            other=np.flatnonzero(mask!=mask[ix]); proposed.append(int(rng.choice(other)))
        delta_cap=sum(capacity[j]*(-1 if mask[j] else 1) for j in proposed)
        if not 20000<=cap+delta_cap<=total_cap-20000: continue
        delta=sum((r[:,j]*(-1 if mask[j] else 1) for j in proposed))
        value=cost(a+delta)
        if value<best-1e-7:
            mask[proposed]=~mask[proposed]; a+=delta; cap+=delta_cap; best=value
            trace.append(dict(iteration=i+1,penalty_krw=best,G1_kw=cap))
    return mask,trace


def render_report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lock=json.loads((out/'selection_lock.json').read_text())
    dev=pd.read_csv(out/'metrics_development.csv'); hold=pd.read_csv(out/'metrics_holdout.csv')
    scores=pd.concat([pd.read_csv(out/'scores_development.csv'),pd.read_csv(out/'scores_holdout.csv')])
    overall=pd.concat([dev,hold]); overall=overall[overall.dimension=='overall']
    overall.to_csv(out/'overall_comparison.csv',index=False)
    compare=[]
    for family in ['B','C']:
        for fold in range(6):
            old=pd.read_parquet(PREVIOUS/f'fold{fold}_V3_{family}_predictions.parquet',columns=['pred_cf','cf'])
            baseline=(old.pred_cf-old.cf).abs().mean()
            for row in overall[(overall.fold==fold)&overall.candidate.str.startswith(family+'_')].itertuples(): compare.append(dict(fold=fold,family=family,candidate=row.candidate,policy=row.policy,cf_mae=row.cf_mae,old_cf_mae=baseline,delta=row.cf_mae-baseline))
    comparison=pd.DataFrame(compare); comparison.to_csv(out/'improvement_vs_stage2.csv',index=False)
    selected=[]
    for f,c in lock['chosen'].items():
        d=overall[(overall.candidate==c['candidate'])&(overall.policy==c['policy'])].copy(); d['family']=f; selected.append(d)
    selected=pd.concat(selected)
    selected.pivot(index='fold',columns='family',values='cf_mae').plot(marker='o',figsize=(9,4)); plt.ylabel('CF MAE'); plt.tight_layout(); plt.savefig(out/'selected_fold_mae.png'); plt.close()
    scale=overall[(overall.policy=='raw')&overall.candidate.str.endswith('_r17')].copy(); scale.pivot(index='fold',columns='candidate',values='cf_mae').plot(figsize=(11,5)); plt.ylabel('CF MAE'); plt.tight_layout(); plt.savefig(out/'scale_comparison.png'); plt.close()
    day=pd.read_csv(out/'march_daily_cf_residual.csv'); day['day']=pd.to_datetime(day.day)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True); axes[0].plot(day.day,day.actual_cf,label='actual'); axes[0].plot(day.day,day.pred_cf,label='forecast'); axes[0].legend(); axes[0].set_ylabel('Daylight CF'); axes[1].plot(day.day,day.residual_cf,label='actual - forecast'); axes[1].plot(day.day,day.conditional_residual_cf,label='weather-calibrated residual'); axes[1].axhline(0,color='black',lw=.5); axes[1].legend(); plt.tight_layout(); plt.savefig(out/'march_daily.png'); plt.close()
    composition=pd.read_csv(out/'selected_composition_metrics.csv')
    text='''# OIBC 2026 3차 실험

원본/외부 데이터 정책: 제공 파일만 읽고 실측 기상·외부 산불 기록·사전학습 모델을 사용하지 않았습니다. raw SHA256을 전후 대조했습니다.

## 검증 경계와 가용 시각

운영일 D는 (D 00:00, D+1 00:00]입니다. 기존 코드의 실제 비교는 `ts > start` / `ts <= end`이므로 3월1일00시는 봄 fold에 포함되지 않았습니다. 기존 start는 배타 경계였으며 이번에는 운영일과 실제 첫/마지막 샘플을 명시합니다. 운영일 기반 재선택과 기존 샘플이 전수 일치하여 경계 수정에 의한 성능 차이는 0입니다. boundary_audit.csv, boundary_metric_comparison.csv와 provenance/fold*.parquet에 각 샘플 발행·학습 종료·예보10/17 발행·lag/rolling 원천 시각을 저장했습니다. 모든 원천 시각은 발행 시각 이하입니다. 실제 발전실적 수신 지연은 제공 파일에 없으므로 즉시 수신 가정은 여전히 조건입니다. 평가용 test 폴더에는 예보만 있고 기간 중 실적이 제공되지 않습니다. 따라서 C의 rolling 성능은 해당 기간에도 과거 실적을 수신할 수 있는 경우에 한하며, 별도 실적 제공이 확인되기 전에는 오프라인 제출 가능 성능으로 간주할 수 없습니다. B는 제공 예보와 정적/시간 피처로 계산 가능합니다. 정적 메타가 과거에도 유효했다는 가정도 남습니다.

## 선택 절차와 holdout 한계

최종 holdout용 튜닝은 fold0~3의 200k CF MAE 평균으로 세 가지 고정 grid를 비교합니다. 과거 fold는 첫 발행 전에 끝난 튜닝 fold만 사용해 별도로 설정을 고릅니다. fold0은 사전 고정 grid0, fold1은 튜닝fold0, fold2/3은 튜닝fold0~1, fold4는 튜닝fold0~2, fold5는 튜닝fold0~3을 사용합니다. 직전 fold의 마지막00시가 첫 발행 이후인 경우 해당 튜닝 구간 전체를 제외했습니다. parameter_schedule.json과 예측의 parameter_source_max에 근거를 저장합니다. 트리 수·learning rate·num_leaves·min_data_in_leaf를 함께 바꾸는 제한된 탐색입니다. 코드의 min_child_samples는 min_data_in_leaf의 sklearn API 별칭입니다. 튜닝에 완전한 fleet 정답 시간이 없어 CF MAE를 대리 지표로 사용했습니다. 이후 학습량·예보라운드·후처리는 fold0~4 중 완전 fleet의 공식 정산 페널티로 선택하며, 해당 시간은 fold4에만 있습니다. 최종 봄 fold5는 튜닝/선택/그룹최적화에서 제외하고, 직전 개발fold4의 마지막7시간(2월28일18시~3월1일00시)도 첫 발행 시각 이후이므로 선택·그룹최적화·3월 잔차 보정 학습에서 제외했습니다. scores_selection.csv가 이 embargo를 적용한 선택용 점수이고 scores_development.csv는 전체 개발기간 기술 점수입니다. 또한 selection_lock.json 및 composition_lock.json을 먼저 저장한 뒤 평가했습니다. 단, 봄 구간은 2차 보고서에서 이미 확인한 자료이므로 완전히 새로 봉인된 미관측 holdout은 아닙니다. 이 한계는 이번 절차로 소급 제거할 수 없습니다. fold4도 선택에 사용했으므로 그 성능은 최종 독립 성능이 아닙니다.

## 학습량·예보·야간 후처리

B/C 각각 200k·1m·전체 학습과 r17·r10·r10+r17·두 라운드+수정량을 비교했습니다. 라운드 비교는 동일한 최대100만 학습 행에서 수행했습니다. 학습량 효과는 r17을 고정한200k/1m/full 비교로 분리했습니다. 같은 fold의 모델들은 같은 정답 마스크·동일 family별 튜닝 설정을 사용합니다. fold별 파라미터는 엄격한 과거 가용성 때문에 다를 수 있습니다. 200k/1m는 이전 실험과 같은 시간순 등간격 표본이며 이상치는 이후 제외합니다. sample_*.csv는 발전소×계절(0=겨울,1=봄,2=여름,3=가을)별 표본 수입니다. 전체행 모델도 각 fold 이전 정답만 사용합니다. 1m는 최대100만 행이며 첫 fold는 과거 유효 데이터가 약96만 행뿐이므로 전체행과 같습니다. 부족한 행을 복제하거나 이후 자료로 채우지 않았습니다. feature importance gain/splits와 변수별 학습/검증 결측률을 저장했습니다. missing_weather별 오차는 관측 분포의 차이를 포함하는 기술 통계이며 결측의 인과효과는 아닙니다. 예보 결측은 native missing branch로 처리했습니다.

raw는 기존과 같은 음수 예측 하한 0입니다. night_only는 시간구간 시작·중앙·끝의 최대 태양고도가 -6도 이하일 때만 0으로 합니다. train_calibrated는 발전소별 학습 관측168시간 이상이며 최대 CF<=.001인 고도 임계값(-12/-9/-6/-3/0)을 찾습니다. 불충분한 발전소는 -6도 fallback입니다. 학습 구간만 사용하고 검증 정답으로 조건을 맞추지 않습니다. 야간 최대고도 검사는 일출·일몰 한 시간 구간에 일부 햇빛이 있는 경우를 보수적으로 보존합니다. metrics의 twilight와 removed_actual_kwh는 실제 발전량 제거량을 보여줍니다. 집합 야간/주간 페널티는 별도 고정 시계시간(07~18시/나머지)이며 발전소별 고도 분류와 구분합니다.

## 사전 선택된 후보

'''
    preferred=freeze_preferred_candidate(out)
    text+='전역 추천도 개발 선택용 페널티만 비교해 결정했습니다. 봄 정답으로 B/C 중 하나를 고르지 않았습니다.\n\n```json\n'+json.dumps(preferred,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='```json\n'+json.dumps(lock['chosen'],ensure_ascii=False,indent=2)+'\n```\n\n'+markdown(selected[['family','fold','candidate','policy','cf_mae','mae_kwh']])
    text+='\n\n## 공식 정산 결과 (각 fold 안에서만 비교)\n\n'+markdown(composition[composition.group=='total'])
    text+='''

공식 산식은 README의 S=ΣSMP·실적−0.1ΣSMP·k·|오차|, k=11~15시3/기타1입니다. 그룹 구성은253개소를 정확히 G1/G2로 분할하며 각20,000kW 이상입니다. 원본 정답253개소가 모두 있는 시간만 집합 채점합니다. 없는 정답을 0으로 채우지 않습니다. scores_*.csv에서 score_available=False 및 NaN인 fold는 점수를 계산할 수 없습니다. fold마다 평가 시간이 달라 총 정산금을 fold 사이에 단순 비교하지 않습니다.

## OOF 그룹 최적화

OOF_B/C.parquet은 fold0~4 중 최종 holdout 첫 발행 시각(2025-02-28 17시)까지의 관측만 포함합니다. 모든 예측은 해당 운영일 발행 전 자료로 학습했고, C는 일별로 수신된 과거 실적을 갱신하는 rolling-origin입니다. 그룹 선택은 이 학습 OOF 중 253개소 완전 관측인 fold4 시간만 사용합니다. 초기 용량 균형 배정에서 고정seed 단일 이동/쌍 교환5000회로 페널티를 낮춥니다. 전역 최적을 보장하지 않습니다. 잔차 상관·공통 관측 수·상반 부호 비율은 pairwise 공통 시각만 사용하고 NaN을0으로 채우지 않습니다. 상반부호 비율의 0은 실제 잔차 부호가0인 경우도 포함합니다. optimization_development.csv에 개별 오차합 대비 상쇄율과 그룹 용량을 저장했습니다. Holdout 그룹 성능은 selected_composition_metrics.csv에 독립적으로 평가했습니다. 다른 family의 예측에 대해 최적화한 그룹을 섞어 선택하지 않았습니다.

## 2025년3월 잔차와 산불 가설

V3_B 계열의 사전 선택 모델 raw 예측(actual−forecast)을 분석했습니다. 이 모델 자체가 기상예보·시간·발전소 메타를 조건으로 예측합니다. 추가로 이전 OOF 잔차에 Ridge를 학습해 기상·태양고도·예측 CF에 따른 체계적 편향을 보정했습니다. 3월 정답은 보정 모델 학습에 사용하지 않았습니다. march_daily_cf_residual.csv에 주간CF·원잔차·조건부 잔차·과거 일평균 대비 경험적 하위꼬리 p 및 BH 보정값을 저장했습니다. 이는 계절/공간/시간상관 및 OOF 보정의 훈련내 편향을 완전히 통제하지 못하는 탐색치입니다. 통계적 인과 검정으로 해석하지 않습니다. 과거 날짜 분포가 봄과 교환가능하다는 보장도 없습니다.

지역 패턴은 제공된 익명 WA지점 및 비식별 좌표의4×4분위수 격자로만 집계합니다. 실제 지역·설비·사업자를 추정하지 않습니다. 특정 날 음의 잔차 또는 지역 동시 감소가 있어도 원인은 계측·예보오차·운영·미관측 기상 등과 구분되지 않습니다. 2025년 월평균 하락은 산불 증거가 아닙니다. 외부 산불 데이터는 사용하지 않았습니다.

## 상세 결과와 재현

- metrics_development/holdout.csv: 모든 실험의 발전소·시간대·월·주야간·신규/기존·기상결측별 CF MAE/kWh MAE, 제거량.
- scores_development/holdout.csv: 각 실험/후처리의 G1/G2 및 전체 MAE, 공식 페널티/정산금, 유효 시간 수.
- settlement_periods_*.csv: 시계시간 기준 집합 주야간 MAE/페널티/정산금.
- improvement_vs_stage2.csv: 같은 fold 원본 V3_B/C 대비 CF MAE 차이.
- models/, predictions/, checkpoints/: 모든 모델·원본 예측·설정 fingerprint. feature_spec.json은 원본+이전 피처 파일 해시를 고정합니다.
- selected_B/C_holdout.parquet: 사전 선택된 모델/후처리 holdout 예측. OOF_B/C는 그룹 최적화용이며 holdout을 포함하지 않습니다.

이 결과는 세 grid의 제한된 튜닝과 겨울 OOF만으로 그룹을 선택한 결과입니다. 다음 우선순위는 새로운 미관측 기간 평가, 발전 실적 수신지연 확인, 더 긴 완전fleet OOF 축적과 그룹배정 안정성 확인입니다.

![fold](selected_fold_mae.png)

![scale](scale_comparison.png)

![march](march_daily.png)
'''
    text+='\n## 기존 대비 개선 및 학습량 비교\n\n'
    chosen_comparisons=[]
    for f,c in lock['chosen'].items():
        chosen_comparisons.append(comparison[(comparison.candidate==c['candidate'])&(comparison.policy==c['policy'])])
    text+=markdown(pd.concat(chosen_comparisons)[['fold','family','cf_mae','old_cf_mae','delta']])
    raw=overall[overall.policy=='raw']
    scale_summary=raw.groupby('candidate').cf_mae.mean().rename('macro_6fold_cf_mae').reset_index()
    # Macro averages below are descriptive; selection never reads holdout scores.
    text+='\n\n6개 fold 평균은 사후 기술 요약이며 모델 선택에는 사용하지 않았습니다.\n\n'+markdown(scale_summary)
    selected_scores=[]
    for f,c in lock['chosen'].items():
        selected_scores.append(scores[(scores.candidate==c['candidate'])&(scores.group=='total')])
    text+='\n## 선택 후보의 야간 후처리별 공식 페널티\n\n'+markdown(pd.concat(selected_scores)[['fold','candidate','policy','scored_hours','penalty_krw','score_krw']])
    detail=pd.concat([dev,hold]); twilight=detail[(detail.dimension=='period')&(detail.label=='twilight')]
    matched=[]
    for f,c in lock['chosen'].items(): matched.append(twilight[twilight.candidate==c['candidate']])
    text+='\n## 일출·일몰 인접 구간 영향\n\n'+markdown(pd.concat(matched)[['fold','candidate','policy','rows','cf_mae','removed_actual_kwh','removed_prediction_kwh']])
    runs=pd.read_csv(out/'march_negative_runs.csv')
    text+='\n## 3월 조건부 음의 잔차 연속 구간 (탐색)\n\n'+markdown(runs)
    baseline=pd.read_parquet(PREVIOUS/'fold5_V3_B_predictions.parquet',columns=['site_id','ts','cf','pred_cf'])
    diagnostic=pd.read_parquet(out/'march_residual_diagnostics.parquet',columns=['site_id','ts','daylight','residual_cf','conditional_residual_cf'])
    baseline=baseline.merge(diagnostic,on=['site_id','ts'],how='inner',validate='one_to_one')
    baseline['day']=operating_day(baseline.ts); baseline['old_residual_cf']=baseline.cf-baseline.pred_cf
    old_daily=baseline[baseline.daylight].groupby('day').agg(old_residual_cf=('old_residual_cf','mean'),new_residual_cf=('residual_cf','mean'),conditional_residual_cf=('conditional_residual_cf','mean'))
    old_daily.to_csv(out/'march_stage2_stage3_residual_comparison.csv')
    text+='\n\n이전 V3_B와 개선 B의 동일 주간 관측 잔차 비교도 march_stage2_stage3_residual_comparison.csv에 저장했습니다. 음의 잔차 날짜는 모델 설정에 따라 달라질 수 있습니다.\n'
    lowest=day.sort_values('conditional_residual_cf').head(7)
    text+='\n\n음의 조건부 잔차가 큰 날짜: \n\n'+markdown(lowest[['day','actual_cf','pred_cf','residual_cf','conditional_residual_cf','empirical_lower_tail_p','BH_q']])
    regional=pd.read_csv(out/'march_regional_anonymous_grid.csv')
    synchrony=regional.groupby('day').conditional_residual_cf.agg(regions='size',negative_regions=lambda z:int((z<0).sum()),mean_residual='mean')
    synchrony['negative_region_fraction']=synchrony.negative_regions/synchrony.regions
    synchrony.to_csv(out/'march_regional_synchrony.csv')
    text+='\n\n익명 격자에서 같은 날 음의 잔차가 나타난 비율: \n\n'+markdown(synchrony.sort_values('mean_residual').head(7))
    text+='\n\n위 날짜나 동시성은 이상 패턴 후보이며 산불 발생일·피해지역의 추정 또는 원인 확정이 아닙니다. 예보 오차와 관측되지 않은 기상/운영 변화가 남아 있습니다.\n'
    # Inspect sample composition against full eligible history in each fold.
    distribution=[]; sample_summary=[]
    for fold in range(6):
        full=pd.read_csv(out/f'sample_B_full_r17_fold{fold}.csv').rename(columns={'rows':'full_rows'})
        for scale in ['200k','1m','full']:
            sample=pd.read_csv(out/f'sample_B_{scale}_r17_fold{fold}.csv').rename(columns={'rows':'selected_rows'})
            merged=full.merge(sample,on=['site_id','season'],how='left').fillna({'selected_rows':0})
            merged['selected_fraction']=merged.selected_rows/merged.full_rows
            merged['fold']=fold; merged['scale']=scale; distribution.append(merged)
            variation=.5*np.abs(merged.selected_rows/merged.selected_rows.sum()-merged.full_rows/merged.full_rows.sum()).sum()
            sample_summary.append(dict(fold=fold,scale=scale,fit_rows=int(merged.selected_rows.sum()),full_eligible_rows=int(merged.full_rows.sum()),sites=int(merged.loc[merged.selected_rows>0,'site_id'].nunique()),missing_site_season_cells=int((merged.selected_rows==0).sum()),site_season_total_variation=variation))
    pd.concat(distribution,ignore_index=True).to_csv(out/'sample_distribution.csv',index=False)
    sample_summary=pd.DataFrame(sample_summary); sample_summary.to_csv(out/'sample_distribution_summary.csv',index=False)
    text+='\n## 발전소·계절별 표본 분포\n\n'+markdown(sample_summary)
    text+='\n\nselected share와 full share의 total variation은0이면 같은 분포입니다. B/C 표본 인덱스는 같으며 분포는 B 파일로 대표했습니다. 세부 발전소×계절 수는 sample_distribution.csv에 있습니다.\n'
    importance=[]
    for family,chosen in lock['chosen'].items():
        gains=pd.read_csv(out/f"importance_{chosen['candidate']}_fold5.csv")
        missing=pd.read_csv(out/f"missingness_{chosen['candidate']}_fold5.csv")
        gains=gains.merge(missing,on='feature'); gains['family']=family
        gains['gain_share']=gains.gain/gains.gain.sum()
        importance.append(gains)
        text+=f'\n## {family} 선택 모델의 기상 중요도 및 결측\n\n'
        weather=gains[gains.feature.str.startswith(('wa_','wb_'))&~gains.feature.isin(['wa_station','wb_station','wa_dist','wb_dist'])]
        text+=markdown(weather.sort_values('gain',ascending=False).head(15))
    pd.concat(importance).to_csv(out/'selected_feature_importance_missingness.csv',index=False)
    text+='\n\nGain은 상관된 예보 변수들 사이에서 분산되며 인과효과나 유일한 중요도를 뜻하지 않습니다. 검증 결측과 오차의 관계는 metrics의 missing_weather 행을 확인하십시오.\n'
    if (out/'offline_C_metrics.csv').exists():
        offline=pd.read_csv(out/'offline_C_metrics.csv')
        policy=lock['chosen']['C']['policy']
        z=offline[(offline.dimension=='overall')&(offline.policy==policy)]
        text+='\n## 평가 기간 중 실적을 받지 못하는 C 민감도\n\n'
        text+='각 fold 첫 발행 시점 이후 실적을 전혀 받지 않는 경우를 별도 재추론했습니다. 모델/하이퍼파라미터/후처리/그룹은 다시 선택하지 않았습니다. 관측되지 않은 lag는 NaN, 빈7일 구간의 평균은 NaN/관측수0입니다. 실적을0으로 채운 것이 아닙니다. 이 결과는 test 파일에 실적 스트림이 없는 조건에서 C 성능을 해석하기 위한 보조 실험입니다.\n\n'
        text+=markdown(z[['fold','policy','cf_mae','mae_kwh']])
        text+='\n\n세부 주야간·신규/기존·정산 점수는 offline_C_metrics.csv와 offline_C_scores.csv에 저장했습니다. 이 민감도나 봄 점수로 사전 추천을 바꾸지 않았습니다.\n'
    if (out/'optimization_cancellation_bounds.csv').exists():
        text+='\n## 그룹 상쇄 및 오차 하한\n\n'+markdown(pd.read_csv(out/'optimization_cancellation_bounds.csv'))
        text+='\n\nfleet 오차 하한은 같은 시각 전체 발전소 잔차 합의 절대값으로, 어떤2개 그룹으로 나누어도 이보다 페널티를 낮출 수 없습니다. 용량 제약 아래에서 달성 가능함을 보장하지 않습니다. 최적화 잔여 여지를 판단하는 하한입니다.\n'
    text+='\n## 데이터 누수 경계 표\n\n'+markdown(pd.read_csv(out/'boundary_audit.csv'))
    sites=pd.read_csv(DATA/'sites.csv')
    categories={c:list(sites[c].unique()) for c in ['site_id','wa_station','wb_station']}
    manifest={'preferred':json.loads((out/'preferred_candidate_lock.json').read_text())['preferred'],'role':'pre-holdout validation model candidates; no final all-period refit or submission','categorical_numeric_codes':categories,'models':{}}
    for family,c in lock['chosen'].items():
        model_file=f"models/{c['candidate']}_fold5.txt"
        booster=lgb.Booster(model_file=str(out/model_file))
        manifest['models'][family]={'model_file':model_file,'model_sha256':sha256(out/model_file),'features':booster.feature_name(),'policy':c['policy'],'night_threshold_file':'night_thresholds_fold5.csv','OOF_file':f'OOF_{family}.parquet','holdout_file':f'selected_{family}_holdout.parquet','composition_file':f'composition_{family}.csv','fit_data_cutoff':'2025-02-28 17:00:00+09:00','generation_stream_required':family=='C','raw_prediction':'max(model_CF,0) * capacity_kw; apply frozen night policy at site level before grouping'}
    write_json(out/'candidate_manifest.json',manifest)
    text+='\n\n후보 모델의 경로·피처 순서·범주 코드·후처리·그룹 파일은 [candidate_manifest.json](candidate_manifest.json)에 있습니다. 이 모델들은 봄 이전 학습의 검증용 후보이며, 봄까지 포함해 재학습한 최종 제출 모델은 아닙니다.\n'
    (out/'report.md').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(OUT)); p.add_argument('--stage',choices=['audit','tune','development','holdout','all'],default='all'); args=p.parse_args()
    exp=Experiment(args.out)
    if args.stage in ['audit','all']: exp.boundary_audit()
    if args.stage=='audit': return
    params=exp.tune() if args.stage in ['tune','all'] else json.loads((exp.out/'tuned_parameters.json').read_text())
    if args.stage=='tune': return
    if args.stage in ['development','all']:
        m,s=exp.experiments(params,list(range(5))); lock=exp.freeze_selection(m,s); comp=exp.oof_and_groups(lock)
    else:
        lock=json.loads((exp.out/'selection_lock.json').read_text())
        comp={f:pd.read_csv(exp.out/f'composition_{f}.csv') for f in ['B','C']}
        assert json.loads((exp.out/'composition_lock.json').read_text())['selection_lock_hash']==sha256(exp.out/'selection_lock.json')
    if args.stage=='development': return
    exp.experiments(params,[5]); exp.evaluate_selected_groups(lock,comp); exp.march_analysis(lock); exp.finalize()
    log('all stages complete')
if __name__=='__main__': main()
