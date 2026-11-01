"""Reproducible day-ahead expanding validation using provided files only."""
from pathlib import Path
import argparse, hashlib, json, importlib.metadata, gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import wilcoxon
from forecast_baseline import build_features, sha256, DATA, METADATA, TIME_FEATURES, forecast_columns
from validation import validate_sites, local_hours, validate_composition

OUT=Path(__file__).resolve().parents[1]/'reports/expanding'
SEED=2026
FOLDS=[('2023-06-01','2023-09-01'),('2023-12-01','2024-03-01'),('2024-06-01','2024-09-01'),('2024-09-01','2024-12-01'),('2024-12-01','2025-03-01'),('2025-03-01','2025-06-01')]

def operating_day(ts):
    return (local_hours(ts)-pd.Timedelta(nanoseconds=1)).normalize()

def issuance(ts):
    return operating_day(ts)-pd.Timedelta(days=1)+pd.Timedelta(hours=17)

def check_asof(source, target):
    s=pd.DatetimeIndex(source); cut=issuance(target)
    if s.tz is None or ((s>cut)&~s.isna()).any():
        raise ValueError('source unavailable at day-ahead issuance')

def check_split(train_ts, validation_ts):
    a=local_hours(train_ts); b=local_hours(validation_ts)
    if len(a)==0 or len(b)==0 or a.max()>issuance(b).min() or a.max()>=b.min():
        raise ValueError('training target crosses validation issuance cutoff')

def validate_generation(g, sites):
    validate_sites(sites)
    local_hours(g.ts)
    if g.duplicated(['site_id','ts']).any() or not g.site_id.isin(sites.site_id).all():
        raise ValueError('generation key schema invalid')
    if not np.isfinite(g.energy).all(): raise ValueError('nonfinite generation')

def interpolate_single(g):
    """Only a single interior missing hour; retain endpoint availability provenance."""
    rows=[]
    for site,d in g.sort_values(['site_id','ts']).groupby('site_id',observed=True):
        d=d.reset_index(drop=True)
        for i in np.flatnonzero(d.ts.diff().eq(pd.Timedelta(hours=2))):
            endpoints=d.iloc[[i-1,i]]
            if not np.isfinite(endpoints.energy).all(): continue
            if 'cf' in d and not endpoints.cf.between(0,1).all(): continue
            rows.append(dict(site_id=site,ts=d.ts.iloc[i]-pd.Timedelta(hours=1),energy=(d.energy.iloc[i]+d.energy.iloc[i-1])/2,available_at=d.ts.iloc[i]))
    return pd.DataFrame(rows,columns=['site_id','ts','energy','available_at'])

def history_features(g, target):
    """Historical raw valid observations only, never interpolated; latest source audited."""
    wide=g.pivot(index='ts',columns='site_id',values='cf').sort_index()
    grid=pd.date_range(wide.index.min(),wide.index.max(),freq='h'); wide=wide.reindex(grid)
    # At issuance t-1 day 17h. Rolling ends exactly at that hour, no backfill.
    mean=wide.rolling(168,min_periods=1).mean()
    count=wide.rolling(168,min_periods=1).count()
    cols=wide.columns.get_indexer(target.site_id)
    def lookup(frame,times):
        idx=frame.index.get_indexer(times); ok=(idx>=0)&(cols>=0)
        a=np.full(len(target),np.nan,dtype='float32')
        a[ok]=frame.to_numpy()[idx[ok],cols[ok]]; return a
    lag_ts=local_hours(target.ts)-pd.Timedelta(hours=48)
    cut=issuance(target.ts)
    check_asof(lag_ts,target.ts); check_asof(cut,target.ts)
    return pd.DataFrame({'lag48_cf':lookup(wide,lag_ts),'history7d_cf':lookup(mean,cut),'history7d_count':lookup(count,cut)})

def solar(ts,lat,lon):
    # Analytic approximation at interval midpoint, no external ephemeris/data.
    t=local_hours(ts)-pd.Timedelta(minutes=30)
    gamma=2*np.pi*(t.dayofyear.to_numpy()-1)/365.25
    dec=np.deg2rad(23.44)*np.sin(gamma-1.39)
    eq=229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)-.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    hour=t.hour.to_numpy()+t.minute.to_numpy()/60+(np.asarray(lon)-135)/15+eq/60
    h=np.deg2rad(15*(hour-12)); p=np.deg2rad(np.asarray(lat))
    return np.rad2deg(np.arcsin(np.clip(np.sin(p)*np.sin(dec)+np.cos(p)*np.cos(dec)*np.cos(h),-1,1))).astype('float32')

def inventory(out):
    rows=[]; weather=[]; cadence=[]; schemas={}
    for path in sorted(DATA.rglob('*')):
        if path.suffix not in ('.csv','.parquet'): continue
        rel=str(path.relative_to(DATA)); print('inventory',rel,flush=True)
        d=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
        schemas[rel]={c:str(t) for c,t in d.dtypes.items()}
        rows.append(dict(file=rel,rows=len(d),columns=len(d.columns),bytes=path.stat().st_size))
        if 'ts' in d:
            d['ts']=pd.to_datetime(d.ts)
        if 'forecast_' in path.name or 'observed_' in path.name:
            station=next(c for c in d if c.endswith('_station'))
            keys=[station]+(['round'] if 'round' in d else [])
            diff=d.sort_values(keys+['ts']).groupby(keys,observed=True).ts.diff().dt.total_seconds()/3600
            cadence.append(dict(file=rel,start=str(d.ts.min()),end=str(d.ts.max()),duplicate_keys=int(d.duplicated(keys+['ts']).sum()),median_interval_hours=float(diff.median()),nonhour_intervals=int((diff.dropna()!=1).sum())))
            for c in d.columns.difference(keys+['ts']):
                weather.append(dict(file=rel,variable=c,missing=int(d[c].isna().sum()),missing_rate=float(d[c].isna().mean()),minimum=float(d[c].min()),maximum=float(d[c].max()),availability='previous operating day round hour' if 'round' in d else 'measurement time known; receipt latency unspecified; excluded'))
        del d; gc.collect()
    pd.DataFrame(rows).to_csv(out/'input_inventory.csv',index=False)
    pd.DataFrame(weather).to_csv(out/'weather_quality.csv',index=False)
    pd.DataFrame(cadence).to_csv(out/'weather_cadence.csv',index=False)
    (out/'schemas.json').write_text(json.dumps(schemas,indent=2))

def audit_patterns(g,out):
    d=g.copy(); day=operating_day(d.ts)
    d['year']=day.year; d['month']=day.month; d['hour']=local_hours(d.ts).hour
    d['season']=np.select([d.month.isin([12,1,2]),d.month.isin([3,4,5]),d.month.isin([6,7,8])],['winter','spring','summer'],default='autumn')
    for keys,name in [(['month','hour'],'month_hour'),(['season','hour'],'season_hour'),(['site_id','year'],'site_year'),(['year','month'],'year_month')]:
        d.groupby(keys,observed=True).cf.agg(['count','mean','std','median',lambda x:x.quantile(.05),lambda x:x.quantile(.95)]).to_csv(out/f'cf_{name}.csv')
    wide=d.pivot(index='ts',columns='site_id',values='cf')
    wide.corr(min_periods=168).to_csv(out/'cf_correlation.csv')
    valid=wide.notna().astype('int32'); (valid.T@valid).to_csv(out/'cf_correlation_overlap.csv')
    wide.loc[(wide.index.hour>=7)&(wide.index.hour<=18)].corr(min_periods=168).to_csv(out/'cf_daylight_correlation.csv')
    # Balanced site × month × hour cells prevent changing plant mix or seasonal clock weights.
    spring=d[d.month.isin([3,4,5])].groupby(['site_id','year','month','hour'],observed=True).cf.agg(['mean','count']).reset_index()
    results=[]
    for year in [2021,2022,2023,2024]:
        paired=spring[spring.year==2025].merge(spring[spring.year==year],on=['site_id','month','hour'],suffixes=('_2025','_other'))
        paired=paired[(paired['count_2025']>=20)&(paired['count_other']>=20)]
        delta=(paired.assign(delta=paired.mean_2025-paired.mean_other).groupby('site_id').delta.mean())
        if len(delta):
            rng=np.random.default_rng(SEED); boot=np.mean(rng.choice(delta.to_numpy(),size=(2000,len(delta)),replace=True),axis=1)
            p=wilcoxon(delta).pvalue if (delta!=0).any() else 1.
            results.append(dict(other_year=year,paired_sites=len(delta),mean_cf_delta=float(delta.mean()),median_cf_delta=float(delta.median()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),wilcoxon_p=float(p),bonferroni_p=min(float(p)*4,1)))
        paired.to_csv(out/f'spring_paired_cells_{year}.csv',index=False)
    pd.DataFrame(results).to_csv(out/'spring_comparison.csv',index=False)


def run(args):
    import lightgbm as lgb
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    inputs=[p for p in DATA.rglob('*') if p.is_file() and p.name!='.DS_Store']
    hashes={str(p.relative_to(DATA)):sha256(p) for p in inputs}
    if args.resume: args.reuse_audit=True
    if args.reuse_audit:
        old=json.loads((out/'config.json').read_text())
        if old['input_hashes']!=hashes: raise ValueError('cached audit input hashes differ')
        for filename in ['schemas.json','weather_quality.csv','spring_comparison.csv','cf_correlation.csv']:
            if not (out/filename).is_file(): raise ValueError('cached audit incomplete')
    else: inventory(out)
    sites=pd.read_csv(DATA/'sites.csv'); g=pd.read_parquet(DATA/'train/generation.parquet')
    g.ts=local_hours(g.ts); g=g.sort_values(['ts','site_id']).reset_index(drop=True)
    validate_generation(g,sites)
    g['capacity_kw']=g.site_id.map(sites.set_index('site_id').capacity_kw); g['cf']=g.energy/g.capacity_kw
    coverage=g.groupby('site_id').agg(first=('ts','min'),last=('ts','max'),observations=('ts','size'))
    coverage['span_hours']=(coverage['last']-coverage['first']).dt.total_seconds()/3600+1
    coverage['internal_missing']=coverage.span_hours-coverage.observations
    coverage.to_csv(out/'site_valid_periods.csv')
    if not args.reuse_audit: audit_patterns(g,out)
    extra=interpolate_single(g); extra.to_csv(out/'single_hour_interpolation.csv',index=False)
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        if old['trees']!=args.trees or old['max_train_rows']!=args.max_train_rows:
            raise ValueError('resume training configuration differs')
        prepared=pd.read_parquet(out/'datasets/V3.parquet')
        if not prepared[['site_id','ts']].equals(g[['site_id','ts']]):
            raise ValueError('prepared dataset key mismatch')
        names=METADATA+TIME_FEATURES+forecast_columns('wa')+forecast_columns('wb')+['solar_elevation','lag48_cf','history7d_cf','history7d_count']
        x=prepared[names].copy(); del prepared
        # Materialized audit keys keep string IDs and full-precision capacity.
        x['site_id']=pd.Categorical(x.site_id,categories=list(sites.site_id)).codes.astype('int16')
        x['capacity_kw']=x.capacity_kw.astype('float32')
        history=x[['lag48_cf','history7d_cf','history7d_count']]
    else:
        x,_=build_features(g[['site_id','ts']],sites.set_index('site_id'))
        # Operational date fixes calendar rollover at midnight for all new experiments.
        day=operating_day(g.ts); check_asof(day-pd.Timedelta(days=1)+pd.Timedelta(hours=17),g.ts)
        x['month']=day.month; x['dayofyear']=day.dayofyear
        x['dayofyear_sin']=np.sin(2*np.pi*(day.dayofyear-1)/365.25).astype('float32')
        x['dayofyear_cos']=np.cos(2*np.pi*(day.dayofyear-1)/365.25).astype('float32')
        x['solar_elevation']=solar(g.ts,x.lat,x.lon)
        history=history_features(g,g); x=pd.concat([x,history],axis=1)
        for c in x.select_dtypes('category'): x[c]=x[c].cat.codes.astype('int16')
    base=[c for c in x if not c.startswith(('wa_','wb_')) and c not in ['solar_elevation']+list(history)]
    # Exclude weather station/dist above by prefix consistently in basic versions.
    weather=[c for c in x if c not in list(history)]
    flags=pd.DataFrame({'negative':g.energy<0,'cf_gt_1':g.cf>1})
    for version in ([] if args.resume else ['V0','V1','V2','V3']):
        dest=out/'datasets'; dest.mkdir(exist_ok=True)
        d=g[['site_id','ts','energy','capacity_kw','cf']].copy()
        d['train_eligible']=True if version in ['V0','V1'] else ~(flags.negative|flags.cf_gt_1)
        d['target']=d.energy if version=='V0' else d.cf
        if version=='V3': d=pd.concat([d,x],axis=1).loc[:,lambda z:~z.columns.duplicated()]
        d.to_parquet(dest/f'{version}.parquet',index=False)
        del d
    # Separate interpolation experiment, never modifies validation labels or history.
    extra.to_parquet(out/'datasets/V2_interpolation_additions.parquet',index=False)
    smp=pd.read_csv(DATA/'train/smp.csv'); smp.ts=pd.to_datetime(smp.ts); prices=smp.set_index('ts').smp
    # Fixed capacity-balanced composition, independent of generation/validation.
    loads={'G1':0.,'G2':0.}; groups={}
    for row in sites.sort_values(['capacity_kw','site_id'],ascending=[False,True]).itertuples():
        group=min(loads,key=loads.get); groups[row.site_id]=group; loads[group]+=row.capacity_kw
    composition=pd.DataFrame({'site_id':list(groups),'group':list(groups.values())})
    validate_composition(composition,sites); composition.to_csv(out/'composition.csv',index=False)
    metrics=[]; details=[]; split_rows=[]; ledgers=[]
    config=vars(args)|{'seed':SEED,'folds':FOLDS,'libraries':{k:importlib.metadata.version(k) for k in ['numpy','pandas','lightgbm','scipy']},'groups_capacity_kw':loads,'input_hashes':hashes}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for fi,(start,end) in enumerate(FOLDS):
        start=pd.Timestamp(start,tz='Asia/Seoul'); end=pd.Timestamp(end,tz='Asia/Seoul')
        cutoff=start-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
        tr=np.flatnonzero((g.ts<=cutoff).to_numpy()); va=np.flatnonzero(((g.ts>start)&(g.ts<=end)).to_numpy())
        check_split(g.ts.iloc[tr],g.ts.iloc[va])
        train_counts=g.iloc[tr].groupby('site_id').size()
        cohort=np.where(g.site_id.iloc[va].map(train_counts).fillna(0).to_numpy()>=2160,'established_90d','new_under90d')
        split_rows.append(dict(fold=fi,validation_start=str(start),validation_end=str(end),training_cutoff=str(cutoff),train_rows=len(tr),validation_rows=len(va),train_max=str(g.ts.iloc[tr].max()),asof_pass=True))
        train=g.iloc[tr].copy(); train['hour']=local_hours(train.ts).hour
        climat=train.groupby(['site_id','hour'],observed=True).cf.mean()
        lookup=pd.MultiIndex.from_arrays([g.site_id.iloc[va],local_hours(g.ts.iloc[va]).hour])
        fallback=train.groupby('hour').cf.mean().reindex(local_hours(g.ts.iloc[va]).hour).to_numpy()
        a=climat.reindex(lookup).to_numpy(); a=np.where(np.isnan(a),fallback,a)
        candidates=[('A_climatology',a)]
        for name,features,clean,raw,interp in [('V0_B',base,False,True,False),('V1_B',base,False,False,False),('V2_B',base,True,False,False),('V2_interp_B',base,True,False,True),('V3_B',weather,True,False,False),('V3_C',list(x),True,False,False)]:
            idx=tr
            if args.max_train_rows and len(idx)>args.max_train_rows:
                idx=idx[np.linspace(0,len(idx)-1,args.max_train_rows,dtype=int)]
            # Keep identical base sample across versions; then remove anomalies.
            if clean: idx=idx[~flags.iloc[idx].any(axis=1).to_numpy()]
            xt=x.iloc[idx][features]; yt=(g.energy if raw else g.cf).iloc[idx]
            if interp and len(extra):
                e=extra[(extra.available_at<=cutoff)&(extra.ts<=cutoff)].copy()
                if len(e):
                    e['capacity_kw']=e.site_id.map(sites.set_index('site_id').capacity_kw)
                    e=e[(e.energy>=0)&(e.energy<=e.capacity_kw)]
                    xe,_=build_features(e[['site_id','ts']],sites.set_index('site_id'))
                    eday=operating_day(e.ts); xe['month']=eday.month; xe['dayofyear']=eday.dayofyear
                    xe['dayofyear_sin']=np.sin(2*np.pi*(eday.dayofyear-1)/365.25).astype('float32')
                    xe['dayofyear_cos']=np.cos(2*np.pi*(eday.dayofyear-1)/365.25).astype('float32')
                    for c in xe.select_dtypes('category'): xe[c]=xe[c].cat.codes.astype('int16')
                    xt=pd.concat([xt,xe[features]],ignore_index=True); yt=pd.concat([yt,e.energy/e.capacity_kw],ignore_index=True)
            model=lgb.LGBMRegressor(n_estimators=args.trees,num_leaves=31,learning_rate=.08,random_state=SEED,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
            model_path=out/f'fold{fi}_{name}.txt'
            if args.resume and model_path.exists():
                model=lgb.Booster(model_file=str(model_path))
                if model.feature_name()!=features: raise ValueError('cached model features differ')
                prediction=model.predict(x.iloc[va][features],num_threads=4)
            else:
                model.fit(xt,yt)
                prediction=model.predict(x.iloc[va][features])
                model.booster_.save_model(str(model_path))
            prediction=np.maximum(prediction,0)
            if raw: prediction=prediction/g.capacity_kw.iloc[va].to_numpy()
            candidates.append((name,prediction))
            details.append(dict(fold=fi,experiment=name,fit_rows=len(xt),features=features))
            print('trained',fi,name,len(xt),flush=True)
        for name,p in candidates:
            d=g.iloc[va][['site_id','ts','energy','cf','capacity_kw']].copy()
            d['pred_cf']=np.maximum(p,0); d['pred_kwh']=d.pred_cf*d.capacity_kw; d['cohort']=cohort
            d['abs_cf']=(d.pred_cf-d.cf).abs(); d['abs_kwh']=(d.pred_kwh-d.energy).abs()
            opday=operating_day(d.ts)
            # Format only the unique operational dates; avoid per-row timezone strftime.
            dates=pd.Series(opday)
            mapping={t:f'{t.year:04d}-{t.month:02d}' for t in dates.unique()}
            d['month']=dates.map(mapping).to_numpy(); d['hour']=local_hours(d.ts).hour
            for key in [None,'site_id','month','hour','cohort']:
                blocks=[('all',d)] if key is None else d.groupby(key,observed=True)
                for label,z in blocks:
                    metrics.append(dict(fold=fi,experiment=name,dimension=key or 'overall',label=label,rows=len(z),cf_mae=z.abs_cf.mean(),mae_kwh=z.abs_kwh.mean()))
            d.to_parquet(out/f'fold{fi}_{name}_predictions.parquet',index=False)
            d['group']=d.site_id.map(groups)
            # Fixed full-fleet mask per fold, identical across experiments; no fill.
            complete=d.groupby('ts').site_id.nunique(); times=complete.index[complete==len(sites)]
            z=d[d.ts.isin(times)].groupby(['group','ts'],observed=True)[['energy','pred_kwh']].sum().reset_index()
            z['smp']=z.ts.map(prices); assert z.smp.notna().all()
            z['k']=np.where(z.ts.dt.hour.between(11,15),3,1)
            z['revenue_krw']=z.smp*z.energy; z['penalty_krw']=.1*z.smp*z.k*(z.pred_kwh-z.energy).abs()
            z['score_krw']=z.revenue_krw-z.penalty_krw; z['experiment']=name; z['fold']=fi
            ledgers.append(z)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(split_rows).to_csv(out/'split_audit.csv',index=False)
    ledger=pd.concat(ledgers,ignore_index=True); ledger.to_parquet(out/'settlement_ledger.parquet',index=False)
    score=ledger.groupby(['fold','experiment'],observed=True).agg(scored_group_hours=('ts','size'),score_krw=('score_krw','sum'),penalty_krw=('penalty_krw','sum'))
    score.to_csv(out/'official_formula_scores.csv')
    (out/'fits.json').write_text(json.dumps(details,indent=2))
    assert hashes=={str(p.relative_to(DATA)):sha256(p) for p in inputs},'raw input modified'
    (out/'verification.json').write_text(json.dumps({'raw_hashes_unchanged':True,'split_checks':len(split_rows),'asof_history_checks':True,'validation_original_only':True,'observed_weather_features':False,'seed':SEED},indent=2))
    report(out,pd.DataFrame(metrics),score,config)

def markdown(frame):
    frame=frame.copy()
    if not isinstance(frame.index,pd.RangeIndex): frame=frame.reset_index()
    def cell(v):
        if isinstance(v,(float,np.floating)): return f'{v:.6g}'
        return str(v).replace('|','/')
    rows=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    rows.extend('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(rows)

def report(out,m,score,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    overall=m[m.dimension=='overall'].pivot(index='fold',columns='experiment',values='cf_mae')
    overall.plot(marker='o',figsize=(11,5)); plt.ylabel('CF MAE'); plt.tight_layout(); plt.savefig(out/'fold_comparison.png'); plt.close()
    monthly=m[m.dimension=='month'].pivot_table(index='label',columns='experiment',values='cf_mae')
    monthly.plot(figsize=(12,5)); plt.xticks(rotation=60); plt.tight_layout(); plt.savefig(out/'monthly_comparison.png'); plt.close()
    spring=pd.read_csv(out/'spring_comparison.csv')
    text='''# OIBC expanding 전처리·검증 결과

제공 README의 공식 산식 S=ΣSMP·실적−0.10ΣSMP·k·|예측−실적|을 적용했습니다. 문제정의서 자체는 저장소에서 발견되지 않아 동점 규정 등은 확인하지 못했습니다. 공식 산식 점수는 고정 용량 균형 G1/G2와 전체 253개소 정답이 모두 존재하는 시각만 사용한 로컬 재현값입니다. 기간·평가 마스크가 달라 fold 간 금액을 직접 비교하지 않습니다. 완전 관측 시간이 없는 fold는 official_formula_scores.csv에 행이 없으며 점수를 0으로 해석하면 안 됩니다. 개별 관측 MAE는 별도 보조 지표입니다.

입력 전체 스키마는 schemas.json, 파일 목록은 input_inventory.csv, 기상 변수별 결측·범위는 weather_quality.csv, 실제 간격은 weather_cadence.csv에 있습니다. 단위 정본은 제공 `handover 2/01_participant_dataset/README.md`의 변수 표입니다. A 풍속 m/s, B 풍속 km/h, A 적설 mm/B 적설 cm를 원래 단위로 유지합니다. 예보 round=17은 전일 17시 이용 가능하다고 제공 README가 명시합니다. 실측 기상 수신 지연과 발전 실적 수신 지연은 불명입니다. 실측 기상은 사용하지 않고, 발전 실적은 해당 우측닫힘 시간이 지나면 즉시 이용 가능하다고 가정했습니다. 운영 적용 전 이 가정 확인이 필요합니다.

관측 이전/이후를 채우지 않았습니다. site_valid_periods.csv에 첫·마지막 관측, 실제 관측 수, 내부 공백을 기록했습니다. 관측 시작을 설비 준공일로 해석하지 않습니다. 학습 종료는 각 fold 첫 발행일 17시 이하로 제한했습니다. fold 내 모델은 고정하며 C의 관측 이력만 매 발행 시점으로 갱신합니다. 00시의 운영일은 전일입니다. 원본 음수/CF>1도 유한 실적이면 검증 정답에 그대로 보존했습니다.

V0는 kWh 타깃, V1은 CF 타깃이며 둘은 메타·시간 피처가 같습니다. V2는 음수/CF>1 학습 정답을 제외합니다. V2_interp_B는 정확히 한 시간인 내부 공백만 두 끝점으로 보간하고 오른쪽 끝점이 학습 마감 이전인 경우만 추가합니다. 연속 7시간 공백은 보간 대상이 아닙니다. V3_B는 round17 예보와 제공 좌표로 근사한 시간구간 중앙 태양고도를 추가합니다. V3_C는 raw 관측 기반 48시간 lag와 전일17시까지 168시간 CF 평균·관측 수를 추가합니다. 예보 결측은 LightGBM native missing branch로 처리하며 검증 통계나 미래 기상으로 채우지 않습니다. 실측 기상은 EDA에서만 읽습니다. 음수 예측은 0으로 제한합니다. 용량은 제공 고정 메타값이며 과거 증설 여부는 불명입니다.

A는 fold 이전 발전소·시간대 CF 평균이며 신규 발전소는 전체 발전소 시간대 평균으로 대체합니다. 신규는 학습 관측 2160시간 미만(90일 상당), 기존은 그 이상입니다. 이 기준은 연속 운영기간과 다릅니다. 동일 검증 관측 집합에서 모든 모델을 평가했습니다. 학습 행 제한은 설정에 기록된 고정 등간격 표본이며 시간순 분할에는 무작위성을 사용하지 않았습니다. 하이퍼파라미터는 고정하고 검증 기반 조기 종료를 하지 않았습니다. 이것은 가벼운 baseline 비교로, 전체 학습·튜닝 결과를 의미하지 않습니다.

실험 config.json, 모델 fold*_*.txt, 각 관측 예측 parquet, fits.json과 metrics.csv를 저장했습니다. metrics.csv에는 fold·월·시간·발전소·신규/기존별 MAE가 있습니다. settlement_ledger.parquet에 공식 산식의 시간·집합별 정산 내역을 저장했습니다. 재학습 시 같은 실행 설정을 사용하십시오.

## 관측된 성능 (CF MAE)

'''
    text+=markdown(overall)+'\n\n## 2025년 3~5월 비교\n\n'+markdown(spring)
    text+='''

같은 발전소·월·시간의 양쪽 연도 관측 20개 이상 셀을 맞추고 발전소별 차이를 집계했습니다. 신뢰구간은 발전소 재표집 2000회, p값은 발전소별 paired Wilcoxon 및 4회 Bonferroni입니다. 발전소 공간 상관과 날씨의 연도별 차이가 남아 있으므로 인과 추론이나 산불의 증거가 아닙니다. 제공 데이터의 평균 CF 변화에 대한 탐색 결과입니다. site_year와 year_month 표 및 주간/전체 CF 상관·동시 관측 수를 함께 보십시오. 평균 변화는 일사·기상 차이, 설비 변화, 계측 변화 등을 구분하지 못합니다.

## 검증 및 다음 단계

verification.json과 split_audit.csv는 입력 SHA256 불변, 발행 시각 이전 학습 정답, lag 원천 시각 제한을 검증합니다. 단위 테스트는 경계 00시 운영일, 미래 소스 차단, 첫 검증일 cutoff, 한 시간 보간, 입력 중복 키를 확인합니다. V3_C에서 이후 검증일 이전 실적을 사용하는 것은 매일 관측이 도착하는 rolling-origin 가정입니다. 관측 지연이 확인되면 cutoff에 그 지연을 더 적용해야 합니다.

우선 개선은 발전실적 수신 지연 확인, 모든 학습 행/충분한 트리 수 재실험, fold별 신규 발전소 오차 확인, 예보 10/17 라운드 차이 활용입니다. 통계 차이의 원인은 확인되지 않았고 외부 자료/사전학습 모델은 사용하지 않았습니다. 아래 차이는 이 설정에서 측정한 결과이며 일반적인 향상으로 확정하지 않습니다.

'''
    for name in overall.columns:
        if name!='A_climatology':
            delta=overall[name]-overall.A_climatology
            text+=f'- {name}: A 대비 fold 평균 CF MAE 차이 {delta.mean():+.6f}, 개선 fold {int((delta<0).sum())}/6.\n'
    text+='\n![fold](fold_comparison.png)\n\n![month](monthly_comparison.png)\n'
    (out/'report.md').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--resume',action='store_true',help='Reuse hash-checked prepared datasets and matching fold models'); p.add_argument('--reuse-audit',action='store_true',help='Reuse EDA tables only after checking unchanged input files'); p.add_argument('--out',default=str(OUT)); p.add_argument('--trees',type=int,default=60); p.add_argument('--max-train-rows',type=int,default=200000,help='0 uses all training observations')
    run(p.parse_args())
