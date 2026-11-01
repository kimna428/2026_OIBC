"""Read-only generation diagnostics and independent artificial-gap experiments.

Run: MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python analysis/analyze_generation.py
No imputed generation or benchmark prediction file is written.
"""
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'handover 2/01_participant_dataset'
OUT = ROOT / 'analysis/results'
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261008)
plt.rcParams.update({'figure.dpi': 140, 'axes.grid': True, 'grid.alpha': .2})

def log(s):
    print(s, flush=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / (name + '.png'), bbox_inches='tight')
    plt.close()

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def stats(d, keys):
    g = d.groupby(keys, observed=True).cf
    result = g.agg(['count', 'mean', 'std', 'min', 'max'])
    return result.join(g.quantile([.05, .25, .5, .75, .95, .99]).unstack().rename(
        columns={.05:'p05', .25:'p25', .5:'p50', .75:'p75', .95:'p95', .99:'p99'}))

def main():
    files = [DATA/'train/generation.parquet', DATA/'sites.csv', DATA/'train/smp.csv',
             DATA/'train/observed_wa.parquet', DATA/'train/observed_wb.parquet']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    sites = pd.read_csv(DATA/'sites.csv').set_index('site_id')
    gen = pd.read_parquet(files[0])
    gen.ts = pd.to_datetime(gen.ts).dt.tz_convert('Asia/Seoul')
    assert not gen.duplicated(['site_id','ts']).any(), 'Duplicate generation keys'
    assert gen.site_id.isin(sites.index).all()
    timeline = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(DATA/'train/smp.csv').ts))
    timeline = timeline.tz_convert('Asia/Seoul').as_unit('ns').sort_values()
    assert timeline.is_unique and (np.diff(timeline.asi8) == 3600*10**9).all()
    assert gen.ts.isin(timeline).all()
    assert (sites.capacity_kw > 0).all()
    wide = gen.pivot(index='ts', columns='site_id', values='energy').reindex(index=timeline, columns=sites.index)
    cf = wide.div(sites.capacity_kw, axis=1)
    y = cf.to_numpy()
    n, ns = y.shape
    hours = timeline.hour.to_numpy()
    # NOAA approximation at the center of the right-closed hourly interval.
    doy = timeline.dayofyear.to_numpy()[:,None]
    hour = hours[:,None]
    gamma = 2*np.pi/365*(doy-1+(hour-12)/24)
    eq = 229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)-.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    decl = .006918-.399912*np.cos(gamma)+.070257*np.sin(gamma)-.006758*np.cos(2*gamma)+.000907*np.sin(2*gamma)-.002697*np.cos(3*gamma)+.00148*np.sin(3*gamma)
    lat = np.radians(sites.lat.to_numpy()[None,:])
    ha = np.radians(((hour-.5)*60+eq+4*(sites.lon.to_numpy()[None,:]-135))/4-180)
    daylight = np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(ha)>0
    log(f'Grid {n:,} hours x {ns} sites; observed {np.isfinite(y).sum():,}')
    site_rows, gaps = [], []
    for j, sid in enumerate(sites.index):
        valid = np.flatnonzero(np.isfinite(y[:,j]))
        if not len(valid):
            raise ValueError(f'No observations for {sid}')
        first,last = valid[0],valid[-1]
        missing = ~np.isfinite(y[:,j])
        edges = np.diff(np.r_[False, missing, False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            length = b-a
            position = 'leading' if a==0 else ('trailing' if b==n else 'internal')
            category = 'leading_long' if a==0 and length>24 else ('1-3h' if length<=3 else ('4-24h' if length<=24 else '>24h'))
            gaps.append([sid,timeline[a],timeline[b-1],int(length),position,category])
        site_rows.append(dict(site_id=sid,first_observed=timeline[first],last_observed=timeline[last],
            observed_hours=len(valid),expected_hours=n,observation_rate=len(valid)/n,
            active_span_hours=int(last-first+1),active_span_observation_rate=len(valid)/(last-first+1),
            leading_missing_hours=int(first),trailing_missing_hours=int(n-1-last),
            internal_missing_hours=int(last-first+1-len(valid)),
            daylight_observation_rate=np.isfinite(y[:,j])[daylight[:,j]].mean()))
    coverage = pd.DataFrame(site_rows).set_index('site_id').join(sites)
    coverage.to_csv(OUT/'site_coverage.csv')
    gaps = pd.DataFrame(gaps,columns=['site_id','start','end','length_hours','position','category'])
    gaps.to_csv(OUT/'missing_runs.csv',index=False)
    gap_summary = gaps.groupby(['position','category']).length_hours.agg(['count','sum','median','max'])
    gap_summary.to_csv(OUT/'missing_summary.csv')
    gaps.groupby('position').length_hours.describe(percentiles=[.5,.75,.9,.95,.99]).to_csv(OUT/'missing_length_distribution.csv')
    gen['cf'] = gen.energy/gen.site_id.map(sites.capacity_kw)
    # Format unique timestamps once, rather than millions of zoned datetimes.
    timestamp_codes = timeline.get_indexer(gen.ts)
    gen['month'] = timeline.strftime('%Y-%m').to_numpy()[timestamp_codes]
    gen['month_of_year'] = timeline.month.to_numpy()[timestamp_codes]
    gen['hour'] = hours[timestamp_codes]
    stats(gen,['site_id']).to_csv(OUT/'cf_by_site.csv')
    stats(gen,['month']).to_csv(OUT/'cf_by_month.csv')
    stats(gen,['month_of_year']).to_csv(OUT/'cf_by_month_of_year.csv')
    stats(gen,['hour']).to_csv(OUT/'cf_by_hour.csv')
    stats(gen,['site_id','month']).to_csv(OUT/'cf_by_site_month.csv')
    stats(gen,['site_id','hour']).to_csv(OUT/'cf_by_site_hour.csv')
    stats(gen,['month_of_year','hour']).to_csv(OUT/'cf_by_month_hour.csv')
    # Original observations only; missing remains missing in this derived CF file.
    gen[['site_id','ts','cf']].to_parquet(OUT/'observed_capacity_factor.parquet',index=False)
    quality = dict(negative_energy=int((gen.energy<0).sum()),cf_above_1=int((gen.cf>1).sum()),
                   cf_above_1_2=int((gen.cf>1.2).sum()),nonfinite_energy=int((~np.isfinite(gen.energy)).sum()),
                   zero_energy=int((gen.energy==0).sum()),max_cf=float(gen.cf.max()))
    plt.figure(figsize=(12,4))
    plt.bar(np.arange(ns),coverage.observation_rate,label='Full training grid')
    plt.plot(np.arange(ns),coverage.active_span_observation_rate,color='orange',label='First-to-last observation')
    plt.xlabel('Site order PV001–PV253'); plt.ylabel('Observation rate'); plt.legend(); savefig('coverage')
    monthly = cf.notna().resample('MS').mean().T
    plt.figure(figsize=(14,7)); plt.imshow(monthly,aspect='auto',vmin=0,vmax=1,cmap='viridis')
    plt.xticks(np.arange(0,len(monthly.columns),3),monthly.columns.strftime('%Y-%m')[::3],rotation=45)
    plt.ylabel('Site order'); plt.colorbar(label='Observation rate'); savefig('coverage_month_heatmap')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    bins=np.logspace(0,np.log10(max(gaps.length_hours.max(),2)),40)
    for pos,d in gaps.groupby('position'):
        axes[0].hist(d.length_hours,bins=bins,histtype='step',label=pos)
    axes[0].set(xscale='log',yscale='log',xlabel='Missing run length (hours)',ylabel='Run count'); axes[0].legend()
    gaps.groupby('category').length_hours.sum().plot.bar(ax=axes[1]); axes[1].set_ylabel('Total missing hours'); savefig('missing_runs')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].hist(gen.cf,bins=100,log=True); axes[0].set(xlabel='Observed hourly CF',ylabel='Count (log)')
    for ax,key in [(axes[1],'month_of_year'),(axes[2],'hour')]:
        q=gen.groupby(key).cf.quantile([.05,.5,.95]).unstack()
        ax.fill_between(q.index,q[.05],q[.95],alpha=.2,label='p05–p95')
        ax.plot(q.index,q[.5],label='Median'); ax.plot(gen.groupby(key).cf.mean(),label='Mean')
        ax.set(xlabel=key,ylabel='CF'); ax.legend()
    savefig('cf_distributions')
    plt.figure(figsize=(14,7)); plt.imshow(cf.groupby(hours).mean().T,aspect='auto',cmap='magma')
    plt.xticks(range(24)); plt.xlabel('KST interval end hour'); plt.ylabel('Site order'); plt.colorbar(label='Mean observed CF'); savefig('cf_site_hour')
    plt.figure(figsize=(10,5)); plt.imshow(gen.groupby(['month_of_year','hour']).cf.mean().unstack(),aspect='auto',cmap='magma')
    plt.xticks(range(24)); plt.yticks(range(12),range(1,13)); plt.xlabel('KST interval end hour'); plt.ylabel('Month'); plt.colorbar(label='Mean observed CF'); savefig('cf_month_hour')
    log('Diagnostics saved; preparing observed-weather daily similarity.')
    dates = timeline.normalize()
    dayindex = pd.DatetimeIndex(dates.unique())
    day_codes = dayindex.get_indexer(dates)
    daily_weather = []
    weather_checks = []
    weather_keys = []
    for provider,cols,bounds in [
        ('wa',['wa_cloud','wa_temp','wa_humidity','wa_rain'],[(0,100),(-50,60),(0,100),(0,300)]),
        ('wb',['wb_cloud','wb_temp','wb_rel_hum','wb_precip_1h'],[(0,100),(-50,60),(0,100),(0,300)])]:
        station = provider+'_station'
        w = pd.read_parquet(DATA/f'train/observed_{provider}.parquet',columns=[station,'ts']+cols)
        duplicate_keys = int(w.duplicated([station,'ts']).sum())
        w.ts = pd.to_datetime(w.ts).dt.tz_convert('Asia/Seoul')
        for col,(lo,hi) in zip(cols,bounds):
            bad=(w[col]<lo)|(w[col]>hi)|~np.isfinite(w[col])
            weather_checks.append(dict(variable=col,excluded_values=int(bad.sum())))
            w.loc[bad,col]=np.nan
        w=w[w.ts.dt.hour.between(7,18)].copy()
        daylight_duplicates = int(w.duplicated([station,'ts']).sum())
        weather_keys.append(dict(provider=provider,duplicate_keys=duplicate_keys,
                                 feature_window_duplicate_keys=daylight_duplicates))
        assert daylight_duplicates == 0, 'Ambiguous duplicate weather keys in feature window'
        w['day']=w.ts.dt.normalize()
        grouped=w.groupby([station,'day'])[cols]
        d=grouped.mean().where(grouped.count()>=8)
        daily_weather.append((provider,d,cols))
    pd.DataFrame(weather_checks).to_csv(OUT/'weather_feature_quality.csv',index=False)
    pd.DataFrame(weather_keys).to_csv(OUT/'weather_key_quality.csv',index=False)
    # At least 6/8 observed daily features. Pairwise distance uses shared features.
    weather_cache={}
    for j,sid in enumerate(sites.index):
        key=(sites.loc[sid,'wa_station'],sites.loc[sid,'wb_station'])
        if key in weather_cache: continue
        parts=[]
        for provider,d,cols in daily_weather:
            parts.append(d.loc[sites.loc[sid,provider+'_station']].reindex(dayindex)[cols].to_numpy())
        raw=np.concatenate(parts,axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            scale=np.nanstd(raw,axis=0); center=np.nanmean(raw,axis=0)
        scale[~np.isfinite(scale)|(scale==0)]=1
        weather_cache[key]=(raw-center)/scale
    log('Running artificial gaps: each experiment independently hides one site block.')
    block_rows=[]; evaluations=[]; skipped=[]
    lengths=[1,2,3,6,12,24,48,72,168]
    for j,sid in enumerate(sites.index):
        a=y[:,j]; valid=np.isfinite(a)
        peer_ids=np.flatnonzero(((sites.wa_station==sites.loc[sid,'wa_station'])|
                                (sites.wb_station==sites.loc[sid,'wb_station'])).to_numpy())
        peers=peer_ids[peer_ids!=j]
        key=(sites.loc[sid,'wa_station'],sites.loc[sid,'wb_station'])
        z=weather_cache[key]
        for scenario,length in [('internal',l) for l in lengths]+[('leading',168),('leading',720)]:
            # A normal block is fully observed, finite and within physical CF [0,1.2].
            normal=valid&(a>=0)&(a<=1.2)
            count=np.convolve(normal.astype(int),np.ones(length,dtype=int),mode='valid')
            starts=np.flatnonzero(count==length)
            if scenario=='internal':
                starts=starts[(starts>0)&(starts+length<n)]
                starts=starts[valid[starts-1]&valid[starts+length]]
            # Ensure the artificial block actually contains daytime.
            daytime_count=np.r_[0,np.cumsum(daylight[:,j])]
            starts=starts[daytime_count[starts+length]-daytime_count[starts]>0]
            if not len(starts):
                skipped.append(dict(site_id=sid,scenario=scenario,length=length)); continue
            chosen=RNG.choice(starts,min(2,len(starts)),replace=False)
            for rep,start in enumerate(chosen):
                end=start+length; idx=np.arange(start,end); truth=a[idx]
                category='leading_long' if scenario=='leading' else ('1-3h' if length<=3 else ('4-24h' if length<=24 else '>24h'))
                bid=len(block_rows)
                block_rows.append(dict(block_id=bid,site_id=sid,scenario=scenario,category=category,length_hours=length,
                    start=timeline[start],end=timeline[end-1],replicate=rep,peer_count=len(peers)))
                available=valid.copy(); available[start:end]=False
                if scenario=='leading': available[:start]=False
                pred={}
                pred['linear']=np.full(length,np.nan) if scenario=='leading' else np.linspace(a[start-1],a[end],length+2)[1:-1]
                # Same-site same-hour: mean observed donors within +/-30 calendar days.
                same=np.full(length,np.nan)
                for offset in range(24):
                    targets=idx[hours[idx]==offset]
                    if not len(targets): continue
                    donors=targets[:,None]+24*np.r_[np.arange(-30,0),np.arange(1,31)][None,:]
                    inside=(donors>=0)&(donors<n); safe=np.clip(donors,0,n-1)
                    vals=np.where(inside&available[safe],a[safe],np.nan)
                    counts=np.isfinite(vals).sum(axis=1)
                    same[targets-start]=np.divide(np.nansum(vals,axis=1),counts,out=np.full(len(targets),np.nan),where=counts>0)
                pred['same_site_same_hour']=same
                # Similar days: daily observed weather, same season (+/-45 days),
                # five nearest donor days; no target-day energy enters distance.
                wp=np.full(length,np.nan)
                excluded_days=np.unique(day_codes[idx])
                for dc in excluded_days:
                    targets=idx[day_codes[idx]==dc]
                    shared=np.isfinite(z)&np.isfinite(z[dc])
                    nshared=shared.sum(axis=1)
                    delta=np.where(shared,z-z[dc],0)
                    distances=np.divide((delta**2).sum(axis=1),nshared,out=np.full(len(z),np.inf),where=nshared>=6)
                    seasonal=np.abs(dayindex.dayofyear.to_numpy()-dayindex[dc].dayofyear)
                    seasonal=np.minimum(seasonal,366-seasonal)
                    distances[(seasonal>45)|np.isin(np.arange(len(z)),excluded_days)]=np.inf
                    rank=np.argsort(distances)
                    rank=rank[np.isfinite(distances[rank])]
                    for t in targets:
                        donor_ts=dayindex[rank]+pd.Timedelta(hours=int(hours[t]))
                        donor_idx=timeline.get_indexer(donor_ts)
                        donor_idx=donor_idx[donor_idx>=0]
                        donor_idx=donor_idx[available[donor_idx]][:5]
                        if len(donor_idx): wp[t-start]=a[donor_idx].mean()
                pred['weather_similar_day']=wp
                # Independent single-site outages: other sites are observable.
                npred=np.full(length,np.nan)
                if len(peers):
                    peer=y[np.ix_(idx,peers)]
                    counts=np.isfinite(peer).sum(axis=1)
                    npred=np.divide(np.nansum(peer,axis=1),counts,out=np.full(length,np.nan),where=counts>0)
                pred['same_station_neighbor_cf']=npred
                common=np.logical_and.reduce([np.isfinite(p) for p in pred.values()])
                for method,p in pred.items():
                    for scope,mask in [('all',np.ones(length,bool)),('daylight',daylight[idx,j]),('night',~daylight[idx,j]),
                                       ('common_all',common),('common_daylight',common&daylight[idx,j])]:
                        good=mask&np.isfinite(p); err=p[good]-truth[good]
                        evaluations.append(dict(block_id=bid,site_id=sid,scenario=scenario,category=category,
                            length_hours=length,method=method,scope=scope,target_hours=int(mask.sum()),
                            restored_hours=int(good.sum()),abs_sum=float(np.abs(err).sum()),
                            sq_sum=float((err**2).sum()),bias_sum=float(err.sum()),
                            kwh_abs_sum=float(np.abs(err).sum()*sites.loc[sid,'capacity_kw'])))
        if (j+1)%25==0: log(f'Benchmark {j+1}/{ns} sites; {len(block_rows):,} blocks')
    blocks=pd.DataFrame(block_rows); ev=pd.DataFrame(evaluations)
    blocks.to_csv(OUT/'artificial_missing_blocks.csv',index=False)
    ev.to_csv(OUT/'benchmark_block_metrics.csv',index=False)
    pd.DataFrame(skipped).to_csv(OUT/'benchmark_skipped.csv',index=False)
    def summarize(keys,filename):
        s=ev.groupby(keys)[['target_hours','restored_hours','abs_sum','sq_sum','bias_sum','kwh_abs_sum']].sum()
        den=s.restored_hours.replace(0,np.nan)
        s['coverage']=s.restored_hours/s.target_hours.replace(0,np.nan)
        s['mae_cf']=s.abs_sum/den; s['rmse_cf']=np.sqrt(s.sq_sum/den)
        s['bias_cf']=s.bias_sum/den; s['mae_kwh']=s.kwh_abs_sum/den
        s.to_csv(OUT/filename)
        return s
    summary=summarize(['scenario','category','method','scope'],'benchmark_summary.csv')
    summarize(['scenario','length_hours','method','scope'],'benchmark_by_length.csv')
    summarize(['site_id','scenario','category','method','scope'],'benchmark_by_site.csv')
    # Site-level bootstrap preserves within-site temporal dependence.
    ci=[]
    for keys,d in ev.groupby(['scenario','category','method','scope']):
        ds=d.groupby('site_id')[['abs_sum','restored_hours']].sum().to_numpy()
        sel=RNG.integers(0,len(ds),(1000,len(ds)))
        sums=ds[sel].sum(axis=1)
        ratios=np.divide(sums[:,0],sums[:,1],out=np.full(1000,np.nan),where=sums[:,1]>0)
        finite=ratios[np.isfinite(ratios)]
        lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
        ci.append(dict(zip(['scenario','category','method','scope'],keys))|dict(mae_ci_low=lo,mae_ci_high=hi,sites=len(ds)))
    pd.DataFrame(ci).to_csv(OUT/'benchmark_mae_site_bootstrap_ci.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,scope in zip(axes,['all','daylight']):
        tab=summary.xs(('internal',scope),level=('scenario','scope')).mae_cf.unstack('method').reindex(['1-3h','4-24h','>24h'])
        tab.plot.bar(ax=ax); ax.set(title=scope+' (method-specific coverage)',ylabel='MAE in CF',xlabel='Gap length')
    savefig('benchmark_errors')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,scope in zip(axes,['all','daylight']):
        tab=summary.xs(('leading',scope),level=('scenario','scope'))
        tab.mae_cf.unstack('method').plot.bar(ax=ax)
        ax.set(title='Synthetic leading gaps: '+scope,ylabel='MAE in CF')
    savefig('benchmark_leading')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    summary.xs(('internal','all'),level=('scenario','scope')).coverage.unstack('method').plot.bar(ax=axes[0])
    summary.xs(('internal','common_daylight'),level=('scenario','scope')).mae_cf.unstack('method').plot.bar(ax=axes[1])
    axes[0].set_ylabel('Restoration coverage'); axes[1].set_ylabel('MAE CF, common daylight targets'); savefig('benchmark_coverage_common')
    overview=dict(training_start=str(timeline[0]),training_end=str(timeline[-1]),sites=ns,
        expected_hours_per_site=n,observed_rows=len(gen),expected_cells=n*ns,
        observation_rate=float(np.isfinite(y).mean()),
        pooled_active_observation_rate=float(coverage.observed_hours.sum()/coverage.active_span_hours.sum()),
        missing_runs=len(gaps),missing_hours=int(gaps.length_hours.sum()),
        leading_missing_hours=int(coverage.leading_missing_hours.sum()),
        internal_missing_hours=int(coverage.internal_missing_hours.sum()),
        trailing_missing_hours=int(coverage.trailing_missing_hours.sum()),
        artificial_blocks=len(blocks),artificial_target_hours=int(blocks.length_hours.sum()),
        quality=quality,input_sha256=hashes,seed=20261008)
    assert all(digest(p)==hashes[str(p.relative_to(ROOT))] for p in files), 'Input changed'
    (OUT/'overview.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
    log(json.dumps(overview,indent=2))
    log(summary.xs('daylight',level='scope')[['coverage','mae_cf','rmse_cf']].to_string())

if __name__=='__main__': main()
