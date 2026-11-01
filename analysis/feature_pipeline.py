"""Chunked round-17 pipeline. No fitting, imputation, observed weather or lags."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
try:
    from .validation import local_hours, split_labels, validate_sites
except ImportError:
    from validation import local_hours, split_labels, validate_sites

META=['capacity_kw','lat','lon','wa_dist','wb_dist']
TIME=['month','operating_hour','dayofyear','hour_sin','hour_cos','year_sin','year_cos']
AUDIT=['ts','operating_day','energy','target_cf','target_present','negative_energy','cf_gt_1','smp','k']

def weather_columns(data):
    result={}
    for provider in ('wa','wb'):
        a=pq.read_schema(data/'train'/f'forecast_{provider}.parquet')
        b=pq.read_schema(data/'test'/f'forecast_{provider}.parquet')
        cols=[n for n in a.names if n not in (provider+'_station','round','ts')]
        if set(cols)!={n for n in b.names if n not in (provider+'_station','round','ts')}:
            raise ValueError('train/test weather schema mismatch')
        if any(a.field(c).type!=b.field(c).type for c in cols):
            raise ValueError('train/test weather dtype mismatch')
        result[provider]=cols
    return result

def feature_columns(cols):
    names=['site_id']+META+TIME
    for p in ('wa','wb'):
        names+=cols[p]+[p+'_row_missing',p+'_any_missing']+[c+'_missing' for c in cols[p]]
    return names

def forecast_times(values):
    t=pd.DatetimeIndex(values)
    if t.tz is None or t.hasnans: raise ValueError("forecast timestamps require timezone")
    return t.tz_convert("Asia/Seoul")

def check_forecast_keys(path, provider):
    station=provider+'_station'
    keys=pd.read_parquet(path,columns=[station,'round','ts'])
    keys['ts']=forecast_times(keys.ts)
    if keys.isna().any().any() or keys.duplicated([station,'round','ts']).any():
        raise ValueError(f'duplicate/null forecast key: {path}')
    return {'rows_all_rounds':len(keys),'rows_round17':int((keys['round']==17).sum()),'duplicate_keys':0,'off_hour_rows':int((keys.ts!=keys.ts.dt.floor('h')).sum())}

def join_weather(base, weather, station, columns):
    """Fixed-size MultiIndex lookup; never many-to-many merges."""
    if weather.duplicated([station,'ts']).any():
        raise ValueError('duplicate selected forecast key')
    index=pd.MultiIndex.from_frame(weather[[station,'ts']])
    lookup=pd.MultiIndex.from_arrays([base[station],base.ts],names=[station,'ts'])
    pos=index.get_indexer(lookup)
    found=pos>=0
    values={}
    for c in columns:
        arr=np.full(len(base),np.nan,dtype='float64')
        arr[found]=weather[c].to_numpy()[pos[found]]
        if np.isinf(arr).any():
            raise ValueError('infinite weather values need explicit source investigation')
        values[c]=arr
        values[c+'_missing']=np.isnan(arr)
    p=station.split('_')[0]
    values[p+'_row_missing']=~found
    values[p+'_any_missing']=np.logical_or.reduce([values[c+'_missing'] for c in columns])
    return values

def build_chunk(times, sites, generation, prices, weather, columns):
    """Build only a bounded time chunk × 253 sites required for prediction.

    Metadata maps and forecast index lookups preserve cardinality exactly.
    All numeric raw values remain float64. Target flags are audit-only.
    """
    validate_sites(sites)
    times=local_hours(times)
    if not times.is_unique: raise ValueError('duplicate requested times')
    ids=sites.site_id.sort_values().to_numpy()
    base=pd.DataFrame({'ts':times.repeat(len(ids)), 'site_id':np.tile(ids,len(times))})
    meta=sites.set_index('site_id')
    for c in META+['wa_station','wb_station']:
        base[c]=base.site_id.map(meta[c])
    t=local_hours(base.ts)
    day=(t-pd.Timedelta(nanoseconds=1)).normalize()
    # Interval ordinal 1..24 within the operating day; midnight ends hour 24.
    hour=np.where(t.hour==0,24,t.hour)
    doy=day.dayofyear.to_numpy()
    year_days=np.where(day.is_leap_year,366.,365.)
    base['operating_day']=day
    base['month']=day.month.astype('int8')
    base['operating_hour']=hour.astype('int8')
    base['dayofyear']=doy.astype('int16')
    for name,angle in [('hour',2*np.pi*hour/24),('year',2*np.pi*(doy-1)/year_days)]:
        base[name+'_sin']=np.sin(angle); base[name+'_cos']=np.cos(angle)
    g=generation.copy()
    g['ts']=local_hours(g.ts)
    if g.duplicated(['site_id','ts']).any(): raise ValueError('duplicate generation keys')
    index=pd.MultiIndex.from_frame(g[['site_id','ts']])
    lookup=pd.MultiIndex.from_frame(base[['site_id','ts']])
    energy=g.set_index(['site_id','ts']).energy.reindex(lookup).to_numpy()
    if np.isinf(energy).any(): raise ValueError('nonfinite target needs investigation')
    base['energy']=energy
    base['target_cf']=energy/base.capacity_kw.to_numpy()
    base['target_present']=np.isfinite(energy)
    base['negative_energy']=energy<0
    base['cf_gt_1']=base.target_cf>1
    price=prices.copy(); price['ts']=local_hours(price.ts)
    if price.ts.duplicated().any(): raise ValueError('duplicate SMP key')
    base['smp']=base.ts.map(price.set_index('ts').smp)
    if base.smp.isna().any(): raise ValueError('missing SMP')
    base['k']=np.where((t.hour>=11)&(t.hour<=15),3,1).astype('int8')
    for p in ('wa','wb'):
        w=weather[p].copy(); w['ts']=forecast_times(w.ts)
        # Caller supplies round-17 only; enforce rather than silently combine rounds.
        if not (w['round']==17).all(): raise ValueError('only round 17 is eligible')
        vals=join_weather(base,w,p+'_station',columns[p])
        for c,v in vals.items(): base[c]=v
    base['site_id']=pd.Categorical(base.site_id,categories=ids)
    base=base[AUDIT+feature_columns(columns)]
    if len(base)!=len(times)*253 or base.duplicated(['site_id','ts']).any():
        raise ValueError('join cardinality violation')
    return base

def load_dataset(path, features_only=False):
    """Read a generated partition. Use manifest feature_columns for model inputs."""
    import json
    path=Path(path)
    cols=json.loads((path.parent/'manifest.json').read_text())['feature_columns'] if features_only else None
    return pd.read_parquet(path,columns=cols)
