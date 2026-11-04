"""Strict keyed adapters; no execution/import of teammate code or pickle models."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.validation import local_hours, hourly_weights
from .core import Grid,START,END,CUTOFF


def file_sha(path):
    with Path(path).open('rb') as stream:
        digest=getattr(hashlib,'file_digest',None)
        if digest is not None:return digest(stream,'sha256').hexdigest()
        # GPU server Python3.10 has no hashlib.file_digest (added in3.11).
        result=hashlib.sha256()
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
        return result.hexdigest()


def read_table(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)


def timestamped(frame):
    d=frame.copy()
    d['ts']=local_hours(pd.to_datetime(d.ts))
    if d.site_id.isna().any() or d.duplicated(['site_id','ts']).any():
        raise ValueError('Null or duplicate site/timestamp keys')
    return d


def load_actual(data):
    data=Path(data)
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id').reset_index(drop=True)
    generation=timestamped(pd.read_parquet(data/'train/generation.parquet'))
    if not generation.site_id.isin(sites.site_id).all():
        raise ValueError('Unknown target sites')
    # No future-target filtering is used to derive training history.
    history=generation.loc[generation.ts<=CUTOFF].copy()
    cf=history.energy.to_numpy()/history.site_id.map(sites.set_index('site_id').capacity_kw).to_numpy()
    seen=history.loc[np.isfinite(cf)&(cf>=0)&(cf<=1)].groupby('site_id').size().reindex(sites.site_id,fill_value=0).to_numpy()
    year=generation.loc[(generation.ts>=START)&(generation.ts<END)]
    times=pd.date_range(START,END-pd.Timedelta(hours=1),freq='h').as_unit('ns')
    actual=year.pivot(index='ts',columns='site_id',values='energy').reindex(index=times,columns=sites.site_id).to_numpy(dtype='float64')
    prices=pd.read_csv(data/'train/smp.csv');prices['ts']=local_hours(pd.to_datetime(prices.ts))
    if prices.ts.duplicated().any():
        raise ValueError('Duplicate SMP timestamps')
    prices=prices.set_index('ts').reindex(times)
    k=prices.k.to_numpy() if 'k' in prices else hourly_weights(times)
    grid=Grid(sites,times,actual,prices.smp.to_numpy(),k)
    audit=dict(data_root=str(data.resolve()),files={str(p.relative_to(data)):file_sha(p) for p in
               [data/'sites.csv',data/'train/generation.parquet',data/'train/smp.csv']},
               raw_year_rows=len(year),negative_targets=int((year.energy<0).sum()),
               raw_precision='float64',teammate_prepared_precision='float32; common evaluation retains original float64',
               float32_max_rounding_kwh=float(np.nanmax(np.abs(actual-actual.astype('float32').astype('float64')))),
               history_cutoff=str(CUTOFF),no_history_sites=sites.loc[seen==0,'site_id'].tolist())
    return grid,seen,audit


def align_frame(frame,grid,columns):
    d=timestamped(frame)
    if not d.site_id.isin(grid.sites.site_id).all():
        raise ValueError('Unknown prediction sites')
    d=d.loc[d.ts.isin(grid.times)]
    if d.empty:
        raise ValueError('No predictions at requested timestamps; check one-hour shift/profile')
    index=pd.MultiIndex.from_product([grid.times,grid.sites.site_id],names=['ts','site_id'])
    aligned=d.set_index(['ts','site_id']).reindex(index)
    shape=grid.actual.shape
    if 'energy' in aligned:
        supplied=aligned.energy.to_numpy(dtype='float64').reshape(shape)
        if not np.array_equal(np.isfinite(supplied),grid.observed):
            raise ValueError('Prediction table actual-mask differs from raw common actuals')
        if not np.allclose(supplied[grid.observed],grid.actual[grid.observed],rtol=1e-6,atol=1e-5):
            raise ValueError('Prediction table actual values differ from common actuals')
    result={}
    for name in columns:
        if name not in aligned:
            raise ValueError(f'Missing prediction column {name}')
        result[name]=grid.validate_prediction(aligned[name].to_numpy(dtype='float64').reshape(shape))
    return result


def load_tsmixer(root,grid):
    root=Path(root)
    axis=json.loads((root/'prediction_axis.json').read_text())
    if axis.get('units')!='capacity_factor' or axis.get('hours')!='01:00 through next-day 00:00 KST':
        raise ValueError('Unknown TSMixer units/hour convention')
    ids=axis['site_ids'];days=pd.DatetimeIndex(pd.to_datetime(axis['operating_days']))
    if days.tz is None or days.has_duplicates or len(ids)!=253 or len(set(ids))!=253 or set(ids)!=set(grid.sites.site_id):
        raise ValueError('Invalid TSMixer canonical axes')
    days=days.tz_convert('Asia/Seoul')
    times=pd.DatetimeIndex([d+pd.Timedelta(hours=h) for d in days for h in range(1,25)]).as_unit('ns')
    if not times.equals(grid.times):
        raise ValueError('TSMixer time axis differs; no one-hour shift/reindex allowed')
    raw=np.load(root/'predictions.npy',mmap_mode='r',allow_pickle=False)
    if raw.shape!=(len(days),253,24) or not np.isfinite(raw).all() or (raw<0).any():
        raise ValueError('Invalid TSMixer prediction shape/values')
    values=raw.transpose(0,2,1).reshape(len(times),253)
    reorder=pd.Index(ids).get_indexer(grid.sites.site_id)
    result=values[:,reorder]*grid.sites.capacity_kw.to_numpy()[None,:]
    provenance={'prediction_sha256':file_sha(root/'predictions.npy'),'axis_sha256':file_sha(root/'prediction_axis.json'),
                'cutoff_verified':False}
    run=root/'run.json'
    if run.is_file():
        r=json.loads(run.read_text());cutoff=pd.Timestamp(r['plans']['refit']['cutoff'])
        if cutoff.tzinfo is None or cutoff!=CUTOFF:
            raise ValueError('TSMixer refit cutoff mismatch')
        provenance.update(cutoff_verified=True,run_sha256=file_sha(run))
    return grid.validate_prediction(result),provenance
