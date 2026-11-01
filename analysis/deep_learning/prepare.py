"""Month-bounded reuse of the existing round17 builder; append round10/revisions to mmap caches."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from analysis.feature_pipeline import build_chunk, join_weather, weather_columns
from analysis.validation import validate_sites, validate_composition, local_hours
from .config import ROOT, path, fingerprint
from .features import STATIC, TIME, contract, operating_day, issuance, assert_asof, temporal_features

REQUIRED=['sites.csv','smp.csv','train/smp.csv','train/generation.parquet','train/forecast_wa.parquet','train/forecast_wb.parquet','test/forecast_wa.parquet','test/forecast_wb.parquet']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def inspect_inputs(data):
    data=Path(data); missing=[str(data/p) for p in REQUIRED if not (data/p).is_file()]
    if missing: raise FileNotFoundError('Missing provided data files:\n'+'\n'.join(missing)+'\nCopy the competition dataset or set OIBC_DATA_ROOT; no data will be synthesized.')
    sites=pd.read_csv(data/'sites.csv').sort_values('site_id').reset_index(drop=True); validate_sites(sites)
    for c in STATIC+['wa_station','wb_station']:
        if c not in sites: raise ValueError(f'Missing site metadata: {c}')
    if set(pq.read_schema(data/'train/generation.parquet').names)!={'site_id','ts','energy'}: raise ValueError('Unexpected generation schema')
    cols=weather_columns(data); expected=contract()['weather_base']
    if cols['wa']+cols['wb']!=expected: raise ValueError('Provided forecast schema differs from the stage-three feature contract')
    return sites,cols

def balanced_composition(sites):
    loads={'G1':0.,'G2':0.}; assignment={}
    for r in sites.sort_values(['capacity_kw','site_id'],ascending=[False,True]).itertuples():
        g=min(loads,key=loads.get); assignment[r.site_id]=g; loads[g]+=r.capacity_kw
    comp=sites[['site_id']].assign(group=sites.site_id.map(assignment)); validate_composition(comp,sites)
    baseline=ROOT/'reports/expanding/composition.csv'
    if baseline.exists():
        old=pd.read_csv(baseline).set_index('site_id').group.sort_index()
        if not old.equals(comp.set_index('site_id').group.sort_index()): raise ValueError('Balanced composition differs from stage-three baseline')
    return comp

def coverage(data,sites):
    bounds={}; finite=0
    for batch in pq.ParquetFile(data/'train/generation.parquet').iter_batches(batch_size=100000):
        d=batch.to_pandas(); d.ts=local_hours(d.ts)
        if not d.site_id.isin(sites.site_id).all() or not np.isfinite(d.energy).all(): raise ValueError('Invalid raw generation')
        finite+=len(d)
        for site,x in d.groupby('site_id',observed=True):
            first,last=x.ts.min(),x.ts.max()
            old=bounds.get(site,(first,last)); bounds[site]=(min(first,old[0]),max(last,old[1]))
    return bounds,finite

def read_range(file,start,end,columns=None):
    d=pd.read_parquet(file,columns=columns,filters=[('ts','>=',start.to_pydatetime()),('ts','<=',end.to_pydatetime())])
    d.ts=pd.to_datetime(d.ts).dt.tz_convert('Asia/Seoul').dt.as_unit('ns')
    return d

def prepared_partition(prepared,start,end,base_columns):
    result=[]
    for name in ['train','dev','holdout']:
        d=read_range(prepared/f'{name}.parquet',start,end,base_columns)
        if len(d): result.append(d)
    return pd.concat(result,ignore_index=True) if result else pd.DataFrame(columns=base_columns)

def prepare(data,cache_root,prepared_root=None,start=None,end=None,partitions=('train','test')):
    data=Path(data); cache_root=Path(cache_root); sites,cols=inspect_inputs(data)
    raw_hashes={p:sha(data/p) for p in REQUIRED}
    reuse=False; prepared=Path(prepared_root) if prepared_root else None
    if prepared and (prepared/'manifest.json').exists() and all((prepared/f'{n}.parquet').exists() for n in ['train','dev','holdout','test']):
        old=json.loads((prepared/'manifest.json').read_text())
        for p,h in old['input_sha256'].items():
            rel=str(Path(p).relative_to('handover 2/01_participant_dataset'))
            if raw_hashes.get(rel)!=h: raise ValueError('Existing Parquet source hashes differ')
        for n,v in old['output_files'].items():
            if sha(prepared/f'{n}.parquet')!=v['sha256']: raise ValueError('Existing Parquet hash differs')
        reuse=True
    bounds,raw_rows=coverage(data,sites)
    prices_by_part={p:pd.read_csv(data/('train/smp.csv' if p=='train' else 'smp.csv')) for p in ['train','test']}
    for p in prices_by_part.values():
        p.ts=local_hours(pd.to_datetime(p.ts)).as_unit('ns')
        if p.ts.duplicated().any() or not np.isfinite(p.smp).all() or (p.smp<0).any(): raise ValueError('Invalid SMP')
    composition=balanced_composition(sites)
    categories={p:sorted(sites[p+'_station'].dropna().unique().tolist()) for p in ['wa','wb']}
    feature_spec={'version':1,'weather_features':contract()['weather_features'],'temporal_features':TIME,'static_features':STATIC,'site_ids':sites.site_id.tolist(),'station_categories':categories,'composition':composition.to_dict('records'),'units':'Provided original units; train-only standardization','input_hashes':raw_hashes}
    feature_signature=fingerprint(feature_spec)
    for partition in partitions:
        prices=prices_by_part[partition]
        all_times=pd.DatetimeIndex(prices.ts)
        if not all_times.equals(pd.date_range(all_times.min(),all_times.max(),freq='h').as_unit('ns')): raise ValueError('SMP must be an hourly continuous timeline')
        days=operating_day(all_times).unique()
        if start: days=days[days>=pd.Timestamp(start,tz='Asia/Seoul')]
        if end: days=days[days<pd.Timestamp(end,tz='Asia/Seoul')]
        if not len(days): raise ValueError(f'No {partition} days selected')
        dest=cache_root/partition
        if dest.exists() and any(dest.iterdir()): raise FileExistsError(f'Cache exists: {dest}; use a new cache directory')
        dest.mkdir(parents=True,exist_ok=True)
        shape=(len(days),253,24); wlen=len(feature_spec['weather_features'])
        arrays={}
        specs={'weather':(shape+(wlen,),'float32',np.nan),'temporal':(shape+(len(TIME),),'float32',np.nan),'target':(shape,'float64',np.nan),'energy':(shape,'float64',np.nan),'missing_kind':(shape,'uint8',5),'smp':((len(days),24),'float64',np.nan),'k':((len(days),24),'float32',np.nan)}
        for name,(size,dtype,fill) in specs.items():
            arrays[name]=np.lib.format.open_memmap(dest/f'{name}.npy',mode='w+',dtype=dtype,shape=size)
            # Initialize by chunks to avoid a separate full-memory allocation.
            for i in range(0,len(days),31): arrays[name][i:i+31]=fill
        np.save(dest/'static.npy',sites[STATIC].to_numpy(dtype='float32'))
        sites.to_csv(dest/'sites.csv',index=False); composition.to_csv(dest/'composition.csv',index=False)
        source=dest/'source_audit.jsonl'; observed=0
        month_labels=[f'{d.year:04d}-{d.month:02d}' for d in days]
        for month in dict.fromkeys(month_labels):
            positions=np.flatnonzero(np.asarray(month_labels)==month); selected=days[positions]
            times=pd.date_range(selected.min()+pd.Timedelta(hours=1),selected.max()+pd.Timedelta(days=1),freq='h').as_unit('ns')
            subset_prices=prices[prices.ts.isin(times)].copy()
            if len(subset_prices)!=len(times): raise ValueError('Partial operating day in price timeline')
            weather={}; weather10={}; key_info={}
            for provider in ['wa','wb']:
                d=read_range(data/partition/f'forecast_{provider}.parquet',times.min(),times.max())
                station=provider+'_station'
                if d.duplicated([station,'round','ts']).any() or not d['round'].isin([10,17]).all(): raise ValueError('Invalid forecast keys/rounds')
                key_info[provider]={'rows':len(d),'off_hour_rows':int((d.ts!=d.ts.dt.floor('h')).sum())}
                weather[provider]=d[d['round']==17]; weather10[provider]=d[d['round']==10]
            if reuse:
                needed=['ts','site_id','energy','smp','k']+STATIC+['wa_station','wb_station'] if False else ['ts','site_id','energy','smp','k']+STATIC+cols['wa']+cols['wb']
                base=read_range(prepared/'test.parquet',times.min(),times.max(),needed) if partition=='test' else prepared_partition(prepared,times.min(),times.max(),needed)
                base=base.sort_values(['ts','site_id']).reset_index(drop=True)
                if len(base)!=len(times)*253 or base.duplicated(['site_id','ts']).any(): raise ValueError('Existing Parquet grid cardinality differs')
                lookup=pd.MultiIndex.from_frame(base[['ts','site_id']]); expected=pd.MultiIndex.from_product([times,sites.site_id],names=['ts','site_id'])
                if not lookup.equals(expected): raise ValueError('Existing Parquet order/grid differs')
            else:
                generation=read_range(data/'train/generation.parquet',times.min(),times.max()) if partition=='train' else pd.DataFrame({'site_id':pd.Series(dtype=str),'ts':pd.Series([],dtype='datetime64[ns, Asia/Seoul]'),'energy':pd.Series(dtype=float)})
                base=build_chunk(times,sites,generation,subset_prices,weather,cols)
            for provider in ['wa','wb']:
                base[provider+'_station']=base.site_id.map(sites.set_index('site_id')[provider+'_station'])
            values={c:base[c].to_numpy(dtype='float32') for c in cols['wa']+cols['wb']}
            for provider in ['wa','wb']:
                joined=join_weather(base,weather10[provider],provider+'_station',cols[provider])
                for c in cols[provider]: values[c+'_r10']=joined[c].astype('float32'); values[c+'_revision']=values[c]-values[c+'_r10']
            for r in [10,17]: assert_asof(issuance(times)-pd.Timedelta(hours=17-r),times)
            raw=np.stack([values[c] for c in feature_spec['weather_features']],axis=-1).reshape(len(selected),24,253,wlen).transpose(0,2,1,3)
            temporal=temporal_features(times,sites).reshape(len(selected),24,253,len(TIME)).transpose(0,2,1,3)
            energy=base.energy.to_numpy(dtype=float).reshape(len(selected),24,253).transpose(0,2,1)
            if partition=='test' and np.isfinite(energy).any(): raise ValueError('Test contains labels')
            target=energy/sites.capacity_kw.to_numpy()[None,:,None]
            arrays['weather'][positions]=raw; arrays['temporal'][positions]=temporal; arrays['energy'][positions]=energy; arrays['target'][positions]=target
            arrays['smp'][positions]=subset_prices.set_index('ts').smp.reindex(times).to_numpy().reshape(len(selected),24)
            from analysis.validation import hourly_weights
            arrays['k'][positions]=hourly_weights(times).reshape(len(selected),24)
            kind=np.zeros_like(energy,dtype='uint8')
            if partition=='test': kind[:]=5
            else:
                for j,site in enumerate(sites.site_id):
                    missing=~np.isfinite(energy[:,j,:]); first,last=bounds.get(site,(None,None))
                    if first is None: kind[:,j,:]=4
                    else:
                        ts=times.to_numpy().reshape(len(selected),24)
                        category=np.where(ts<first,1,np.where(ts>last,3,2))
                        kind[:,j,:]=np.where(missing,category,0)
            arrays['missing_kind'][positions]=kind; observed+=int(np.isfinite(energy).sum())
            with source.open('a') as f: f.write(json.dumps({'month':month,'days':len(selected),'rounds':[10,17],'forecast_keys':key_info,'asof_violations':0,'reused_round17_parquet':reuse})+'\n')
            print(f'{partition} {month}: {len(selected)} days',flush=True)
        for a in arrays.values(): a.flush()
        if partition=='train' and not start and not end and observed!=raw_rows: raise ValueError('Cache target coverage differs from original generation')
        manifest=feature_spec|{'kind':'provided','partition':partition,'start':str(days.min().date()),'end_exclusive':str((days.max()+pd.Timedelta(days=1)).date()),'test_start':str(operating_day(prices_by_part['test'].ts).min().date()),'days':[str(d.date()) for d in days],'num_sites':253,'shape':list(shape),'feature_signature':feature_signature,'observed_targets':observed,'missing_kind_codes':{'observed':0,'pre_first':1,'internal':2,'post_last':3,'no_history':4,'test_unlabelled':5},'bounds':{k:[str(v[0]),str(v[1])] for k,v in bounds.items()},'reused_round17_parquet':reuse}
        manifest['signature']=fingerprint(manifest)
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if raw_hashes!={p:sha(data/p) for p in REQUIRED}: raise ValueError('Original inputs changed')

def main():
    import os
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default=os.environ.get('OIBC_DATA_ROOT',str(ROOT/'handover 2/01_participant_dataset'))); p.add_argument('--cache-root',default=os.environ.get('OIBC_CACHE_ROOT',str(ROOT/'analysis/deep_learning/cache'))); p.add_argument('--prepared-root',default=str(ROOT/'analysis/datasets')); p.add_argument('--check-only',action='store_true'); p.add_argument('--start'); p.add_argument('--end'); p.add_argument('--partition',choices=['train','test','both'],default='both'); args=p.parse_args()
    inspect_inputs(Path(args.data_root))
    if args.check_only: print('Provided files and schemas are present; no cache generated.'); return
    prepare(args.data_root,args.cache_root,args.prepared_root,args.start,args.end,('train','test') if args.partition=='both' else (args.partition,))
if __name__=='__main__': main()
