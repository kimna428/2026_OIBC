"""Lossless monthly station/round mmap storage; reconstruct only requested site/day features."""
from collections import OrderedDict
import numpy as np
from .features import contract, temporal_features, day_hours

NAMES=['weather','temporal','target','energy','missing_kind','smp','k']

def estimate(days,sites,layout='compact'):
    """Upper bound includes .npy headers/metadata; no float16 or lossy packing."""
    if layout=='legacy': per_day=len(sites)*24*(90*4+8*4+8+8+1)+24*(8+4)
    else:
        per_day=sum(sites[p+'_station'].nunique(dropna=True)*24*sum(c.startswith(p+'_') for c in contract()['weather_base'])*2*4 for p in ['wa','wb'])+len(sites)*24*(8+1)+24*(8+1)
    return int(per_day*days+2**20+((days+27)//28)*4096)

class MonthlyWriter:
    def __init__(self,root,sites,features): self.root=root; self.sites=sites; self.features=features; self.chunks=[]
    def write(self,month,positions,raw,energy,kind,smp,k):
        directory=self.root/'months'/month; directory.mkdir(parents=True,exist_ok=False)
        cats={p:sorted(self.sites[p+'_station'].dropna().unique().tolist()) for p in ['wa','wb']}
        base=contract()['weather_base']
        for p in ['wa','wb']:
            columns=[c for c in base if c.startswith(p+'_')]
            indices=[self.features.index(c) for c in columns]+[self.features.index(c+'_r10') for c in columns]
            station_values=np.empty((len(positions),len(cats[p]),24,len(indices)),dtype='float32')
            for j,station in enumerate(cats[p]):
                members=np.flatnonzero(self.sites[p+'_station'].to_numpy()==station)
                values=np.take(raw[:,members[0]],indices,axis=-1)
                for member in members[1:]:
                    if not np.array_equal(values,np.take(raw[:,member],indices,axis=-1),equal_nan=True): raise ValueError('Forecasts differ within the same station; cannot deduplicate losslessly')
                station_values[:,j]=values
            # Unmapped station must already have all-NaN weather.
            missing=self.sites[p+'_station'].isna().to_numpy()
            if missing.any() and np.isfinite(np.take(raw[:,missing],indices,axis=-1)).any(): raise ValueError('Nonmissing forecast without station mapping')
            np.save(directory/(p+'.npy'),station_values,allow_pickle=False)
        for name,value in [('energy',energy.astype('float64')),('missing_kind',kind.astype('uint8')),('smp',smp.astype('float64')),('k',k.astype('uint8'))]: np.save(directory/(name+'.npy'),value,allow_pickle=False)
        self.chunks.append({'month':month,'start':int(positions[0]),'count':len(positions)})
    def metadata(self):
        return {'layout':'station_month_v1','chunks':self.chunks,'station_ids':{p:sorted(self.sites[p+'_station'].dropna().unique().tolist()) for p in ['wa','wb']},'weather_dtype':'float32','energy_dtype':'float64','revision':'computed float32 round17 - round10','temporal':'computed with existing temporal_features','target':'computed float64 energy / original capacity_kw'}

class CompactArray:
    def __init__(self,cache,name):
        self.cache=cache; self.name=name; shape=(len(cache.days),cache.num_sites,24)
        self.shape=shape+({'weather':(90,),'temporal':(8,)}.get(name,()))
        if name in ['smp','k']: self.shape=(len(cache.days),24)
        self.dtype=np.dtype('float64' if name in ['target','energy','smp'] else 'uint8' if name=='missing_kind' else 'float32')
    def __getitem__(self,key):
        keys=key if isinstance(key,tuple) else (key,)
        d=keys[0]
        if not isinstance(d,(int,np.integer)):
            ids=np.arange(self.shape[0])[d]
            return np.stack([self[int(i)] for i in np.atleast_1d(ids)])[(slice(None),)+keys[1:]]
        d=int(d)
        if d<0: d+=self.shape[0]
        if not 0<=d<self.shape[0]: raise IndexError('Day index out of range')
        c=self.cache; name=self.name
        if name=='temporal':
            if c._temporal_day is None or c._temporal_day[0]!=d:
                value=temporal_features(day_hours(c.days[d]),c.sites).reshape(24,c.num_sites,8).transpose(1,0,2)
                c._temporal_day=(d,value)
            result=c._temporal_day[1]
            return result[keys[1:]] if len(keys)>1 else result
        arrays,offset=c._month(d)
        if name=='weather':
            selection=keys[1] if len(keys)>1 else slice(None)
            positions=np.arange(c.num_sites)[selection]; scalar=np.ndim(positions)==0; positions=np.atleast_1d(positions)
            values={}; base=c._weather_base
            for p in ['wa','wb']:
                columns=[v for v in base if v.startswith(p+'_')]; n=len(columns)
                station=c._station_map[p][positions]; valid=station>=0
                a=np.full((len(positions),24,n*2),np.nan,dtype='float32')
                if valid.any(): a[valid]=arrays[p][offset,station[valid]]
                for j,v in enumerate(columns): values[v]=a[...,j]; values[v+'_r10']=a[...,n+j]; values[v+'_revision']=a[...,j]-a[...,n+j]
            result=np.stack([values[v] for v in c.manifest['weather_features']],axis=-1)
            if scalar: result=result[0]
            return result[keys[2:]] if len(keys)>2 else result
        if name=='target': result=arrays['energy'][offset]/c.sites.capacity_kw.to_numpy()[:,None]
        else: result=arrays[name][offset]
        if name=='k': result=result.astype('float32')
        return result[keys[1:]] if len(keys)>1 else result

def init_compact(cache):
    cache._months=OrderedDict(); cache._temporal_day=None; cache._weather_base=contract()['weather_base']
    if cache.manifest['station_ids']!=cache.manifest['station_categories']: raise ValueError('Station mapping differs from the feature contract')
    cache._station_map={p:cache.sites[p+'_station'].map({v:i for i,v in enumerate(cache.manifest['station_ids'][p])}).fillna(-1).to_numpy(dtype=int) for p in ['wa','wb']}
    cursor=0
    for chunk in cache.manifest['chunks']:
        if chunk['start']!=cursor or chunk['count']<1: raise ValueError('Invalid monthly day index')
        cursor+=chunk['count']
        days=cache.days[chunk['start']:cursor]
        if any(f'{d.year:04d}-{d.month:02d}'!=chunk['month'] for d in days): raise ValueError('Monthly day axis mismatch')
        for name in ['wa','wb','energy','missing_kind','smp','k']:
            file=cache.root/'months'/chunk['month']/(name+'.npy')
            if not file.is_file(): raise FileNotFoundError(f'Missing monthly cache {file}; rerun prepare into a new directory')
            a=np.load(file,mmap_mode='r',allow_pickle=False)
            if name in ['wa','wb']:
                shape=(chunk['count'],len(cache.manifest['station_ids'][name]),24,sum(c.startswith(name+'_') for c in contract()['weather_base'])*2); dtype='float32'
            elif name in ['smp','k']: shape=(chunk['count'],24); dtype='float64' if name=='smp' else 'uint8'
            else: shape=(chunk['count'],cache.num_sites,24); dtype='float64' if name=='energy' else 'uint8'
            if a.shape!=shape or a.dtype!=np.dtype(dtype): raise ValueError(f'Invalid monthly array schema: {file}')
    if cursor!=len(cache.days): raise ValueError('Monthly index does not cover all days')

def month(cache,day):
    chunk=next(x for x in cache.manifest['chunks'] if x['start']<=day<x['start']+x['count']); label=chunk['month']
    if label not in cache._months:
        cache._months[label]={n:np.load(cache.root/'months'/label/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ['wa','wb','energy','missing_kind','smp','k']}
        while len(cache._months)>2: cache._months.popitem(last=False)
    cache._months.move_to_end(label)
    return cache._months[label],day-chunk['start']
