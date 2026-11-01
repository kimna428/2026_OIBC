"""Small Python metadata + lazily reopened read-only mmap arrays, safe for worker spawn."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import aware, day_hours, normalize, static_features, split_plan
from .config import fingerprint
from analysis.validation import validate_sites, validate_composition

class DayCache:
    def __init__(self,root):
        self.root=Path(root); self._arrays={}
        required=['manifest.json','sites.csv','composition.csv']+[n+'.npy' for n in ['weather','temporal','target','energy','missing_kind','static','smp','k']]
        missing=[str(self.root/n) for n in required if not (self.root/n).is_file()]
        if missing: raise FileNotFoundError('Incomplete cache:\n'+'\n'.join(missing)+'\nRun python -m analysis.deep_learning.prepare with the provided data first.')
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        signed={k:v for k,v in self.manifest.items() if k!='signature'}
        if fingerprint(signed)!=self.manifest['signature']: raise ValueError('Cache manifest signature mismatch')
        self.sites=pd.read_csv(self.root/'sites.csv'); validate_sites(self.sites)
        if self.sites.site_id.tolist()!=self.manifest['site_ids']: raise ValueError('Canonical site order differs')
        self.composition=pd.read_csv(self.root/'composition.csv'); validate_composition(self.composition,self.sites)
        if self.composition.to_dict('records')!=self.manifest['composition']: raise ValueError('Composition changed')
        self.days=pd.DatetimeIndex([aware(x) for x in self.manifest['days']]).as_unit('ns')
        self.num_sites=len(self.sites)
        if self.static.shape!=(self.num_sites,5) or not np.allclose(self.static,self.sites[['capacity_kw','lat','lon','wa_dist','wb_dist']].to_numpy(),equal_nan=True): raise ValueError('Static mmap and canonical site metadata differ')
        shape=(len(self.days),self.num_sites,24)
        if self.weather.shape!=shape+(len(self.manifest['weather_features']),) or self.target.shape!=shape or self.energy.shape!=shape or self.temporal.shape!=shape+(len(self.manifest['temporal_features']),): raise ValueError('Cache array schema mismatch')
    def __getstate__(self): return {k:v for k,v in self.__dict__.items() if k!='_arrays'}|{'_arrays':{}}
    def __getattr__(self,name):
        if name in ['weather','temporal','target','energy','missing_kind','static','smp','k']:
            if name not in self._arrays: self._arrays[name]=np.load(self.root/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            return self._arrays[name]
        raise AttributeError(name)

class DayDataset(Dataset):
    def __init__(self,cache,normalizer,plan,split,full_fleet=False):
        self.cache=cache; self.normalizer=normalizer; self.plan=plan; self.split=split; self.full_fleet=full_fleet
        if split!='test' and normalizer['cache_signature']!=cache.manifest['signature']: raise ValueError('Normalizer/cache mismatch')
        cutoff=aware(plan['cutoff']); self.cutoff=cutoff
        days=cache.days
        if split=='train': selected=np.flatnonzero(days+pd.Timedelta(hours=1)<=cutoff)
        elif split=='dev': selected=np.flatnonzero((days>=aware(plan['dev_start']))&(days<aware(plan['dev_end'])))
        elif split=='holdout': selected=np.flatnonzero((days>=aware(plan['holdout_start']))&(days<aware(plan['holdout_end'])))
        elif split=='test': selected=np.arange(len(days))
        else: raise ValueError('Unknown split')
        if not len(selected): raise ValueError(f'No days for {split}; cache does not cover configured dates')
        if split in ['dev','holdout']:
            expected=pd.date_range(aware(plan[split+'_start']),aware(plan[split+'_end'])-pd.Timedelta(days=1),freq='D').as_unit('ns')
            if not days[selected].equals(expected): raise ValueError(f'Incomplete {split} day axis')
            first_issue=days[selected[0]]-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
            if cutoff>first_issue: raise ValueError('Model training cutoff after validation issuance')
        self.day_indices=selected
        self.items=[]
        for d in selected:
            if split=='train':
                y=np.asarray(cache.target[d]); hours=day_hours(days[d]); valid=np.isfinite(y)&(y>=0)&(y<=1)&(hours.asi8[None,:]<=cutoff.value)
                if full_fleet:
                    if valid.any(): self.items.append((int(d),-1))
                else:
                    self.items.extend((int(d),int(s)) for s in np.flatnonzero(valid.any(axis=1)))
            elif full_fleet: self.items.append((int(d),-1))
            else: self.items.extend((int(d),s) for s in range(cache.num_sites))
        if not self.items: raise ValueError('No supervised samples in training range')
        self.items=np.asarray(self.items,dtype='int64')
        raw,flags,onehot,valid=static_features(cache.sites,cache.manifest['station_categories'],normalizer.get('station_seen'))
        scaled,_=normalize(raw,normalizer['static'])
        self.static_values=np.concatenate([scaled,flags.astype('float32'),onehot],axis=-1)
        self.metadata_valid=valid; self.seen=np.asarray(normalizer['seen_hours'])
    def __len__(self): return len(self.items)
    def __getitem__(self,index):
        day,site=self.items[index]; selection=slice(None) if self.full_fleet else int(site)
        w=np.array(self.cache.weather[day,selection],copy=True); w,wm=normalize(w,self.normalizer['weather'])
        temporal,_=normalize(np.array(self.cache.temporal[day,selection],copy=True),self.normalizer['temporal'])
        y=np.array(self.cache.target[day,selection],dtype='float32',copy=True)
        observed=np.isfinite(y); train_mask=observed&(y>=0)&(y<=1)
        time_ok=day_hours(self.cache.days[day]).asi8<=self.cutoff.value
        if self.split=='train':
            train_mask&=time_ok
            y[...,~time_ok]=np.nan
            observed&=time_ok
        else: train_mask=observed.copy()
        positions=np.arange(self.cache.num_sites) if self.full_fleet else np.asarray(site)
        ids=np.where(self.seen[positions]>0,positions+1,0).astype('int64')
        static=self.static_values[selection].copy(); valid=np.asarray(self.metadata_valid[selection]).copy()
        smp=np.array(self.cache.smp[day],dtype='float32',copy=True); k=np.array(self.cache.k[day],dtype='float32',copy=True)
        capacity=self.cache.sites.capacity_kw.to_numpy(dtype='float32')[selection].copy()
        complete=np.isfinite(self.cache.target[day]).all(axis=0)
        if self.split=='train': complete&=time_ok & ((self.cache.target[day]>=0)&(self.cache.target[day]<=1)).all(axis=0)
        arrays={'weather':w,'weather_missing':wm,'temporal':temporal,'static':static,'site_index':ids,'site_pos':positions,'metadata_valid':valid,'target_cf':y,'target_mask':observed,'loss_mask':train_mask,'capacity_kw':capacity,'smp':smp,'k':k,'group_complete':complete.copy(),'missing_kind':np.array(self.cache.missing_kind[day,selection],copy=True),'day_index':np.asarray(day)}
        return {name:torch.from_numpy(np.array(a,copy=True)) for name,a in arrays.items()}
