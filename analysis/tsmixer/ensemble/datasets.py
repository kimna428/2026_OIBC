import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from analysis.deep_learning.features import day_hours
from .features import auxiliary_day,FC,AUX_ALL
from .sj_scaling import feature_transform,transform_rule


def outage_mask(cache,day,records):
    if records is None:return np.zeros((cache.num_sites,24),bool)
    keys=pd.MultiIndex.from_frame(records[['site_id','date']])
    times=day_hours(cache.days[day])
    ix=pd.MultiIndex.from_arrays([np.repeat(cache.sites.site_id,24),np.tile((times-pd.Timedelta(hours=1)).normalize() if records.attrs.get('operating_day') else times.normalize(),cache.num_sites)])
    return ix.isin(keys).reshape(cache.num_sites,24)


def masks(history,features,d,family,outage=None):
    source=int(history.indices[d]);target=np.asarray(history.target[d]);observed=np.isfinite(target)
    weather=features.sj(source) if family=='sj_ensemble' else features.state(source)
    valid=observed&weather['day']
    if family=='sj_ensemble':
        # Original SJ suspicious-zero filtering; all criteria are historical.
        from .sj_weather import solar_position
        times=day_hours(history.days[d]);lat=np.tile(history.sites.lat,24);lon=np.tile(history.sites.lon,24)
        elevation=solar_position((times-pd.Timedelta(minutes=30)).repeat(history.num_sites),lat,lon)[0].reshape(24,-1).T
        valid&=~((target==0)&(elevation>15))
    valid&=~outage_mask(history.base,source,outage)
    return valid,weather


def fit_stats(history,features,family,outage=None,obs=None):
    if family=='sj_ensemble':
        names=features.spec['columns'];cats=features.spec['categorical'];numeric=[n for n in names if n not in cats]
        ni=[names.index(n) for n in numeric];ci=[names.index(n) for n in cats]
        chunks=[];levels=[set() for _ in cats]
        for d in range(len(history.days)):
            valid,weather=masks(history,features,d,family,outage)
            raw=weather['raw'];v=raw[valid]
            if not len(v):continue
            values=v[:,ni].astype('float32')
            for j,n in enumerate(numeric):values[:,j]=feature_transform(values[:,j],n)
            chunks.append(values)
            for j,index in enumerate(ci):levels[j].update(v[:,index].astype(int).tolist())
        if not chunks:raise ValueError('No historical supervised rows')
        matrix=np.concatenate(chunks);med=[];scale=[]
        for j,n in enumerate(numeric):
            v=matrix[:,j];v=v[np.isfinite(v)];q=np.percentile(v,[25,50,75]) if len(v) else [0,0,1]
            _,unit=transform_rule(n);med.append(float(q[1]));scale.append(float(unit if unit is not None else max(q[2]-q[0],np.std(v) if len(v) else 0,1)))
        return dict(family=family,numeric=numeric,categorical=cats,numeric_indices=ni,cat_indices=ci,medians=med,scales=scale,
                    levels=[sorted(s) for s in levels],fit_cutoff=str(history.cutoff))
    count=np.zeros(15);total=np.zeros(15);square=np.zeros(15)
    oc=np.zeros(15);ot=np.zeros(15);os=np.zeros(15)
    for d in range(len(history.days)):
        source=int(history.indices[d]);raw=features.state(source)['raw']
        for x in (raw[...,:15],raw[...,17:32]):
            valid=np.isfinite(x);count+=valid.sum((0,1));total+=np.where(valid,x,0).sum((0,1),dtype='float64');square+=np.where(valid,x.astype('float64')**2,0).sum((0,1))
        o=auxiliary_day(obs,history.base,source)
        o[...,day_hours(history.days[d])>history.cutoff,:]=np.nan
        valid=np.isfinite(o);oc+=valid.sum((0,1));ot+=np.where(valid,o,0).sum((0,1),dtype='float64');os+=np.where(valid,o.astype('float64')**2,0).sum((0,1))
    mean=total/np.maximum(count,1);sd=np.sqrt(np.maximum(square/np.maximum(count,1)-mean**2,0))+1e-6
    om=ot/np.maximum(oc,1);sdobs=np.sqrt(np.maximum(os/np.maximum(oc,1)-om**2,0))+1e-6
    return dict(family=family,mean=mean.tolist(),std=sd.tolist(),obs_mean=om.tolist(),obs_std=sdobs.tolist(),fit_cutoff=str(history.cutoff))


class NeuralDays(Dataset):
    def __init__(self,view,features,stats,family,indices,training=False,outage=None,obs=None):
        self.view,self.features,self.stats,self.family=view,features,stats,family
        self.indices=list(map(int,indices));self.training=training;self.outage=outage;self.obs=obs
        self.items=[]
        for d in self.indices:
            if family=='sj_ensemble':
                if training:
                    valid,_=masks(view,features,d,family,outage)
                    self.items.extend((d,int(s)) for s in np.flatnonzero(valid.any(1)))
                else:self.items.extend((d,s) for s in range(view.num_sites))
            elif not training or masks(view,features,d,family,outage)[0].any():self.items.append((d,-1))
        if not self.items:raise ValueError('No usable neural samples')
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        d,site=self.items[i];v=self.view;source=int(v.indices[d]) if hasattr(v,'indices') else d
        w=self.features.sj(source) if self.family=='sj_ensemble' else self.features.state(source)
        y=np.asarray(v.target[d]);observed=np.isfinite(y)
        valid=masks(v,self.features,d,self.family,self.outage)[0] if self.training else observed
        target=np.where(observed,np.clip(y,0,1.2) if self.training and self.family=='state01' else y,0).astype('float32')
        if self.family=='sj_ensemble':
            raw=w['raw'][site];s=self.stats;x=raw[:,s['numeric_indices']].copy()
            for j,n in enumerate(s['numeric']):x[:,j]=feature_transform(x[:,j],n)
            missing=~np.isfinite(x);x=np.where(missing,s['medians'],x);x=np.clip((x-s['medians'])/s['scales'],-8,8).astype('float32')
            categorical=np.zeros((24,len(s['categorical'])),dtype='int64')
            for j,levels in enumerate(s['levels']):
                original=raw[:,s['cat_indices'][j]]
                for k,value in enumerate(levels,1):categorical[:,j][original==value]=k
            arrays=dict(numeric=x,missing=missing,categorical=categorical,target=target[site],mask=valid[site],observed=observed[site],day=w['day'][site],site=np.asarray(site),day_index=np.asarray(d))
        else:
            x=w['raw'].copy();s=self.stats
            for selection in (slice(0,15),slice(17,32)):
                x[...,selection]=np.nan_to_num((x[...,selection]-s['mean'])/s['std'])
            o=np.full((v.num_sites,24,15),np.nan,dtype='float32') if self.obs is None else auxiliary_day(self.obs,v.base,source)
            if hasattr(v,'cutoff'):o[...,day_hours(v.days[d])>v.cutoff,:]=np.nan
            ov=np.isfinite(o)&w['day'][...,None];o=np.where(ov,(o-s['obs_mean'])/s['obs_std'],0)
            arrays=dict(x=x.astype('float32'),target=target,mask=valid,observed=observed,day=w['day'],obs=o.astype('float32'),aux_valid=ov,
                        price=np.asarray(v.smp[d]*v.k[d],dtype='float32'),day_index=np.asarray(d))
        return {k:torch.from_numpy(np.array(a,copy=True)) for k,a in arrays.items()}
