"""Explicit B-family allowlist, operation-day clock and training-only normalization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.validation import local_hours
from .config import ROOT, fingerprint

STATIC=['capacity_kw','lat','lon','wa_dist','wb_dist']
TIME=['hour','month','dayofyear','hour_sin','hour_cos','dayofyear_sin','dayofyear_cos','solar_elevation']
FORBIDDEN={'energy','energy_kwh','cf','target_cf','target','target_mask','loss_mask','target_present','smp','k','group_complete','current_group','lag48_cf','history7d_cf','history7d_count'}
FOLDS=[('2023-06-01','2023-09-01'),('2023-12-01','2024-03-01'),('2024-06-01','2024-09-01'),('2024-09-01','2024-12-01'),('2024-12-01','2025-03-01'),('2025-03-01','2025-06-01')]

def aware(value):
    t=pd.Timestamp(value)
    return t.tz_localize('Asia/Seoul') if t.tzinfo is None else t.tz_convert('Asia/Seoul')

def operating_day(times): return (local_hours(times).as_unit('ns')-pd.Timedelta(nanoseconds=1)).normalize()
def issuance(times): return operating_day(times)-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
def day_hours(day): return pd.date_range(aware(day)+pd.Timedelta(hours=1),periods=24,freq='h').as_unit('ns')

def assert_asof(source,target):
    s=pd.DatetimeIndex(source)
    if s.tz is None or (s>issuance(target)).any(): raise ValueError('Feature source after issuance')

def contract():
    c=json.loads((ROOT/'configs/deep_learning/feature_contract.json').read_text())
    assert_allowed(c['weather_features']+TIME+STATIC)
    return c

def assert_allowed(names):
    if FORBIDDEN.intersection(names): raise ValueError(f'Forbidden model features: {FORBIDDEN.intersection(names)}')
    if len(set(names))!=len(names): raise ValueError('Duplicate model features')

def solar_elevation(times,lat,lon):
    # Same analytic midpoint approximation as expanding_pipeline.solar; no CatBoost dependency.
    t=local_hours(times)-pd.Timedelta(minutes=30)
    gamma=2*np.pi*(t.dayofyear.to_numpy()-1)/365.25
    dec=np.deg2rad(23.44)*np.sin(gamma-1.39)
    eq=229.18*(.000075+.001868*np.cos(gamma)-.032077*np.sin(gamma)-.014615*np.cos(2*gamma)-.040849*np.sin(2*gamma))
    hour=t.hour.to_numpy()+t.minute.to_numpy()/60+(np.asarray(lon)-135)/15+eq/60
    angle=np.deg2rad(15*(hour-12)); p=np.deg2rad(np.asarray(lat))
    return np.rad2deg(np.arcsin(np.clip(np.sin(p)*np.sin(dec)+np.cos(p)*np.cos(dec)*np.cos(angle),-1,1))).astype('float32')

def temporal_features(times,sites):
    t=local_hours(times); day=operating_day(t); s=len(sites)
    hour=t.hour.to_numpy(); doy=day.dayofyear.to_numpy()
    a=np.stack([hour,day.month,doy,np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*(doy-1)/365.25),np.cos(2*np.pi*(doy-1)/365.25)],axis=-1)
    a=np.broadcast_to(a[:,None,:],(len(t),s,7)).copy()
    elevation=solar_elevation(t.repeat(s),np.tile(sites.lat,len(t)),np.tile(sites.lon,len(t))).reshape(len(t),s)
    return np.concatenate([a,elevation[...,None]],axis=-1).astype('float32')

def static_features(sites,categories):
    values=sites[STATIC].to_numpy(dtype='float32')
    flags=~np.isfinite(values)
    onehot=[]
    for p in ['wa','wb']:
        ids=sites[p+'_station'].map({v:i+1 for i,v in enumerate(categories[p])}).fillna(0).to_numpy(dtype=int)
        onehot.append(np.eye(len(categories[p])+1,dtype='float32')[ids])
    valid=np.isfinite(values[:,:3]).all(axis=1)
    return values,flags,np.concatenate(onehot,axis=1),valid

class Moments:
    def __init__(self,n): self.total=np.zeros(n); self.squared=np.zeros(n); self.count=np.zeros(n)
    def add(self,x):
        a=np.asarray(x,dtype='float64').reshape(-1,len(self.total)); valid=np.isfinite(a)
        self.total+=np.where(valid,a,0).sum(0); self.squared+=np.where(valid,a*a,0).sum(0); self.count+=valid.sum(0)
    def finish(self):
        mean=self.total/np.maximum(self.count,1); variance=np.maximum(self.squared/np.maximum(self.count,1)-mean*mean,0)
        std=np.sqrt(variance); std[std<1e-6]=1
        return {'mean':mean.tolist(),'std':std.tolist(),'count':self.count.tolist()}

def normalize(x,stats):
    x=np.asarray(x,dtype='float32')
    if np.isinf(x).any(): raise ValueError('Infinite input feature')
    missing=~np.isfinite(x)
    value=(x-np.asarray(stats['mean'],dtype='float32'))/np.asarray(stats['std'],dtype='float32')
    return np.where(missing,0,value).astype('float32'),missing

def split_plan(config,cache,phase):
    split=config['split']; start,end=FOLDS[split['outer_fold']]
    outer_issue=aware(start)-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    dev_start,dev_end=aware(split['dev_start']),aware(split['dev_end'])
    if dev_end<=dev_start or dev_end>outer_issue: raise ValueError('Development targets cross outer issuance cutoff')
    cutoff=dev_start-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    if phase=='refit': cutoff=outer_issue
    if phase=='production': cutoff=aware(cache.manifest['test_start'])-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    return {'cutoff':str(cutoff),'dev_start':str(dev_start),'dev_end':str(dev_end),'holdout_start':str(aware(start)),'holdout_end':str(aware(end)),'outer_issue':str(outer_issue),'phase':phase}

def fit_normalizer(cache,cutoff):
    cutoff=aware(cutoff); w=Moments(cache.weather.shape[-1]); t=Moments(len(TIME)); counts=np.zeros(cache.num_sites,dtype='int64')
    first=cache.manifest['start']
    for d,day in enumerate(cache.days):
        hours=day_hours(day)
        if hours.min()>cutoff: break
        target=np.asarray(cache.target[d]); eligible=np.isfinite(target)&(target>=0)&(target<=1)&(hours.asi8[None,:]<=cutoff.value)
        selected=eligible.any(axis=1)
        if selected.any(): w.add(cache.weather[d,selected]); t.add(cache.temporal[d,selected])
        counts+=eligible.sum(axis=1)
    if not counts.sum(): raise ValueError('No valid historical training targets before cutoff')
    static=Moments(len(STATIC)); static.add(cache.static[counts>0])
    return {'weather':w.finish(),'temporal':t.finish(),'static':static.finish(),'seen_hours':counts.tolist(),'fit_cutoff':str(cutoff),'cache_signature':cache.manifest['signature']}
