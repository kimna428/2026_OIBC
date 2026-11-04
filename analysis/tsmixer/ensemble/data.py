"""Cut off targets at the data view, before masks/statistics/model selection."""
import numpy as np
import pandas as pd
from analysis.deep_learning.features import aware,day_hours
from analysis.tsmixer.data import DayArrayView
from analysis.tsmixer.comparison.core import Grid
from analysis.validation import local_hours,hourly_weights
from analysis.tsmixer.comparison.io import file_sha

class BoundedArray:
    def __init__(self,view,name): self.view,self.name=view,name;self.shape=(len(view.days),)+getattr(view.base,name).shape[1:]
    def __getitem__(self,key):
        keys=key if isinstance(key,tuple) else (key,);d=int(keys[0]);source=int(self.view.indices[d])
        a=np.array(getattr(self.view.base,self.name)[source],copy=True)
        if self.name in ('target','energy'):
            a[...,day_hours(self.view.days[d])>self.view.cutoff]=np.nan
        elif self.name in ('smp','k'): a[day_hours(self.view.days[d])>self.view.cutoff]=0
        return a[keys[1:]] if len(keys)>1 else a

class History:
    def __init__(self,base,cutoff,start='2021-01-01'):
        self.base,self.cutoff=base,aware(cutoff)
        self.indices=np.flatnonzero((base.days>=aware(start))&(base.days+pd.Timedelta(hours=1)<=self.cutoff))
        if not len(self.indices): raise ValueError('No historical data before cutoff')
        self.days=base.days[self.indices]
        self.sites,self.composition=base.sites,base.composition
        self.num_sites,self.static,self.manifest=base.num_sites,base.static,base.manifest
        self.root=base.root
        for name in ('target','energy','smp','k'): setattr(self,name,BoundedArray(self,name))
        for name in ('weather','temporal','missing_kind'): setattr(self,name,DayArrayView(getattr(base,name),self.indices))


def actual_grid(data_root,fold):
    from pathlib import Path
    root=Path(data_root)
    sites=pd.read_csv(root/'sites.csv').sort_values('site_id').reset_index(drop=True)
    generation=pd.read_parquet(root/'train/generation.parquet')
    generation['ts']=local_hours(pd.to_datetime(generation.ts))
    if generation.duplicated(['site_id','ts']).any() or not generation.site_id.isin(sites.site_id).all(): raise ValueError('Actual keys invalid')
    selected=generation.loc[(generation.ts>=fold.start)&(generation.ts<fold.end)]
    actual=selected.pivot(index='ts',columns='site_id',values='energy').reindex(index=fold.times,columns=sites.site_id).to_numpy()
    prices=pd.read_csv(root/'train/smp.csv');prices['ts']=local_hours(pd.to_datetime(prices.ts))
    if prices.ts.duplicated().any(): raise ValueError('SMP duplicate keys')
    prices=prices.set_index('ts').reindex(fold.times)
    grid=Grid(sites,fold.times,actual,prices.smp.to_numpy(),prices.k.to_numpy() if 'k' in prices else hourly_weights(fold.times))
    grid.source_hashes={name:file_sha(root/name) for name in ('sites.csv','train/generation.parquet','train/smp.csv')}
    h=generation.loc[generation.ts<=fold.cutoff]
    cf=h.energy/h.site_id.map(sites.set_index('site_id').capacity_kw)
    seen=h.loc[np.isfinite(cf)&cf.between(0,1)].groupby('site_id').size().reindex(sites.site_id,fill_value=0).to_numpy()
    return grid,seen,dict(first_target=str(generation.ts.min()),first_smp=str(local_hours(pd.to_datetime(pd.read_csv(root/'train/smp.csv').ts)).min()),
                          train_rows=len(h),valid_train_rows=int((np.isfinite(cf)&cf.between(0,1)).sum()),validation_rows=int(grid.observed.sum()),hours=len(fold.times))
