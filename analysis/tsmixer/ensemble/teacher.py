"""SJ XGB auxiliary cloud predictor: historical held-forward, outer model fixed."""
from pathlib import Path
import json
import time
from .persistence import atomic_json
from analysis.tsmixer.comparison.io import file_sha
import numpy as np
import pandas as pd
from analysis.deep_learning.features import aware

MEAN_VARS=['wb_cloud','wb_uv_idx','wb_temp','wb_rel_hum','wb_dew_point','wb_precip_prob','wb_ceiling','wb_wind_speed']
A_VARS=['wa_cloud','wa_temp','wa_humidity','wa_rain','wa_wind_speed']


def daily_inputs(features):
    features.build_sj_weather();cache=features.cache;pieces=[]
    extra=['wb_temp__source','wb_temp__missing17','wb_temp__age_days','wb_precip_prob__source']
    pairs=cache.sites[['wa_station','wb_station']].drop_duplicates()
    for station,b in features.fused['wb'].items():
        sources=pairs.loc[pairs.wb_station==station,'wa_station'].dropna().unique()
        linked={}
        for name in A_VARS:
            stack=np.stack([features.fused['wa'][s][name].to_numpy().reshape(-1,24) for s in sources])
            linked['linked_'+name]=stack.mean(0)
            if name=='wa_cloud':linked['linked_wa_cloud_range']=stack.max(0)-stack.min(0)
        arrays={n:b[n].to_numpy().reshape(-1,24) for n in MEAN_VARS+extra};arrays.update(linked)
        values={}
        for scope,indices in [('all24',slice(None)),('peak5',slice(10,15))]:
            for name,a in arrays.items():
                z=a[:,indices]
                valid=np.isfinite(z);count=valid.sum(1);total=np.where(valid,z,0).sum(1,dtype='float64')
                mean=np.divide(total,count,out=np.full(len(z),np.nan),where=count>0)
                squared=np.where(valid,(z-mean[:,None])**2,0).sum(1)
                std=np.sqrt(np.divide(squared,count-1,out=np.full(len(z),np.nan),where=count>1))
                minimum=np.where(count>0,np.where(valid,z,np.inf).min(1),np.nan)
                maximum=np.where(count>0,np.where(valid,z,-np.inf).max(1),np.nan)
                for suffix,value in [('mean',mean),('std',std),('min',minimum),('max',maximum)]:
                    values[f'{name}__{scope}_{suffix}']=value
        for hour in range(11,16):values[f'b_peak_hour{hour}_cloud']=arrays['wb_cloud'][:,hour-1]
        days=cache.days
        d=pd.DataFrame(dict(wb_station=station,day=days,**values))
        d['doy_sin']=np.sin(days.dayofyear*2*np.pi/365.25);d['doy_cos']=np.cos(days.dayofyear*2*np.pi/365.25)
        d['year']=days.year;d['month']=days.month
        d['station_code']=sorted(features.fused['wb']).index(station)
        pieces.append(d)
    return pd.concat(pieces,ignore_index=True)


def build_teacher(features,history,data_root,output):
    import lightgbm as lgb
    output=Path(output);output.mkdir(exist_ok=True)
    daily=daily_inputs(features)
    obs=pd.read_parquet(Path(data_root)/'train/observed_wb.parquet',columns=['wb_station','ts','wb_cloud'],
                      filters=[('ts','<=',history.cutoff.to_pydatetime())])
    obs['ts']=pd.to_datetime(obs.ts).dt.tz_convert('Asia/Seoul')
    conflicts=obs.groupby(['wb_station','ts']).wb_cloud.nunique(dropna=False)
    badkeys=conflicts.index[conflicts>1]
    obs.loc[~obs.wb_cloud.between(0,100),'wb_cloud']=np.nan
    obs=obs.drop_duplicates(['wb_station','ts'])
    obs.loc[pd.MultiIndex.from_frame(obs[['wb_station','ts']]).isin(badkeys),'wb_cloud']=np.nan
    obs['day']=(obs.ts-pd.Timedelta(hours=1)).dt.normalize()
    obs=obs.loc[obs.ts.dt.hour.between(11,15)]
    target=obs.groupby(['wb_station','day']).wb_cloud.agg(['mean','count'])
    target.loc[target['count']!=5,'mean']=np.nan
    ix=pd.MultiIndex.from_frame(daily[['wb_station','day']]);y=target['mean'].reindex(ix).to_numpy()
    names=features.spec['weather_columns'];x=daily[names].astype('float32')
    predictions=np.full(len(daily),np.nan,dtype='float32');records=[]
    params=features.spec['weather_params'];rounds=features.spec['weather_trees']
    def model_at(cutoff,name):
        fit=np.isfinite(y)&((daily.day+pd.Timedelta(hours=15))<=cutoff).to_numpy()
        if fit.sum()<100:return None
        marker=output/f'{name}.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if record['cutoff']!=str(cutoff) or file_sha(output/f'{name}.txt')!=record['sha256']:raise ValueError('Teacher provenance differs')
            records.append(record);return lgb.Booster(model_file=str(output/f'{name}.txt'))
        model=lgb.train(params,lgb.Dataset(x.loc[fit],label=y[fit],categorical_feature=['station_code']),num_boost_round=rounds)
        file=output/f'{name}.txt'
        if file.exists():file.rename(file.with_name(file.name+f'.uncommitted_{time.time_ns()}'))
        model.save_model(str(file))
        record=dict(name=name,cutoff=str(cutoff),rows=int(fit.sum()),last_label=str((daily.loc[fit,'day']+pd.Timedelta(hours=15)).max()),sha256=file_sha(file))
        atomic_json(marker,record);records.append(record)
        return model
    # Never advance a teacher with observations from the outer evaluation year.
    for start in pd.date_range('2021-03-01',history.cutoff.normalize(),freq='3MS',tz='Asia/Seoul'):
        end=start+pd.DateOffset(months=3);cutoff=start-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
        model=model_at(cutoff,f'held_{start:%Y%m}')
        selected=daily.day.between(start,end,inclusive='left').to_numpy()&(daily.day<=history.cutoff.normalize()).to_numpy()
        if model is not None:predictions[selected]=np.clip(model.predict(x.loc[selected]),0,100)
    outer=model_at(history.cutoff,'fixed_outer')
    future=(daily.day>history.cutoff.normalize()).to_numpy()
    if outer is not None:predictions[future]=np.clip(outer.predict(x.loc[future]),0,100)
    indexed=pd.Series(predictions,index=ix)
    values=[]
    for day in features.cache.days:
        keys=pd.MultiIndex.from_arrays([features.cache.sites.wb_station,np.repeat(day,features.cache.num_sites)])
        values.append(indexed.reindex(keys).to_numpy())
    features.teacher=np.asarray(values,dtype='float32');features._day=None
    return dict(policy='historical_quarter_OOF_and_cutoff_fixed_outer',cutoff=str(history.cutoff),records=records)
