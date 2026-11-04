"""Portable own model files; no joblib/pickle loading of teammate artifacts."""
from pathlib import Path
import gc
import time
import json
from .persistence import atomic_json
from analysis.tsmixer.comparison.io import file_sha
import numpy as np
import pandas as pd
from analysis.deep_learning.features import day_hours
from .datasets import masks,outage_mask
from .features import FC


def state_matrix(features,d,round):
    c=features.cache;raw=np.asarray(c.weather[d]);names=features.names
    w=raw[...,[names.index(n if round==17 else n+'_r10') for n in FC]]
    g=features.state(d);times=day_hours(c.days[d]);doy=times.dayofyear.to_numpy();hour=times.hour.to_numpy()
    shape=g['day'].shape
    x=np.concatenate([w,g['elevation'][...,None],g['azimuth'][...,None],g['cosz'][...,None],
           np.broadcast_to(np.sin(2*np.pi*doy/365.25),shape)[...,None],np.broadcast_to(np.cos(2*np.pi*doy/365.25),shape)[...,None],
           np.broadcast_to(hour,shape)[...,None],np.broadcast_to(c.sites.capacity_kw.to_numpy()[:,None],shape)[...,None],
           np.broadcast_to(c.sites.lat.to_numpy()[:,None],shape)[...,None],np.broadcast_to(c.sites.lon.to_numpy()[:,None],shape)[...,None]],-1)
    return x.astype('float32'),g['day']


def training_matrix(history,features,family,name,outage,round=None):
    xs=[];ys=[];weights=[];stops=[]
    for d in range(len(history.days)):
        source=int(history.indices[d]);target=np.asarray(history.target[d]);observed=np.isfinite(target)
        if family=='sj_ensemble':
            valid,w=masks(history,features,d,family,outage);x=w['raw'] if name=='lightgbm' else w['xgb']
        else:
            x,day=state_matrix(features,source,round);valid=observed&day
        if not valid.any():continue
        xs.append(x[valid]);ys.append((np.clip(target,0,1.2) if family=='state01' else target)[valid])
        weighted=np.asarray(history.smp[d]*history.k[d])[None,:]*history.sites.capacity_kw.to_numpy()[:,None]
        weights.append(np.broadcast_to(weighted,valid.shape)[valid]);stops.append(outage_mask(history.base,source,outage)[valid])
    if not xs:raise ValueError('No supervised tree rows')
    return np.concatenate(xs),np.concatenate(ys).astype('float32'),np.concatenate(weights),np.concatenate(stops)


def fit_trees(history,features,family,outage,output,seed):
    import lightgbm as lgb
    import xgboost as xgb
    output=Path(output);output.mkdir(exist_ok=True);models=[]
    for round in ([None] if family=='sj_ensemble' else [10,17]):
        for name in ('lightgbm','xgboost'):
            tag=name if round is None else f'{name}_{round}'
            marker=output/f'{tag}.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                if file_sha(record['path'])!=record['sha256']:raise ValueError('Tree checkpoint hash differs')
                models.append(record);continue
            x,y,w,stops=training_matrix(history,features,family,name,outage,round)
            levels=None
            if family=='sj_ensemble':
                names=features.spec['columns'] if name=='lightgbm' else features.spec['xgb_columns']
                frame=pd.DataFrame(x,columns=names);cats=features.spec['categorical']
                if name=='lightgbm':
                    params={**features.spec['lgb_params'],'seed':seed}
                    model=lgb.train(params,lgb.Dataset(frame,label=y,categorical_feature=cats),num_boost_round=500)
                else:
                    levels={c:sorted(frame[c].dropna().unique().astype(int).tolist()) for c in cats}
                    for c in cats:frame[c]=pd.Categorical(frame[c],categories=levels[c])
                    params={**features.spec['xgb_params'],'random_state':seed,'device':'cpu'}
                    model=xgb.XGBRegressor(**params).fit(frame,y)
            else:
                # Historical-only outage targets; identical imputer is not trained on stopped rows.
                if stops.any():
                    imputer=lgb.LGBMRegressor(objective='regression',n_estimators=600,learning_rate=.06,num_leaves=127,min_child_samples=200,
                        subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1,n_jobs=4,random_state=seed).fit(x[~stops],y[~stops])
                    y[stops]=np.clip(imputer.predict(x[stops]),0,1.2)
                    imputer_file=output/f'{tag}_imputer.txt'
                    if imputer_file.exists():imputer_file.rename(imputer_file.with_name(imputer_file.name+f'.uncommitted_{time.time_ns()}'))
                    imputer.booster_.save_model(str(imputer_file))
                    del imputer
                w=w/max(float(w.mean()),1e-12)
                if name=='lightgbm':
                    model=lgb.LGBMRegressor(objective='l1',n_estimators=75,learning_rate=.06,num_leaves=127,min_child_samples=200,
                         subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=0,verbose=-1,n_jobs=4,random_state=seed).fit(x,y,sample_weight=w)
                    model=model.booster_
                else:
                    model=xgb.XGBRegressor(objective='reg:absoluteerror',n_estimators=100,learning_rate=.06,max_leaves=127,max_depth=0,
                       grow_policy='lossguide',tree_method='hist',min_child_weight=200,subsample=.8,colsample_bytree=.8,reg_lambda=0,
                       device='cpu',n_jobs=4,random_state=seed).fit(x,y,sample_weight=w)
            file=output/f'{tag}.txt' if name=='lightgbm' else output/f'{tag}.ubj'
            if file.exists():file.rename(file.with_name(file.name+f'.uncommitted_{time.time_ns()}'))
            model.save_model(str(file))
            record=dict(name=name,round=round,path=str(file),levels=levels,sha256=file_sha(file))
            atomic_json(marker,record);models.append(record)
            del x,y,w,stops,model;gc.collect()
    return models


def tree_predict(models,features,days):
    import lightgbm as lgb
    import xgboost as xgb
    result={};capacity=features.cache.sites.capacity_kw.to_numpy()[:,None]
    for record in models:
        if record['name']=='lightgbm':model=lgb.Booster(model_file=record['path'])
        else:model=xgb.XGBRegressor();model.load_model(record['path'])
        rows=[]
        for d in days:
            if record['round'] is None:
                w=features.sj(int(d));values=w['raw'] if record['name']=='lightgbm' else w['xgb']
                names=features.spec['columns'] if record['name']=='lightgbm' else features.spec['xgb_columns']
                x=pd.DataFrame(values.reshape(-1,values.shape[-1]),columns=names)
                if record['levels']:
                    for c,levels in record['levels'].items():x[c]=pd.Categorical(x[c],categories=levels)
                day=w['day']
            else:
                values,day=state_matrix(features,int(d),record['round']);x=values.reshape(-1,values.shape[-1])
            cf=np.asarray(model.predict(x)).reshape(features.cache.num_sites,24)
            rows.append(np.where(day,np.maximum(cf,0),0)*capacity)
        result[(record['name'],record['round'])]=np.asarray(rows)
        del model
    return {name:np.mean([v for (n,r),v in result.items() if n==name],axis=0) for name in ('lightgbm','xgboost')}
