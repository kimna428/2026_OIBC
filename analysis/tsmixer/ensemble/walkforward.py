"""One explicit model/fold per server invocation; completed jobs are immutable."""
import argparse
import gc
import json
import platform
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.features import fit_normalizer
from analysis.tsmixer.config import resolve
from analysis.tsmixer.data import open_cache,DayDataset
from analysis.tsmixer.year_backtest import PriceFreeCache,predict as tsmixer_predict
from analysis.tsmixer.runtime import GPULock,environment
from analysis.tsmixer.comparison.io import file_sha
from . import config as configuration
from .folds import FOLDS,validate_folds
from .data import History,actual_grid
from .features import Features,outages,sj_zero_spells,observed_targets
from .datasets import NeuralDays
from .trainer import fit_component,neural_predict
from .teacher import build_teacher
from .trees import fit_trees,tree_predict
from .predictions import save_prediction,load_prediction
from .persistence import reserve,atomic_json,complete,is_complete,sources
from .reuse import legacy_assessment,reuse_refit

MODELS=('tsmixer','sj_ensemble','state01')
_active_job=None


def recipe(c,cache,fold,family):
    root=resolve(c['data_root'])
    raw={str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.csv','.parquet')}
    return dict(config=c,model_config=configuration.model_config(c,family),sj_feature_spec_sha256=file_sha(resolve(c['sj_features'])),fold=fold.record(),model=family,cache_signature=cache.manifest['signature'],
                sources=sources(),raw_input_sha256=raw,libraries=dict(torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,python=platform.python_version()))


def train(c,fold,family,resume=False,smoke=False,updates=5):
    global _active_job
    _active_job=None
    if platform.system()!='Linux' or not torch.cuda.is_available():raise ValueError('Real-data training/smoke requires Linux CUDA; Mac uses fixture smoke only')
    if family!='tsmixer':
        import importlib.util
        missing=[name for name in ('lightgbm','xgboost') if importlib.util.find_spec(name) is None]
        if missing:raise RuntimeError(f'Missing server dependencies before any training: {missing}')
    if updates not in range(1,11):raise ValueError('Smoke updates1..10 only')
    job_started=time.perf_counter()
    cache=open_cache(resolve(c['cache_root'])/'train')
    if cache.manifest['kind']!='provided':raise ValueError('Full training cannot use synthetic cache')
    from analysis.deep_learning.features import operating_day
    prices=pd.read_csv(resolve(c['data_root'])/'train/smp.csv')
    if not cache.days.equals(operating_day(pd.to_datetime(prices.ts)).unique()):raise ValueError('Require complete original historical cache axis')
    audit=cache.root/'source_audit.jsonl'
    if not audit.is_file():raise ValueError('Require forecast issuance audit')
    rows=[json.loads(line) for line in audit.read_text().splitlines()]
    if sum(r['days'] for r in rows)!=len(cache.days) or any(r['asof_violations']!=0 or r['rounds']!=[10,17] for r in rows):raise ValueError('Cache forecast issuance audit failed')
    if not fold.times.isin(pd.DatetimeIndex([h for d in cache.days for h in __import__('analysis.deep_learning.features',fromlist=['day_hours']).day_hours(d)])).all():raise ValueError('Incomplete fold cache axis')
    torch.set_num_threads(c['cpu_threads']);device=torch.device('cuda')
    cfg=configuration.model_config(c,family)
    path=resolve(c['run_root'])/('smoke' if smoke else 'models')/fold.name/family
    with GPULock(device):
        plan_recipe=recipe(c,cache,fold,family)
        for name,sha in cache.manifest.get('input_hashes',{}).items():
            if plan_recipe['raw_input_sha256'].get(name)!=sha:raise ValueError(f'Cache/raw source hash differs: {name}')
        path=reserve(path,plan_recipe,resume)
        if is_complete(path):print('Verified complete; no training:',path);return
        _active_job=path
        atomic_json(path/'status.json',dict(state='running',phase='preparing',pid=os.getpid()))
        features=Features(cache,json.loads(resolve(c['sj_features']).read_text()))
        selected={};models={};stats_by={};records={};outage=None;obs=None
        components=['tsmixer'] if family=='tsmixer' else (['sj_bigru'] if family=='sj_ensemble' else ['cloud','all'])
        for stage in ('selection','refit'):
            if smoke and stage=='refit':break
            atomic_json(path/'status.json',dict(state='running',phase=stage,pid=os.getpid()))
            cutoff=fold.inner_cutoff if stage=='selection' else fold.cutoff
            history=History(cache,cutoff,c['train_start']);dev_history=History(cache,fold.cutoff,c['train_start']) if stage=='selection' else None
            plan=fold.plan(stage);directory=path/stage;directory.mkdir(exist_ok=True)
            if family!='tsmixer':
                outage=sj_zero_spells(history) if family=='sj_ensemble' else outages(history)
                if family=='state01':obs,bounds=observed_targets(resolve(c['data_root']),history,features)
                outage_file=directory/'outage_days.parquet'
                if outage_file.exists():
                    prior=pd.read_parquet(outage_file).sort_values(['site_id','date']).reset_index(drop=True)
                    current=outage.sort_values(['site_id','date']).reset_index(drop=True)
                    if not np.array_equal(prior.site_id.to_numpy(),current.site_id.to_numpy()) or not pd.DatetimeIndex(prior.date).as_unit('ns').equals(pd.DatetimeIndex(current.date).as_unit('ns')):raise ValueError('Outage preprocessing changed during resume')
                else:outage.to_parquet(outage_file,index=False)
                records[stage]=dict(outage_sha256=file_sha(outage_file),fit_cutoff=str(cutoff),outage_rows=len(outage),aux_bounds=bounds if family=='state01' else None)
            for component in components:
                atomic_json(path/'status.json',dict(state='running',phase=stage,component=component,pid=os.getpid()))
                if stage=='refit' and component=='tsmixer' and not smoke:
                    fresh=fit_normalizer(PriceFreeCache(history),str(cutoff))
                    reused=reuse_refit(resolve(c['legacy_tsmixer']),cache,cfg,fresh,selected[component],fold,device,directory)
                    if reused:
                        models[component],stats_by[component]=reused;continue
                model,stats,epoch=fit_component(component,cfg,cache,history,dev_history,features,plan,directory/component,device,
                       outage,obs,epochs=selected.get(component) if stage=='refit' else None,smoke_updates=updates if smoke else None)
                if stage=='selection':selected[component]=epoch;del model;gc.collect();torch.cuda.empty_cache()
                else:models[component]=model;stats_by[component]=stats
            if stage=='selection':atomic_json(path/'epoch_selection.json',dict(selected=selected,target_end=str(fold.cutoff),plan=plan))
        if smoke:
            atomic_json(path/'smoke_summary.json',dict(smoke_only=True,environment=environment(device),updates=updates,total_wall_seconds=time.perf_counter()-job_started,profile='bounded_updates; not OOF'))
            print('Smoke only:',path);return
        if family!='tsmixer':
            if family=='sj_ensemble':records['teacher']=build_teacher(features,history,resolve(c['data_root']),path/'teacher')
            tree_records=fit_trees(history,features,family,outage,path/'trees',c['seed'])
            atomic_json(path/'tree_records.json',tree_records)
        atomic_json(path/'status.json',dict(state='running',phase='predicting',pid=os.getpid()))
        blind=PriceFreeCache(cache,blind=True);days=np.flatnonzero((cache.days>=fold.start.normalize())&(cache.days<fold.end.normalize()))
        parts={}
        for component in components:
            if component=='tsmixer':
                dataset=DayDataset(blind,stats_by[component],fold.plan('refit'),'dev')
                cf=tsmixer_predict(models[component],dataset,cfg['training'],device)
            else:
                dataset=NeuralDays(blind,features,stats_by[component],family,days)
                cf=neural_predict(models[component],dataset,component,cfg['training'],device)
            parts[component]=cf*cache.sites.capacity_kw.to_numpy()[None,:,None]
        if family=='tsmixer':value=parts['tsmixer']
        else:
            parts.update(tree_predict(tree_records,features,days))
            value=.25*parts['lightgbm']+.25*parts['xgboost']+.5*parts['sj_bigru'] if family=='sj_ensemble' else .1*parts['lightgbm']+.1*parts['xgboost']+.4*parts['cloud']+.4*parts['all']
        meta=dict(model=family,fold=fold.name,start=str(fold.start),end_exclusive=str(fold.end),
                  training_grid_start=str(history.days[0]),training_cutoff=str(fold.cutoff),normalization_cutoff=str(fold.cutoff),auxiliary_cutoff=str(fold.cutoff) if family!='tsmixer' else None,
                  outage_cutoff=str(fold.cutoff) if family!='tsmixer' else None,calibration_cutoff=None,selection_cutoff=str(fold.cutoff),
                  inference_inputs=['forecasts','solar_geometry','calendar','site_metadata'],protocol='past_only_walkforward_v1',smoke_only=False,
                  data_hashes={n:plan_recipe['raw_input_sha256'][n] for n in ('sites.csv','train/generation.parquet','train/smp.csv')},seed=c['seed'],epoch_selection=selected,calibration='none; raw blend',calibration_factor=1.,feature_provenance=dict(sj_spec_sha256=plan_recipe['sj_feature_spec_sha256'],implementation='reviewed repository variant; lazy float32 inputs'),preprocessing=records,
                  environment=environment(device),recipe_sha256=file_sha(path/'recipe.json'),
                  checkpoints={str(p.relative_to(path)):file_sha(p) for p in path.rglob('*') if p.is_file() and (p.suffix in ('.pt','.txt','.ubj') or p.name in ('recipe.json','epoch_selection.json','reused_refit.json','tree_records.json','normalizer.json'))})
        # A prediction write interrupted before its hash marker is never promoted.
        for name,array in {family:value,**{family+'_'+k:v for k,v in parts.items()}}.items():
            file=path/('predictions.parquet' if name==family else name+'.parquet')
            save_prediction(file,array.transpose(0,2,1).reshape(-1,cache.num_sites),fold.times,cache.sites,meta,resume=True)
        atomic_json(path/'status.json',dict(state='complete',model=family,fold=fold.name))
        files=[str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='complete.json']
        complete(path,files)
        print('Completed:',path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','smoke','audit'])
    p.add_argument('--config',default='configs/tsmixer/ensemble_walkforward.yaml');p.add_argument('--fold',choices=list(FOLDS));p.add_argument('--model',choices=MODELS)
    p.add_argument('--resume',action='store_true');p.add_argument('--full-train',action='store_true');p.add_argument('--fixture',action='store_true');p.add_argument('--updates',type=int,default=5);p.add_argument('--output')
    a=p.parse_args();c=configuration.load(a.config);validate_folds()
    if a.command=='audit':
        from .evaluation import audit
        audit(c,a.output);return
    if a.command=='smoke' and a.fixture:
        from .smoke import run
        run(c,a.output,a.updates);return
    if a.output:p.error('For training use OIBC_ENSEMBLE_ROOT/config run_root; --output is audit/fixture-only')
    if not a.fold or not a.model:p.error('Choose exactly one --fold and --model')
    if a.command=='train' and not a.full_train:p.error('Full training requires explicit --full-train')
    try:
        train(c,FOLDS[a.fold],a.model,a.resume,a.command=='smoke',a.updates)
    except BaseException as error:
        path=resolve(c['run_root'])/('smoke' if a.command=='smoke' else 'models')/a.fold/a.model
        if _active_job==path and (path/'recipe.json').is_file() and not (path/'complete.json').is_file():
            atomic_json(path/'status.json',dict(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',reason=str(error),pid=os.getpid()))
        raise

if __name__=='__main__':main()
