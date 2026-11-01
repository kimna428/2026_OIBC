"""Freeze nonnegative blend weights using development labels only; apply to refit models."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
from .config import path,fingerprint
from .checkpoint import load_checkpoint
from .dataset import DayCache,DayDataset
from .features import aware,day_hours,issuance
from .runtime import device_for,GPULock
from .inference import predict_grid,model_from_checkpoint
from .predict import export_predictions

def simplex_weights(n,divisions):
    if n==1:
        yield [1.]; return
    for cuts in itertools.combinations(range(divisions+n-1),n-1):
        edges=(-1,)+cuts+(divisions+n-1,)
        yield [(edges[i+1]-edges[i]-1)/divisions for i in range(n)]

def objective(grids,cache,days,weights):
    penalty=0.; hours=0
    membership=cache.composition.set_index('site_id').group.reindex(cache.sites.site_id).to_numpy()
    cap=cache.sites.capacity_kw.to_numpy()
    for i,d in enumerate(days):
        truth=np.asarray(cache.energy[d]); complete=np.isfinite(truth).all(0)
        if not complete.any(): continue
        mixed=sum(w*np.asarray(g[i]) for w,g in zip(weights,grids))*cap[:,None]
        price=np.asarray(cache.smp[d])*np.asarray(cache.k[d]); hours+=int(complete.sum())
        for label in ['G1','G2']:
            mask=membership==label
            # Select complete target columns BEFORE summing; never impute missing targets.
            error=mixed[mask][:,complete].sum(0)-truth[mask][:,complete].sum(0)
            penalty+=float(.1*np.sum(price[complete]*np.abs(error)))
    if not hours: raise ValueError('No complete development group truth for ensemble weight selection')
    return penalty/hours

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='action',required=True)
    for action in ['fit','predict','evaluate']:
        q=sub.add_parser(action); q.add_argument('--checkpoints',nargs='+',required=True); q.add_argument('--cache-root'); q.add_argument('--output',required=True); q.add_argument('--device',default='cuda'); q.add_argument('--smoke-test',action='store_true')
        if action=='fit': q.add_argument('--step',type=float,default=.1)
        else:
            q.add_argument('--weights',required=True); q.add_argument('--split',choices=['dev','holdout'] if action=='evaluate' else ['dev','holdout','test'],default='holdout' if action=='evaluate' else 'test')
    args=p.parse_args(); checkpoints=[load_checkpoint(path(f)) for f in args.checkpoints]; device=device_for(args.device)
    if any(c['smoke_test'] for c in checkpoints) and not args.smoke_test: raise ValueError('Smoke ensemble requires explicit --smoke-test')
    root=path(args.cache_root or checkpoints[0]['config']['data']['cache_root'])
    split='dev' if args.action=='fit' else args.split
    cache=DayCache(root/('test' if split=='test' else 'train'))
    output=path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    stored=json.loads(path(args.weights).read_text()) if args.action!='fit' else None
    if stored and (not np.isfinite(stored['weights']).all() or min(stored['weights'])<0 or not np.isclose(sum(stored['weights']),1)): raise ValueError('Invalid frozen ensemble weights')
    if stored and (stored['feature_signature']!=cache.manifest['feature_signature'] or len(stored['weights'])!=len(checkpoints)): raise ValueError('Ensemble schema/component mismatch')
    grids=[]; days=None
    with GPULock(device):
        for i,c in enumerate(checkpoints):
            if args.action=='fit' and c['plan']['phase'] not in ['development','finetune']: raise ValueError('Ensemble weights must be selected on development checkpoints, never refit/holdout/test')
            if stored:
                if c['selected_dev_signature']!=stored['components'][i]['selected_dev_signature']: raise ValueError('Refit checkpoint does not descend from selected development component')
                if aware(stored['selection_end'])>aware(c['plan']['outer_issue']): raise ValueError('Ensemble weights use future labels')
            dataset=DayDataset(cache,c['normalizer'],c['plan'],split,c['config']['model']['name']=='cross_site')
            if aware(c['plan']['cutoff'])>issuance(day_hours(cache.days[dataset.day_indices[0]])).min(): raise ValueError('Checkpoint exceeds prediction issuance')
            if days is not None and not np.array_equal(days,dataset.day_indices): raise ValueError('Ensemble components have different validation windows')
            days=dataset.day_indices; model=model_from_checkpoint(c,cache,device)
            grids.append(predict_grid(model,dataset,c['config']['training'],device,output.parent/(output.stem+f'.component{i}.npy'),c['config']['training']['precision'],c['config']['training'].get('postprocess','none')))
            del model
            if device.type=='cuda': torch.cuda.empty_cache()
        if args.action=='fit':
            if not np.isfinite(args.step) or args.step<=0: raise ValueError('Ensemble step must be positive')
            divisions=round(1/args.step)
            if not 1<=divisions<=100 or abs(divisions*args.step-1)>1e-8 or len(checkpoints)>4: raise ValueError('Use 1..4 components and a reciprocal-integer grid step between .01 and 1')
            candidates=[(objective(grids,cache,days,w),w) for w in simplex_weights(len(grids),divisions)]
            best,weights=min(candidates,key=lambda t:t[0])
            end=max(aware(c['selection_end']) for c in checkpoints)
            if any(end>aware(c['plan']['outer_issue']) for c in checkpoints): raise ValueError('Blend development labels cross holdout issuance')
            result={'weights':weights,'criterion':'development official penalty per hour','development_penalty_per_hour':best,'selection_end':str(end),'holdout_used':False,'feature_signature':cache.manifest['feature_signature'],'components':[{'selected_dev_signature':c['selected_dev_signature'],'architecture':c['config']['model'],'seed':c['config']['training']['seed']} for c in checkpoints],'smoke_test':args.smoke_test}
            if output.exists() and json.loads(output.read_text())!=result: raise FileExistsError('Frozen ensemble weights differ; use a new output path')
            output.write_text(json.dumps(result,indent=2)); print('Development weights frozen:',weights)
        else:
            if stored['smoke_test'] and not args.smoke_test: raise ValueError('Synthetic weights cannot be used for real predictions')
            if split=='test' and cache.manifest['kind']=='provided' and len(days)*24!=8760: raise ValueError('Incomplete official test grid')
            from .storage import result_guard
            result_guard(output,int(np.prod(grids[0].shape))*4+4096)
            grid=np.lib.format.open_memmap(output.with_suffix('.npy'),mode='w+',dtype='float32',shape=grids[0].shape)
            for i in range(len(days)): grid[i]=sum(w*np.asarray(g[i]) for w,g in zip(stored['weights'],grids))
            grid.flush()
            if args.action=='evaluate':
                if output.suffix!='.json': raise ValueError('Ensemble evaluation --output must end in .json')
                from .metrics import Evaluator
                evaluator=Evaluator(cache,checkpoints[0]['normalizer']['seen_hours'])
                for i,d in enumerate(days): evaluator.add_day(int(d),grid[i])
                metrics,detail,groups,ledger=evaluator.finish()
                metrics.update({'split':split,'weights':stored['weights'],'synthetic':cache.manifest['kind']=='synthetic_smoke','not_model_performance':args.smoke_test,'selection_end':stored['selection_end'],'holdout_used_for_weight_selection':False})
                output.write_text(json.dumps(metrics,indent=2,allow_nan=False)); detail.to_csv(output.with_suffix('.site_metrics.csv'),index=False); groups.to_csv(output.with_suffix('.group_metrics.csv'),index=False)
                if len(ledger): ledger.to_parquet(output.with_suffix('.ledger.parquet'),index=False)
                print('Frozen ensemble evaluation:',output)
            else: print(export_predictions(grid,cache,days,output))
if __name__=='__main__': main()
