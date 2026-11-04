"""Audit, import, score past OOF, freeze F1/F2 weights, then evaluate a later fold."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.deep_learning.storage import result_guard
from analysis.tsmixer.config import resolve
from analysis.tsmixer.comparison.io import file_sha
from analysis.tsmixer.comparison.diagnostics import residual_analysis,correlation
from analysis.tsmixer.comparison.three_diagnostics import complementarity
from analysis.tsmixer.comparison.core import evaluate,Grid,verify_reference
from .config import load
from .folds import FOLDS
from .data import actual_grid
from .predictions import load_prediction,save_prediction
from .evaluation import groups,evaluate_model,discover,site_metrics,component_predictions,reference_compatible
from .blend import candidates,mix,ablations,MODELS
from .weight_selection import select,freeze,read_frozen
from .persistence import atomic_json,reserve,complete,is_complete,sources


def prepare_output(path):
    path=resolve(path);result_guard(path,128*1024**2)
    path.mkdir(parents=True,exist_ok=False);return path


def score_fold(c,fold,output,include_components=False):
    output=prepare_output(output)
    grid,seen,_=actual_grid(resolve(c['data_root']),fold);predictions,availability,hashes=discover(c,fold,grid)
    if include_components:predictions.update(component_predictions(availability,grid,fold))
    assignments,provenance=groups(c,grid.sites,fold);trials=[];individual={}
    for model,value in predictions.items():individual[model]=evaluate_model(grid,value,assignments,fold,seen,output,model,reference_compatible(c,grid,fold))
    target_end=FOLDS['F2'].cutoff if fold.name=='F1' else FOLDS['F3'].cutoff
    eligible=grid.times<=target_end
    selection_grid=Grid(grid.sites,grid.times[eligible],grid.actual[eligible],grid.smp[eligible],grid.k[eligible]) if fold.name!='F3' else None
    for weights in ([] if fold.name=='F3' else candidates(c['grid_step'])):
        try:prediction=mix(predictions,weights)
        except ValueError:continue
        accuracy=site_metrics(selection_grid,prediction[eligible],seen)
        settlement,_=evaluate(selection_grid,prediction[eligible],assignments['metadata_primary'])
        trials.append(dict(weights=weights,target_end=str(target_end),**accuracy['all'],day_site_nmae_pct=accuracy['day']['site_nmae_pct'],
                    W=settlement['W'],valid_hours=settlement['valid_hours'],S=settlement['S']))
    if predictions:
        diagnostic=residual_analysis(grid,predictions,seen,output)
        ledger={name:evaluate(grid,p,assignments['metadata_primary'])[1] for name,p in predictions.items()}
        diagnostic['complementarity']=complementarity(grid,predictions,{'metadata_primary':ledger},output)
        day_mask=__import__('analysis.deep_learning.features',fromlist=['interval_max']).interval_max(grid.times,grid.sites).T>0
        pairs=[]
        import itertools
        for a,b in itertools.combinations(predictions,2):
            mask=grid.observed&day_mask
            ra=(predictions[a]-grid.actual)/grid.sites.capacity_kw.to_numpy();rb=(predictions[b]-grid.actual)/grid.sites.capacity_kw.to_numpy()
            pairs.append(dict(model_a=a,model_b=b,day_cf_residual_correlation=correlation(ra[mask],rb[mask]),rows=int(mask.sum())))
        atomic_json(output/'diagnostics.json',dict(**diagnostic,day_correlations=pairs))
    record=dict(fold=fold.name,fold_record=fold.record(),availability=availability,input_hashes=hashes,groups=provenance,
                individual=individual,trials=trials,selection_target_end=str(target_end) if fold.name!='F3' else None,selection_hours=int(eligible.sum()) if fold.name!='F3' else 0,use='F1/F2 candidate selection only; F3 individual scores and frozen applications only',source_hashes=sources())
    atomic_json(output/'metrics.json',record)
    pd.DataFrame([{**{n:weights for n,weights in r['weights'].items()},**{k:v for k,v in r.items() if k!='weights'}} for r in trials]).to_csv(output/'grid.csv',index=False)
    return record


def select_weights(c,input_paths,output,strategy):
    tables={};hashes={}
    for path in input_paths:
        path=resolve(path);r=json.loads(path.read_text());name=r['fold']
        if name=='F3' or name in tables:raise ValueError('F3 or duplicate fold cannot select weights')
        if name not in FOLDS or r.get('fold_record')!=FOLDS[name].record():raise ValueError('Selection table fold boundaries differ')
        next_fold=FOLDS['F2'] if name=='F1' else FOLDS['F3']
        if r.get('selection_target_end')!=str(next_fold.cutoff):raise ValueError('Selection data must end at next fold17:00 cutoff')
        tables[name]=r['trials'];hashes[name]=dict(metrics_path=str(path),metrics=file_sha(path),predictions=r['input_hashes'])
    value=select(tables,list(tables),strategy,(c['joint_site_weight'],c['joint_penalty_weight']))
    output=resolve(output);output.parent.mkdir(parents=True,exist_ok=True);freeze(output,value,hashes);return value


def apply_weights(c,fold,path,output):
    frozen=read_frozen(resolve(path));selected=frozen['selection']
    order={'F1':1,'F2':2,'F3':3}
    if max((order[f] for f in selected['selection_folds']),default=0)>=order[fold.name]:raise ValueError('Evaluate weights only on a strictly later fold')
    if selected.get('selection_data_end') is not None and pd.Timestamp(selected['selection_data_end'])>fold.cutoff:raise ValueError('Weight selection used labels after target fold cutoff')
    output=prepare_output(output);grid,seen,_=actual_grid(resolve(c['data_root']),fold)
    predictions,availability,hashes=discover(c,fold,grid);assignments,provenance=groups(c,grid.sites,fold)
    if any(w>0 and m not in predictions for m,w in selected['weights'].items()):
        record=dict(status='unavailable',availability=availability,weights=selected['weights'],reason='No zero fill/renormalization')
    else:
        blend=mix(predictions,selected['weights']);result=evaluate_model(grid,blend,assignments,fold,seen,output,'frozen',reference_compatible(c,grid,fold))
        removal={}
        for removed,weights in ablations(selected['weights']).items():
            try:removal[removed]=evaluate_model(grid,mix(predictions,weights),assignments,fold,seen)
            except ValueError:removal[removed]=dict(status='unavailable')
        record=dict(status='computed',strategy=selected['strategy'],selection_folds=selected['selection_folds'],fold=fold.name,weights=selected['weights'],frozen_sha256=file_sha(resolve(path)),input_hashes=hashes,
                result=result,remove_one_exploratory=removal,groups=provenance,
                independent_new_holdout=False,F3_previously_explored=fold.name=='F3')
    atomic_json(output/'frozen_evaluation.json',record);return record


def import_prediction(c,fold,model,path,metadata,resume=False):
    grid,_,_=actual_grid(resolve(c['data_root']),fold)
    value,meta=load_prediction(resolve(path),resolve(metadata),grid,fold,model)
    output=resolve(c['run_root'])/'imports'/fold.name/model
    output=reserve(output,dict(prediction=file_sha(resolve(path)),metadata=file_sha(resolve(metadata)),fold=fold.name,model=model),resume)
    if is_complete(output):return
    save_prediction(output/'predictions.parquet',value,grid.times,grid.sites,{**meta,'imported_from':str(resolve(path))},observed=grid.observed,resume=resume)
    complete(output,['recipe.json','predictions.parquet','predictions.json'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['score','select','fixed','apply','import','summary','verify-reference'])
    p.add_argument('--config',default='configs/tsmixer/ensemble_walkforward.yaml');p.add_argument('--fold',choices=list(FOLDS));p.add_argument('--output',required=True)
    p.add_argument('--inputs',nargs='+');p.add_argument('--weights');p.add_argument('--strategy',choices=['site','penalty','joint'],default='site')
    p.add_argument('--model',choices=MODELS);p.add_argument('--prediction');p.add_argument('--metadata');p.add_argument('--resume',action='store_true')
    p.add_argument('--coefficients',nargs=3,type=float,help='Fixed a b c, no outcome fitting')
    p.add_argument('--components',action='store_true',help='Separate component diagnostics; never duplicate family/component simplex axes')
    p.add_argument('--reference',help='Previously recomputed original baseline proof; never a bare scalar')
    a=p.parse_args();c=load(a.config)
    if a.reference:c['baseline_reference']=a.reference
    if not resolve(a.output).is_relative_to(resolve(c['run_root'])):p.error('--output must stay under configured run_root')
    if a.command=='verify-reference':
        if not a.prediction or not a.metadata:p.error('Provide baseline predictions and past-only provenance')
        fold=FOLDS['F3'];grid,_,_=actual_grid(resolve(c['data_root']),fold)
        value,_=load_prediction(resolve(a.prediction),resolve(a.metadata),grid,fold,'baseline_reference')
        original=grid.sites[['site_id']].assign(group=grid.sites.current_group.map({'A':'G1','B':'G2'}))
        reference=verify_reference(grid,value,original);reference['data_hashes']=grid.source_hashes
        path=resolve(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Reference proof already exists')
        atomic_json(path,reference)
    elif a.command=='score':
        if not a.fold:p.error('--fold required')
        score_fold(c,FOLDS[a.fold],a.output,a.components)
    elif a.command=='select':
        if not a.inputs:p.error('F1 or F1/F2 metrics required')
        select_weights(c,a.inputs,a.output,a.strategy)
    elif a.command=='fixed':
        if not a.coefficients:p.error('Provide --coefficients a b c; e.g. equal thirds or .1 .8 .1')
        weights=dict(zip(MODELS,a.coefficients));mix({n:np.zeros((1,1)) for n in MODELS},weights)
        selection=dict(weights=weights,strategy='fixed_'+('_'.join(map(str,a.coefficients))),selection_folds=[],F3_used=False)
        path=resolve(a.output);path.parent.mkdir(parents=True,exist_ok=True);freeze(path,selection,{})
    elif a.command=='apply':
        if not a.fold or not a.weights:p.error('--fold and --weights required')
        apply_weights(c,FOLDS[a.fold],a.weights,a.output)
    elif a.command=='import':
        if not all((a.fold,a.model,a.prediction,a.metadata)):p.error('Import needs --fold --model --prediction --metadata')
        # output denotes the new run root, not an arbitrary file outside policy.
        c['run_root']=str(resolve(a.output));import_prediction(c,FOLDS[a.fold],a.model,a.prediction,a.metadata,a.resume)
    else:
        output=prepare_output(a.output);rows=[]
        for path in a.inputs or []:
            r=json.loads(resolve(path).read_text())
            if 'individual' in r:
                for model,value in r['individual'].items():
                    for scope in ('all','day','night'):rows.append(dict(label=model,scope=scope,fold=r['fold'],**value['site'][scope]))
            elif r.get('status')=='computed':
                for scope in ('all','day','night'):rows.append(dict(label='frozen_'+r['strategy'],scope=scope,fold=r['fold'],**r['result']['site'][scope]))
        summaries=[]
        for label,scope in sorted({(r['label'],r['scope']) for r in rows}):
            group=[r for r in rows if r['label']==label and r['scope']==scope]
            if len({r['fold'] for r in group})!=len(group):raise ValueError('Duplicate strategy/model fold')
            group_valid=[r for r in group if r['site_nmae_pct'] is not None]
            summaries.append(dict(label=label,scope=scope,folds=[r['fold'] for r in group],
                 mean_site_nmae_pct=float(np.mean([r['site_nmae_pct'] for r in group_valid])) if group_valid else None,
                 worst_site_nmae_pct=max((r['site_nmae_pct'] for r in group_valid),default=None),mean_site_mae_kwh=float(np.mean([r['site_mae_kwh'] for r in group_valid])) if group_valid else None,
                 worst_site_mae_kwh=max((r['site_mae_kwh'] for r in group_valid),default=None),complete_three_folds=len(group)==3))
        atomic_json(output/'summary.json',dict(rows=rows,by_strategy=summaries,note='Never pool unrelated models or weight strategies'))


if __name__=='__main__':main()
