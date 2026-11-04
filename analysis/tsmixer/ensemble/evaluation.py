"""Site accuracy first, common settlement as a separate diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.validation import validate_composition
from analysis.deep_learning.features import interval_max
from analysis.tsmixer.comparison.core import evaluate,REFERENCE
from analysis.tsmixer.comparison.io import file_sha,load_tsmixer
from analysis.tsmixer.comparison.diagnostics import residual_analysis,correlation
from analysis.tsmixer.comparison.three_diagnostics import complementarity
from .folds import FOLDS
from .data import actual_grid
from .predictions import load_prediction
from .persistence import atomic_json,is_complete
from .reuse import legacy_assessment
from analysis.tsmixer.config import resolve


def capacity_groups(sites):
    loads={'G1':0.,'G2':0.};assignment={}
    for row in sites.sort_values(['capacity_kw','site_id'],ascending=[False,True]).itertuples():
        group=min(loads,key=loads.get);assignment[row.site_id]=group;loads[group]+=row.capacity_kw
    result=sites[['site_id']].assign(group=sites.site_id.map(assignment))
    validate_composition(result,sites);return result


def groups(c,sites,fold):
    safe=capacity_groups(sites)
    result={'metadata_primary':safe};provenance={'metadata_primary':dict(basis='capacity_only_metadata',eligible_for_weight_selection=True)}
    sj=resolve(c['sj_group'])
    if sj.is_file():
        result['sj_final_reference']=pd.read_csv(sj)
        provenance['sj_final_reference']=dict(basis='historical_residuals',declared_cutoff='2024-05-31 17:00:00+09:00',
                retrospective=fold.cutoff<pd.Timestamp('2024-05-31 17:00:00+09:00'),
                eligible_for_weight_selection=False,time_provenance='team declaration; grouping source logs missing',sha256=file_sha(sj))
    for value in result.values():validate_composition(value,sites)
    return result,provenance


def site_metrics(grid,prediction,seen=None):
    p=grid.validate_prediction(prediction);observed=grid.observed;cap=grid.sites.capacity_kw.to_numpy()
    r=p-grid.actual;day=interval_max(grid.times,grid.sites).T>0
    rows={}
    slices={'all':observed,'day':observed&day,'night':observed&~day,
            'high_actual_cf_ge_0.5_diagnostic':observed&(grid.actual/cap>=.5)}
    if seen is not None:
        slices.update(no_training_history=observed&(seen==0)[None,:],under90days=observed&((seen>0)&(seen<2160))[None,:])
    for name,mask in slices.items():
        n=int(mask.sum());e=r[mask];cf=(r/cap)[mask]
        rows[name]=dict(observed_rows=n,site_nmae_pct=float(100*np.abs(cf).mean()) if n else None,
                  site_cf_mae=float(np.abs(cf).mean()) if n else None,site_mae_kwh=float(np.abs(e).mean()) if n else None,
                  mean_bias_kwh=float(e.mean()) if n else None,mean_bias_cf=float(cf.mean()) if n else None,
                  absolute_cf_error_p95=float(np.quantile(np.abs(cf),.95)) if n else None,absolute_cf_error_p99=float(np.quantile(np.abs(cf),.99)) if n else None)
    return rows


def reference_compatible(c,grid,fold):
    if fold.name!='F3' or not c.get('baseline_reference'):return False
    reference=json.loads(resolve(c['baseline_reference']).read_text())
    original=grid.sites[['site_id']].assign(group=grid.sites.current_group.map({'A':'G1','B':'G2'}))
    return bool(reference.get('verified') and reference.get('expected_W')==REFERENCE
        and abs(reference.get('recomputed_W',float('inf'))-REFERENCE)<=1
        and reference.get('contract')==grid.contract(original,'observed_only')
        and reference.get('data_hashes')==getattr(grid,'source_hashes',None))


def evaluate_model(grid,prediction,assignments,fold,seen=None,output=None,label='model',reference_verified=False):
    result=dict(site=site_metrics(grid,prediction,seen),settlement={})
    for name,composition in assignments.items():
        for mode in ('observed_only','complete_fleet'):
            metrics,ledger=evaluate(grid,prediction,composition,mode,reference_verified=reference_verified)
            if not reference_verified or mode!='observed_only' or fold.name!='F3':
                metrics['internal_score']=None
                metrics['internal_score_status']='unverified_reference_or_different_mask' if fold.name=='F3' else 'not_applicable_different_year'
            result['settlement'][name+'/'+mode]=metrics
            if output is not None:ledger.to_csv(Path(output)/f'{label}_{name}_{mode}_hourly.csv',index=False)
    return result


def discover(c,fold,grid):
    predictions={};availability={};hashes={}
    for model in ('tsmixer','sj_ensemble','state01'):
        paths=[resolve(c['run_root'])/section/fold.name/model for section in ('models','imports')]
        found=False
        for path in paths:
            if not path.exists():continue
            if not is_complete(path):
                availability[model]=dict(status='incomplete',path=str(path));continue
            value,meta=load_prediction(path/'predictions.parquet',path/'predictions.json',grid,fold,model)
            predictions[model]=value;hashes[model]=dict(predictions=file_sha(path/'predictions.parquet'),metadata=file_sha(path/'predictions.json'))
            availability[model]=dict(status='available',path=str(path));found=True;break
        if not found:availability.setdefault(model,dict(status='unavailable',reason='No verified past-only fold predictions'))
    return predictions,availability,hashes


def component_predictions(availability,grid,fold):
    result={}
    for family,record in availability.items():
        if record['status']!='available' or family=='tsmixer':continue
        root=Path(record['path'])
        for path in sorted(root.glob(f'{family}_*.parquet')):
            meta=path.with_suffix('.json')
            if not meta.is_file():raise ValueError('Missing component provenance')
            value,_=load_prediction(path,meta,grid,fold,family)
            result[path.stem]=value
    return result


def audit(c,output=None):
    output=resolve(output or str(resolve(c['run_root'])/'audit'))
    from analysis.deep_learning.storage import result_guard
    if not output.is_relative_to(resolve(c['run_root'])):raise ValueError('Audit output outside run root')
    result_guard(output,64*1024**2);output.mkdir(parents=True,exist_ok=False)
    records={}
    for name,fold in FOLDS.items():
        grid,seen,data=actual_grid(resolve(c['data_root']),fold)
        _,availability,hashes=discover(c,fold,grid)
        assignments,provenance=groups(c,grid.sites,fold)
        records[name]=dict(fold=fold.record(),data=data,seen_sites=int((seen>0).sum()),no_history_sites=int((seen==0).sum()),
            complete_fleet_hours=int(grid.observed.all(1).sum()),availability=availability,hashes=hashes,groups=provenance,
            legacy=legacy_assessment(resolve(c['legacy_tsmixer']),fold))
        if name=='F3' and records[name]['legacy'].get('predicted'):
            legacy,_=load_tsmixer(resolve(c['legacy_tsmixer']),grid)
            atomic_json(output/'legacy_F3_reference.json',dict(status='not_strict_OOF; cannot select/freeze weights',
                 results=evaluate_model(grid,legacy,assignments,fold,seen)))
    atomic_json(output/'audit.json',records)
    print(json.dumps(records,ensure_ascii=False,indent=2));return records
