"""Three-provider saved-prediction comparison. No foreign code/model execution."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.deep_learning.storage import result_guard
from analysis.validation import validate_composition
from .core import evaluate, verify_reference
from .io import load_actual,load_tsmixer,read_table,align_frame,file_sha
from .audit import recover_composition
from .providers import load_sj,load_state01_table,load_state01_run,STATE_PARTS
from .state01_audit import audit_state01,inventory
from .diagnostics import residual_analysis,group_pair_analysis
from .three_diagnostics import complementarity
from .multi_ensemble import sweep,select_oof,mix,MODELS,validate_oof_group
from .__main__ import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state01-root',type=Path,default=Path('/Users/kimna/Downloads/01_state'))
    parser.add_argument('--sj-package',type=Path,default=Path('/Users/kimna/Downloads/oibc_repro_sj_2026_11_03'))
    parser.add_argument('--sj-source',type=Path,default=Path('/Users/kimna/Downloads/oibc_train_source_2026_11_03'))
    parser.add_argument('--sj-composition',type=Path,default=Path('runs/tsmixer/comparison/local_audit_20261104_002/teammate_composition.csv'))
    parser.add_argument('--data-root',type=Path,default=Path('handover 2/01_participant_dataset'))
    parser.add_argument('--tsmixer-run',type=Path,default=Path('runs/tsmixer/year_backtest/seed2026_001'))
    parser.add_argument('--inbox',type=Path,default=Path('runs/tsmixer/comparison_inputs'))
    parser.add_argument('--sj-predictions',type=Path)
    parser.add_argument('--sj-metadata',type=Path)
    state=parser.add_mutually_exclusive_group()
    state.add_argument('--state01-predictions',type=Path)
    state.add_argument('--state01-run',type=Path,help='Corrected main_all per-seed result root; never execute main.py')
    parser.add_argument('--state01-metadata',type=Path)
    parser.add_argument('--baseline-predictions',type=Path)
    parser.add_argument('--oof-predictions',type=Path)
    parser.add_argument('--oof-composition',type=Path)
    parser.add_argument('--oof-group-provenance',type=Path)
    parser.add_argument('--grid-step',type=float,choices=[.05,.1],default=.1)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.oof_predictions and (not args.oof_composition or not args.oof_group_provenance):
        parser.error('OOF selection requires --oof-composition and --oof-group-provenance; do not reuse a later residual-selected group')
    output=args.output.resolve();allowed=Path('runs/tsmixer/comparison').resolve()
    if output==allowed or not output.is_relative_to(allowed) or output.exists():
        parser.error('Choose a fresh child of runs/tsmixer/comparison; never overwrite')
    for root in (args.state01_root,args.sj_package,args.sj_source,args.data_root,args.tsmixer_run,args.inbox):
        if output.is_relative_to(root.resolve()):parser.error('Output overlaps a read-only input')
    result_guard(output,256*1024**2)
    grid,seen,data_audit=load_actual(args.data_root)
    state_audit=audit_state01(args.state01_root,grid)
    groups={};group_availability={}
    if args.sj_composition.is_file():
        groups['sj_final']=pd.read_csv(args.sj_composition)[['site_id','group']]
        group_availability['sj_final']=dict(status='available',path=str(args.sj_composition),sha256=file_sha(args.sj_composition))
    else:
        comp,evidence=recover_composition(args.sj_package,args.sj_source,args.sj_package.parent/'Clustering_sj.md',grid.sites)
        groups['sj_final']=comp;group_availability['sj_final']=dict(status='available',recovery=evidence)
    for label,name in [('baseline','baseline_composition.csv'),('tsmixer_metadata_candidate','tsmixer_metadata_candidate_composition.csv')]:
        path=args.tsmixer_run/name
        if not path.is_file():path=Path('reports/tsmixer/year_backtest/seed2026_001')/name
        group_availability[label]=dict(status='available' if path.is_file() else 'unavailable',path=str(path))
        if path.is_file():groups[label]=pd.read_csv(path)[['site_id','group']]
    for composition in groups.values():validate_composition(composition,grid.sites)
    predictions={};availability={};provenance={}
    def accept(names, function):
        try:
            values,evidence=function()
            if not values:raise ValueError('No usable model columns')
            for name,value in values.items():
                predictions[name]=grid.validate_prediction(value)
                availability[name]=dict(status='available',provenance=evidence)
            for name in names:
                availability.setdefault(name,dict(status='unavailable',reason='Required component columns/seeds absent'))
            provenance[names[0]]=evidence
        except (ValueError,KeyError,FileNotFoundError) as error:
            for name in names:availability[name]=dict(status='ineligible_or_invalid',reason=str(error))
    if (args.tsmixer_run/'predictions.npy').is_file() and (args.tsmixer_run/'prediction_axis.json').is_file():
        def ts():
            value,evidence=load_tsmixer(args.tsmixer_run,grid)
            if not evidence['cutoff_verified']:raise ValueError('TSMixer run.json with refit cutoff required for three-model comparison')
            return dict(tsmixer=value),evidence
        accept(['tsmixer'],ts)
    else:availability['tsmixer']=dict(status='unavailable',reason='Missing TSMixer year predictions/axis; no retraining attempted')
    sj_path=args.sj_predictions or args.inbox/'sj_validation.parquet'
    if not sj_path.is_file() and not args.sj_predictions:
        sj_path=args.sj_package/'reference/selected_stack_all_oof.parquet'
    sj_names=['sj_ensemble','sj_lightgbm','sj_xgboost','sj_bigru']
    if sj_path.is_file():
        sj_config=json.loads((args.sj_package/'config.json').read_text())
        sj_meta=args.sj_metadata or args.inbox/'sj_validation.json'
        accept(sj_names,lambda:load_sj(sj_path,grid,sj_config,sj_meta if sj_meta.is_file() or args.sj_metadata else None))
    else:
        for name in sj_names:availability[name]=dict(status='unavailable',reason='SJ validation predictions not supplied',expected=str(sj_path))
    state_path=args.state01_predictions or args.inbox/'state01_validation.parquet'
    state_meta=args.state01_metadata or args.inbox/'state01_validation.json'
    state_names=['state01',*STATE_PARTS]
    if args.state01_run or state_path.is_file():
        if not state_meta.is_file():
            for name in state_names:availability[name]=dict(status='ineligible_or_invalid',reason='Require cutoff-safe provenance JSON; original main_all is ineligible')
        else:
            accept(state_names,lambda:load_state01_run(args.state01_run,grid,state_meta) if args.state01_run else load_state01_table(state_path,grid,state_meta))
    else:
        for name in state_names:availability[name]=dict(status='unavailable',reason='No saved state01 year predictions',expected=str(state_path))
    reference=dict(verified=False,reason='No original baseline predictions; all internal scores provisional')
    if args.baseline_predictions:
        original=grid.sites[['site_id']].assign(group=grid.sites.current_group.map({'A':'G1','B':'G2'}))
        baseline=align_frame(read_table(args.baseline_predictions),grid,['forecast_kwh'])['forecast_kwh']
        reference=verify_reference(grid,baseline,original)
    output.mkdir(parents=True,exist_ok=False)
    for label,composition in groups.items():composition.to_csv(output/f'{label}_composition.csv',index=False)
    scores={};summary=[];ledgers={label:{} for label in groups}
    for model,prediction in predictions.items():
        scores[model]={}
        for label,composition in groups.items():
            for mode in ('observed_only','complete_fleet'):
                result,ledger=evaluate(grid,prediction,composition,mode,reference['verified'])
                scores[model][f'{label}/{mode}']=result
                summary.append(dict(model=model,assignment=label,mode=mode,**{k:v for k,v in result.items() if k!='contract'}))
                ledger.to_csv(output/f'{model}_{label}_{mode}_hourly.csv',index=False)
                by_group=ledger.groupby('group').agg(hours=('ts','size'),observed_rows=('observed_rows','sum'),
                      abs_error_kwh=('abs_error_kwh','sum'),capacity_sum=('observed_capacity_kw','sum'),
                      mae_kwh=('abs_error_kwh','mean'),macro_nmae_pct=('nmae_pct','mean'),W=('penalty_krw','sum'),S=('settlement_krw','sum'))
                by_group['ratio_nmae_pct']=100*by_group.abs_error_kwh/by_group.capacity_sum
                by_group.to_csv(output/f'{model}_{label}_{mode}_groups.csv')
                if mode=='observed_only':ledgers[label][model]=ledger
    diagnostics=residual_analysis(grid,predictions,seen,output) if predictions else dict(status='unavailable')
    if predictions:
        group_pair_analysis(ledgers,output)
        diagnostics['complementarity']=complementarity(grid,predictions,ledgers,output)
    ensemble=sweep(grid,predictions,groups,args.grid_step,reference['verified'])
    trial_rows=[]
    for trial in ensemble['trials']:
        trial_rows.append({**{k:v for k,v in trial.items() if k not in ('weights','contract')},**trial['weights']})
    pd.DataFrame(trial_rows).to_csv(output/'ensemble_grid.csv',index=False)
    oof=None;oof_applied=[]
    if args.oof_predictions:
        oof_frame=read_table(args.oof_predictions)
        oof_comp=pd.read_csv(args.oof_composition)
        validate_composition(oof_comp,grid.sites)
        group_check=validate_oof_group(oof_frame,oof_comp,json.loads(args.oof_group_provenance.read_text()))
        oof=select_oof(oof_frame,grid.sites,oof_comp,args.grid_step)
        oof['group_selection_provenance']=group_check
        try:
            blended=mix(predictions,oof['selected_weights'])
            for label,composition in groups.items():
                result,_=evaluate(grid,blended,composition,reference_verified=reference['verified'])
                oof_applied.append(dict(assignment=label,weights=oof['selected_weights'],**result))
        except ValueError as error:
            oof_applied=[dict(status='unavailable',reason=str(error))]
    # Penalty-heavy timestamps and each model's signed residual/penalty contribution.
    contributions=[]
    for label,models in ledgers.items():
        if not models:continue
        joined=pd.concat([d.assign(model=name) for name,d in models.items()],ignore_index=True)
        hard=joined.groupby(['group','ts']).penalty_krw.max().nlargest(50).index
        take=pd.MultiIndex.from_frame(joined[['group','ts']]).isin(hard)
        contributions.append(joined.loc[take].assign(assignment=label))
    if contributions:pd.concat(contributions,ignore_index=True).to_csv(output/'high_penalty_contributions.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
    pd.DataFrame([dict(model=name,**value) for name,value in availability.items()]).to_csv(output/'availability.csv',index=False)
    write(output/'metrics.json',dict(availability=availability,results=scores,reference=reference,
          primary_assignment='sj_final',groups=group_availability,observed_rows=int(grid.observed.sum()),
          valid_hours=int(grid.observed.any(axis=1).sum()),complete_fleet_hours=int(grid.observed.all(axis=1).sum()),
          common_contract=grid.contract(groups['sj_final'],'observed_only'),training_provenance=provenance))
    write(output/'ensemble.json',ensemble);write(output/'oof_selection.json',dict(selection=oof,year_fixed_evaluation=oof_applied))
    write(output/'diagnostics.json',diagnostics)
    write(output/'audit.json',dict(state01=state_audit,data=data_audit))
    # Hash source files again: fail if another process changed the source during the audit.
    after=inventory(args.state01_root)
    if state_audit.get('files')!=after.get('files'):raise RuntimeError('01_state source changed during analysis')
    write(output/'execution.json',dict(argv=sys.argv,python=sys.version,external_code_executed=False,
          checkpoint_deserialization=False,training_executed=False,source_unchanged=True,
          source_hashes={str(p):file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    lines=['# 세 모델 공통 평가 실행 결과','',f'실제 평가 가능 모델: {list(predictions)}',
           '모든 가중치 grid는 탐색용이며 검증연도에서 최적 가중치를 채택하지 않았습니다.',
           f"W_ref 호환 검증: {reference['verified']}; 검증 전 점수는 잠정치입니다.",'']
    lines.extend(f"- {name}: {value['status']} — {value.get('reason','')}" for name,value in availability.items())
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output),available=list(predictions),observed_rows=int(grid.observed.sum())),ensure_ascii=False))


if __name__=='__main__':main()
