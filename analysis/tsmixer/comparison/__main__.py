"""Compare supplied predictions only. No model fitting, inference, SSH or installs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.deep_learning.storage import result_guard
from analysis.validation import validate_composition
from .core import evaluate,verify_reference,select_oof_alpha,START,END,CUTOFF
from .io import load_actual,load_tsmixer,align_frame,read_table,file_sha
from .audit import audit_manifest,recover_composition,group_geometry,archive_inventory
from .diagnostics import residual_analysis,group_pair_analysis,ensemble_sweep


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--train-source',type=Path,required=True)
    p.add_argument('--clustering',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('handover 2/01_participant_dataset'))
    p.add_argument('--tsmixer-run',type=Path,default=Path('runs/tsmixer/year_backtest/seed2026_001'))
    p.add_argument('--existing-reports',type=Path,default=Path('reports/tsmixer/year_backtest/seed2026_001'))
    p.add_argument('--teammate-predictions',type=Path,help='Observed site predictions with lightgbm/xgboost/bigru, kWh; validation only')
    p.add_argument('--baseline-predictions',type=Path,help='Original provided-baseline validation predictions, forecast_kwh; for denominator verification')
    p.add_argument('--oof-predictions',type=Path,help='Strict pre-cutoff OOF table documented in README')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve()
    allowed=Path('runs/tsmixer/comparison').resolve()
    if not output.is_relative_to(allowed) or output==allowed:
        p.error('Choose a new child of runs/tsmixer/comparison')
    if output.exists():
        raise FileExistsError('Never reuse or overwrite an existing comparison directory')
    protected=[args.package.resolve(),args.train_source.resolve(),args.data_root.resolve(),args.tsmixer_run.resolve()]
    if any(output.is_relative_to(root) for root in protected):
        raise ValueError('Output overlaps a read-only source')
    result_guard(output,256*1024**2)
    grid,seen,data_audit=load_actual(args.data_root)
    repro=audit_manifest(args.package);source=audit_manifest(args.train_source)
    if repro['mismatch'] or source['mismatch']:
        raise ValueError('Package file hashes changed; investigate before analysis')
    clustering=args.clustering or args.package.parent/'Clustering_sj.md'
    comp,recovery=recover_composition(args.package,args.train_source,clustering,grid.sites)
    groups={'teammate':comp}
    for name,file in [('baseline','baseline_composition.csv'),('tsmixer_metadata_candidate','tsmixer_metadata_candidate_composition.csv')]:
        path=args.tsmixer_run/file
        if not path.is_file():path=args.existing_reports/file
        if path.is_file():
            groups[name]=pd.read_csv(path)[['site_id','group']]
            validate_composition(groups[name],grid.sites)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in groups.items():
        value.to_csv(output/f'{name}_composition.csv',index=False,lineterminator='\r\n' if name=='teammate' else '\n')
    package_config=json.loads((args.package/'config.json').read_text())
    if pd.Timestamp(package_config['profiles']['validation']['cutoff'])!=CUTOFF:
        raise ValueError('Team validation cutoff differs')
    if pd.Timestamp(package_config['profiles']['validation']['start'])!=START or pd.Timestamp(package_config['profiles']['validation']['end_exclusive'])!=END:
        raise ValueError('Team validation timestamp window differs')
    prediction={};availability={};provenance={}
    if (args.tsmixer_run/'predictions.npy').is_file() and (args.tsmixer_run/'prediction_axis.json').is_file():
        prediction['tsmixer'],provenance['tsmixer']=load_tsmixer(args.tsmixer_run,grid)
        availability['tsmixer']=dict(status='available',source=str(args.tsmixer_run))
    else:
        availability['tsmixer']=dict(status='unavailable',reason='Missing year predictions.npy and/or prediction_axis.json; dev/smoke predictions cannot substitute')
    table=args.teammate_predictions
    if table is None:
        candidate=args.package/'reference/selected_stack_all_oof.parquet'
        if candidate.is_file():table=candidate
    if table:
        frame=read_table(table)
        if 'fold' in frame:
            frame=frame.loc[frame.fold=='year_2024'].copy()
        available=[c for c in ['lightgbm','xgboost','bigru'] if c in frame]
        prediction.update(align_frame(frame,grid,available))
        for name in available:availability[name]=dict(status='available',source=str(table))
        weights=package_config['weights'];factor=package_config['profiles']['validation']['factor']
        if all(name in prediction for name in weights):
            prediction['teammate_ensemble']=sum(prediction[name]*weight for name,weight in weights.items())*factor
            if 'forecast_kwh' in frame:
                saved=align_frame(frame,grid,['forecast_kwh'])['forecast_kwh']
                if not np.allclose(saved[grid.observed],prediction['teammate_ensemble'][grid.observed],rtol=1e-6,atol=1e-4):
                    raise ValueError('Saved ensemble differs from validation factor/weights')
            availability['teammate_ensemble']=dict(status='available',factor=factor,weights=weights)
        provenance['teammate_predictions']=dict(path=str(table),sha256=file_sha(table),
                  training_cutoff_status='config declares cutoff; missing training records do not prove checkpoint provenance')
    for name in ['lightgbm','xgboost','bigru','teammate_ensemble','separate_gru1','separate_gru2']:
        availability.setdefault(name,dict(status='unavailable',reason='No saved validation predictions' if not name.startswith('separate') else 'No implementation/checkpoint for this described experiment'))
    reference=dict(verified=False,reason='No original baseline validation predictions; fixed denominator in config alone is insufficient')
    if args.baseline_predictions:
        base=align_frame(read_table(args.baseline_predictions),grid,['forecast_kwh'])['forecast_kwh']
        original=grid.sites[['site_id']].assign(group=grid.sites.current_group.map({'A':'G1','B':'G2'}))
        reference=verify_reference(grid,base,original)
    results={};ledgers={name:{} for name in groups};summary=[]
    for name,pred in prediction.items():
        results[name]={}
        for assignment,composition in groups.items():
            for mode in ('observed_only','complete_fleet'):
                result,ledger=evaluate(grid,pred,composition,mode,reference['verified'])
                results[name][f'{assignment}/{mode}']=result
                summary.append(dict(model=name,assignment=assignment,mode=mode,**{k:v for k,v in result.items() if k!='contract'}))
                ledger.to_csv(output/f'{name}_{assignment}_{mode}_hourly.csv',index=False)
                group_rows=[]
                for group,d in ledger.groupby('group'):
                    group_rows.append(dict(group=group,valid_hours=len(d),observed_rows=int(d.observed_rows.sum()),
                         nmae_pct=float(100*d.abs_error_kwh.sum()/d.observed_capacity_kw.sum()),
                         nmae_macro_pct=float(d.nmae_pct.mean()),mae_kwh=float(d.abs_error_kwh.mean()),
                         W=float(d.penalty_krw.sum()),S=float(d.settlement_krw.sum())))
                pd.DataFrame(group_rows).to_csv(output/f'{name}_{assignment}_{mode}_groups.csv',index=False)
                if mode=='observed_only':ledgers[assignment][name]=ledger
    diagnostics=residual_analysis(grid,prediction,seen,output) if prediction else dict(status='unavailable',reason='No year predictions locally')
    if prediction:group_pair_analysis(ledgers,output)
    oof=select_oof_alpha(read_table(args.oof_predictions),grid.sites,comp) if args.oof_predictions else None
    ensemble=ensemble_sweep(grid,prediction,groups,reference['verified'],oof)
    if ensemble['trials']:
        pd.DataFrame([{k:v for k,v in r.items() if k!='contract'} for r in ensemble['trials']]).to_csv(output/'ensemble_sweep.csv',index=False)
    imported={}
    legacy_mask_sha=hashlib.sha256(grid.observed.reshape(365,24,253).transpose(0,2,1).tobytes()).hexdigest()
    reported=args.existing_reports/'metrics.json'
    if reported.is_file():
        existing=json.loads(reported.read_text())
        contract=existing['assignments']['baseline/observed_only']['contract']
        imported=dict(status='reported_only_not_recomputed',source=str(reported),sha256=file_sha(reported),metrics=existing,
                      raw_observed_mask_matches=legacy_mask_sha==contract['observed_mask_sha256'],
                      raw_observed_rows_match=int(grid.observed.sum())==contract['observed_rows'])
    # Safe weights-only checkpoint inspection, no teammate modules executed.
    checkpoints={}
    import torch
    for profile in ['validation','final']:
        path=args.package/f'models/{profile}/full_bigru_outer.pt'
        if path.is_file():
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            checkpoints[profile]=dict(path=str(path),sha256=file_sha(path),config=checkpoint['config'],epoch=checkpoint['epoch'],
                    parameter_count=sum(v.numel() for v in checkpoint['state_dict'].values()),
                    tensor_shapes={k:list(v.shape) for k,v in checkpoint['state_dict'].items()})
    audit=dict(repro=repro,train_source=source,data=data_audit,composition=recovery,
               geometry={name:group_geometry(grid.sites,c) for name,c in groups.items()},
               archive=archive_inventory(args.package.with_suffix('.zip')),checkpoints=checkpoints,
               config_sha256=file_sha(args.package/'config.json'),prediction_provenance=provenance)
    write(output/'audit.json',audit)
    write(output/'metrics.json',dict(status='computed' if prediction else 'blocked_missing_predictions',availability=availability,
          reference=reference,results=results,common_contract=grid.contract(comp,'observed_only'),
          observed_rows=int(grid.observed.sum()),valid_hours=int(grid.observed.any(axis=1).sum()),complete_fleet_hours=int(grid.observed.all(axis=1).sum()),
          teammate_expected_reported_only=package_config.get('validation_expected'),imported_tsmixer=imported))
    write(output/'diagnostics.json',diagnostics);write(output/'ensemble.json',ensemble)
    pd.DataFrame(summary,columns=['model','assignment','mode','site_nmae_pct','site_mae_kwh','group_nmae_pct','group_nmae_macro_pct','group_mae_kwh','W','S','revenue_krw','penalty_per_hour','internal_score','internal_score_status','valid_hours','observed_rows','group_hour_rows','group_cancellation_fraction']).to_csv(output/'summary.csv',index=False)
    write(output/'execution.json',dict(argv=sys.argv,python=sys.version,package_code_executed=False,training_executed=False,
          source_hashes={str(path):file_sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}))
    text=['# 공통 평가 실행 상태','',
          f"원본 관측행 {int(grid.observed.sum()):,}, 유효시간 {int(grid.observed.any(axis=1).sum()):,}, complete-fleet {int(grid.observed.all(axis=1).sum()):,}시간.",
          f"기준 페널티 호환 검증: {reference['verified']}. 수치 점수는 검증 전까지 잠정치.",
          '팀원 설정의 기대 점수와 기존 TSMixer 보고값은 재계산 결과와 분리해 metrics.json에 보관했습니다.','']
    text += [f"- {name}: {v['status']} — {v.get('reason',v.get('source',''))}" for name,v in availability.items()]
    (output/'REPORT_KO.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(output=str(output),prediction_models=list(prediction),observed_rows=int(grid.observed.sum()),reference_verified=reference['verified']),ensure_ascii=False))


if __name__=='__main__':main()
