"""Static source/data inventory. No external module imports or checkpoint loads."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .audit import archive_inventory
from .core import START, END, CUTOFF
from .io import file_sha
from analysis.validation import local_hours


def literal_assignment(file, name):
    for node in ast.parse(Path(file).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'No literal assignment {name}')


def parameter_counts(width=192, channels=45):
    projection=(channels+1)*width
    recurrent=sum(2*(3*width*input_width+3*width*width+6*width) for input_width in (width,2*width))
    pv=4*width+2*width+1
    return {f'aux_{n}':dict(parameters=projection+recurrent+pv+4*width+(2*width+1)*n,
                           projection=projection,gru=recurrent,pv_head=pv,aux_head=4*width+(2*width+1)*n)
            for n in (2,14)}


def inventory(root):
    root=Path(root).resolve()
    if not root.is_dir():
        return dict(status='unavailable',root=str(root),reason='01_state root inaccessible')
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=file_sha(p))
           for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and p.name!='.DS_Store'}
    return dict(status='available',root=str(root),files=files)


def audit_state01(root, grid):
    root=Path(root).resolve();audit=inventory(root)
    if audit['status']!='available':return audit
    code=root/'01_state'
    audit['splits']=literal_assignment(code/'common.py','SPLITS')
    audit['historical_reference_scores']=[]
    ref=literal_assignment(code/'report.py','REF')
    audit['historical_reference_scores']=[dict(fold=key[0],combo=key[1],score=value,source='code REF constant; not recomputed') for key,value in ref.items()]
    audit['parameter_counts_analytic_not_checkpoint']=parameter_counts()
    audit['artifact_paths']=[p for p in audit['files'] if p.endswith(('.pt','.pth','.pkl','.joblib','.ckpt')) or '/pred_' in p or '/eval_' in p]
    audit['packaged_default_ROOT']=str((code/'common.py').resolve().parents[2])
    audit['archive']=archive_inventory(root.with_suffix('.zip'))
    data=root/'dataset';meta=pd.read_csv(data/'sites.csv')
    audit['metadata_matches_common']=bool(meta.set_index('site_id')[['capacity_kw','lat','lon','wa_station','wb_station']].sort_index().equals(
        grid.sites.set_index('site_id')[['capacity_kw','lat','lon','wa_station','wb_station']].sort_index()))
    genfile=data/'train/generation.parquet'
    year=pd.read_parquet(genfile,filters=[('ts','>=',START.to_pydatetime()),('ts','<',END.to_pydatetime())])
    year['ts']=local_hours(pd.to_datetime(year.ts))
    if year.duplicated(['site_id','ts']).any() or not year.site_id.isin(grid.sites.site_id).all():
        raise ValueError('01_state actual keys invalid')
    actual=year.pivot(index='ts',columns='site_id',values='energy').reindex(index=grid.times,columns=grid.sites.site_id).to_numpy()
    drop=pd.read_csv(data/'dropped_train_hours.csv');drop['ts']=local_hours(pd.to_datetime(drop.ts))
    late=pd.read_parquet(genfile,filters=[('ts','>',CUTOFF.to_pydatetime()),('ts','<',START.to_pydatetime())])
    audit['data_comparison']=dict(total_generation_rows=pq.ParquetFile(genfile).metadata.num_rows,
        year_rows=len(year),year_mask_matches=bool(np.array_equal(np.isfinite(actual),grid.observed)),
        year_actuals_match=bool(np.array_equal(actual,grid.actual,equal_nan=True)),
        removed_hours=len(drop),removed_first=str(drop.ts.min()),removed_last=str(drop.ts.max()),
        removed_hours_in_common_year=int(((drop.ts>=START)&(drop.ts<END)).sum()),
        late_training_rows_after_required_cutoff=len(late),late_first=str(late.ts.min()),late_last=str(late.ts.max()))
    audit['protocol_assessment']=dict(eligible_as_original_common_year=False,
        reasons=['Split.gen_tr includes targets through 2024-06-01 00:00, seven hours after required cutoff',
                 'outage_days is computed on all generation before split; boundary run qualification can depend on later targets',
                 'observed_like_forecast uses min/max of fa_all/fb_all spanning the validation year for auxiliary target filtering',
                 'no saved predictions/checkpoints/run logs supplied'],
        inference_observed_weather_input=False,
        inference_past_generation_input=False,
        inference_inputs=['forecasts','solar_geometry','calendar','site_metadata'],
        source_review_not_checkpoint_proof=True)
    return audit
