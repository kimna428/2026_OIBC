import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .config import path
from .checkpoint import load_checkpoint
from .dataset import DayCache,DayDataset
from .inference import predict_grid,model_from_checkpoint
from .features import day_hours,issuance,aware
from .runtime import device_for,GPULock

def export_predictions(grid,cache,days,output):
    """Site Parquet plus a validated group CSV in the official submission shape."""
    output=Path(output)
    from .storage import result_guard
    result_guard(output,len(days)*cache.num_sites*24*80+2**20)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.suffix!='.parquet': raise ValueError('--output must end in .parquet (site predictions)')
    groupfile=output.with_suffix('.forecast.csv'); capacity=cache.sites.capacity_kw.to_numpy(); group=cache.composition.set_index('site_id').group.reindex(cache.sites.site_id)
    writer=None; group_rows=[]
    try:
        for i,d in enumerate(days):
            pred=np.asarray(grid[i]); ts=day_hours(cache.days[d]); kwh=pred*capacity[:,None]
            if not np.isfinite(pred).all() or (pred<0).any(): raise ValueError('Invalid exported prediction')
            frame=pd.DataFrame({'site_id':np.repeat(cache.sites.site_id.to_numpy(),24),'ts':np.tile(ts,cache.num_sites),'pred_cf':pred.ravel(),'pred_kwh':kwh.ravel()})
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(output,table.schema)
            writer.write_table(table)
            for label in ['G1','G2']:
                values=kwh[group.eq(label).to_numpy()].sum(axis=0)
                group_rows.append(pd.DataFrame({'group':label,'ts':ts,'forecast_kwh':values}))
    finally:
        if writer: writer.close()
    groups=pd.concat(group_rows,ignore_index=True).sort_values(['group','ts'])
    expected=len(days)*24
    if groups.duplicated(['group','ts']).any() or len(groups)!=2*expected or groups.ts.dt.tz is None: raise ValueError('Submission key schema invalid')
    if groups.forecast_kwh.isna().any() or (groups.forecast_kwh<0).any(): raise ValueError('Invalid submission energy')
    for g,d in groups.groupby('group'):
        expected_axis=pd.DatetimeIndex(np.concatenate([day_hours(cache.days[x]).to_numpy() for x in days]))
        if not pd.DatetimeIndex(d.ts).equals(expected_axis): raise ValueError('Submission timeline differs from supplied SMP grid')
    groups.to_csv(groupfile,index=False)
    cache.composition.to_csv(output.with_suffix('.composition.csv'),index=False)
    (output.with_suffix('.json')).write_text(json.dumps({'site_prediction':str(output),'forecast_csv':str(groupfile),'rows_per_group':expected,'official_test_rows_expected':8760,'full_official_test':expected==8760,'synthetic':cache.manifest['kind']=='synthetic_smoke','scoring_performed':False},indent=2))
    return groupfile

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--split',choices=['dev','holdout','test'],default='test'); p.add_argument('--output',required=True); p.add_argument('--cache-root'); p.add_argument('--device',default='cuda'); p.add_argument('--smoke-test',action='store_true'); args=p.parse_args()
    ck=load_checkpoint(path(args.checkpoint))
    if ck['smoke_test'] and not args.smoke_test: raise ValueError('Smoke checkpoint cannot be used for a real submission')
    root=path(args.cache_root or ck['config']['data']['cache_root']); cache=DayCache(root/('test' if args.split=='test' else 'train'))
    if args.split=='test' and cache.manifest['partition']!='test': raise ValueError('Test cache expected')
    dataset=DayDataset(cache,ck['normalizer'],ck['plan'],args.split,ck['config']['model']['name']=='cross_site')
    first_issue=issuance(day_hours(cache.days[dataset.day_indices[0]])).min()
    if aware(ck['plan']['cutoff'])>first_issue: raise ValueError('Checkpoint trained beyond first prediction issuance')
    if args.split=='test' and cache.manifest['kind']=='provided' and len(dataset.day_indices)*24!=8760: raise ValueError('Official test must have all 8760 SMP hours')
    output=path(args.output); device=device_for(args.device)
    with GPULock(device):
        model=model_from_checkpoint(ck,cache,device)
        grid=predict_grid(model,dataset,ck['config']['training'],device,output.with_suffix('.npy'),ck['config']['training']['precision'],ck['config']['training'].get('postprocess','none'))
        groupfile=export_predictions(grid,cache,dataset.day_indices,output)
    print(f'Site predictions: {output}; group submission: {groupfile}; test scoring not performed')
if __name__=='__main__': main()
