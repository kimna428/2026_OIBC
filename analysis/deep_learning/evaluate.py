import argparse,json
from pathlib import Path
from .config import path
from .dataset import DayCache,DayDataset
from .checkpoint import load_checkpoint
from .inference import predict_grid,model_from_checkpoint
from .runtime import device_for,GPULock
from .metrics import Evaluator
from .features import aware

def evaluate_checkpoint(checkpoint,cache,split,device,output):
    if split=='test': raise ValueError('Test has no labels; use predict, never evaluation')
    if split=='holdout' and checkpoint['plan']['phase']=='production': raise ValueError('Production model trained on holdout; holdout evaluation forbidden')
    dataset=DayDataset(cache,checkpoint['normalizer'],checkpoint['plan'],split,checkpoint['config']['model']['name']=='cross_site')
    model=model_from_checkpoint(checkpoint,cache,device)
    prediction=predict_grid(model,dataset,checkpoint['config']['training'],device,output/'predictions.npy',checkpoint['config']['training']['precision'],checkpoint['config']['training'].get('postprocess','none'))
    evaluator=Evaluator(cache,checkpoint['normalizer']['seen_hours'])
    for i,d in enumerate(dataset.day_indices): evaluator.add_day(int(d),prediction[i])
    metrics,detail,groups,ledger=evaluator.finish()
    metrics.update({'split':split,'model_role':checkpoint['plan']['phase'],'training_cutoff':checkpoint['plan']['cutoff'],'training_observed_site_hours':int(sum(checkpoint['normalizer']['seen_hours'])),'composition':cache.manifest['composition'],'feature_signature':cache.manifest['feature_signature'],'synthetic':cache.manifest['kind']=='synthetic_smoke','not_model_performance':checkpoint['smoke_test'],'postprocess':checkpoint['config']['training'].get('postprocess','none')})
    detail.to_csv(output/'site_and_cohort_metrics.csv',index=False); groups.to_csv(output/'group_metrics.csv',index=False)
    if len(ledger): ledger.to_parquet(output/'settlement_ledger.parquet',index=False)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--split',choices=['dev','holdout'],default='dev'); p.add_argument('--cache-root'); p.add_argument('--device',default='cuda'); p.add_argument('--output'); p.add_argument('--smoke-test',action='store_true'); args=p.parse_args()
    checkpoint=load_checkpoint(path(args.checkpoint))
    if checkpoint['smoke_test'] and not args.smoke_test: raise ValueError('Synthetic/debug checkpoint requires --smoke-test')
    root=path(args.cache_root or checkpoint['config']['data']['cache_root']); cache=DayCache(root/'train')
    device=device_for(args.device); out=path(args.output) if args.output else path(args.checkpoint).parent/('smoke_eval_'+args.split if checkpoint['smoke_test'] else 'eval_'+args.split); out.mkdir(parents=True,exist_ok=True)
    with GPULock(device): metrics=evaluate_checkpoint(checkpoint,cache,args.split,device,out)
    print(json.dumps({k:v for k,v in metrics.items() if k!='composition'},indent=2))
if __name__=='__main__': main()
