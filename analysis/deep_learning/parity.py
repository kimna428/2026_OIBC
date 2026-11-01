"""Bounded real-data feature/mask equivalence and loader timing, never model training."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .dataset import DayCache,DayDataset
from .features import fit_normalizer,aware
from .storage import tree_bytes

def compare(legacy,compact):
    a=DayCache(legacy); b=DayCache(compact)
    if len(a.days)>7: raise ValueError('Feature parity tool is limited to <=7 days; no full dataset scan')
    assert a.days.equals(b.days)
    assert a.manifest['feature_signature']==b.manifest['feature_signature']
    errors={}; masks={}
    for name in ['weather','temporal','target','energy','missing_kind','smp','k']:
        maximum=0.
        for d in range(len(a.days)):
            x=np.asarray(getattr(a,name)[d]); y=np.asarray(getattr(b,name)[d])
            assert np.array_equal(np.isnan(x),np.isnan(y)),(name,'missing mask')
            assert np.array_equal(x,y,equal_nan=True),(name,'values')
            valid=np.isfinite(x)
            if valid.any(): maximum=max(maximum,float(np.max(np.abs(x[valid].astype('float64')-y[valid].astype('float64')))))
        errors[name]=maximum; masks[name]=True
    # Train-only normalization and the partial last-day label mask are compared separately.
    cutoff=aware(a.days[0])+__import__('pandas').Timedelta(hours=17)
    if a.manifest['partition']=='train':
        x=fit_normalizer(a,cutoff); y=fit_normalizer(b,cutoff)
        assert {k:v for k,v in x.items() if k!='cache_signature'}=={k:v for k,v in y.items() if k!='cache_signature'}
    else:
        from .features import static_features,TIME,STATIC
        x={'cache_signature':a.manifest['signature'],'seen_hours':[1]*a.num_sites,'weather':{'mean':[0]*90,'std':[1]*90},'temporal':{'mean':[0]*len(TIME),'std':[1]*len(TIME)},'static':{'mean':[0]*len(STATIC),'std':[1]*len(STATIC)}}; y=x|{'cache_signature':b.manifest['signature']}
    plan={'cutoff':str(cutoff),'dev_start':str(a.days[1]),'dev_end':str(a.days[-1]+__import__('pandas').Timedelta(days=1))}
    timing={}
    for fleet in [False,True]:
        for split in (['train','dev'] if a.manifest['partition']=='train' else ['test']):
            da=DayDataset(a,x,plan,split,fleet); db=DayDataset(b,y,plan,split,fleet)
            assert np.array_equal(da.items,db.items)
            indices=np.random.default_rng(2026).integers(len(da),size=min(32,len(da)))
            for i in indices:
                aa=da[int(i)]; bb=db[int(i)]
                for key in aa: torch.testing.assert_close(aa[key],bb[key],rtol=0,atol=0,equal_nan=True)
            samples=[]
            for ds in [da,db]:
                rounds=[]
                for repeat in range(5):
                    start=time.perf_counter()
                    for i in indices: ds[int(i)]
                    rounds.append((time.perf_counter()-start)/len(indices)*1000)
                samples.append(float(np.median(rounds)))
            timing[f'{split}_{"fleet" if fleet else "site"}_ms_per_sample']={'legacy':samples[0],'compact':samples[1]}
    return {'kind':'feature_parity_only_no_training','days':len(a.days),'site_hours':len(a.days)*a.num_sites*24,'partition':a.manifest['partition'],'max_absolute_error':errors,'missing_masks_equal':masks,'normalized_tensors_and_all_masks_exact':True,'legacy_bytes':tree_bytes(legacy),'compact_bytes':tree_bytes(compact),'single_process_warm_timing':timing,'timing_is_not_gpu_training_throughput':True,'timing_method':'median of 5 rounds, <=32 fixed sample indices, one CPU process'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--legacy-cache',required=True); p.add_argument('--compact-cache',required=True); p.add_argument('--output',required=True); args=p.parse_args()
    result=compare(args.legacy_cache,args.compact_cache)
    from .storage import result_guard
    result_guard(args.output,65536)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
