"""Validate supplied mmap cache headers, optional bounded scans and original hashes."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .config import ROOT,path
from .dataset import DayCache
from .storage import tree_bytes
from .prepare import inspect_inputs,sha

def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache-root',default=os.environ.get('OIBC_CACHE_ROOT',str(ROOT/'analysis/deep_learning/cache'))); p.add_argument('--partition',choices=['train','test','both'],default='both'); p.add_argument('--scan',action='store_true'); p.add_argument('--verify-originals',action='store_true'); p.add_argument('--data-root',default=os.environ.get('OIBC_DATA_ROOT',str(ROOT/'handover 2/01_participant_dataset'))); args=p.parse_args()
    signatures=[]
    for partition in (['train','test'] if args.partition=='both' else [args.partition]):
        c=DayCache(path(args.cache_root)/partition); signatures.append(c.manifest['feature_signature'])
        summary={'partition':partition,'kind':c.manifest['kind'],'days':len(c.days),'sites':c.num_sites,'weather_shape':list(c.weather.shape),'target_shape':list(c.target.shape),'observed_targets':c.manifest['observed_targets'],'schema':'passed','layout':c.manifest.get('layout','legacy'),'actual_cache_bytes':tree_bytes(c.root),'synthetic_performance_claim':False}
        if args.scan:
            count=0
            for day in range(len(c.days)):
                if np.isinf(c.weather[day]).any(): raise ValueError('Infinite cached forecast')
                if not np.array_equal(np.isfinite(c.target[day]),np.isfinite(c.energy[day])): raise ValueError('Energy/CF presence masks differ')
                if not np.allclose(c.target[day]*c.sites.capacity_kw.to_numpy()[:,None],c.energy[day],equal_nan=True): raise ValueError('Target capacity conversion differs')
                count+=int(np.isfinite(c.target[day]).sum())
            if count!=c.manifest['observed_targets']: raise ValueError('Observed target count differs')
            if partition=='test' and count: raise ValueError('Test cache contains labels')
            summary['scan']='passed'
        if args.verify_originals:
            inspect_inputs(path(args.data_root))
            for name,value in c.manifest['input_hashes'].items():
                if sha(path(args.data_root)/name)!=value: raise ValueError(f'Input hash differs: {name}')
            summary['original_hashes']='passed'
        print(json.dumps(summary,indent=2))
    if len(set(signatures))!=1: raise ValueError('Train/test feature/site/composition signatures differ')
if __name__=='__main__': main()
