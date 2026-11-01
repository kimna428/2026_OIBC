"""Explicit synthetic-only fixture and <=2-update CLI tests. Never a fallback for missing data."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .config import ROOT,fingerprint
from .features import STATIC,TIME,contract,temporal_features,day_hours
from analysis.validation import hourly_weights


def synthetic_cache(root):
    root=Path(root); rng=np.random.default_rng(2026)
    sites=pd.DataFrame({'site_id':[f'PV{i:03d}' for i in range(1,254)],'capacity_kw':160.,'lat':np.linspace(35.,36.,253),'lon':np.linspace(127.,129.,253),'wa_dist':1000.,'wb_dist':2000.,'wa_station':[f'WA{i%61+1:03d}' for i in range(253)],'wb_station':[f'WB{i%73+1:03d}' for i in range(253)]})
    comp=sites[['site_id']].assign(group=['G1']*126+['G2']*127)
    categories={p:sorted(sites[p+'_station'].unique().tolist()) for p in ['wa','wb']}
    spec={'version':1,'weather_features':contract()['weather_features'],'temporal_features':TIME,'static_features':STATIC,'site_ids':sites.site_id.tolist(),'station_categories':categories,'composition':comp.to_dict('records'),'units':'SYNTHETIC TEST DATA ONLY','input_hashes':{'synthetic':'explicitly_generated_fixture'}}
    signature=fingerprint(spec)
    for part,start,periods in [('train','2025-01-29',6),('test','2025-02-04',1)]:
        dest=root/part; dest.mkdir(parents=True,exist_ok=True); days=pd.date_range(start,periods=periods,tz='Asia/Seoul',freq='D').as_unit('ns')
        shape=(periods,253,24); base=rng.normal(size=shape+(30,)).astype('float32'); round10=base+rng.normal(scale=.1,size=base.shape).astype('float32')
        weather=np.concatenate([base,round10,base-round10],axis=-1); weather[rng.random(weather.shape)<.03]=np.nan
        target=rng.uniform(0,.6,size=shape); target[rng.random(shape)<.12]=np.nan
        if part=='train':
            target[:2,:10,:]=np.nan; target[2:,:,10:13]=.2
            target[2,20,8]=np.nan
        else: target[:]=np.nan
        energy=target*160.; kind=np.where(np.isfinite(target),0,2).astype('uint8')
        if part=='train': kind[:2,:10,:]=1
        else: kind[:]=5
        temporal=[]; price=[]; weights=[]
        for day in days:
            times=day_hours(day); temporal.append(temporal_features(times,sites).transpose(1,0,2)); price.append(np.full(24,100.)); weights.append(hourly_weights(times))
        arrays={'weather':weather,'temporal':np.asarray(temporal,dtype='float32'),'target':target,'energy':energy,'missing_kind':kind,'static':sites[STATIC].to_numpy(dtype='float32'),'smp':np.asarray(price),'k':np.asarray(weights,dtype='float32')}
        for key,value in arrays.items(): np.save(dest/(key+'.npy'),value)
        sites.to_csv(dest/'sites.csv',index=False); comp.to_csv(dest/'composition.csv',index=False)
        manifest=spec|{'kind':'synthetic_smoke','partition':part,'start':start,'end_exclusive':str((days[-1]+pd.Timedelta(days=1)).date()),'test_start':'2025-02-04','days':[str(d.date()) for d in days],'num_sites':253,'shape':list(shape),'feature_signature':signature,'observed_targets':int(np.isfinite(target).sum()),'missing_kind_codes':{'observed':0,'pre_first':1,'internal':2,'post_last':3,'no_history':4,'test_unlabelled':5}}
        manifest['signature']=fingerprint(manifest); (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return root


def smoke_config(model,cache,folder):
    c=yaml.safe_load((ROOT/'configs/deep_learning'/({'cnn':'cnn_small','transformer':'transformer_small','cross_site':'cross_site_base'}[model]+'.yaml')).read_text())
    c['name']=model+'_synthetic_smoke'; c['data']['cache_root']=str(cache)
    common={'name':model,'dropout':.05,'gradient_checkpointing':True,'site_dropout':.1}
    if model=='transformer': common.update(d_model=32,heads=4,layers=2,feedforward=64)
    else: common.update(width=32,depth=2,kernel_size=3,expansion=2)
    if model=='cross_site': common.update(cross_layers=1,heads=4,feedforward=64,cross_attention=True)
    c['model']=common
    c['split'].update(dev_start='2025-02-01',dev_end='2025-02-02',smoke_outer_start='2025-02-03',smoke_outer_end='2025-02-04')
    c['training'].update(device='cpu',epochs=2,batch_size=2,num_workers=0,accumulation_steps=1,warmup_steps=0,precision='fp32',postprocess='night_only',save_root=str(folder))
    return c


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default=str(ROOT/'runs/deep_learning/smoke')); p.add_argument('--extended',action='store_true'); p.add_argument('--device',choices=['cpu','cuda'],default='cpu'); p.add_argument('--precision',choices=['fp32','auto','bf16','fp16'],default='fp32'); args=p.parse_args(); out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise FileExistsError('Use a fresh --output directory for repeat smoke tests')
    cache=synthetic_cache(out/'cache'); results={}
    for name in ['cnn','transformer','cross_site']:
        config=smoke_config(name,cache,out); config['training']['precision']=args.precision; filename=out/(name+'.yaml'); filename.write_text(yaml.safe_dump(config,sort_keys=False)); run=out/name
        commands=[['train','--config',str(filename),'--run-dir',str(run),'--device',args.device,'--smoke-test','--max-steps','1'],['train','--config',str(filename),'--run-dir',str(run),'--resume',str(run/'last.pt'),'--device',args.device,'--smoke-test','--max-steps','2'],['evaluate','--checkpoint',str(run/'best.pt'),'--split','dev','--device',args.device,'--smoke-test'],['predict','--checkpoint',str(run/'best.pt'),'--split','test','--output',str(run/'test.parquet'),'--device',args.device,'--smoke-test']]
        for command in commands:
            with (out/(name+'_cli.log')).open('a') as log:
                subprocess.run([sys.executable,'-m','analysis.deep_learning.'+command[0],*command[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[name]={'train_one_update':True,'resume_to_two_updates':True,'evaluate_cli':True,'predict_and_submission_cli':True}
    if args.extended:
        import torch
        def run_cli(module,arguments):
            with (out/'extended_cli.log').open('a') as log:
                subprocess.run([sys.executable,'-m','analysis.deep_learning.'+module,*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        cpu=['--device',args.device,'--smoke-test']; cap=['--max-steps','2']
        refits=[]
        for name in ['cnn','transformer','cross_site']:
            folder=out/(name+'_refit'); refits.append(str(folder/'best.pt'))
            run_cli('train',['--config',str(out/(name+'.yaml')),'--phase','refit','--selection-checkpoint',str(out/name/'best.pt'),'--run-dir',str(folder),*cpu,*cap])
            run_cli('evaluate',['--checkpoint',str(folder/'best.pt'),'--split','holdout',*cpu])
        run_cli('train',['--config',str(out/'cnn.yaml'),'--phase','production','--selection-checkpoint',str(out/'cnn/best.pt'),'--run-dir',str(out/'cnn_production'),*cpu,*cap])
        run_cli('predict',['--checkpoint',str(out/'cnn_production/best.pt'),'--split','test','--output',str(out/'production_test.parquet'),*cpu])
        # Compare exact continuous two-update weights to the one-update+resume trajectory.
        run_cli('train',['--config',str(out/'cnn.yaml'),'--run-dir',str(out/'cnn_continuous'),*cpu,*cap])
        from .checkpoint import load_checkpoint
        resumed=load_checkpoint(out/'cnn/last.pt'); continuous=load_checkpoint(out/'cnn_continuous/last.pt')
        for key in resumed['model']: torch.testing.assert_close(resumed['model'][key],continuous['model'][key],rtol=0,atol=0)
        fine=smoke_config('cross_site',cache,out); fine['training']['batch_size']=8
        prefile=out/'recipe_pretrain.yaml'; prefile.write_text(yaml.safe_dump(fine,sort_keys=False))
        run_cli('train',['--config',str(prefile),'--run-dir',str(out/'recipe_pretrain'),*cpu,'--max-steps','1'])
        fine['name']='cross_site_group_smoke'; fine['group'].update(enabled=True,beta=.2); fine['training'].update(phase='finetune',learning_rate=.0001)
        finefile=out/'recipe_finetune.yaml'; finefile.write_text(yaml.safe_dump(fine,sort_keys=False))
        run_cli('train',['--config',str(finefile),'--init-checkpoint',str(out/'recipe_pretrain/best.pt'),'--run-dir',str(out/'recipe_finetune'),*cpu,'--max-steps','1'])
        run_cli('train',['--config',str(finefile),'--phase','refit','--selection-checkpoint',str(out/'recipe_finetune/best.pt'),'--run-dir',str(out/'recipe_refit'),*cpu,*cap])
        two_stage=load_checkpoint(out/'recipe_refit/last.pt')
        if two_stage['cursor']['stage']!=1 or len(two_stage['recipe'])!=2: raise AssertionError('Both refit recipe stages must execute')
        checkpoints=[str(out/name/'best.pt') for name in ['cnn','transformer','cross_site']]
        run_cli('ensemble',['fit','--checkpoints',*checkpoints,'--output',str(out/'blend.json'),'--step','.5',*cpu])
        run_cli('ensemble',['predict','--checkpoints',*refits,'--weights',str(out/'blend.json'),'--split','test','--output',str(out/'blend_test.parquet'),*cpu])
        run_cli('ensemble',['evaluate','--checkpoints',*refits,'--weights',str(out/'blend.json'),'--split','holdout','--output',str(out/'blend_holdout_metrics.json'),*cpu])
        results['extended']={'frozen_ensemble_evaluation':True,'refit_and_holdout_cli':True,'production_and_test_cli':True,'exact_resume_matches_continuous_two_updates':True,'group_finetune_and_two_stage_refit':True,'development_ensemble_and_refit_inference':True}
    report={'kind':'synthetic_smoke_not_model_performance','device':args.device,'large_training_executed':False,'remote_connection_attempted':False,'results':results}
    (out/'summary.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
