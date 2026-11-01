"""Single-GPU training. CPU is deliberately limited to explicit <=2-update smoke tests."""
import argparse,copy,json,math,os,time,contextlib,sys,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import load_config,path,fingerprint
from .dataset import DayCache,DayDataset
from .features import split_plan,fit_normalizer,aware,FOLDS
from .models import build_model,model_inputs,parameter_count
from .losses import compute_loss
from .checkpoint import capture_rng,restore_rng,save_checkpoint,load_checkpoint
from .runtime import seed_everything,device_for,amp_settings,autocast,to_device,loader,GPULock,append_log
from .inference import dimensions,predict_grid
from .metrics import Evaluator
from analysis.validation import validate_composition


def loss_composition(config,cache,cutoff):
    group=config['group']; comp=cache.composition
    if group.get('composition_path'):
        p=path(group['composition_path']); comp=pd.read_csv(p)
        if not group.get('available_at'): raise ValueError('Custom loss composition requires available_at provenance')
        available=aware(group['available_at'])
        lock=p.parent/'composition_lock.json'
        if lock.exists():
            info=json.loads(lock.read_text())
            if 'source_max' in info and available<aware(info['source_max']): raise ValueError('Declared composition timestamp predates recorded source data')
        if available>aware(cutoff): raise ValueError('Group assignment used labels after training cutoff')
    validate_composition(comp,cache.sites)
    return comp


def selection_value(metrics,name):
    if name not in ['site_cf_mae','site_kwh_mae','penalty_per_hour']: raise ValueError('Unsupported selection metric')
    value=metrics[name]
    if value is None: raise ValueError('Requested selection metric unavailable: no complete group truth. Use an explicit site metric or another earlier development window.')
    return float(value)


def validate(model,dataset,config,device,out,group_index):
    pred=predict_grid(model,dataset,config['training'],device,out/'validation_predictions.npy',config['training']['precision'],config['training'].get('postprocess','none'))
    evaluator=Evaluator(dataset.cache,dataset.normalizer['seen_hours'])
    for i,d in enumerate(dataset.day_indices): evaluator.add_day(int(d),pred[i])
    metrics,detail,groups,ledger=evaluator.finish()
    # Log the same un-postprocessed forward objective used by training, separately from settlement.
    numerator_site=0.; numerator_group=0.; mass_site=0.; mass_group=0.
    with torch.inference_mode():
        for batch in loader(dataset,False,config['training'],torch.Generator().manual_seed(12345)):
            batch=to_device(batch,device)
            with autocast(device,amp_settings(device,config['training']['precision'])): raw=model(model_inputs(batch))
            _,info=compute_loss(raw,batch,config['loss'],config['group'],group_index)
            numerator_site+=info['site_loss']*info['site_mass']; mass_site+=info['site_mass']
            numerator_group+=info['group_loss']*info['group_mass']; mass_group+=info['group_mass']
    metrics['validation_site_loss']=numerator_site/max(mass_site,1e-12); metrics['validation_group_loss']=numerator_group/max(mass_group,1e-12)
    metrics['validation_loss_raw_forward']=config['group']['alpha']*metrics['validation_site_loss']+config['group']['beta']*metrics['validation_group_loss']
    return metrics


def optimizer_to(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor): state[key]=value.to(device)


def train_worker(args):
    c=load_config(args.config); c['_smoke_test']=bool(args.smoke_test)
    resume=load_checkpoint(path(args.resume)) if args.resume else None
    phase=args.phase or (resume['plan']['phase'] if resume else c['training'].get('phase','development'))
    if args.fold is not None:
        c['split']['outer_fold']=args.fold
        start=aware(FOLDS[args.fold][0]); end=start-pd.Timedelta(days=1)
        c['split']['dev_start']=str((end-pd.offsets.MonthBegin(1)).date()); c['split']['dev_end']=str(end.date())
    if args.device: c['training']['device']=args.device
    requested=c['training'].get('device','cuda')
    if args.smoke_test:
        if args.max_steps not in [1,2]: raise ValueError('Smoke tests require --max-steps 1 or 2')
        c['training']['num_workers']=0; c['training']['batch_size']=min(c['training']['batch_size'],8 if c['model']['name']=='cross_site' and c['model'].get('width',256)<=64 else 2); c['training']['accumulation_steps']=1
        torch.set_num_threads(2)
    elif requested=='cpu' or args.max_steps is not None: raise ValueError('Full local CPU training is disabled. Explicit --smoke-test --device cpu --max-steps 1/2 is required.')
    device=device_for(requested)
    cache=DayCache(path(c['data']['cache_root'])/'train')
    if cache.manifest['kind']=='synthetic_smoke' and not args.smoke_test: raise ValueError('Synthetic cache is only for explicit smoke tests')
    plan=split_plan(c,cache,phase)
    print(f'Preparing {c["name"]}: phase={phase}, data={cache.manifest["kind"]}, cutoff={plan["cutoff"]}, smoke={args.smoke_test}',flush=True)
    seed=c['training']['seed']; seed_everything(seed,c['training'].get('deterministic',True))
    initial=load_checkpoint(path(args.init_checkpoint)) if args.init_checkpoint else None
    selection=load_checkpoint(path(args.selection_checkpoint)) if args.selection_checkpoint else None
    if phase=='finetune':
        if c['model']['name']!='cross_site' or not c['group']['enabled'] or (initial is None and resume is None): raise ValueError('Group fine-tuning requires cross_site, enabled group loss and --init-checkpoint')
    elif c['group']['enabled'] and phase=='development': raise ValueError('Pretrain site-only first; use phase finetune for the group-loss experiment')
    if phase in ['refit','production']:
        if selection is None and resume is None: raise ValueError('Fixed-recipe refit/production requires --selection-checkpoint selected on dev')
        if selection and selection['smoke_test'] and not args.smoke_test: raise ValueError('Smoke selection cannot be promoted to a real model')
        if selection and aware(selection['selection_end'])>aware(plan['cutoff']): raise ValueError('Selection includes future labels relative to refit cutoff')
        if selection and selection['config']['model']!=c['model']: raise ValueError('Architecture differs from selected recipe')
    if selection and (selection['recipe'][-1]['loss']!=c['loss'] or selection['recipe'][-1]['group']!=c['group']): raise ValueError('Use the selected loss/group recipe config for refit')
    if initial and (initial['config']['model']!=c['model'] or initial['plan']['cutoff']!=plan['cutoff']): raise ValueError('Fine-tune must keep the pretrained architecture/training cutoff')
    normalizer=resume['normalizer'] if resume else (initial['normalizer'] if initial else fit_normalizer(cache,plan['cutoff']))
    if normalizer['fit_cutoff']!=plan['cutoff'] or normalizer['cache_signature']!=cache.manifest['signature']: raise ValueError('Scaler source period differs')
    train_data=DayDataset(cache,normalizer,plan,'train',c['model']['name']=='cross_site')
    dev_data=DayDataset(cache,normalizer,plan,'dev',c['model']['name']=='cross_site') if phase in ['development','finetune'] else None
    comp=loss_composition(c,cache,plan['cutoff']); group_index=torch.tensor(comp.set_index('site_id').group.reindex(cache.sites.site_id).map({'G1':0,'G2':1}).to_numpy(),device=device)
    if c['group']['enabled']:
        complete=0
        for day in train_data.day_indices:
            y=np.asarray(cache.target[day]); from .features import day_hours
            complete+=int((np.isfinite(y).all(0)&((y>=0)&(y<=1)).all(0)&(day_hours(cache.days[day]).asi8<=aware(plan['cutoff']).value)).sum())
        if not complete: raise ValueError('No full-fleet training hours before cutoff; group fine-tuning is impossible for this fold. Do not use future targets.')
    dims=dimensions(cache); model=build_model(c['model'],dims).to(device)
    if initial: model.load_state_dict(initial['model'],strict=True)
    if resume: model.load_state_dict(resume['model'],strict=True)
    run=path(args.run_dir) if args.run_dir else path(c['training']['save_root'])/f"{c['name']}_seed{seed}_fold{c['split']['outer_fold']}_{phase}"
    if (run/'last.pt').exists() and not resume: raise FileExistsError(f'Existing run {run}; explicitly --resume or choose --run-dir')
    run.mkdir(parents=True,exist_ok=True)
    run_spec={'config':c,'plan':plan,'feature_signature':cache.manifest['feature_signature'],'cache_signature':cache.manifest['signature'],'normalizer':normalizer,'composition':comp.to_dict('records'),'dimensions':dims,'phase':phase,'selection_checkpoint_signature':selection.get('run_signature') if selection else (resume['provenance']['selection_checkpoint_signature'] if resume else None),'initial_checkpoint_signature':initial.get('run_signature') if initial else (resume['provenance']['initial_checkpoint_signature'] if resume else None)}
    signature=fingerprint(run_spec)
    if resume and resume['run_signature']!=signature: raise ValueError('Resume config/data/normalizer differs from checkpoint; use the same args/config')
    dtype=amp_settings(device,c['training']['precision'])
    if resume and resume.get('amp_dtype')!=str(dtype): raise ValueError('Effective AMP precision changed; exact resume requires the same precision')
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and dtype==torch.float16)
    generator=torch.Generator().manual_seed(seed)
    train_loader=loader(train_data,True,c['training'],generator)
    if phase in ['refit','production']:
        stages=copy.deepcopy(selection['recipe'] if selection else resume['recipe'])
    else:
        opt_keys=['batch_size','accumulation_steps','learning_rate','weight_decay','warmup_steps','gradient_clip']
        stages=[{'epochs':c['training']['epochs'],'scheduler_epochs':c['training']['epochs'],'loss':c['loss'],'group':c['group'],'optimization':{k:c['training'][k] for k in opt_keys}}]
    parents=initial['recipe'] if initial else (resume['parent_recipe'] if resume else [])
    (run/'config.json').write_text(json.dumps(run_spec,indent=2)); (run/'environment.json').write_text(json.dumps({'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','pyarrow','PyYAML']},'source_sha256':{str(f.relative_to(Path(__file__).parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(__file__).parent.rglob('*.py')) if 'tests' not in f.parts},'torch':str(torch.__version__),'cuda_build':torch.version.cuda,'device':str(device),'gpu_name':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'parameters':parameter_count(model),'amp_dtype':str(dtype),'smoke_test':args.smoke_test,'synthetic':cache.manifest['kind']=='synthetic_smoke'},indent=2))
    print(f'Training samples={len(train_data)}, parameters={parameter_count(model):,}, AMP={dtype}; device={device}',flush=True)
    global_step=resume['global_step'] if resume else 0; best=resume['best_value'] if resume else float('inf'); bad_epochs=resume['bad_epochs'] if resume else 0
    start_stage=resume['cursor']['stage'] if resume else 0
    accumulation=c['training']['accumulation_steps']; stopped=False
    if args.max_steps is not None and global_step>=args.max_steps: raise ValueError('Checkpoint already reached --max-steps')
    with contextlib.nullcontext():
        for stage_index in range(start_stage,len(stages)):
            stage=stages[stage_index]; opt={**c['training'],**stage.get('optimization',{})}
            if args.smoke_test: opt['batch_size']=min(opt['batch_size'],8 if c['model']['name']=='cross_site' and c['model'].get('width',256)<=64 else 2)
            accumulation=opt['accumulation_steps']; train_loader=loader(train_data,True,opt,generator)
            optimizer=torch.optim.AdamW(model.parameters(),lr=opt['learning_rate'],weight_decay=opt['weight_decay'])
            total_updates=max(1,math.ceil(len(train_loader)/accumulation)*stage.get('scheduler_epochs',stage['epochs'])); warmup=min(opt.get('warmup_steps',50),max(0,total_updates-1))
            def schedule(step):
                if step<warmup: return (step+1)/max(1,warmup)
                return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_updates-warmup))))
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
            start_epoch=0; cursor=0; epoch_total=0.; epoch_count=0
            if resume and stage_index==start_stage:
                optimizer.load_state_dict(resume['optimizer']); optimizer_to(optimizer,device); scheduler.load_state_dict(resume['scheduler']); scaler.load_state_dict(resume['scaler'])
                start_epoch=resume['cursor']['epoch']; cursor=resume['cursor']['batch']; epoch_total=resume.get('epoch_total',0.); epoch_count=resume.get('epoch_count',0)
                generator.set_state(resume['epoch_generator_state'] if cursor else resume['loader_generator_state']); restore_rng(resume['rng'])
            for epoch in range(start_epoch,stage['epochs']):
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
                epoch_start=generator.get_state().clone(); model.train(); optimizer.zero_grad(set_to_none=True)
                if not cursor: epoch_total=0.; epoch_count=0
                micro=0; next_batch=0
                for bi,batch in enumerate(train_loader):
                    if bi<cursor: continue
                    next_batch=bi+1; batch=to_device(batch,device)
                    with autocast(device,dtype): pred=model(model_inputs(batch))
                    loss,parts=compute_loss(pred,batch,stage['loss'],stage['group'],group_index)
                    if parts['site_mass']==0 and parts['group_mass']==0: continue
                    scaler.scale(loss/accumulation).backward(); micro+=1
                    epoch_total+=float(loss.detach()); epoch_count+=1
                    update=micro==accumulation or next_batch==len(train_loader)
                    if not update: continue
                    scaler.unscale_(optimizer)
                    if micro<accumulation:
                        for p in model.parameters():
                            if p.grad is not None: p.grad.mul_(accumulation/micro)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip'],error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); micro=0; global_step+=1
                    append_log(run/'steps.jsonl',{'stage':stage_index,'epoch':epoch,'step':global_step,'loss':float(loss.detach()),**parts,'lr':scheduler.get_last_lr()[0],'smoke_test':args.smoke_test,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0})
                    if args.max_steps is not None and global_step>=args.max_steps: stopped=True; break
                # Flush a final accumulation window if skipped unsupervised batches ended the loader.
                if micro:
                    scaler.unscale_(optimizer)
                    for p in model.parameters():
                        if p.grad is not None: p.grad.mul_(accumulation/micro)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),opt['gradient_clip'],error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1
                    if args.max_steps is not None and global_step>=args.max_steps: stopped=True
                if not epoch_count: raise ValueError('No effective supervised loss in this training epoch')
                metrics=validate(model,dev_data,c,device,run,group_index) if dev_data else None
                value=selection_value(metrics,c['training']['selection_metric']) if metrics else 0.
                improved=value<best-c['training'].get('min_delta',0.)
                if improved: best=value; bad_epochs=0
                else: bad_epochs+=1
                completed=next_batch>=len(train_loader)
                next_cursor={'stage':stage_index,'epoch':epoch+1 if completed else epoch,'batch':0 if completed else next_batch}
                recipe=copy.deepcopy(parents+stages[:stage_index]+[dict(stage,epochs=epoch+1)]) if phase in ['development','finetune'] else copy.deepcopy(stages)
                state={'format_version':1,'amp_dtype':str(dtype),'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'normalizer':normalizer,'dimensions':dims,'config':c,'plan':plan,'feature_signature':cache.manifest['feature_signature'],'run_signature':signature,'recipe':recipe,'selection_end':plan['dev_end'] if phase in ['development','finetune'] else (selection['selection_end'] if selection else resume['selection_end']),'global_step':global_step,'best_value':best,'bad_epochs':bad_epochs,'cursor':next_cursor,'epoch_total':epoch_total if not completed else 0.,'epoch_count':epoch_count if not completed else 0,'epoch_generator_state':epoch_start,'loader_generator_state':generator.get_state(),'rng':capture_rng(),'smoke_test':args.smoke_test,'validation':metrics,'parent_recipe':parents,'provenance':run_spec,'selected_dev_signature':signature if phase in ['development','finetune'] else (selection['selected_dev_signature'] if selection else resume['selected_dev_signature'])}
                save_checkpoint(run/'last.pt',state)
                if improved or phase in ['refit','production']: save_checkpoint(run/'best.pt',state)
                append_log(run/'epochs.jsonl',{'stage':stage_index,'epoch':epoch,'step':global_step,'train_loss':epoch_total/max(1,epoch_count),'validation':metrics,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else 0,'smoke_test':args.smoke_test,'synthetic':cache.manifest['kind']=='synthetic_smoke'})
                print(f'{c["name"]} {phase} stage={stage_index} epoch={epoch} step={global_step} train_loss={epoch_total/max(1,epoch_count):.6f}',flush=True)
                cursor=0
                if stopped or (dev_data and bad_epochs>=c['training']['patience']): break
            if stopped: break
    return run


def train(args):
    config=load_config(args.config)
    device=device_for(args.device or config['training'].get('device','cuda'))
    # Acquire BEFORE any GPU parameter/state allocation, including resumed models.
    with GPULock(device): return train_worker(args)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--phase',choices=['development','finetune','refit','production']); p.add_argument('--resume'); p.add_argument('--init-checkpoint'); p.add_argument('--selection-checkpoint'); p.add_argument('--run-dir'); p.add_argument('--device'); p.add_argument('--max-steps',type=int); p.add_argument('--fold',type=int,choices=range(6)); p.add_argument('--smoke-test',action='store_true'); args=p.parse_args()
    try: train(args)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('GPU OOM: lower model-specific batch_size, enable gradient_checkpointing, and increase accumulation_steps. Do not launch models concurrently. Resume only with matching settings; changed batch settings require a new experiment.') from exc
if __name__=='__main__': main()
