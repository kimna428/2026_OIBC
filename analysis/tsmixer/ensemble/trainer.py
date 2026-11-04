"""Epoch-boundary resumable component training, separate from original v0 CLI."""
import json
import math
import time
import platform
import resource
from pathlib import Path
import numpy as np
import torch
from analysis.deep_learning.storage import result_guard
from torch.utils.data import DataLoader,Sampler
from analysis.deep_learning.features import fit_normalizer,apply_postprocess
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.inference import dimensions
from analysis.deep_learning.models.common import model_inputs
from analysis.tsmixer.model import TSMixer
from analysis.tsmixer.data import DayDataset
from analysis.tsmixer.year_backtest import PriceFreeCache,predict as tsmixer_predict
from analysis.tsmixer.runtime import seed_everything,loader
from analysis.tsmixer.train import train_epoch
from analysis.tsmixer.checkpoint import save,load,capture_rng,restore_rng
from .models import SJGRU,StateGRU,state_loss
from .datasets import NeuralDays,fit_stats
from .features import AUX_ALL
from .persistence import atomic_json,is_complete,complete

class GroupedDays(Sampler):
    """Shuffle days and sites within days; bounded lazy feature reconstruction."""
    def __init__(self,dataset,seed):self.dataset,self.seed=dataset,seed
    def __iter__(self):
        rng=np.random.default_rng(self.seed);groups={}
        for i,(day,site) in enumerate(self.dataset.items):groups.setdefault(day,[]).append(i)
        for day in rng.permutation(list(groups)):
            yield from rng.permutation(groups[day]).tolist()
    def __len__(self):return len(self.dataset)


def make_model(component,config,stats,cache):
    if component=='tsmixer':return TSMixer(**dimensions(cache),**config['model'])
    if component=='sj_bigru':
        return SJGRU(len(stats['numeric']),[len(v)+1 for v in stats['levels']],
                     [8 if n=='site_id' else 4 if n in ('wa_station','wb_station') else 2 for n in stats['categorical']])
    return StateGRU(2 if component=='cloud' else 14,width=config.get('state_width',192))


def neural_predict(model,dataset,component,training,device):
    model.eval();c=dataset.view
    predictions={};present={}
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=training['batch_size'] if component=='sj_bigru' else training.get('state_batch_days',1),shuffle=False,num_workers=0):
            if component=='sj_bigru':p=model({k:batch[k].to(device) for k in ('numeric','missing','categorical')})
            else:
                x=batch['x'];b,s=x.shape[:2];p,_=model(x.reshape(b*s,24,45).to(device));p=p.reshape(b,s,24)
            p=torch.where(batch['day'].to(device),p.clamp_min(0),0).cpu().numpy()
            for j,d in enumerate(batch['day_index'].tolist()):
                if component=='sj_bigru':
                    predictions.setdefault(d,np.empty((c.num_sites,24),dtype='float32'));present.setdefault(d,set())
                    site=int(batch['site'][j]);
                    if site in present[d]:raise ValueError('Duplicate site prediction')
                    present[d].add(site);predictions[d][site]=p[j]
                else:predictions[d]=p[j];present[d]=set(range(c.num_sites))
    if any(len(present[d])!=c.num_sites for d in predictions):raise ValueError('Incomplete neural day')
    return np.stack([predictions[d] for d in dataset.indices])


def fit_component(component,config,cache,train_view,dev_view,features,plan,output,device,
                  outage=None,obs=None,epochs=None,smoke_updates=None):
    """selection labels are physically bounded by fold.cutoff via dev_view."""
    output=Path(output);output.mkdir(exist_ok=True)
    t=config['training'];family='sj_ensemble' if component=='sj_bigru' else 'state01'
    if is_complete(output):
        old=load(output/'best.pt' if epochs is None else output/'last.pt')
        stats=old['stats'];model=make_model(component,config,stats,cache).to(device);model.load_state_dict(old['model'])
        return model,stats,old['best_epoch']
    seed_everything(t['seed'],True)
    start=time.perf_counter()
    for pending in output.glob('*.pt.tmp'):
        pending.rename(pending.with_name(pending.name+f'.interrupted_{time.time_ns()}'))
    old=load(output/'last.pt') if (output/'last.pt').is_file() else None
    if old and old.get('epoch_record'):
        journal=output/'epochs.jsonl'
        records=[json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        if not any(r['epoch']==old['epoch'] for r in records):
            with journal.open('a') as f:f.write(json.dumps(old['epoch_record'],allow_nan=False)+'\n')
    stats=old['stats'] if old else (fit_normalizer(PriceFreeCache(train_view),plan['cutoff']) if component=='tsmixer' else fit_stats(train_view,features,family,outage,obs))
    stats_file=output/'normalizer.json'
    if stats_file.exists():
        if fingerprint(json.loads(stats_file.read_text()))!=fingerprint(stats):raise ValueError('Normalizer changed during resume')
    else:atomic_json(stats_file,stats)
    if component=='tsmixer':
        train=DayDataset(PriceFreeCache(train_view),stats,plan,'train')
        dev=DayDataset(PriceFreeCache(dev_view),stats,plan,'dev') if dev_view else None
    else:
        train=NeuralDays(train_view,features,stats,family,range(len(train_view.days)),True,outage,obs)
        if dev_view:
            selected=np.flatnonzero((dev_view.days>=__import__('pandas').Timestamp(plan['dev_start']))&(dev_view.days<__import__('pandas').Timestamp(plan['dev_end'])))
            dev=NeuralDays(dev_view,features,stats,family,selected)
        else:dev=None
    model=make_model(component,config,stats,cache).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'] if component!='sj_bigru' else .0001)
    total_epochs=epochs or t['epochs'];batch=t['batch_size'] if component in ('tsmixer','sj_bigru') else t.get('state_batch_days',1)
    updates_per_epoch=math.ceil(len(train)/batch)
    if component=='tsmixer':scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    elif component=='sj_bigru':scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t['epochs'],eta_min=.0001)
    else:scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,t['learning_rate'],total_steps=max(1,total_epochs*updates_per_epoch),pct_start=.1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    generator=torch.Generator().manual_seed(t['seed'])
    best,best_epoch,bad,start_epoch=float('inf'),0,0,0
    if old:
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scheduler.load_state_dict(old['scheduler'])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor) and key!='step':state[key]=value.to(device)
        generator.set_state(old['loader_rng']);restore_rng(old['rng'])
        best,best_epoch,bad,start_epoch=(float('inf') if old['best'] is None else old['best']),old['best_epoch'],old['bad'],old['epoch']
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    prep_seconds=time.perf_counter()-start;began=time.perf_counter()
    capacity=torch.tensor(cache.sites.capacity_kw.to_numpy(),dtype=torch.float32,device=device)
    groups=torch.tensor((cache.sites.current_group=='B').to_numpy(),device=device)
    for epoch in range(start_epoch,start_epoch if dev is not None and bad>=t['patience'] else total_epochs):
        model.train()
        if component=='tsmixer':
            info=train_epoch(model,loader(train,True,t,generator),optimizer,scaler,config,device,max_updates=smoke_updates)
        else:
            batches=DataLoader(train,batch_size=batch,sampler=GroupedDays(train,t['seed']+epoch) if component=='sj_bigru' else None,
                           shuffle=component!='sj_bigru',generator=generator,num_workers=0)
            loss_sum=0.;updates=0
            for b in batches:
                optimizer.zero_grad(set_to_none=True)
                if component=='sj_bigru':
                    p=model({k:b[k].to(device) for k in ('numeric','missing','categorical')});mask=b['mask'].to(device)
                    loss=(p-b['target'].to(device)).abs()[mask].mean()
                else:
                    x=b['x'].to(device);bn,s=x.shape[:2];p,aux=model(x.reshape(bn*s,24,45));p=p.reshape(bn,s,24)*b['day'].to(device)
                    cols=[0,6] if component=='cloud' else AUX_ALL;aux=aux.reshape(bn,s,24,-1)
                    loss=state_loss(p,b['target'].to(device),b['mask'].to(device),capacity,b['price'].to(device),groups,
                          aux,b['obs'][...,cols].to(device),b['aux_valid'][...,cols].to(device),.5 if component=='cloud' else .8)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if component not in ('sj_bigru','tsmixer'):scheduler.step()
                loss_sum+=float(loss.detach());updates+=1
                if smoke_updates and updates>=smoke_updates:break
            info=dict(loss=loss_sum/max(updates,1),updates=updates)
        if component in ('sj_bigru','tsmixer'):scheduler.step()
        value=None
        if dev is not None and smoke_updates is None:
            grid=tsmixer_predict(model,dev,t,device) if component=='tsmixer' else neural_predict(model,dev,component,t,device)
            total,count=0.,0
            for j,d in enumerate(dev.day_indices if component=='tsmixer' else dev.indices):
                y=np.asarray(dev_view.target[d]);mask=np.isfinite(y);total+=float(np.abs(grid[j]-y)[mask].sum());count+=int(mask.sum())
            value=total/count if count else float('inf')
            if not math.isfinite(value):raise ValueError('No internal selection labels')
            if value<best-t['min_delta']:best,best_epoch,bad=value,epoch+1,0
            else:bad+=1
        else:best_epoch=epoch+1
        record=dict(parameters=sum(p.numel() for p in model.parameters()),epoch=epoch+1,component=component,**info,internal_cf_mae=value,prep_seconds=prep_seconds,elapsed_seconds=time.perf_counter()-began)
        record['cpu_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024)
        if device.type=='cuda':record.update(peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
        state=dict(config=config,seed=t['seed'],model=model.state_dict(),stats=stats,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   loader_rng=generator.get_state(),rng=capture_rng(),epoch=epoch+1,best=best if math.isfinite(best) else None,
                   best_epoch=best_epoch,bad=bad,epoch_record=record,smoke_only=bool(smoke_updates),plan=plan)
        # None best is only used by refit; selection always has a finite score.
        result_guard(output,32*1024**2)
        save(output/'last.pt',state)
        if value is not None and best_epoch==epoch+1:save(output/'best.pt',state)
        with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
        if smoke_updates or (dev is not None and bad>=t['patience']):break
    if smoke_updates:
        atomic_json(output/'smoke.json',record)
    else:
        if epochs is None:
            state=load(output/'best.pt');model.load_state_dict(state['model'])
        complete(output,['last.pt','epochs.jsonl','normalizer.json']+(['best.pt'] if epochs is None else []))
    return model,stats,best_epoch
