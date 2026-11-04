"""Bounded synthetic optimizer steps; never a source of OOF predictions."""
import time
from pathlib import Path
import numpy as np
import torch
from analysis.tsmixer.model import TSMixer
from analysis.tsmixer.runtime import seed_everything
from .models import SJGRU,StateGRU,state_loss
from .persistence import atomic_json
from analysis.deep_learning.storage import result_guard


def run(config,output=None,updates=5):
    if not 1<=updates<=10:raise ValueError('Fixture smoke updates1..10')
    output=Path(output or Path(config['run_root'])/'fixture_smoke').resolve()
    result_guard(output,16*1024**2);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);seed_everything(config['seed'],True);records=[]
    for family in ('tsmixer','sj_ensemble','state01'):
        start=time.perf_counter()
        if family=='tsmixer':
            model=TSMixer(hidden_dim=16,blocks=1,time_hidden=8,feature_hidden=24,dropout=0)
            inputs=dict(weather=torch.randn(4,24,90),weather_missing=torch.zeros(4,24,90,dtype=torch.bool),temporal=torch.randn(4,24,8),static=torch.randn(4,146),site_index=torch.arange(4))
            forward=lambda:model(inputs)
        elif family=='sj_ensemble':
            model=SJGRU(80,[5]*6,[8,4,4,2,2,2]);inputs=dict(numeric=torch.randn(4,24,80),missing=torch.zeros(4,24,80,dtype=torch.bool),categorical=torch.zeros(4,24,6,dtype=torch.long))
            forward=lambda:model(inputs)
        else:
            model=StateGRU(14,width=16);forward=lambda:model(torch.randn(4,24,45))[0]
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True);loss=forward().abs().mean();loss.backward();optimizer.step()
        records.append(dict(model=family,updates=updates,seconds=time.perf_counter()-start,final_loss=float(loss.detach()),small_fixture_architecture=True))
    result=dict(smoke_only=True,eligible_for_OOF=False,records=records,device='cpu')
    atomic_json(output/'smoke.json',result);print(result);return result
