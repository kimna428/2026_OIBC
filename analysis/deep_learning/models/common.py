"""Only these explicit feature tensors may enter a model; targets/prices stay outside."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

INPUT_KEYS=('weather','weather_missing','temporal','static','site_index','site_pos','metadata_valid')

def model_inputs(batch): return {k:batch[k] for k in INPUT_KEYS}

def validate_inputs(batch):
    if set(batch)!=set(INPUT_KEYS): raise ValueError('Model accepts only the explicit feature allowlist')
    w=batch['weather']
    if w.shape!=batch['weather_missing'].shape or w.shape[-2]!=24: raise ValueError('Weather shape must end in [24,W] with a separate missing mask')
    if not torch.isfinite(w).all() or not torch.isfinite(batch['static']).all() or not torch.isfinite(batch['temporal']).all(): raise ValueError('Nonfinite normalized inputs')

class InputEncoder(nn.Module):
    def __init__(self,weather_dim,temporal_dim,static_dim,num_sites,width,site_dim=32,static_hidden=128,site_dropout=.1):
        super().__init__(); self.site_dropout=site_dropout
        self.embedding=nn.Embedding(num_sites+1,site_dim,padding_idx=0)
        self.static=nn.Sequential(nn.Linear(static_dim,static_hidden),nn.GELU(),nn.LayerNorm(static_hidden),nn.Linear(static_hidden,static_hidden))
        self.condition=nn.Linear(site_dim+static_hidden,width)
        self.weather=nn.Linear(weather_dim*2+temporal_dim,width)
    def forward(self,b):
        validate_inputs(b)
        ids=b['site_index'].long()
        if self.training and self.site_dropout:
            ids=torch.where(torch.rand(ids.shape,device=ids.device)<self.site_dropout,torch.zeros_like(ids),ids)
        condition=self.condition(torch.cat([self.embedding(ids),self.static(b['static'])],dim=-1))
        weather=torch.cat([b['weather'],b['weather_missing'].to(b['weather'].dtype),b['temporal']],dim=-1)
        return self.weather(weather)+condition.unsqueeze(-2)

class ConvBlock(nn.Module):
    def __init__(self,width,kernel_size=7,expansion=4,dropout=.1):
        super().__init__()
        if kernel_size%2!=1: raise ValueError('Kernel size must be odd for residual shape preservation')
        self.dw=nn.Conv1d(width,width,kernel_size,padding=kernel_size//2,groups=width)
        self.mix=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,int(width*expansion)),nn.GELU(),nn.Dropout(dropout),nn.Linear(int(width*expansion),width),nn.Dropout(dropout))
        self.scale=nn.Parameter(torch.ones(width)*1e-3)
    def forward(self,x): return x+self.mix(self.dw(x.transpose(-1,-2)).transpose(-1,-2))*self.scale

def run_block(block,x,enabled):
    return checkpoint(block,x,use_reentrant=False) if enabled and torch.is_grad_enabled() else block(x)

def head(width,dropout): return nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width//2,1))
