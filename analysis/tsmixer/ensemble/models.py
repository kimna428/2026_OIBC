"""Repository-owned reviewed SJ and corrected state01 networks."""
import torch
from torch import nn

class SJGRU(nn.Module):
    def __init__(self,numeric_features,cat_sizes,cat_dims):
        super().__init__();self.embeddings=nn.ModuleList([nn.Embedding(s,d,padding_idx=0) for s,d in zip(cat_sizes,cat_dims)])
        self.projection=nn.Linear(numeric_features*2+sum(cat_dims),64)
        self.gru=nn.GRU(64,32,2,batch_first=True,bidirectional=True,dropout=.05)
        self.norm=nn.LayerNorm(64);self.pv=nn.Linear(64,1);nn.init.constant_(self.pv.bias,.25)
    def forward(self,b):
        x=torch.cat([b['numeric'],b['missing'].float(),*[e(b['categorical'][...,i]) for i,e in enumerate(self.embeddings)]],-1)
        h=self.projection(x);r,_=self.gru(h);return self.pv(self.norm(h+r)).squeeze(-1)

class StateGRU(nn.Module):
    def __init__(self,n_aux,width=192):
        super().__init__();self.inp=nn.Linear(45,width)
        self.gru=nn.GRU(width,width,2,batch_first=True,bidirectional=True,dropout=.1)
        self.head=nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,1))
        self.aux=nn.Sequential(nn.LayerNorm(2*width),nn.Linear(2*width,n_aux))
    def forward(self,x):
        h,_=self.gru(torch.nn.functional.gelu(self.inp(x)))
        return 1.2*torch.sigmoid(self.head(h).squeeze(-1)),self.aux(h)


def state_loss(p,y,mask,capacity,price,groups,aux,obs,aux_valid,lam):
    y=torch.where(mask,y,torch.zeros_like(y));m=mask.float();cap=capacity[None,:,None]
    w=price[:,None,:]*cap*m
    site=(w*(p-y).abs()).sum()/w.sum().clamp_min(1e-6)
    errors=[];mass=[]
    for group in (0,1):
        use=groups==group
        error=((p[:,use]-y[:,use])*cap[:,use]*m[:,use]).sum(1).abs()
        errors.append((price*error).sum());mass.append((price*(cap[:,use]*m[:,use]).sum(1)).sum())
    group=sum(errors)/sum(mass).clamp_min(1e-6)
    safe=torch.where(aux_valid,obs,torch.zeros_like(obs))
    auxiliary=((aux-safe).abs()*aux_valid).sum()/aux_valid.sum().clamp_min(1)
    return .5*site+.5*group+lam*auxiliary
