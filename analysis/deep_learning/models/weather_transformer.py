import torch
from torch import nn
from .common import InputEncoder,run_block,head

class WeatherTransformer(nn.Module):
    def __init__(self,weather_dim,temporal_dim,static_dim,num_sites,d_model=448,heads=8,layers=6,feedforward=2048,dropout=.1,gradient_checkpointing=False,site_dropout=.1,**unused):
        super().__init__()
        if d_model%heads: raise ValueError('d_model must be divisible by heads')
        self.checkpointing=gradient_checkpointing
        self.encoder=InputEncoder(weather_dim,temporal_dim,static_dim,num_sites,d_model,site_dropout=site_dropout)
        self.position=nn.Parameter(torch.zeros(1,24,d_model)); nn.init.normal_(self.position,std=.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d_model,heads,feedforward,dropout,activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
        self.head=head(d_model,dropout)
    def forward(self,batch):
        x=self.encoder(batch)
        if x.ndim!=3: raise ValueError('Transformer expects site/day batches')
        x=x+self.position
        # Bidirectional attention sees only the 24 pre-issued forecast hours, never targets.
        for layer in self.layers: x=run_block(layer,x,self.checkpointing and self.training)
        return self.head(x).squeeze(-1)
