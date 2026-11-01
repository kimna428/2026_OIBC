import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .common import InputEncoder,ConvBlock,run_block,head

class CrossSiteAttention(nn.Module):
    def __init__(self,weather_dim,temporal_dim,static_dim,num_sites,width=256,depth=4,kernel_size=7,expansion=4,cross_layers=2,heads=8,feedforward=1024,dropout=.1,cross_attention=True,gradient_checkpointing=False,site_dropout=.1,**unused):
        super().__init__(); self.num_sites=num_sites; self.checkpointing=gradient_checkpointing
        self.encoder=InputEncoder(weather_dim,temporal_dim,static_dim,num_sites,width,site_dropout=site_dropout)
        self.blocks=nn.ModuleList([ConvBlock(width,kernel_size,expansion,dropout) for _ in range(depth)])
        self.attention=nn.ModuleList([nn.TransformerEncoderLayer(width,heads,feedforward,dropout,activation='gelu',batch_first=True,norm_first=True) for _ in range(cross_layers if cross_attention else 0)])
        self.fusion=nn.Linear(width*2,width); self.head=head(width,dropout)
    def forward(self,batch):
        x=self.encoder(batch)
        if x.ndim!=4 or x.shape[1]!=self.num_sites: raise ValueError('Cross-site model requires [batch_days,S,24,features]')
        expected=torch.arange(self.num_sites,device=x.device).expand(x.shape[0],-1)
        if not torch.equal(batch['site_pos'].long(),expected): raise ValueError('Cross-site inputs must use canonical manifest site order')
        batch_size,sites,hours,width=x.shape
        temporal=x.reshape(batch_size*sites,hours,width)
        for block in self.blocks: temporal=run_block(block,temporal,self.checkpointing and self.training)
        temporal=temporal.reshape(batch_size,sites,hours,width)
        tokens=temporal.mean(dim=2)  # Only S tokens, never S*24 full attention.
        valid=batch['metadata_valid'].bool(); active=valid.any(dim=1)
        for layer in self.attention:
            # Avoid NaN attention for an all-invalid-metadata day. Targets never mask attention.
            if active.any():
                selected=tokens[active]; padding=~valid[active]
                if self.checkpointing and self.training:
                    def block(value,module=layer,mask=padding): return module(value,src_key_padding_mask=mask)
                    selected=checkpoint(block,selected,use_reentrant=False)
                else: selected=layer(selected,src_key_padding_mask=padding)
                tokens=tokens.index_copy(0,active.nonzero().flatten(),selected)
        context=tokens.unsqueeze(2).expand(-1,-1,hours,-1)
        return self.head(self.fusion(torch.cat([temporal,context],dim=-1))).squeeze(-1)
