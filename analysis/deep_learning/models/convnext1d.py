from torch import nn
from .common import InputEncoder,ConvBlock,run_block,head

class ConvNeXt1D(nn.Module):
    def __init__(self,weather_dim,temporal_dim,static_dim,num_sites,width=352,depth=5,kernel_size=7,expansion=4,dropout=.1,gradient_checkpointing=False,site_dropout=.1,**unused):
        super().__init__(); self.checkpointing=gradient_checkpointing
        self.encoder=InputEncoder(weather_dim,temporal_dim,static_dim,num_sites,width,site_dropout=site_dropout)
        self.blocks=nn.ModuleList([ConvBlock(width,kernel_size,expansion,dropout) for _ in range(depth)])
        self.head=head(width,dropout)
    def forward(self,batch):
        x=self.encoder(batch)
        if x.ndim!=3: raise ValueError('CNN expects [batch,24,features]')
        for block in self.blocks: x=run_block(block,x,self.checkpointing and self.training)
        return self.head(x).squeeze(-1)
