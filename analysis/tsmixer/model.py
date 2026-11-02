"""Site/day TSMixer; axes are always [..., 24, hidden], never batch mixing."""
import torch
from torch import nn
from analysis.deep_learning.models.common import InputEncoder, INPUT_KEYS, head


class TimeMixing(nn.Module):
    """One shared 24 -> hidden_time -> 24 MLP per feature channel."""
    def __init__(self, width, hidden_time, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(24, hidden_time), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_time, 24),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return x + self.mlp(self.norm(x).transpose(-1, -2)).transpose(-1, -2)


class FeatureMixing(nn.Module):
    """One shared hidden -> hidden_feature -> hidden MLP per time step."""
    def __init__(self, width, hidden_feature, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, hidden_feature), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_feature, width),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class TSMixer(nn.Module):
    def __init__(self, weather_dim=90, temporal_dim=8, static_dim=146,
                 num_sites=253, hidden_dim=128, blocks=3, time_hidden=64,
                 feature_hidden=256, dropout=0.1, site_dropout=0.1):
        super().__init__()
        sizes = (weather_dim, temporal_dim, static_dim, num_sites, hidden_dim,
                 blocks, time_hidden, feature_hidden)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in sizes):
            raise ValueError('All model dimensions and blocks must be positive integers')
        if hidden_dim < 2 or not 0 <= dropout < 1 or not 0 <= site_dropout < 1:
            raise ValueError('hidden_dim >= 2 and dropout/site_dropout in [0,1) required')
        self.input_dims = (weather_dim, temporal_dim, static_dim)
        self.encoder = InputEncoder(weather_dim, temporal_dim, static_dim, num_sites,
                                    hidden_dim, site_dropout=site_dropout)
        self.blocks = nn.ModuleList([
            nn.Sequential(TimeMixing(hidden_dim, time_hidden, dropout),
                          FeatureMixing(hidden_dim, feature_hidden, dropout))
            for _ in range(blocks)])
        self.head = head(hidden_dim, dropout)

    def forward(self, batch):
        # Public site/day API accepts the five actual feature inputs. Existing
        # pipeline metadata may also be supplied; no labels/prices are accepted.
        required = {'weather', 'weather_missing', 'temporal', 'static', 'site_index'}
        if not required <= set(batch) or set(batch) - set(INPUT_KEYS):
            raise ValueError('Only normalized model features and pipeline metadata are allowed')
        w, t, s = batch['weather'], batch['temporal'], batch['static']
        wd, td, sd = self.input_dims
        if w.ndim != 3 or w.shape[1:] != (24, wd):
            raise ValueError('Site/day model requires weather [B,24,W]; use forward_fleet explicitly')
        n = w.shape[0]
        if t.shape != (n, 24, td) or s.shape != (n, sd):
            raise ValueError('Temporal/static dimensions differ from model contract')
        if batch['site_index'].shape != (n,) or batch['weather_missing'].shape != w.shape:
            raise ValueError('Invalid site_index or weather_missing shape')
        if batch['weather_missing'].dtype != torch.bool:
            raise ValueError('weather_missing must be a boolean mask')
        b = dict(batch)
        b.setdefault('site_pos', torch.arange(n, device=w.device))
        b.setdefault('metadata_valid', torch.ones(n, device=w.device, dtype=torch.bool))
        x = self.encoder(b)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)

    def forward_fleet(self, batch):
        """Future group-loss interface: require ALL canonical sites, never a subset."""
        w = batch['weather']
        if w.ndim != 4:
            raise ValueError('Fleet API requires [B_days,S,24,W]')
        b, s = w.shape[:2]
        if s != self.encoder.embedding.num_embeddings - 1:
            raise ValueError('Fleet API requires every canonical site')
        expected = torch.arange(s, device=w.device).expand(b, -1)
        if 'site_pos' not in batch or not torch.equal(batch['site_pos'], expected):
            raise ValueError('Fleet site_pos must follow canonical order')
        flat = {k: v.reshape(b*s, *v.shape[2:]) for k, v in batch.items()}
        return self(flat).reshape(b, s, 24)
