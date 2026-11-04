from dataclasses import dataclass
import pandas as pd
from analysis.deep_learning.features import aware

@dataclass(frozen=True)
class Fold:
    name: str
    year: int
    @property
    def start(self): return aware(f'{self.year}-06-01 01:00')
    @property
    def end(self): return aware(f'{self.year+1}-06-01 01:00')
    @property
    def cutoff(self): return aware(f'{self.year}-05-31 17:00')
    @property
    def inner_start(self): return aware(f'{self.year}-04-01')
    @property
    def inner_cutoff(self): return self.inner_start-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    @property
    def times(self): return pd.date_range(self.start,self.end-pd.Timedelta(hours=1),freq='h').as_unit('ns')
    def plan(self, stage):
        if stage not in ('selection','refit'): raise ValueError('Unknown stage')
        return dict(cutoff=str(self.inner_cutoff if stage=='selection' else self.cutoff),
             dev_start=str(self.inner_start if stage=='selection' else self.start.normalize()),
             dev_end=str(self.start.normalize() if stage=='selection' else self.end.normalize()),
             selection_target_end=str(self.cutoff),phase=stage)
    def record(self):
        return dict(name=self.name,start=str(self.start),end_exclusive=str(self.end),hours=len(self.times),
                    cutoff=str(self.cutoff),inner_cutoff=str(self.inner_cutoff),inner_start=str(self.inner_start),
                    inner_target_end=str(self.cutoff))

FOLDS={f'F{i}':Fold(f'F{i}',2021+i) for i in (1,2,3)}

def validate_folds():
    for i,f in enumerate(FOLDS.values()):
        if not f.inner_cutoff<f.inner_start<f.cutoff<f.start<f.end: raise ValueError('Fold boundaries reversed')
        if i and list(FOLDS.values())[i-1].end!=f.start: raise ValueError('Overlapping/gapped folds')
    return True
