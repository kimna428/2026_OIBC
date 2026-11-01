"""Bounded batch inference into mmap output, shared by evaluate and predict."""
from pathlib import Path
import numpy as np
import torch
from .dataset import DayCache,DayDataset
from .models import build_model,model_inputs
from .runtime import loader,to_device,autocast,amp_settings
from .features import apply_postprocess

def dimensions(cache):
    return {'weather_dim':len(cache.manifest['weather_features']),'temporal_dim':len(cache.manifest['temporal_features']),'static_dim':len(cache.manifest['static_features'])*2+sum(len(v)+1 for v in cache.manifest['station_categories'].values()),'num_sites':cache.num_sites}

def predict_grid(model,dataset,training_config,device,filename,precision='auto',postprocess='none'):
    filename=Path(filename); filename.parent.mkdir(parents=True,exist_ok=True)
    days=dataset.day_indices; mapping={int(d):i for i,d in enumerate(days)}
    pred=np.lib.format.open_memmap(filename,mode='w+',dtype='float32',shape=(len(days),dataset.cache.num_sites,24)); pred[:]=np.nan
    iterator=loader(dataset,False,training_config,torch.Generator().manual_seed(12345)); model.eval(); dtype=amp_settings(device,precision)
    with torch.inference_mode():
        for batch in iterator:
            ids=batch['day_index'].numpy(); site=batch['site_pos'].numpy()
            with autocast(device,dtype): values=model(model_inputs(to_device(batch,device)))
            values=values.float().cpu().numpy()
            if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite inference')
            for j,day in enumerate(ids):
                if dataset.full_fleet: pred[mapping[int(day)]]=values[j]
                else: pred[mapping[int(day)],site[j]]=values[j]
    for i,d in enumerate(days):
        if not np.isfinite(pred[i]).all(): raise ValueError('Prediction missing for at least one site/hour')
        pred[i]=apply_postprocess(pred[i],dataset.cache,int(d),dataset.normalizer,postprocess)
    pred.flush()
    return pred

def model_from_checkpoint(checkpoint,cache,device):
    if checkpoint['feature_signature']!=cache.manifest['feature_signature']: raise ValueError('Feature/site/composition signature differs from checkpoint')
    model=build_model(checkpoint['config']['model'],checkpoint['dimensions'])
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.to(device)
