"""One row per canonical (timestamp,site_id); bounded provenance for every fold."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.tsmixer.comparison.io import file_sha,align_frame,read_table
from .persistence import atomic_json

STAGES=('training','normalization','auxiliary','outage','calibration','selection')

def validate_provenance(meta, fold, model, path=None):
    if meta.get('fold')!=fold.name or meta.get('model')!=model or meta.get('smoke_only') is not False:
        raise ValueError('Wrong fold/model or smoke artifacts cannot be promoted')
    if meta.get('start')!=str(fold.start) or meta.get('end_exclusive')!=str(fold.end): raise ValueError('Prediction period differs')
    if meta.get('timestamp_convention')!='interval_end' or meta.get('units') not in ('kWh','capacity_factor'): raise ValueError('Unknown timestamp/units')
    for stage in STAGES:
        value=meta.get(stage+'_cutoff','missing')
        if value is None and stage in ('auxiliary','outage','calibration'): continue
        if value=='missing': raise ValueError(f'Missing {stage} cutoff')
        t=pd.Timestamp(value)
        if pd.isna(t) or t.tzinfo is None or t>fold.cutoff: raise ValueError(f'{stage} sees future data')
    factor=meta.get('calibration_factor',1.)
    if not np.isfinite(factor) or factor<=0 or (factor!=1 and meta.get('calibration_cutoff') is None):raise ValueError('Calibration factor needs prior-fold fitting provenance')
    if not meta.get('inference_inputs') or not set(meta['inference_inputs'])<= {'forecasts','solar_geometry','calendar','site_metadata','predicted_weather'}:
        raise ValueError('Unknown or unavailable inference inputs')
    if meta.get('protocol')!='past_only_walkforward_v1': raise ValueError('Require audited past-only protocol, not original state01 main_all')
    if path and meta.get('prediction_sha256')!=file_sha(path): raise ValueError('Prediction hash differs')
    return True


def load_prediction(path,meta_path,grid,fold,model):
    path=Path(path);meta=json.loads(Path(meta_path).read_text())
    validate_provenance(meta,fold,model,path)
    if hasattr(grid,'source_hashes') and meta.get('data_hashes')!=grid.source_hashes:raise ValueError('Prediction data edition differs from evaluation data')
    value=align_frame(read_table(path),grid,[meta.get('column','prediction')])[meta.get('column','prediction')]
    if meta['units']=='capacity_factor': value=value*grid.sites.capacity_kw.to_numpy()[None,:]
    return grid.validate_prediction(value),meta


def save_prediction(path, value, times, sites, meta, observed=None, resume=False):
    path=Path(path);value=np.asarray(value)
    mask=np.ones(value.shape,dtype=bool) if observed is None else np.asarray(observed,dtype=bool)
    if (value.shape!=(len(times),len(sites)) or mask.shape!=value.shape or np.isinf(value).any()
            or not np.isfinite(value[mask]).all() or (value[np.isfinite(value)]<0).any()):raise ValueError('Invalid canonical predictions')
    frame=pd.DataFrame(dict(ts=times.repeat(len(sites)),site_id=np.tile(sites.site_id,len(times)),prediction=value.ravel()))
    if path.exists():
        if not resume:raise FileExistsError('Never overwrite a prediction')
        existing=read_table(path)
        if (len(existing)!=len(frame) or not np.array_equal(existing.site_id.to_numpy(),frame.site_id.to_numpy())
            or not pd.DatetimeIndex(existing.ts).equals(pd.DatetimeIndex(frame.ts))
            or not np.array_equal(existing.prediction.to_numpy(),value.ravel(),equal_nan=True)):
            raise ValueError('Existing prediction differs; never overwrite')
        metadata=path.with_suffix('.json')
        if metadata.exists():
            prior=json.loads(metadata.read_text())
            if prior['prediction_sha256']!=file_sha(path):raise ValueError('Corrupt prediction hash')
            for key in ('fold','model','start','end_exclusive','protocol','training_cutoff','normalization_cutoff','selection_cutoff'):
                if prior.get(key)!=meta.get(key):raise ValueError('Existing prediction provenance differs')
            return
    else:
        tmp=path.with_suffix('.pending.parquet')
        if tmp.exists():tmp.rename(tmp.with_name(tmp.name+f'.interrupted_{time.time_ns()}'))
        frame.to_parquet(tmp,index=False);tmp.replace(path)
    atomic_json(path.with_suffix('.json'),{**meta,'units':'kWh','column':'prediction','timestamp_convention':'interval_end','prediction_sha256':file_sha(path)})
