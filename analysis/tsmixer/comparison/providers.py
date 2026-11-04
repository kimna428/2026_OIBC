"""Prediction-only provider adapters. Never import or execute teammate code."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .core import START, END, CUTOFF
from .io import read_table, align_frame, file_sha

SAFE_INPUTS = {'forecasts', 'solar_geometry', 'calendar', 'site_metadata'}
FIT_STAGES = ('training_cutoff', 'normalization_cutoff', 'auxiliary_cutoff',
              'outage_detection_cutoff', 'calibration_cutoff', 'selection_target_end')
STATE_PARTS = {'state01_lgb':'lgb_n75', 'state01_xgb':'xgb_n100',
               'state01_cloud':'cloud_l05', 'state01_all':'all_l08'}


def validate_protocol(metadata, prediction_path=None):
    """Check declared bounds, not certify that unseen training records are true.

    Original state01 main_all ends 7h too late and has global outage/aux bounds.
    Such predictions must NOT silently become eligible through row alignment.
    """
    if metadata.get('profile') != 'year_validation':
        raise ValueError('Only year_validation predictions are eligible; final/test models overlap the evaluation year')
    if metadata.get('protocol') != 'cutoff_safe_v1':
        raise ValueError('Require cutoff_safe_v1; original state01 main_all protocol needs correction')
    if metadata.get('timestamp_convention') != 'interval_end' or metadata.get('timezone') != 'Asia/Seoul':
        raise ValueError('Declare interval_end timestamps in Asia/Seoul')
    if metadata.get('units') not in ('kWh', 'capacity_factor'):
        raise ValueError('Declare kWh or capacity_factor; no unit guessing')
    for key, expected in [('prediction_start', START), ('prediction_end_exclusive', END)]:
        value = pd.Timestamp(metadata.get(key))
        if value.tzinfo is None or value != expected:
            raise ValueError(f'{key} differs from the common year')
    for key in FIT_STAGES:
        if key not in metadata:
            raise ValueError(f'Missing provenance bound: {key}; use null only for a documented unused stage')
        if metadata[key] is None and key in ('auxiliary_cutoff', 'calibration_cutoff', 'selection_target_end'):
            continue
        value = pd.Timestamp(metadata[key])
        if value.tzinfo is None or pd.isna(value) or value > CUTOFF:
            raise ValueError(f'{key} is missing or after 2024-05-31 17:00 KST')
    inputs = metadata.get('inference_inputs', [])
    if not inputs or not set(inputs) <= SAFE_INPUTS:
        raise ValueError('Inference inputs include unavailable targets/observed weather or are undeclared')
    controls = metadata.get('leakage_controls', {})
    for key in ('outages_fit_before_cutoff', 'aux_bounds_fit_before_cutoff', 'no_target_inputs'):
        if controls.get(key) is not True:
            raise ValueError(f'Unresolved leakage control: {key}')
    if prediction_path is not None and metadata.get('prediction_sha256') != file_sha(prediction_path):
        raise ValueError('Prediction hash differs from provenance; bind records to this exact file')
    return dict(status='declared_protocol_bounds_checked', independently_audited=False,
                caution='JSON declarations do not prove actual training provenance; inspect source/run records independently')


def load_sj(path, grid, config, metadata_path=None):
    cfg = config['profiles']['validation']
    if (pd.Timestamp(cfg['cutoff']) != CUTOFF or pd.Timestamp(cfg['start']) != START
            or pd.Timestamp(cfg['end_exclusive']) != END):
        raise ValueError('SJ config does not describe the common validation year')
    frame = read_table(path)
    if 'fold' in frame:
        frame = frame.loc[frame.fold == 'year_2024'].copy()
    columns = [c for c in ('lightgbm', 'xgboost', 'bigru') if c in frame]
    result = {f'sj_{name}': value for name, value in align_frame(frame, grid, columns).items()}
    weights = config['weights']
    if set(weights) != {'lightgbm','xgboost','bigru'} or not np.isclose(sum(weights.values()),1) or min(weights.values()) < 0:
        raise ValueError('Invalid SJ component weights')
    factor = float(cfg['factor'])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Invalid SJ validation calibration')
    if len(columns) == 3:
        result['sj_ensemble'] = sum(result[f'sj_{k}']*v for k,v in weights.items())*factor
        if 'forecast_kwh' in frame:
            saved=align_frame(frame,grid,['forecast_kwh'])['forecast_kwh']
            if not np.allclose(saved[grid.observed],result['sj_ensemble'][grid.observed],rtol=1e-6,atol=1e-4):
                raise ValueError('SJ saved ensemble does not match VALIDATION factor/weights')
    elif 'forecast_kwh' in frame:
        result['sj_ensemble'] = align_frame(frame,grid,['forecast_kwh'])['forecast_kwh']
    evidence = dict(path=str(path), sha256=file_sha(path), factor=factor, weights=weights,
                    cutoff_status='config_declaration_only; original training/calibration records still needed')
    if metadata_path:
        metadata=json.loads(Path(metadata_path).read_text())
        if metadata.get('units') != 'kWh':
            raise ValueError('SJ native component columns are kWh')
        evidence['protocol']=validate_protocol(metadata,path)
    return result, evidence


def load_state01_table(path, grid, metadata_path):
    metadata=json.loads(Path(metadata_path).read_text())
    evidence=validate_protocol(metadata,path)
    frame=read_table(path)
    column=metadata.get('prediction_column','kwh')
    mappings={'state01':column, **metadata.get('components',{})}
    if any(not re.fullmatch(r'state01_[A-Za-z0-9_]+', name) for name in mappings if name!='state01'):
        raise ValueError('Component names must start with state01_')
    values=align_frame(frame,grid,list(dict.fromkeys(mappings.values())))
    scale=grid.sites.capacity_kw.to_numpy()[None,:] if metadata['units']=='capacity_factor' else 1.
    result={name:grid.validate_prediction(values[c]*scale) for name,c in mappings.items()}
    return result, dict(path=str(path),sha256=file_sha(path),metadata_sha256=file_sha(metadata_path),protocol=evidence)


def load_state01_run(run, grid, metadata_path):
    """Read explicit main_all per-seed files, never average incomplete seeds.

    Require corrected protocol and per-file hashes in an external comparison
    provenance JSON; original main.py does not produce this safe protocol.
    """
    run=Path(run);metadata=json.loads(Path(metadata_path).read_text())
    evidence=validate_protocol(metadata)
    if metadata.get('units') != 'kWh':
        raise ValueError('Native state01 per-seed kwh files require units=kWh')
    seeds=metadata.get('seeds')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Declare unique integer seeds; no seed inference/partial renormalization')
    result={};files={}
    for name,part in STATE_PARTS.items():
        directory='gbdt' if name in ('state01_lgb','state01_xgb') else 'gru'
        arrays=[]
        for seed in seeds:
            relative=f'main_all/{directory}/pred_{part}_s{seed}.parquet'
            path=run/relative
            expected=metadata.get('prediction_files',{}).get(relative)
            if not path.is_file() or expected!=file_sha(path):
                raise ValueError(f'Missing seed file or hash: {relative}')
            # Each constituent independently covers every common observed row.
            value=align_frame(read_table(path),grid,['kwh'])['kwh']
            arrays.append(value);files[relative]=expected
        result[name]=np.mean(arrays,axis=0)
    result['state01']=.1*result['state01_lgb']+.1*result['state01_xgb']+.4*result['state01_cloud']+.4*result['state01_all']
    return result,dict(run=str(run),seeds=seeds,prediction_files=files,protocol=evidence,
                       metadata_sha256=file_sha(metadata_path),weights=dict(lgb=.1,xgb=.1,cloud=.4,all=.4))
