import copy
import os
from pathlib import Path
import yaml
from analysis.deep_learning.config import expand,fingerprint,ROOT
from analysis.tsmixer.config import load_config,resolve


def load(file='configs/tsmixer/ensemble_walkforward.yaml'):
    c=expand(yaml.safe_load(resolve(file).read_text()))
    if c['version']!='past_only_walkforward_v1' or c['sj_calibration']!='none':raise ValueError('Do not carry SJ year-specific calibration to folds')
    if c['grid_step'] not in (.1,.05) or abs(c['joint_site_weight']+c['joint_penalty_weight']-1)>1e-12:raise ValueError('Invalid predeclared objective/grid')
    for n in ('max_epochs','patience','state_batch_days','sj_batch_size','cpu_threads'):
        if type(c[n]) is not int or c[n]<1:raise ValueError(f'{n} must be positive integer')
    if type(c['seed']) is not int or c['seed']<0:raise ValueError('Nonnegative integer seed required')
    if c['selection_policy']!='macro_mean_site_nmae_then_worst_fold':raise ValueError('Unknown selection policy')
    run=resolve(c['run_root'])
    if run.is_relative_to(ROOT) and not run.is_relative_to(ROOT/'runs/tsmixer/ensemble'):raise ValueError('Generated repository artifacts must remain under ignored runs/tsmixer/ensemble')
    for key in ('data_root','cache_root','legacy_tsmixer'):
        protected=resolve(c[key])
        if run.is_relative_to(protected) or protected.is_relative_to(run):raise ValueError('Run root overlaps protected source/cache/legacy model')
    budget=Path(os.environ.get('OIBC_RUN_ROOT',str(run))).expanduser().resolve()
    if not run.is_relative_to(budget):budget=run
    os.environ['OIBC_RUN_ROOT']=str(budget)
    return c


def model_config(c,family):
    result=copy.deepcopy(load_config(c['base_config']))
    result['training'].update(seed=c['seed'],epochs=c['max_epochs'],patience=c['patience'],num_workers=0,
        cpu_threads=c['cpu_threads'],state_batch_days=c['state_batch_days'],precision='fp32',selection_metric='site_cf_mae')
    if family=='sj_ensemble':result['training']['batch_size']=c['sj_batch_size']
    return result
