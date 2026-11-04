import json
from pathlib import Path
import torch
from analysis.deep_learning.config import fingerprint
from analysis.tsmixer.checkpoint import load
from analysis.tsmixer.comparison.io import file_sha,load_tsmixer
from .persistence import atomic_json


def legacy_assessment(root,fold):
    root=Path(root)
    if fold.name!='F3' or not (root/'run.json').is_file():return dict(status='unavailable')
    run=json.loads((root/'run.json').read_text())
    return dict(status='legacy_reference_not_strict_OOF',root=str(root),
        reason='Internal selection included May31 ending hours18..24 beyond the required17:00 cutoff',
        recorded_selection_end=run['plans']['selection']['dev_end'],
        refit_candidate=(root/'refit/last.pt').is_file(),
        predicted=(root/'predictions.npy').is_file(),run_sha256=file_sha(root/'run.json'))


def reuse_refit(root,cache,config,stats,epoch,fold,device,output):
    """Only AFTER new strict inner selection; original source weights stay read-only."""
    root=Path(root)
    if fold.name!='F3' or not (root/'refit/last.pt').is_file():return None
    old=load(root/'refit/last.pt')
    if (old['epoch']!=epoch or old['plan']['cutoff']!=str(fold.cutoff) or old['cache_signature']!=cache.manifest['signature']
        or fingerprint(old['normalizer'])!=fingerprint(stats) or old['config']['model']!=config['model']
        or old['config']['loss']!=config['loss']):return None
    for key in ('seed','learning_rate','weight_decay','batch_size','accumulation_steps','deterministic','gradient_clip','precision','postprocess'):
        if old['config']['training'][key]!=config['training'][key]:return None
    run=json.loads((root/'run.json').read_text())
    for name,expected in run['sources'].items():
        source=Path(__file__).parents[2]/name
        if not source.is_file() or expected!=file_sha(source):return None
    from analysis.tsmixer.model import TSMixer
    from analysis.deep_learning.inference import dimensions
    model=TSMixer(**dimensions(cache),**config['model']).to(device);model.load_state_dict(old['model'])
    record=dict(path=str(root/'refit/last.pt'),sha256=file_sha(root/'refit/last.pt'),epoch=epoch,
                condition='New strict inner Epoch equals original fixed-refit Epoch; normalizer/config/cache/source match')
    atomic_json(Path(output)/'reused_refit.json',record)
    return model,stats
