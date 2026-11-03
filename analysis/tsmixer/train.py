"""Independent AdamW loop. Full development training is explicit Linux/CUDA only."""
import json
import math
import platform
import hashlib
from pathlib import Path
import numpy as np
import torch
from analysis.deep_learning.config import fingerprint
from analysis.deep_learning.features import split_plan, fit_normalizer, aware
from analysis.deep_learning.models.common import model_inputs
from analysis.deep_learning.losses import compute_loss
from analysis.deep_learning.inference import dimensions
from .model import TSMixer
from .data import open_cache, DayDataset, SmokeCache
from .runtime import device_for, seed_everything, loader, portable_batch, amp_context, environment, GPULock
from .checkpoint import save, load, capture_rng, restore_rng
from .config import resolve
from .evaluate import evaluate


def write_json(file, value):
    Path(file).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    roots = [Path(__file__).parent, Path(__file__).parent.parent/'deep_learning']
    result = {}
    for root in roots:
        for file in sorted(root.rglob('*.py')):
            if 'tests' not in file.parts:
                result[str(file.relative_to(root.parent))] = hashlib.sha256(file.read_bytes()).hexdigest()
    from analysis import validation
    file = Path(validation.__file__)
    result['validation.py'] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def train_epoch(model, batches, optimizer, scaler, config, device, max_updates=None):
    """One complete epoch, or bounded smoke updates. Accumulate loss MASS exactly.

    weighted_l1 is a weighted mean: sum its numerator gradients over microbatches
    and divide by the total denominator before clipping/stepping. This avoids
    weighting small/missing-heavy microbatches like fully supervised ones.
    """
    t = config['training']
    model.train()
    optimizer.zero_grad(set_to_none=True)
    updates, micro, mass, numerator = 0, 0, 0., 0.
    total_num, total_mass = 0., 0.

    def step():
        nonlocal updates, micro, mass, numerator
        scaler.unscale_(optimizer)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(mass)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        micro, mass, numerator = 0, 0., 0.
        return float(norm)

    grad_norm = None
    for batch in batches:
        batch = portable_batch(batch, device)
        with amp_context(device, t['precision']):
            pred = model(model_inputs(batch))
        loss, info = compute_loss(pred, batch, config['loss'], config['group'], None)
        m = info['site_mass']
        if not m:
            continue
        # Normalize numerator magnitudes before backward to avoid huge fp16 grads.
        scale_mass = m / 1e6
        scaler.scale(loss * scale_mass).backward()
        mass += scale_mass
        numerator += float(loss.detach()) * m
        total_num += float(loss.detach()) * m
        total_mass += m
        micro += 1
        if micro == t['accumulation_steps']:
            grad_norm = step()
            if max_updates is not None and updates >= max_updates:
                break
    if micro and (max_updates is None or updates < max_updates):
        grad_norm = step()
    if not updates:
        raise ValueError('No effective supervised updates (check target masks and SMP)')
    return {'loss': total_num / total_mass, 'updates': updates,
            'site_mass': total_mass, 'gradient_norm': grad_norm}


def run(config, output, requested_device=None, smoke=False, steps=5,
        real_cache=False, resume=None, full_train=False, backup_root=None):
    device = device_for(requested_device or config['training']['device'])
    if not smoke and (not full_train or platform.system() != 'Linux' or device.type != 'cuda'):
        raise ValueError('Full training requires Linux, CUDA and explicit --full-train; Mac is smoke-only')
    if smoke and not 5 <= steps <= 10:
        raise ValueError('Smoke test is limited to 5..10 optimizer updates')
    if device.type != 'cuda' and config['training']['precision'] != 'fp32':
        raise ValueError('CPU/MPS requires training.precision=fp32')
    if device.type == 'mps' and config['training']['deterministic']:
        raise ValueError('MPS embedding backward has no deterministic implementation; '
                         'use explicit --set training.deterministic=false for optional MPS smoke, '
                         'or keep deterministic=true and use CPU')
    torch.set_num_threads(config['training']['cpu_threads'])
    with GPULock(device):
        return _run(config, resolve(output), device, smoke, steps, real_cache, resume,
                    resolve(backup_root) if backup_root else None)


def _run(config, output, device, smoke, steps, real_cache, resume, backup_root=None):
    t = config['training']
    old = load(resolve(resume)) if resume else None
    if output.exists() and any(output.iterdir()) and old is None:
        raise FileExistsError(f'Output exists; choose a fresh run directory: {output}')
    if old is not None and resolve(resume).parent != output:
        raise ValueError('Resume must use the original checkpoint run directory')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(t['seed'], t['deterministic'])
    cache_root = resolve(config['data']['cache_root'])
    if smoke and not real_cache:
        from analysis.deep_learning.smoke import synthetic_cache
        cache_root = output/'synthetic_cache'
        if not (cache_root/'train/manifest.json').is_file():
            if cache_root.exists() and any(cache_root.iterdir()):
                raise FileExistsError('Partial synthetic cache exists; restore into a fresh local directory')
            synthetic_cache(cache_root)
    cache = open_cache(cache_root/'train')
    if not smoke and cache.manifest['kind'] != 'provided':
        raise ValueError('Synthetic checkpoints/caches cannot be used for full development training')
    if not smoke:
        import pandas as pd
        from analysis.deep_learning.features import operating_day
        prices = pd.read_csv(resolve(config['data']['data_root'])/'train/smp.csv')
        expected = operating_day(pd.to_datetime(prices.ts)).unique()
        if not cache.days.equals(expected):
            raise ValueError('Full training requires the complete original historical day axis, not a partial cache')
    plan = split_plan(config, cache, 'development')
    view = SmokeCache(cache, plan['cutoff']) if smoke else cache
    normalizer = old['normalizer'] if old else fit_normalizer(view, plan['cutoff'])
    if normalizer['fit_cutoff'] != plan['cutoff'] or normalizer['cache_signature'] != cache.manifest['signature']:
        raise ValueError('Normalizer cutoff/cache signature mismatch')
    dims = dimensions(cache)
    provenance = {'config': config, 'plan': plan, 'cache_signature': cache.manifest['signature'],
                  'feature_signature': cache.manifest['feature_signature'], 'dimensions': dims,
                  'smoke_test': smoke, 'real_cache': real_cache,
                  'smoke_days': getattr(view, 'source_day_indices', None), 'sources': source_hashes()}
    signature = fingerprint(provenance)
    if old and (old['run_signature'] != signature or old['smoke_test'] != smoke):
        raise ValueError('Resume config/cache/source/smoke signature differs; use the original recipe')
    dataset = DayDataset(view, normalizer, plan, 'train', full_fleet=False)
    if smoke:
        # At most 128 (CUDA) or 32 (CPU/MPS) site/day items, never a fleet sum.
        from torch.utils.data import Subset
        chosen = np.linspace(0, len(dataset)-1,
                             min(128 if device.type == 'cuda' else 32, len(dataset)), dtype=int)
        dataset = Subset(dataset, chosen.tolist())
    dev = DayDataset(cache, normalizer, plan, 'dev', full_fleet=False) if not smoke else None
    local_training = dict(t)
    if smoke:
        local_training.update(batch_size=min(t['batch_size'], 4), num_workers=0,
                              persistent_workers=False)
        if device.type == 'cuda':
            local_training['batch_size'] = min(t['batch_size'], 128)
    execution = {**config, 'training': local_training}
    generator = torch.Generator().manual_seed(t['seed'])
    model = TSMixer(**dims, **config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    # The original v0 learning rate is constant. Persist a constant scheduler
    # explicitly without introducing decay or changing the optimization recipe.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and t['precision'] == 'fp16')
    env = environment(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start_epoch, global_step, best, bad = 0, 0, float('inf'), 0
    if old:
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        if old.get('scheduler') is None:
            raise ValueError('Checkpoint predates scheduler persistence; resume with its original source commit')
        scheduler.load_state_dict(old['scheduler'])
        for state in optimizer.state.values():
            for k, value in state.items():
                if isinstance(value, torch.Tensor) and k != 'step':
                    state[k] = value.to(device)
        scaler.load_state_dict(old['scaler'])
        generator.set_state(old['loader_rng'])
        restore_rng(old['rng'])
        if device.type == 'mps' and old.get('mps_rng') is not None:
            torch.mps.set_rng_state(old['mps_rng'])
        start_epoch, global_step = old['epoch'], old['global_step']
        best, bad = old['best'], old['bad_epochs']
        if old['environment']['device'] != str(device):
            raise ValueError('Exact resume requires the original device; evaluation may restore on CPU')
    if smoke and global_step >= steps:
        raise ValueError('Resume smoke --steps must exceed the saved optimizer update count')
    if not smoke and (start_epoch >= t['epochs'] or bad >= t['patience']):
        raise ValueError('This run already completed its epoch/early-stop recipe')
    write_json(output/'run.json', {**provenance, 'run_signature': signature,
                                  'config_fingerprint': fingerprint(config), 'environment': env,
                                  'parameters': sum(p.numel() for p in model.parameters()),
                                  'effective_training': local_training,
                                  'checkpoint': str(output/'last.pt'), 'full_training': not smoke})
    last_metrics = None
    # Synthetic/real smoke traverses at most 32 small samples repeatedly.
    remaining = steps-global_step if smoke else None
    epoch_limit = start_epoch + 10 if smoke else t['epochs']
    for epoch in range(start_epoch, epoch_limit):
        info = train_epoch(model, loader(dataset, True, local_training, generator), optimizer,
                           scaler, execution, device, max_updates=remaining)
        scheduler.step()
        global_step += info['updates']
        if smoke:
            remaining = steps-global_step
        metrics = None
        if dev is not None:
            metrics, _, _, _ = evaluate(model, dev, t, device)
            value = metrics[t['selection_metric']]
            if value is None or not math.isfinite(value):
                raise ValueError('Development selection metric unavailable; choose an explicit valid site metric')
        else:
            value = info['loss']
        improved = value < best-t['min_delta']
        best, bad = (value, 0) if improved else (best, bad+1)
        state = {**provenance, 'model': model.state_dict(), 'normalizer': normalizer,
                 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scheduler_name': 'constant',
                 'run_signature': signature, 'config_fingerprint': fingerprint(config),
                 'epoch': epoch+1, 'global_step': global_step, 'best': best, 'bad_epochs': bad,
                 'best_validation_score': best if not smoke else None,
                 'training_complete': (remaining <= 0 if smoke else
                                       epoch+1 >= t['epochs'] or bad >= t['patience']),
                 'loader_rng': generator.get_state(), 'rng': capture_rng(),
                 'mps_rng': torch.mps.get_rng_state() if device.type == 'mps' else None,
                 'environment': env, 'metrics': metrics, 'selection_end': plan['dev_end'],
                 'resume_boundary': 'epoch' if not smoke else 'smoke_only_nonexact_partial_epoch'}
        save(output/'last.pt', state)
        if improved:
            save(output/'best.pt', state)
        record = {'epoch': epoch+1, 'global_step': global_step, **info,
                  'development': metrics, 'smoke_test': smoke,
                  'learning_rate': optimizer.param_groups[0]['lr']}
        if device.type == 'cuda':
            record.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        with (output/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(record, allow_nan=False), flush=True)
        if backup_root:
            from .persistence import backup_run
            backup_run(output, backup_root)
        last_metrics = record
        if (smoke and remaining <= 0) or (not smoke and bad >= t['patience']):
            break
    result = {'smoke_test': smoke, 'global_step': global_step,
              'checkpoint': str(output/'last.pt'), 'best_checkpoint': str(output/'best.pt'),
              'device': str(device), 'last_epoch': last_metrics}
    # Restore on CPU and compare a small normalized sample on the original device.
    saved = load(output/'last.pt')
    restored = TSMixer(**dims, **config['model']).to(device)
    restored.load_state_dict(saved['model'])
    model.eval(); restored.eval()
    sample = next(iter(loader(dataset, False, local_training, torch.Generator().manual_seed(7))))
    inputs = model_inputs(portable_batch(sample, device))
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=1e-5, atol=1e-6)
    result['checkpoint_reload_matches'] = True
    write_json(output/'summary.json', result)
    if backup_root:
        from .persistence import backup_run
        backup_run(output, backup_root)
    return result
