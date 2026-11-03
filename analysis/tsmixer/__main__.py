"""Run from repository root: python -m analysis.tsmixer --help."""
import argparse
import json
import platform
import sys
from pathlib import Path
import torch
from .config import require_root, load_config, resolve
from .runtime import device_for, environment, GPULock
from .train import run, write_json


def parser():
    p = argparse.ArgumentParser(description='OIBC TSMixer v0; Mac smoke only, explicit CUDA full training')
    sub = p.add_subparsers(dest='action', required=True)
    for name in ('environment', 'check-data', 'prepare-cache', 'smoke', 'train', 'evaluate'):
        q = sub.add_parser(name)
        q.add_argument('--config', default='configs/tsmixer/base.yaml')
        q.add_argument('--set', action='append', default=[], metavar='section.key=value')
        q.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
        if name in ('check-data', 'prepare-cache', 'smoke', 'train', 'evaluate'):
            q.add_argument('--data-root')
            q.add_argument('--cache-root')
        if name in ('smoke', 'train', 'evaluate', 'check-data', 'environment'):
            q.add_argument('--output')
        if name in ('check-data', 'prepare-cache'):
            q.add_argument('--checksums', action='store_true')
            q.add_argument('--reference')
        if name == 'prepare-cache':
            q.add_argument('--dry-run', action='store_true')
            q.add_argument('--cache-max-gib', type=float, default=2.)
        if name == 'smoke':
            q.add_argument('--steps', type=int, default=5)
            q.add_argument('--real-cache', action='store_true')
            q.add_argument('--resume')
        if name in ('smoke', 'train'):
            q.add_argument('--backup-root', help='Optional verified generation backup directory (e.g. mounted Drive)')
        if name == 'train':
            q.add_argument('--full-train', action='store_true')
            q.add_argument('--resume')
        if name == 'evaluate':
            q.add_argument('--checkpoint', required=True)
            q.add_argument('--export-csv', action='store_true', help='Export development site/group predictions; not test/OOF')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require_root()
    overrides = list(args.set)
    for name in ('data_root', 'cache_root'):
        if getattr(args, name, None):
            # Route path overrides directly, avoiding YAML interpretation of paths.
            overrides.append('data.'+name+'='+json.dumps(getattr(args, name)))
    c = load_config(args.config, overrides)
    if args.action == 'environment':
        result = environment(device_for(args.device or c['training']['device']))
    elif args.action in ('check-data', 'prepare-cache'):
        from .inventory import inspect_data
        data = resolve(c['data']['data_root'])
        cache = resolve(c['data']['cache_root'])
        if args.action == 'check-data':
            result = inspect_data(data, cache if args.cache_root else None,
                                  args.checksums, args.reference)
        else:
            device = device_for(args.device or c['training']['device'])
            if platform.system() != 'Linux' or device.type != 'cuda':
                raise ValueError('Cache generation is Linux/CUDA-server only; forbidden on Mac')
            inspect_data(data, checksums=args.checksums, reference=args.reference)
            from analysis.deep_learning.prepare import prepare
            if (cache/'train/manifest.json').exists():
                result = inspect_data(data, cache, checksums=args.checksums)
            else:
                # Raw source only; never silently reuse unrelated prepared Parquets.
                with GPULock(device):
                    prepare(data, cache, prepared_root=None, partitions=('train',), layout='compact',
                            cache_max_gib=args.cache_max_gib, dry_run=args.dry_run)
                result = {'cache': str(cache), 'dry_run': args.dry_run}
    elif args.action in ('smoke', 'train'):
        if not args.output:
            raise ValueError('--output must name a fresh run directory (or original resume directory)')
        if args.action == 'train':
            # Check guard BEFORE reading any real data or doing preprocessing.
            device = device_for(args.device or c['training']['device'])
            if not args.full_train or platform.system() != 'Linux' or device.type != 'cuda':
                raise ValueError('Full training requires Linux CUDA and --full-train; use smoke on Mac')
            from .inventory import inspect_data
            inspect_data(resolve(c['data']['data_root']), resolve(c['data']['cache_root']))
        try:
            result = run(c, args.output, args.device, smoke=args.action == 'smoke',
                         steps=getattr(args, 'steps', 5), real_cache=getattr(args, 'real_cache', False),
                         resume=args.resume, full_train=getattr(args, 'full_train', False), backup_root=args.backup_root)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError('CUDA OOM: choose a fresh run, lower training.batch_size and optionally '
                               'increase training.accumulation_steps. Inspect epochs.jsonl VRAM; no automatic retry.') from exc
        # run() already owns all output files.
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    else:
        from .checkpoint import load
        from .data import open_cache, DayDataset
        from .model import TSMixer
        from .evaluate import evaluate
        from .train import source_hashes
        from analysis.deep_learning.config import fingerprint
        device = device_for(args.device or c['training']['device'])
        if platform.system() != 'Linux' or device.type != 'cuda':
            raise ValueError('Full development evaluation runs on Linux CUDA; Mac uses bounded unit/smoke tests')
        ck = load(resolve(args.checkpoint))
        if ck['smoke_test']:
            raise ValueError('Smoke checkpoints cannot be evaluated as model performance')
        if ck['sources'] != source_hashes():
            raise ValueError('Source code differs from checkpoint; checkout its original commit')
        cache = open_cache(resolve(args.cache_root or ck['config']['data']['cache_root'])/'train')
        if cache.manifest['signature'] != ck['cache_signature']:
            raise ValueError('Cache signature differs from checkpoint')
        if ck['normalizer']['fit_cutoff'] != ck['plan']['cutoff']:
            raise ValueError('Normalizer cutoff differs from training plan')
        if not args.output:
            raise ValueError('Evaluation requires a fresh --output directory')
        output = resolve(args.output)
        if output.exists():
            raise FileExistsError('Use a fresh evaluation output directory')
        with GPULock(device):
            model = TSMixer(**ck['dimensions'], **ck['config']['model']).to(device)
            model.load_state_dict(ck['model'])
            dataset = DayDataset(cache, ck['normalizer'], ck['plan'], 'dev', False)
            training = dict(ck['config']['training'])
            if device.type != 'cuda':
                training.update(precision='fp32', num_workers=0)
            metrics, detail, groups, ledger = evaluate(model, dataset, training, device, output)
        detail.to_csv(output/'site_and_cohort_metrics.csv', index=False)
        groups.to_csv(output/'group_metrics.csv', index=False)
        if len(ledger):
            ledger.to_parquet(output/'settlement_ledger.parquet', index=False)
        result = {'split': 'dev', 'metrics': metrics, 'training_cutoff': ck['plan']['cutoff'],
                  'selection_end': ck['selection_end'], 'cache_signature': ck['cache_signature'],
                  'config_fingerprint': ck['config_fingerprint'], 'checkpoint': str(resolve(args.checkpoint)),
                  'checkpoint_git_commit': ck['environment']['git_commit'],
                  'environment': environment(device), 'holdout_used': False}
        from .persistence import sha256
        result['checkpoint_sha256'] = sha256(resolve(args.checkpoint))
        write_json(output/'metrics.json', result)
        if args.export_csv:
            from .colab import export_development
            export_development(output, cache, ck['plan'])
    if getattr(args, 'output', None) and args.action in ('environment', 'check-data'):
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        print(f'TSMixer error: {exc}', file=sys.stderr)
        sys.exit(2)
