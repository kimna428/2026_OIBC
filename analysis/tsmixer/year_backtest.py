"""Isolated, issuance-safe year experiment. Never calls the development split planner."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.deep_learning.features import aware, day_hours, fit_normalizer, apply_postprocess
from analysis.deep_learning.inference import dimensions
from analysis.deep_learning.models.common import model_inputs
from analysis.deep_learning.storage import result_guard
from analysis.validation import validate_composition, hourly_weights
from .config import load_config, resolve
from .data import open_cache, DayDataset
from .model import TSMixer
from .runtime import seed_everything, loader, portable_batch, GPULock, environment
from .train import train_epoch, write_json, source_hashes
from .checkpoint import save, capture_rng

REFERENCE = 223892580.02742994


def plan(phase):
    if phase not in ('selection', 'refit'):
        raise ValueError('Unknown backtest phase')
    return dict(phase=phase, cutoff=str(aware('2024-03-31 17:00' if phase == 'selection' else '2024-05-31 17:00')),
                dev_start=str(aware('2024-04-01' if phase == 'selection' else '2024-06-01')),
                dev_end=str(aware('2024-06-01' if phase == 'selection' else '2025-06-01')))


class PriceFreeCache:
    """Training does not read SMP/k. Prediction also cannot read target/energy."""
    def __init__(self, cache, blind=False):
        self.cache, self.blind = cache, blind
        self.smp = np.ones((len(cache.days), 24), dtype='float32')
        self.k = self.smp
        if blind:
            self.target = BlindTargets((len(cache.days), cache.num_sites, 24))
            self.energy = self.target
    def __getattr__(self, name):
        return getattr(self.cache, name)


class BlindTargets:
    def __init__(self, shape):
        self.shape = shape
    def __getitem__(self, key):
        # DayDataset requests a day and optionally a site; never materialize year.
        keys = key if isinstance(key, tuple) else (key,)
        return np.full(self.shape[1:], np.nan, dtype='float32')[keys[1:]]


def validate_cache(cache):
    if cache.manifest['kind'] != 'provided':
        raise ValueError('Backtest requires provided historical data')
    required = pd.date_range(aware('2024-03-31'), aware('2025-05-31'), freq='D').as_unit('ns')
    if not required.isin(cache.days).all() or cache.days.has_duplicates or not cache.days.is_monotonic_increasing:
        raise ValueError('Incomplete or unordered historical cache axis')
    audit = cache.root/'source_audit.jsonl'
    if not audit.is_file():
        raise ValueError('Cache lacks forecast issuance audit; prepare a fresh audited cache')
    rows = [json.loads(x) for x in audit.read_text().splitlines()]
    if (not rows or sum(r['days'] for r in rows) != len(cache.days) or
            any(r['asof_violations'] != 0 or r['rounds'] != [10, 17] for r in rows)):
        raise ValueError('Forecast issuance audit failed')
    return dict(layout=cache.manifest.get('layout', 'legacy'), signature=cache.manifest['signature'],
                source_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                reused_read_only=True, feature_signature=cache.manifest['feature_signature'])


def composition_hash(c):
    return hashlib.sha256(c[['site_id', 'group']].sort_values('site_id').to_csv(index=False).encode()).hexdigest()


def metadata_composition(sites, seed):
    """TSMixer experiment candidate: seeded capacity balance, metadata only.

    No validation actuals, residuals, forecast grids or prices enter this API.
    This candidate is NOT claimed to be an optimized performance result.
    """
    rng = np.random.default_rng(seed)
    order = sites.assign(tie=rng.random(len(sites))).sort_values(['capacity_kw', 'tie'], ascending=[False, True])
    loads = [0., 0.]; rows = []
    for row in order.itertuples():
        g = int(np.argmin(loads)); loads[g] += row.capacity_kw
        rows.append((row.site_id, f'G{g+1}'))
    candidate = pd.DataFrame(rows, columns=['site_id', 'group']).set_index('site_id').reindex(sites.site_id)
    # Fixed seeded local search: balance capacity AND weather-station exposure.
    # This is a metadata objective, never a backtest settlement objective.
    capacity = sites.capacity_kw.to_numpy()
    columns = [capacity / capacity.sum()]
    for provider in ('wa_station', 'wb_station'):
        if provider in sites:
            for station in sites[provider].dropna().unique():
                exposure = capacity * (sites[provider].to_numpy() == station)
                columns.append(exposure / exposure.sum())
    features = np.stack(columns, axis=1)
    membership = (candidate.group.to_numpy() == 'G2').astype(int)
    def objective(m):
        imbalance = features.T @ (2*m-1)
        return float(4*imbalance[0]**2 + np.mean(imbalance[1:]**2)) if len(imbalance)>1 else float(imbalance[0]**2)
    best = objective(membership)
    for _ in range(2000):
        a = int(rng.choice(np.flatnonzero(membership == 0)))
        b = int(rng.choice(np.flatnonzero(membership == 1)))
        proposed = membership.copy(); proposed[a], proposed[b] = 1, 0
        if min(capacity[proposed == g].sum() for g in (0,1)) < 20000:
            continue
        value = objective(proposed)
        if value < best:
            membership, best = proposed, value
    return pd.DataFrame({'site_id': sites.site_id, 'group': np.where(membership, 'G2', 'G1')})


def compositions(cache, teammate=None, optimize=False, seed=2026):
    result = {'baseline': cache.composition.copy()}
    if teammate:
        result['teammate'] = pd.read_csv(resolve(teammate), dtype={'site_id': str})[['site_id', 'group']]
    if optimize:
        result['tsmixer_metadata_candidate'] = metadata_composition(cache.sites, seed)
    for value in result.values():
        validate_composition(value, cache.sites)
    return result


def predict(model, dataset, training, device, path=None):
    shape = (len(dataset.day_indices), dataset.cache.num_sites, 24)
    grid = (np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=shape)
            if path else np.empty(shape, dtype='float32'))
    present = np.zeros(shape[:2], dtype=bool)
    mapping = {int(d): i for i, d in enumerate(dataset.day_indices)}
    model.eval()
    with torch.inference_mode():
        for batch in loader(dataset, False, training, torch.Generator().manual_seed(12345)):
            values = model(model_inputs(portable_batch(batch, device))).float().cpu().numpy()
            for j, (d, s) in enumerate(zip(batch['day_index'].tolist(), batch['site_pos'].tolist())):
                i = mapping[d]
                if present[i, s] or not np.isfinite(values[j]).all():
                    raise ValueError('Duplicate/nonfinite prediction')
                grid[i, s] = values[j]; present[i, s] = True
    if not present.all():
        raise ValueError('Incomplete prediction grid')
    for i, d in enumerate(dataset.day_indices):
        grid[i] = apply_postprocess(grid[i], dataset.cache, int(d), dataset.normalizer, training['postprocess'])
    if path:
        grid.flush()
    return grid


def score(cache, indices, predictions, assignments, output, reference=None):
    """Observed-only sums actual AND forecast over observed members, never zero targets."""
    output = Path(output)
    if (predictions.shape != (len(indices), cache.num_sites, 24) or
            not np.isfinite(predictions).all() or (predictions < 0).any()):
        raise ValueError('Require complete finite nonnegative canonical prediction grid')
    caps = cache.sites.capacity_kw.to_numpy()
    results = {}; site_rows = []
    errors = np.zeros(cache.num_sites); counts = np.zeros(cache.num_sites, dtype=int)
    times_all = pd.DatetimeIndex(np.concatenate([day_hours(cache.days[d]).to_numpy() for d in indices]))
    for label, comp in assignments.items():
        capacities = validate_composition(comp, cache.sites)
        group = comp.set_index('site_id').group.reindex(cache.sites.site_id).to_numpy()
        for mode in ('complete_fleet', 'observed_only'):
            rows = []; observed_rows = 0; valid_times = set(); mode_cf_error = 0.
            mask_digest = hashlib.sha256()
            for i, d in enumerate(indices):
                actual = np.asarray(cache.energy[d]); forecast = predictions[i]*caps[:, None]
                observed = np.isfinite(actual)
                mask_digest.update(observed.tobytes())
                times = day_hours(cache.days[d]); price = np.asarray(cache.smp[d], dtype=float)
                k = np.asarray(cache.k[d], dtype=float)
                if not np.isfinite(price).all() or (price < 0).any() or not np.array_equal(k, hourly_weights(times)):
                    raise ValueError('Invalid settlement SMP/k')
                complete = observed.all(axis=0)
                used = observed & (complete[None, :] if mode == 'complete_fleet' else True)
                observed_rows += int(used.sum())
                mode_cf_error += float(np.where(used, np.abs(forecast-actual)/caps[:, None], 0).sum())
                if label == 'baseline' and mode == 'observed_only':
                    errors += np.where(observed, np.abs(forecast-actual)/caps[:, None], 0).sum(axis=1)
                    counts += observed.sum(axis=1)
                for g in ('G1', 'G2'):
                    members = group == g
                    for h in np.flatnonzero(used[members].any(axis=0)):
                        selected = members & used[:, h]
                        a = float(actual[selected, h].sum()); p = float(forecast[selected, h].sum())
                        capacity = float(caps[selected].sum())
                        w = .1*price[h]*k[h]*abs(p-a); revenue = price[h]*a
                        valid_times.add(str(times[h]))
                        rows.append(dict(assignment=label, mode=mode, group=g, ts=str(times[h]),
                                         observed_rows=int(selected.sum()), capacity_kw=capacity,
                                         actual_kwh=a, pred_kwh=p, nmae_pct=100*abs(p-a)/capacity,
                                         penalty_krw=w, revenue_krw=revenue, settlement_krw=revenue-w))
            ledger = pd.DataFrame(rows)
            key = f'{label}/{mode}'
            w = float(ledger.penalty_krw.sum()) if rows else None
            contract = dict(composition_sha256=composition_hash(comp), mode=mode,
                            timestamps_sha256=hashlib.sha256('\n'.join(sorted(valid_times)).encode()).hexdigest(),
                            observed_mask_sha256=mask_digest.hexdigest(), observed_rows=observed_rows,
                            valid_hours=len(valid_times), group_hour_rows=len(rows),
                            start=str(times_all[0]), end=str(times_all[-1]),
                            observed_aggregation='sum_actual_and_prediction_over_observed_members',
                            nmae_denominator='observed_member_capacity', k='11..15 KST:3, otherwise:1')
            comparable = bool(reference and reference.get('penalty_reference') == REFERENCE and reference.get('contract') == contract)
            results[key] = dict(contract=contract, group_capacity_kw=capacities.to_dict(),
                                site_nmae_pct=100*mode_cf_error/observed_rows if observed_rows else None,
                                group_nmae_pct=float(ledger.nmae_pct.mean()) if rows else None,
                                W=w, S=float(ledger.settlement_krw.sum()) if rows else None,
                                revenue_krw=float(ledger.revenue_krw.sum()) if rows else None,
                                penalty_per_hour=w/len(valid_times) if valid_times else None,
                                internal_score=100*(1-w/REFERENCE) if w is not None else None,
                                comparable_internal_score=comparable,
                                comparison_status='verified_contract_match' if comparable else 'unverified_do_not_compare')
            if rows:
                ledger.to_csv(output/f'{label}_{mode}_hourly.csv', index=False)
                summary = ledger.groupby('group').agg(valid_hours=('ts','size'), observed_rows=('observed_rows','sum'),
                          nmae_pct=('nmae_pct','mean'), W=('penalty_krw','sum'), S=('settlement_krw','sum'))
                summary.to_csv(output/f'{label}_{mode}_groups.csv')
    for sid, err, n in zip(cache.sites.site_id, errors, counts):
        site_rows.append(dict(site_id=sid, observed_rows=int(n), nmae_pct=100*err/n if n else None))
    pd.DataFrame(site_rows).to_csv(output/'sites.csv', index=False)
    result = dict(assignments=results, site_nmae_pct=100*float(errors.sum()/counts.sum()) if counts.sum() else None,
                  observed_site_rows=int(counts.sum()), teammate_available='teammate' in assignments,
                  tsmixer_candidate_status='metadata_only_experimental_not_performance_optimized',
                  reference_penalty=REFERENCE)
    write_json(output/'metrics.json', result)
    return result


def fit(config, cache, output, phase, epochs=None):
    """Fresh seed/model/optimizer and cutoff-fitted statistics for EACH stage."""
    output.mkdir(exist_ok=False)
    p = plan(phase); t = config['training']; device = torch.device('cuda')
    seed_everything(t['seed'], t['deterministic'])
    view = PriceFreeCache(cache)
    normalizer = fit_normalizer(view, p['cutoff'])
    train = DayDataset(view, normalizer, p, 'train')
    dev = DayDataset(view, normalizer, p, 'dev') if phase == 'selection' else None
    model = TSMixer(**dimensions(cache), **config['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generator = torch.Generator().manual_seed(t['seed'])
    best, best_epoch, bad = float('inf'), None, 0
    for epoch in range(1, (epochs if epochs else t['epochs'])+1):
        result_guard(output, 16*1024**2)
        info = train_epoch(model, loader(train, True, t, generator), optimizer, scaler, config, device)
        value = None
        if dev:
            grid = predict(model, dev, t, device)
            total, count = 0., 0
            for i, d in enumerate(dev.day_indices):
                target = np.asarray(cache.target[d]); mask = np.isfinite(target)
                total += np.abs(grid[i]-target)[mask].sum(); count += mask.sum()
            value = float(total/count) if count else float('inf')
            if not np.isfinite(value):
                raise ValueError('No finite internal selection metric')
            if value < best-t['min_delta']:
                best, best_epoch, bad = value, epoch, 0
            else:
                bad += 1
        record = dict(epoch=epoch, **info, validation_site_cf_mae=value)
        with (output/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        print(json.dumps(dict(phase=phase, **record)), flush=True)
        state = dict(model=model.state_dict(), normalizer=normalizer, config=config, plan=p,
                     epoch=epoch, selected_epoch=best_epoch, optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), rng=capture_rng(), loader_rng=generator.get_state(),
                     cache_signature=cache.manifest['signature'], year_backtest=True)
        save(output/'last.pt', state)
        if dev and best_epoch == epoch:
            save(output/'best.pt', state)
        if dev and bad >= t['patience']:
            break
    return model, normalizer, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/tsmixer/year_backtest.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--teammate-composition')
    parser.add_argument('--reference-contract', help='Team-provided reference penalty and exact evaluation contract JSON')
    parser.add_argument('--metadata-candidate', action='store_true')
    parser.add_argument('--full-train', action='store_true')
    args = parser.parse_args()
    if not args.full_train or platform.system() != 'Linux' or not torch.cuda.is_available():
        parser.error('Full backtest requires explicit --full-train and Linux CUDA; Mac tests only')
    config = load_config(args.config)
    if config['training']['precision'] != 'fp32' or config['training']['selection_metric'] != 'site_cf_mae':
        raise ValueError('Backtest uses fp32 and price-free site_cf_mae selection')
    output = resolve(args.output)
    # Reserve a fresh directory atomically; never reuse any existing checkpoint directory.
    if not output.is_relative_to(resolve(config['training']['save_root'])/'year_backtest'):
        raise ValueError('Output must be under training.save_root/year_backtest')
    cache = open_cache(resolve(config['data']['cache_root'])/'train')
    audit = validate_cache(cache)
    # A cache starting shortly before the split loses earlier training history.
    from analysis.deep_learning.features import operating_day
    from analysis.deep_learning.prepare import balanced_composition
    prices = pd.read_csv(resolve(config['data']['data_root'])/'train/smp.csv', usecols=['ts'])
    expected = operating_day(pd.to_datetime(prices.ts)).unique()
    if not cache.days.equals(expected):
        raise ValueError('Require complete original historical day axis; partial caches are not reusable')
    if composition_hash(cache.composition) != composition_hash(balanced_composition(cache.sites)):
        raise ValueError('Cache baseline differs from original capacity-balanced assignment')
    groups = compositions(cache, args.teammate_composition, args.metadata_candidate, config['training']['seed'])
    reference = json.loads(resolve(args.reference_contract).read_text()) if args.reference_contract else None
    result_guard(output, 256*1024**2)
    output.mkdir(parents=True, exist_ok=False)
    for label, comp in groups.items():
        comp.to_csv(output/f'{label}_composition.csv', index=False)
    write_json(output/'run.json', dict(config=config, cache=audit, sources=source_hashes(), environment=environment(torch.device('cuda')),
               loss_policy='capacity weighted L1; SMP=k=1 without reading settlement arrays',
               selection_policy='internal observed site CF MAE', group_selection='metadata_only',
               plans={p:plan(p) for p in ('selection','refit')}))
    torch.set_num_threads(config['training']['cpu_threads'])
    with GPULock(torch.device('cuda')):
        _, _, selected = fit(config, cache, output/'selection', 'selection')
        write_json(output/'epoch_selection.json', dict(selected_epoch=selected, metric='site_cf_mae', plan=plan('selection')))
        model, normalizer, _ = fit(config, cache, output/'refit', 'refit', selected)
        dataset = DayDataset(PriceFreeCache(cache, blind=True), normalizer, plan('refit'), 'dev')
        grid = predict(model, dataset, config['training'], torch.device('cuda'), output/'predictions.npy')
        write_json(output/'prediction_axis.json', dict(site_ids=cache.sites.site_id.tolist(),
                   operating_days=[str(cache.days[d]) for d in dataset.day_indices], hours='01:00 through next-day 00:00 KST', units='capacity_factor'))
        metrics = score(cache, dataset.day_indices, grid, groups, output, reference)
    lines = ['# TSMixer 1년 백테스트 결과', '', '내부검증으로 Epoch 선택 후 새 모델로 재학습했습니다. SMP/k는 정산에만 사용했습니다.',
             'S는 발전량×SMP 합계에서 W를 차감한 정산금입니다. 원시 매출은 revenue_krw입니다.',
             '비교 가능 표시는 팀원의 원본 평가 계약과 완전히 일치할 때만 활성화됩니다.',
             '팀원 배정: '+('제공됨' if metrics['teammate_available'] else '미제공 — 결과 없음'),
             'TSMixer 전용 배정은 메타데이터 기반 실험 후보이며 성능 최적화 완료를 의미하지 않습니다.', '']
    for key, value in metrics['assignments'].items():
        lines.append(f"- {key}: W={value['W']}, S={value['S']}, 내부점수={value['internal_score']}, 비교가능={value['comparable_internal_score']}, 유효시간={value['contract']['valid_hours']}, 관측행={value['contract']['observed_rows']}")
    (output/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
