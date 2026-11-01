"""Fixed-mask OIBC evaluation; energy kWh, prices KRW/kWh, money KRW.

No fitting, imputation, target clipping, or forecast generation occurs here.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

TZ = 'Asia/Seoul'
TRAIN_END = pd.Timestamp('2025-01-01 00:00', tz=TZ)
DEV_END = pd.Timestamp('2025-04-01 00:00', tz=TZ)
HOLDOUT_END = pd.Timestamp('2025-06-01 00:00', tz=TZ)


def local_hours(values):
    """Accept aware timestamps (including UTC), reject naive or sub-hour keys."""
    try:
        t = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as e:
        raise ValueError('timestamps must have a consistent explicit timezone') from e
    if t.tz is None or t.hasnans:
        raise ValueError('timestamps must be non-null and timezone aware')
    t = t.tz_convert(TZ)
    if not (t == t.floor('h')).all():
        raise ValueError('timestamps must lie on hourly boundaries')
    return t


def split_labels(values):
    t = local_hours(values)
    if (t > HOLDOUT_END).any():
        raise ValueError('timestamp outside historical split horizon')
    return np.where(t <= TRAIN_END, 'train', np.where(t <= DEV_END, 'dev', 'holdout'))


def hourly_weights(values):
    t = local_hours(values)
    return np.where((t.hour >= 11) & (t.hour <= 15), 3, 1)


def keyed(frame, keys):
    d = frame.copy()
    d['ts'] = local_hours(d.ts)
    if d[keys].isna().any().any() or d.duplicated(keys).any():
        raise ValueError(f'null or duplicate keys: {keys}')
    return d


def validate_sites(sites):
    if len(sites) != 253 or sites.site_id.isna().any() or not sites.site_id.is_unique:
        raise ValueError('sites must contain exactly 253 unique non-null IDs')
    if not np.isfinite(sites.capacity_kw).all() or not (sites.capacity_kw > 0).all():
        raise ValueError('invalid capacity')


def validate_composition(composition, sites):
    validate_sites(sites)
    c = composition[['site_id', 'group']].copy()
    if (not c.site_id.is_unique or c.isna().any().any()
            or set(c.site_id) != set(sites.site_id) or set(c.group) != {'G1', 'G2'}):
        raise ValueError('G1/G2 must partition all 253 sites exactly once')
    caps = c.merge(sites[['site_id', 'capacity_kw']], validate='one_to_one').groupby('group').capacity_kw.sum()
    if (caps < 20000).any():
        raise ValueError('each group needs at least 20,000 kW')
    return caps


@dataclass
class EvaluationWindow:
    name: str
    actual: pd.DataFrame  # full requested grid, missing retained
    smp: pd.Series
    completeness: pd.DataFrame

    @property
    def times(self):
        return self.completeness.index[self.completeness.included]


def build_windows(generation, prices, sites):
    """Build experiment-independent masks from actuals only; reject broken SMP."""
    validate_sites(sites)
    g = keyed(generation, ['site_id', 'ts'])
    p = keyed(prices, ['ts']).sort_values('ts')
    if not g.site_id.isin(sites.site_id).all():
        raise ValueError('unknown generation site')
    if not np.isfinite(p.smp).all() or (p.smp < 0).any():
        raise ValueError('SMP must be finite and nonnegative')
    timeline = pd.DatetimeIndex(p.ts)
    if not timeline.equals(pd.date_range(timeline.min(), HOLDOUT_END, freq='h')):
        raise ValueError('historical SMP must be a contiguous hourly grid ending at holdout end')
    if not g.ts.isin(timeline).all():
        raise ValueError('generation outside historical SMP grid')
    split_labels(g.ts)  # validates every observed row against horizon
    wide = g.pivot(index='ts', columns='site_id', values='energy').reindex(index=timeline, columns=sorted(sites.site_id))
    wide = wide.where(np.isfinite(wide))
    labels = split_labels(timeline)
    windows = {}
    for name in ('train', 'dev', 'holdout'):
        w = wide.loc[labels == name].copy()
        observed = w.notna().sum(axis=1)
        detail = pd.DataFrame({'observed_sites': observed, 'missing_sites': 253-observed})
        detail['included'] = observed == 253
        detail['reason'] = np.where(detail.included, 'complete', 'missing_or_nonfinite_actual')
        windows[name] = EvaluationWindow(name, w, p.set_index('ts').smp.reindex(w.index), detail)
    return windows


def error_metrics(actual, pred):
    error = np.asarray(pred) - np.asarray(actual)
    return {'mae_kwh': float(np.mean(np.abs(error))) if len(error) else np.nan,
            'rmse_kwh': float(np.sqrt(np.mean(error**2))) if len(error) else np.nan}


def score_groups(predictions, composition, sites, window):
    """predictions: group, ts, pred_kwh. Require both groups at ALL fixed times.

    Extra times within this window are allowed and ignored by the fixed mask.
    Returns group metrics, summed score, and the exact scored hourly ledger.
    """
    caps = validate_composition(composition, sites)
    pred = keyed(predictions, ['group', 'ts'])
    if not pred.group.isin(['G1', 'G2']).all() or not pred.ts.isin(window.actual.index).all():
        raise ValueError('unknown group or prediction outside requested window')
    if not np.isfinite(pred.pred_kwh).all() or (pred.pred_kwh < 0).any():
        raise ValueError('group forecasts must be finite and nonnegative')
    if len(window.times) == 0:
        raise ValueError('no complete hours: settlement unavailable')
    rows = []
    for group in ('G1', 'G2'):
        ids = composition.loc[composition.group == group, 'site_id']
        actual = window.actual.loc[window.times, ids].sum(axis=1, min_count=len(ids))
        forecast = pred.loc[pred.group == group].set_index('ts').pred_kwh.reindex(window.times)
        if forecast.isna().any():
            raise ValueError('missing forecast on fixed settlement grid')
        smp = window.smp.reindex(window.times)
        rows.append(pd.DataFrame({'group': group, 'ts': window.times, 'actual_kwh': actual.to_numpy(),
                                  'pred_kwh': forecast.to_numpy(), 'smp': smp.to_numpy(), 'k': hourly_weights(window.times)}))
    ledger = pd.concat(rows, ignore_index=True)
    ledger['revenue_krw'] = ledger.smp * ledger.actual_kwh
    ledger['penalty_krw'] = .10 * ledger.smp * ledger.k * (ledger.pred_kwh-ledger.actual_kwh).abs()
    ledger['score_krw'] = ledger.revenue_krw-ledger.penalty_krw
    metrics = []
    for group, d in ledger.groupby('group'):
        metrics.append(dict(group=group, capacity_kw=float(caps[group]), used_hours=len(d),
                            **error_metrics(d.actual_kwh, d.pred_kwh),
                            **{c: float(d[c].sum()) for c in ('revenue_krw','penalty_krw','score_krw')}))
    result = pd.DataFrame(metrics).set_index('group')
    total = {c: float(result[c].sum()) for c in ('revenue_krw','penalty_krw','score_krw')}
    total.update(error_metrics(ledger.actual_kwh, ledger.pred_kwh))
    total.update(used_hours=len(window.times), excluded_hours=len(window.actual)-len(window.times), group_hour_rows=len(ledger))
    return result, total, ledger


def aggregate_site_predictions(predictions, composition, sites, window):
    """Require 253 finite site forecasts on every fixed settlement hour."""
    validate_composition(composition, sites)
    p = keyed(predictions, ['site_id', 'ts'])
    if not p.site_id.isin(sites.site_id).all() or not p.ts.isin(window.actual.index).all():
        raise ValueError('unknown site or timestamp outside window')
    if not np.isfinite(p.pred_kwh).all():
        raise ValueError('nonfinite site forecast')
    wide = p.pivot(index='ts', columns='site_id', values='pred_kwh').reindex(index=window.times, columns=window.actual.columns)
    if wide.isna().any().any():
        raise ValueError('missing site prediction on fixed settlement grid')
    rows = []
    for group in ('G1', 'G2'):
        ids = composition.loc[composition.group == group, 'site_id']
        rows.append(pd.DataFrame({'group':group, 'ts':window.times, 'pred_kwh':wide[ids].sum(axis=1, min_count=len(ids)).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def evaluate_sites(predictions, window):
    """Score only finite observed targets; require forecasts for every such row.

    No complete-fleet mask is applied here. Missing targets are never filled.
    """
    p = keyed(predictions, ['site_id', 'ts'])
    if not p.site_id.isin(window.actual.columns).all() or not p.ts.isin(window.actual.index).all():
        raise ValueError('unknown site or timestamp outside window')
    if not np.isfinite(p.pred_kwh).all():
        raise ValueError('nonfinite site forecast')
    actual = window.actual.rename_axis(index='ts', columns='site_id').stack().dropna().rename('actual_kwh').reset_index()
    d = actual.merge(p[['site_id','ts','pred_kwh']], on=['site_id','ts'], how='left', validate='one_to_one')
    if d.pred_kwh.isna().any():
        raise ValueError('missing forecast for observed site target')
    rows = [dict(site_id=sid, observed_rows=len(x), **error_metrics(x.actual_kwh,x.pred_kwh)) for sid,x in d.groupby('site_id')]
    result = pd.DataFrame(rows).set_index('site_id').reindex(window.actual.columns)
    result['observed_rows'] = result.observed_rows.fillna(0).astype(int)
    return result, dict(observed_rows=len(d), unobserved_rows=window.actual.size-len(d), **error_metrics(d.actual_kwh,d.pred_kwh))
