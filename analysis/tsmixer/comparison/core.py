"""Canonical time x site evaluation with an actual-only, model-independent mask."""
import hashlib
import json
from dataclasses import dataclass
import numpy as np
import pandas as pd
from analysis.validation import validate_sites, validate_composition, local_hours, hourly_weights

START = pd.Timestamp('2024-06-01 01:00', tz='Asia/Seoul')
END = pd.Timestamp('2025-06-01 01:00', tz='Asia/Seoul')
CUTOFF = pd.Timestamp('2024-05-31 17:00', tz='Asia/Seoul')
REFERENCE = 223892580.02742994
ALPHAS = (0., .05, .10, .15, .20, .30, .40, .50)


def sha_array(a, dtype='<f8'):
    a = np.asarray(a, dtype=dtype).copy()
    if a.dtype.kind == 'f':
        a[np.isnan(a)] = np.nan  # canonical quiet NaN
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def mapping_hash(composition):
    return hashlib.sha256(composition[['site_id','group']].sort_values('site_id').to_csv(index=False).encode()).hexdigest()


@dataclass
class Grid:
    sites: pd.DataFrame
    times: pd.DatetimeIndex
    actual: np.ndarray
    smp: np.ndarray
    k: np.ndarray

    def __post_init__(self):
        validate_sites(self.sites)
        self.times = local_hours(self.times).as_unit('ns')
        if not self.times.is_monotonic_increasing or self.times.has_duplicates:
            raise ValueError('Time keys must be unique and increasing')
        self.actual = np.asarray(self.actual, dtype='float64')
        self.smp, self.k = np.asarray(self.smp, dtype='float64'), np.asarray(self.k, dtype='float64')
        if self.actual.shape != (len(self.times),len(self.sites)):
            raise ValueError('Actual shape/order mismatch')
        if np.isinf(self.actual).any():
            raise ValueError('Infinite actuals must be audited, not silently converted')
        if self.smp.shape != (len(self.times),) or not np.isfinite(self.smp).all() or (self.smp < 0).any():
            raise ValueError('Invalid SMP grid')
        if self.k.shape != self.smp.shape or not np.array_equal(self.k, hourly_weights(self.times)):
            raise ValueError('k differs from interval-end KST rule')

    @property
    def observed(self):
        return np.isfinite(self.actual)

    def validate_prediction(self, prediction):
        prediction = np.asarray(prediction, dtype='float64')
        if prediction.shape != self.actual.shape:
            raise ValueError('Prediction shape/order mismatch')
        if (np.isinf(prediction).any() or (prediction[np.isfinite(prediction)] < 0).any()
                or not np.isfinite(prediction[self.observed]).all()):
            raise ValueError('Require finite nonnegative prediction at EVERY observed target')
        return prediction

    def contract(self, composition, mode):
        validate_composition(composition,self.sites)
        if mode not in ('observed_only','complete_fleet'):
            raise ValueError('Unknown evaluation mode')
        return dict(schema='oibc_common_v1', mode=mode, start=str(self.times[0]), last=str(self.times[-1]),
                    time_sha256=sha_array(self.times.asi8,'<i8'),
                    site_order_sha256=hashlib.sha256('\n'.join(self.sites.site_id).encode()).hexdigest(),
                    actual_sha256=sha_array(self.actual), observed_mask_sha256=sha_array(self.observed,'u1'),
                    smp_sha256=sha_array(self.smp), k_sha256=sha_array(self.k),
                    composition_sha256=mapping_hash(composition),
                    site_nmae='mean(100*abs(pred-actual)/site_capacity) over used observed rows',
                    group_nmae='100*sum(abs(group_residual))/sum(observed_group_capacity)',
                    group_nmae_macro='mean(100*abs(group_residual)/observed_group_capacity)',
                    penalty='.1*SMP*k*abs(sum_observed(pred-actual))',
                    settlement='sum_observed(actual*SMP)-W', missing='no imputation; same observed members for actual and prediction')


def evaluate(grid, prediction, composition, mode='observed_only', reference_verified=False):
    """Primary group NMAE reproduces teammate pipeline.score's ratio of sums.

    Also retain TSMixer's macro mean for diagnosing formula-only differences.
    """
    prediction = grid.validate_prediction(prediction)
    contract = grid.contract(composition,mode)
    observed = grid.observed
    used = observed if mode=='observed_only' else observed & observed.all(axis=1)[:,None]
    count = int(used.sum())
    caps = grid.sites.capacity_kw.to_numpy(dtype='float64')
    residual = np.where(used, prediction-grid.actual, 0.)
    error = np.abs(residual)
    group = composition.set_index('site_id').group.reindex(grid.sites.site_id).to_numpy()
    rows = []
    for g in ('G1','G2'):
        members = group==g
        actual = np.where(used[:,members],grid.actual[:,members],0.).sum(axis=1)
        pred = np.where(used[:,members],prediction[:,members],0.).sum(axis=1)
        cap = (used[:,members]*caps[members]).sum(axis=1)
        n = used[:,members].sum(axis=1)
        valid = n>0
        e = pred-actual
        frame = pd.DataFrame(dict(group=g, ts=grid.times[valid], actual_kwh=actual[valid],
                     pred_kwh=pred[valid], residual_kwh=e[valid], abs_error_kwh=np.abs(e[valid]),
                     observed_capacity_kw=cap[valid], observed_rows=n[valid], smp=grid.smp[valid], k=grid.k[valid],
                     site_abs_error_kwh=error[:,members].sum(axis=1)[valid]))
        frame['penalty_krw'] = .1*frame.smp*frame.k*frame.abs_error_kwh
        frame['revenue_krw'] = frame.smp*frame.actual_kwh
        frame['settlement_krw'] = frame.revenue_krw-frame.penalty_krw
        frame['nmae_pct'] = 100*frame.abs_error_kwh/frame.observed_capacity_kw
        rows.append(frame)
    ledger = pd.concat(rows,ignore_index=True)
    valid_hours = int(used.any(axis=1).sum())
    w = float(ledger.penalty_krw.sum()) if len(ledger) else None
    result = dict(site_nmae_pct=100*float((error/caps).sum()/count) if count else None,
                  site_mae_kwh=float(error.sum()/count) if count else None,
                  group_nmae_pct=100*float(ledger.abs_error_kwh.sum()/ledger.observed_capacity_kw.sum()) if len(ledger) else None,
                  group_nmae_macro_pct=float(ledger.nmae_pct.mean()) if len(ledger) else None,
                  group_mae_kwh=float(ledger.abs_error_kwh.mean()) if len(ledger) else None,
                  W=w, S=float(ledger.settlement_krw.sum()) if len(ledger) else None,
                  revenue_krw=float(ledger.revenue_krw.sum()) if len(ledger) else None,
                  penalty_per_hour=w/valid_hours if valid_hours else None,
                  internal_score=100*(1-w/REFERENCE) if w is not None else None,
                  internal_score_status='reference_verified' if reference_verified and mode=='observed_only' else 'provisional',
                  valid_hours=valid_hours, observed_rows=count, group_hour_rows=len(ledger),
                  group_cancellation_fraction=1-float(ledger.abs_error_kwh.sum()/error.sum()) if error.sum() else None,
                  contract=contract)
    return result,ledger


def verify_reference(grid, baseline_prediction, original_composition):
    """Recompute the provided baseline's W on the SAME actuals, mask and clock.

    This verifies numeric denominator compatibility; training provenance is a
    separate audit. A config constant or another model's W is never sufficient.
    """
    result,_ = evaluate(grid,baseline_prediction,original_composition)
    return dict(verified=result['W'] is not None and abs(result['W']-REFERENCE)<=1.,
                expected_W=REFERENCE, recomputed_W=result['W'], tolerance_krw=1.,
                baseline_prediction_sha256=sha_array(baseline_prediction),contract=result['contract'])


def select_oof_alpha(frame, sites, composition):
    """Strict pre-cutoff out-of-time rows; never accept year-evaluation targets."""
    required={'site_id','ts','energy','teammate','tsmixer','teammate_trained_until','tsmixer_trained_until','smp','k'}
    if not required <= set(frame):
        raise ValueError(f'OOF needs {sorted(required)}')
    d=frame.copy(); d['ts']=local_hours(d.ts)
    days=(pd.DatetimeIndex(d.ts)-pd.Timedelta(nanoseconds=1)).normalize()
    issues=days-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    if (d.ts>CUTOFF).any() or d.duplicated(['site_id','ts']).any():
        raise ValueError('OOF contains future targets or duplicate keys')
    for column in ('teammate_trained_until','tsmixer_trained_until'):
        trained=pd.DatetimeIndex(pd.to_datetime(d[column]))
        if trained.tz is None or trained.hasnans or (trained>issues).any():
            raise ValueError('OOF model cutoffs missing or after issuance')
    if not d.site_id.isin(sites.site_id).all() or not np.isfinite(d.energy).all():
        raise ValueError('Invalid OOF targets/sites')
    time=pd.DatetimeIndex(sorted(d.ts.unique()))
    ix=pd.MultiIndex.from_product([time,sites.site_id],names=['ts','site_id'])
    aligned=d.set_index(['ts','site_id']).reindex(ix)
    prices=d[['ts','smp','k']].drop_duplicates()
    if prices.ts.duplicated().any():
        raise ValueError('Conflicting OOF settlement inputs')
    prices=prices.set_index('ts').reindex(time)
    shape=(len(time),len(sites))
    grid=Grid(sites,time,aligned.energy.to_numpy().reshape(shape),prices.smp.to_numpy(),prices.k.to_numpy())
    a=aligned.teammate.to_numpy().reshape(shape);b=aligned.tsmixer.to_numpy().reshape(shape)
    trials=[]
    for alpha in ALPHAS:
        value,_=evaluate(grid,(1-alpha)*a+alpha*b,composition)
        trials.append(dict(alpha=alpha,W=value['W']))
    if not any(t['W'] is not None for t in trials):
        raise ValueError('No OOF supervised rows')
    return dict(selected_alpha=min(trials,key=lambda t:(t['W'],t['alpha']))['alpha'],trials=trials,
                status='pre_cutoff_oof_selection; provenance of both constituent model cutoffs must be independently audited',
                last_target=str(time[-1]),rows=len(d))
