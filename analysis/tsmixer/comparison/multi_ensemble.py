"""Small convex grids; never choose weights with evaluation-year targets."""
import numpy as np
import pandas as pd
from analysis.validation import local_hours
from .core import Grid, CUTOFF, evaluate, mapping_hash

MODELS=('tsmixer','sj_ensemble','state01')


def weight_grid(step=.1):
    if step not in (.05,.1):
        raise ValueError('Use a small fixed grid step: .05 or .10')
    n=round(1/step)
    return [{MODELS[0]:i/n,MODELS[1]:j/n,MODELS[2]:(n-i-j)/n}
            for i in range(n+1) for j in range(n-i+1)]


def mix(predictions, weights):
    if (set(weights)-set(MODELS) or not weights or
            not all(np.isfinite(w) and w>=0 for w in weights.values()) or
            not np.isclose(sum(weights.values()),1.,atol=1e-12,rtol=0)):
        raise ValueError('Require finite nonnegative weights summing to 1')
    # Do not let 0*NaN contaminate an absent model; never renormalize positive weights.
    active=[(name,w) for name,w in weights.items() if w>0]
    if any(name not in predictions for name,w in active):
        raise ValueError('Positive-weight model unavailable')
    arrays=[np.asarray(predictions[name],dtype='float64') for name,w in active]
    if any(a.shape!=arrays[0].shape for a in arrays):
        raise ValueError('Model prediction shapes differ')
    return sum(a*w for a,(_,w) in zip(arrays,active))


def sweep(grid, predictions, assignments, step=.1, reference_verified=False):
    available={name:grid.validate_prediction(predictions[name]) for name in MODELS if name in predictions}
    plans=weight_grid(step);rows=[];unavailable=[]
    for weights in plans:
        active=[name for name,w in weights.items() if w>0]
        if any(name not in available for name in active):
            unavailable.append(dict(weights=weights,reason='Missing '+','.join(n for n in active if n not in available)))
            continue
        prediction=mix(available,weights)
        kind='single' if len(active)==1 else ('pair' if len(active)==2 else 'triple')
        for label,composition in assignments.items():
            result,_=evaluate(grid,prediction,composition,reference_verified=reference_verified)
            rows.append(dict(weights=weights,kind=kind,assignment=label,
                             use='exploratory_evaluation_year; no best-weight selection',**result))
    return dict(status='computed' if rows else 'unavailable',grid_step=step,
                candidate_weight_vectors=len(plans),available_models=list(available),
                trials=rows,unavailable=unavailable,independent_final_evaluation=False,
                weight_selection_performed=False)


def select_oof(frame, sites, composition, step=.1):
    """Choose weights ONLY on strictly pre-year, out-of-time predictions.

    Each *_trained_until must cover every constituent's normalizer, auxiliary,
    outage detection, selection and calibration. Declared records need review.
    """
    required={'site_id','ts','energy','smp','k',*MODELS,*(f'{name}_trained_until' for name in MODELS)}
    if not required<=set(frame):
        raise ValueError(f'OOF missing {sorted(required-set(frame))}')
    d=frame.copy();d['ts']=local_hours(pd.to_datetime(d.ts))
    if d.empty or d.duplicated(['site_id','ts']).any() or (d.ts>CUTOFF).any():
        raise ValueError('OOF empty, duplicate or includes evaluation-year targets')
    if not d.site_id.isin(sites.site_id).all() or not np.isfinite(d.energy).all():
        raise ValueError('Invalid OOF sites/actuals')
    issues=(pd.DatetimeIndex(d.ts)-pd.Timedelta(nanoseconds=1)).normalize()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    for name in MODELS:
        trained=pd.DatetimeIndex(pd.to_datetime(d[f'{name}_trained_until']))
        if trained.tz is None or trained.hasnans or (trained>issues).any():
            raise ValueError(f'{name} OOF cutoff is after issuance or unknown')
    times=pd.DatetimeIndex(sorted(d.ts.unique()))
    keys=pd.MultiIndex.from_product([times,sites.site_id],names=['ts','site_id'])
    aligned=d.set_index(['ts','site_id']).reindex(keys);shape=(len(times),len(sites))
    prices=d[['ts','smp','k']].drop_duplicates()
    if prices.ts.duplicated().any():
        raise ValueError('Conflicting OOF price rows')
    prices=prices.set_index('ts').reindex(times)
    grid=Grid(sites,times,aligned.energy.to_numpy().reshape(shape),prices.smp.to_numpy(),prices.k.to_numpy())
    predictions={name:aligned[name].to_numpy().reshape(shape) for name in MODELS}
    result=sweep(grid,predictions,{'selection_group':composition},step)
    valid=[r for r in result['trials'] if r['W'] is not None]
    if not valid:
        raise ValueError('OOF has no effective targets')
    selected=min(valid,key=lambda r:(r['W'],tuple(r['weights'][n] for n in MODELS)))
    return dict(selected_weights=selected['weights'],selection_W=selected['W'],grid_step=step,
                last_target=str(times[-1]),observed_rows=int(grid.observed.sum()),
                trials=[dict(weights=r['weights'],W=r['W']) for r in valid],
                provenance='declared row cutoffs checked; audit source records and fold-specific grouping separately',
                independent_final_evaluation=False)


def validate_oof_group(frame, composition, provenance):
    """Avoid choosing OOF weights using a group built from later-fold residuals."""
    if provenance.get('composition_sha256') != mapping_hash(composition):
        raise ValueError('OOF group provenance membership hash differs')
    basis=provenance.get('basis')
    if basis=='metadata_only':
        return dict(status='declared_metadata_only; source review still required')
    if basis!='historical_residuals':
        raise ValueError('Declare metadata_only or historical_residuals group basis')
    times=local_hours(pd.to_datetime(frame.ts))
    first_issue=(times-pd.Timedelta(nanoseconds=1)).normalize().min()-pd.Timedelta(days=1)+pd.Timedelta(hours=17)
    cutoff=pd.Timestamp(provenance.get('cutoff'))
    if cutoff.tzinfo is None or pd.isna(cutoff) or cutoff>first_issue:
        raise ValueError('OOF group selected using data after earliest OOF issuance')
    return dict(status='declared_group_cutoff_checked',cutoff=str(cutoff),first_issue=str(first_issue))
