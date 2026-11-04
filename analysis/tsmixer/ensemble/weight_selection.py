"""Selection API cannot inspect F3; frozen artifacts bind F1/F2 inputs."""
import json
from pathlib import Path
import pandas as pd
from .folds import FOLDS
import numpy as np
from analysis.deep_learning.config import fingerprint
from .blend import MODELS
from .persistence import atomic_json


def select(tables,folds,strategy='site',joint=(.8,.2)):
    if not folds or len(folds)!=len(set(folds)) or set(folds) not in ({'F1'},{'F1','F2'}) or set(tables)!=set(folds):raise ValueError('Weight selection accepts only explicit F1/F2; F3 forbidden')
    folds=sorted(folds)
    if strategy not in ('site','penalty','joint'):raise ValueError('Unknown selection strategy')
    if abs(sum(joint)-1)>1e-12 or min(joint)<0:raise ValueError('Predeclare nonnegative objective weights')
    # Candidate must be available at every selection fold. Never impute a model.
    ends=[]
    for fold in folds:
        bound=FOLDS['F2'].cutoff if fold=='F1' else FOLDS['F3'].cutoff
        for row in tables[fold]:
            end=pd.Timestamp(row.get('target_end'))
            if pd.isna(end) or end.tzinfo is None or end>bound:raise ValueError('Weight-selection labels extend beyond next fold cutoff')
            ends.append(end)
    keyed=[]
    for fold in folds:
        records={tuple(r['weights'][n] for n in MODELS):r for r in tables[fold] if r.get('site_nmae_pct') is not None and r.get('W') is not None}
        if not records:raise ValueError(f'No available OOF candidates in {fold}')
        keyed.append(records)
    common=set(keyed[0]).intersection(*(set(k) for k in keyed[1:]))
    if not common:raise ValueError('No consistent candidate across selection folds')
    # Fixed equal-weight reference when available; otherwise per-fold SJ single.
    denominators=[]
    for records in keyed:
        reference=next((r for k,r in records.items() if np.allclose(k,[1/3]*3)),records.get((0.,1.,0.)))
        if strategy=='joint' and reference is None:raise ValueError('Joint objective requires equal or SJ reference in every fold')
        denominators.append(reference)
    scored=[]
    for weights in sorted(common):
        rows=[records[weights] for records in keyed]
        site=float(np.mean([r['site_nmae_pct'] for r in rows]));worst=max(r['site_nmae_pct'] for r in rows)
        if strategy=='site':objective=site
        elif strategy=='penalty':objective=float(np.mean([r['W']/r['valid_hours'] for r in rows]))
        else:objective=float(np.mean([joint[0]*r['site_nmae_pct']/max(ref['site_nmae_pct'],1e-12)+joint[1]*r['W']/max(ref['W'],1e-12) for r,ref in zip(rows,denominators)]))
        scored.append(dict(weights=dict(zip(MODELS,weights)),objective=objective,mean_site_nmae_pct=site,worst_site_nmae_pct=worst))
    winner=min(scored,key=lambda r:(r['objective'],r['worst_site_nmae_pct'],tuple(r['weights'][n] for n in MODELS)))
    return dict(**winner,strategy=strategy,selection_folds=list(folds),objective_coefficients=list(joint),
                ranking=scored,selection_data_end=str(max(ends)),F3_used=False,selection_unit='macro mean across folds; not pooled years')


def freeze(path,selection,input_hashes):
    path=Path(path)
    if path.exists():raise FileExistsError('Frozen weights are immutable; choose a new experiment')
    if selection['F3_used'] or set(selection['selection_folds'])-{'F1','F2'}:raise ValueError('F3 cannot select weights')
    record=dict(selection=selection,input_hashes=input_hashes)
    record['signature']=fingerprint(record);atomic_json(path,record)
    return record


def read_frozen(path):
    value=json.loads(Path(path).read_text());signature=value.pop('signature')
    if fingerprint(value)!=signature or value['selection']['F3_used'] or set(value['selection']['selection_folds'])-{'F1','F2'}:raise ValueError('Invalid frozen selection/F3 contamination')
    selected=value['selection']
    if selected['selection_folds']:
        end=pd.Timestamp(selected.get('selection_data_end'))
        last=max(selected['selection_folds'])
        bound=FOLDS['F2'].cutoff if last=='F1' else FOLDS['F3'].cutoff
        if pd.isna(end) or end.tzinfo is None or end>bound:raise ValueError('Frozen weights use late selection labels')
    return {**value,'signature':signature}
