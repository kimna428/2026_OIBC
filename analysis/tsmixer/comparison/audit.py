"""Bundle integrity and composition recovery, entirely read-only at source."""
import json
import hashlib
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.validation import validate_composition
from .core import mapping_hash
from .io import file_sha


def audit_manifest(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    present={};missing=[];mismatch=[]
    for relative,expected in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe manifest path')
        if not path.is_file():
            missing.append(relative);continue
        actual=file_sha(path);present[relative]=dict(expected=expected,actual=actual,bytes=path.stat().st_size)
        if expected!=actual:
            mismatch.append(relative)
    return dict(root=str(root.resolve()),manifest_sha256=file_sha(root/'manifest.json'),
                listed=len(manifest['files']),present=present,missing=missing,mismatch=mismatch,
                intact=not missing and not mismatch)


def recover_composition(repro,train_source,clustering,sites):
    repro,train_source,clustering=Path(repro),Path(train_source),Path(clustering)
    candidates={}
    # Explicit semantic locations only: never mistake unrelated composition files.
    for relative in ['reference/composition.csv','runs/predict_validation/composition.csv','runs/predict_final/composition.csv']:
        path=repro/relative
        if path.is_file():
            candidates[str(path)]=pd.read_csv(path)[['site_id','group']]
    cfg=json.loads((train_source/'config.json').read_text())
    if 'composition' in cfg:
        candidates[str(train_source/'config.json')]=pd.DataFrame(list(cfg['composition'].items()),columns=['site_id','group'])
    # Markdown is independent corroboration; never let it override a supplied CSV.
    if clustering.is_file():
        group=None;rows=[]
        for line in clustering.read_text().splitlines():
            match=re.match(r'## (G[12]) 전체 발전소 목록',line)
            if match: group=match.group(1)
            match=re.match(r'\| (PV\d{3}) \|',line)
            if group and match: rows.append((match.group(1),group))
        if rows:
            candidates[str(clustering)]=pd.DataFrame(rows,columns=['site_id','group'])
    if not candidates:
        raise FileNotFoundError('No original CSV, configuration assignment, or complete Markdown assignment')
    evidence={}
    for source,comp in candidates.items():
        capacities=validate_composition(comp,sites)
        evidence[source]=dict(source_sha256=file_sha(source),membership_sha256=mapping_hash(comp),
                              counts=comp.group.value_counts().sort_index().to_dict(),capacity_kw=capacities.to_dict())
    hashes={v['membership_sha256'] for v in evidence.values()}
    if len(hashes)!=1:
        raise ValueError('Composition sources disagree; do not silently choose a final grouping')
    chosen=next(iter(candidates))
    restored=candidates[chosen].sort_values('site_id').reset_index(drop=True)
    crlf_sha=hashlib.sha256(restored.to_csv(index=False,lineterminator='\r\n').encode()).hexdigest()
    manifest=json.loads((repro/'manifest.json').read_text()) if (repro/'manifest.json').is_file() else {}
    expected=manifest.get('files',{}).get('reference/composition.csv')
    return restored,dict(chosen=chosen,sources=evidence,
           all_memberships_match=True,original_csv_present=any(s.endswith('.csv') for s in candidates),
           recovered_crlf_sha256=crlf_sha,expected_reference_csv_sha256=expected,
           exact_reference_bytes_reconstructed=bool(expected and crlf_sha==expected))


def group_geometry(sites,composition):
    group=composition.set_index('site_id').group.reindex(sites.site_id).to_numpy()
    lat=np.deg2rad(sites.lat.to_numpy());lon=np.deg2rad(sites.lon.to_numpy())
    a=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)*np.sin((lon[:,None]-lon)/2)**2
    distance=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    upper=np.triu(np.ones(distance.shape,bool),1);separate=group[:,None]!=group
    result={}
    for label,mask in [('within_10km',distance<=10),('within_30km',distance<=30)]+[
        (f'same_{p}',sites[p].to_numpy()[:,None]==sites[p].to_numpy()) for p in ['wa_station','wb_station']]:
        selected=mask&upper
        result[label]=dict(pairs=int(selected.sum()),separation_pct=float(100*separate[selected].mean()) if selected.any() else None)
    return result


def archive_inventory(path):
    path=Path(path)
    if not path.is_file():return None
    with zipfile.ZipFile(path) as archive:
        return dict(path=str(path),sha256=file_sha(path),entries=[dict(path=i.filename,bytes=i.file_size) for i in archive.infolist()])
