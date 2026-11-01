"""Conservative application budgets; never remove other files or follow directory symlinks."""
import os, shutil
from pathlib import Path
from .config import ROOT
GIB=1024**3

def tree_bytes(root):
    root=Path(root)
    if not root.exists(): return 0
    if root.is_file(): return root.stat().st_size
    total=0; seen=set()
    for directory,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
        for name in files:
            p=Path(directory)/name
            if p.is_symlink(): continue
            s=p.stat(); key=(s.st_dev,s.st_ino)
            if key not in seen: total+=s.st_size; seen.add(key)
    return total

def existing_parent(p):
    p=Path(p).resolve()
    while not p.exists(): p=p.parent
    return p

def guard(destination,additional=0,*,budget_root=None,limit=None):
    """Check before allocation. Limits are logical bytes, deliberately conservative for sparse files."""
    destination=Path(destination).resolve(); additional=max(0,int(additional))
    personal=Path(os.environ.get('OIBC_STORAGE_ROOT',str(ROOT))).expanduser().resolve()
    if not destination.is_relative_to(personal): raise ValueError(f'{destination} is outside OIBC_STORAGE_ROOT={personal}; set it to your own quota directory, never a shared parent')
    reserve=int(float(os.environ.get('OIBC_MIN_FREE_GIB','0.5'))*GIB)
    quota=int(float(os.environ.get('OIBC_QUOTA_GIB','12'))*GIB)
    used=tree_bytes(personal); outside=int(float(os.environ.get('OIBC_EXTERNAL_USED_GIB','0'))*GIB)
    if outside<0 or additional<0: raise ValueError('Storage accounting must be nonnegative')
    used+=outside; free=shutil.disk_usage(existing_parent(destination)).free
    if used+additional+reserve>quota or free<additional+reserve:
        raise OSError(f'Insufficient storage before write: personal_used={used}, additional={additional}, reserve={reserve}, quota={quota}, filesystem_free={free}. No files deleted. Check personal quota with your server administrator too.')
    if budget_root is not None:
        root=Path(budget_root).resolve(); used_budget=tree_bytes(root)
        if used_budget+additional>limit: raise OSError(f'Storage budget exceeded: {root}: {used_budget}+{additional}>{limit}; no files deleted')
    return {'personal_used_bytes':used,'filesystem_free_bytes':free,'quota_bytes':quota,'reserve_bytes':reserve}

def result_guard(destination,additional=0):
    # All result paths must be below the configured results root, so experiments cannot evade the cap.
    root=Path(os.environ.get('OIBC_RUN_ROOT',str(ROOT/'runs/deep_learning'))).expanduser().resolve()
    p=Path(destination).resolve()
    # Explicit unit/smoke temp paths retain a local per-directory cap, inside personal storage.
    if not p.is_relative_to(root): root=p.parent
    limit=int(float(os.environ.get('OIBC_RESULTS_MAX_GIB','4'))*GIB)
    return guard(p,additional,budget_root=root,limit=limit)|{'results_root':str(root),'results_limit_bytes':limit}


def main():
    import argparse,json
    p=argparse.ArgumentParser(); p.add_argument('--additional-gib',type=float,default=0); args=p.parse_args()
    root=Path(os.environ.get('OIBC_STORAGE_ROOT',str(ROOT))).expanduser().resolve()
    print(json.dumps({'storage_root':str(root),**guard(root,int(args.additional_gib*GIB))},indent=2))
if __name__=='__main__': main()
