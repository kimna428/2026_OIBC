"""Atomic tensor/state-dict checkpoints; safe weights-only loads, complete RNG state."""
import os, random
from pathlib import Path
import numpy as np
import torch

def capture_rng():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']); n=state['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype='uint32'),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(filename,state):
    filename=Path(filename); filename.parent.mkdir(parents=True,exist_ok=True)
    from .storage import result_guard
    def tensor_bytes(value):
        if isinstance(value,torch.Tensor): return value.numel()*value.element_size()
        if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value,(list,tuple)): return sum(tensor_bytes(v) for v in value)
        return 0
    # Atomic replacement requires space for old last/best AND one new temporary file.
    result_guard(filename,tensor_bytes(state)+8*1024**2)
    temporary=filename.with_suffix(filename.suffix+'.tmp')
    if temporary.exists(): raise FileExistsError(f'Interrupted checkpoint temporary exists: {temporary}; inspect your own file before retrying')
    try:
        torch.save(state,temporary)
        result_guard(filename,0)
        os.replace(temporary,filename)
    finally:
        if temporary.exists(): temporary.unlink()

def load_checkpoint(filename):
    state=torch.load(filename,map_location='cpu',weights_only=True)
    if state.get('format_version')!=1: raise ValueError('Unsupported checkpoint schema')
    return state
