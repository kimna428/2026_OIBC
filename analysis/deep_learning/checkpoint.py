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
    temporary=filename.with_suffix(filename.suffix+'.tmp'); torch.save(state,temporary); os.replace(temporary,filename)

def load_checkpoint(filename):
    state=torch.load(filename,map_location='cpu',weights_only=True)
    if state.get('format_version')!=1: raise ValueError('Unsupported checkpoint schema')
    return state
