import contextlib, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

class GPULock:
    def __init__(self,device): self.device=device; self.file=None
    def __enter__(self):
        if self.device.type=='cuda':
            import fcntl
            self.file=open(os.environ.get('OIBC_GPU_LOCK','/tmp/oibc-single-gpu.lock'),'a+')
            try: fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: self.file.close(); raise RuntimeError('Another OIBC GPU command is running; run models sequentially')
        return self
    def __exit__(self,*args):
        if self.file:
            import fcntl
            fcntl.flock(self.file,fcntl.LOCK_UN); self.file.close()

def seed_everything(seed,deterministic=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=deterministic
    torch.use_deterministic_algorithms(deterministic)

def worker_init(worker):
    seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)

def device_for(requested):
    if requested.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Install a CUDA-enabled PyTorch wheel on the server; local smoke tests require --device cpu --max-steps 1 or 2.')
    if requested not in ['cpu','cuda','cuda:0']: raise ValueError('Single CPU or CUDA GPU 0 supported')
    return torch.device(requested)

def amp_settings(device,precision):
    if device.type!='cuda' or precision=='fp32': return None
    if precision=='bf16' and not torch.cuda.is_bf16_supported(): return torch.float16
    if precision=='auto': return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision not in ['bf16','fp16']: raise ValueError('precision must be auto/bf16/fp16/fp32')
    return torch.bfloat16 if precision=='bf16' else torch.float16

def autocast(device,dtype): return torch.autocast(device_type=device.type,dtype=dtype) if dtype else contextlib.nullcontext()
def to_device(batch,device): return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

def loader(dataset,training,config,generator=None):
    workers=config['num_workers']
    return DataLoader(dataset,batch_size=config['batch_size'],shuffle=training,num_workers=workers,pin_memory=config.get('pin_memory',True) and torch.cuda.is_available(),persistent_workers=workers>0 and config.get('persistent_workers',True),worker_init_fn=worker_init,generator=generator,drop_last=False,multiprocessing_context='spawn' if workers>0 else None)

def append_log(file,value):
    from .storage import result_guard
    text=json.dumps(value,allow_nan=False)+'\n'
    if Path(file).exists() and Path(file).stat().st_size+len(text.encode())>64*1024**2: raise OSError(f'Log size limit (64 MiB): {file}')
    result_guard(file,len(text.encode()))
    with Path(file).open('a') as f: f.write(text)
