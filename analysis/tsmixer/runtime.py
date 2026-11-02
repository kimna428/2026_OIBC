"""Portable execution. CUDA API calls are gated by the selected device."""
import contextlib
import platform
import subprocess
import sys
import torch
import numpy as np
from analysis.deep_learning.runtime import seed_everything, loader, GPULock


def device_for(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if platform.system() == 'Linux' and torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: install a driver-compatible CUDA PyTorch wheel on Linux')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable (PyTorch build, macOS, hardware or execution sandbox); use --device cpu')
    if requested not in ('cpu', 'mps', 'cuda'):
        raise ValueError('device must be auto|cpu|mps|cuda')
    return torch.device(requested)


def portable_batch(batch, device):
    # DayDataset carries float64 audit fields. MPS has no float64 support.
    # Preserve model/loss float32 values; leave unused audit tensors on CPU.
    used = {'weather', 'weather_missing', 'temporal', 'static', 'site_index',
            'site_pos', 'metadata_valid', 'target_cf', 'loss_mask', 'capacity_kw',
            'smp', 'k', 'group_complete'}
    return {k: (v.to(device=device, dtype=torch.float32 if v.dtype == torch.float64 else v.dtype)
                if k in used else v) for k, v in batch.items()}


def amp_context(device, precision):
    if precision == 'fp32':
        return contextlib.nullcontext()
    if device.type != 'cuda':
        raise ValueError('AMP is CUDA-only; CPU/MPS tests require fp32')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 is not supported by this CUDA device; use fp32 or fp16')
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def environment(device):
    def git(*args):
        p = subprocess.run(['git', *args], text=True, capture_output=True, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    result = {'python': sys.version, 'platform': platform.platform(),
              'machine': platform.machine(), 'numpy': str(np.__version__),
              'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(),
              'mps_built': torch.backends.mps.is_built(),
              'mps_available': torch.backends.mps.is_available(),
              'device': str(device), 'git_commit': git('rev-parse', 'HEAD'),
              'git_status': git('status', '--short')}
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_name=torch.cuda.get_device_name(), free_vram_bytes=free,
                      total_vram_bytes=total, capability=list(torch.cuda.get_device_capability()))
    return result
