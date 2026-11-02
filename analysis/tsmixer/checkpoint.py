"""Atomic, weights-only-loadable independent checkpoints; epoch-boundary resume."""
import os
from pathlib import Path
import torch
from analysis.deep_learning.checkpoint import capture_rng, restore_rng


def save(file, state):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + '.tmp')
    # Exclusive creation protects interrupted/other-process temporary files.
    with tmp.open('xb') as stream:
        torch.save({'tsmixer_format': 1, **state}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, file)


def load(file):
    state = torch.load(file, map_location='cpu', weights_only=True)
    if state.get('tsmixer_format') != 1:
        raise ValueError('Not a TSMixer v0 checkpoint')
    return state
