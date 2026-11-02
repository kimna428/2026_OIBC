"""Development-only streaming canonical predictions with existing postprocessing/scorer."""
from pathlib import Path
import numpy as np
import torch
from analysis.deep_learning.models.common import model_inputs
from analysis.deep_learning.features import apply_postprocess
from analysis.deep_learning.metrics import Evaluator
from .runtime import loader, portable_batch, amp_context


def canonical_day(values, positions, num_sites=253):
    """Reject partial/duplicate site batches instead of treating them as fleet sums."""
    values, positions = np.asarray(values), np.asarray(positions)
    if values.shape != (num_sites, 24) or positions.shape != (num_sites,):
        raise ValueError('Complete canonical site/day predictions required')
    if not np.array_equal(np.sort(positions), np.arange(num_sites)):
        raise ValueError('Duplicate, missing or unknown site positions')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite predictions')
    result = np.empty_like(values)
    result[positions] = values
    return result


def evaluate(model, dataset, training, device, output=None):
    """Holdout intentionally unavailable in v0 to keep it out of architecture selection."""
    if dataset.split != 'dev':
        raise ValueError('TSMixer v0 evaluation accepts development data only')
    model.eval()
    cache = dataset.cache
    evaluator = Evaluator(cache, dataset.normalizer['seen_hours'])
    pred = np.empty((cache.num_sites, 24), dtype='float32')
    present = np.zeros(cache.num_sites, dtype=bool)
    current = None
    grid = None
    mapping = {int(d): i for i, d in enumerate(dataset.day_indices)}
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        grid = np.lib.format.open_memmap(output/'predictions.npy', mode='w+', dtype='float32',
                                        shape=(len(mapping), cache.num_sites, 24))

    def finish(day):
        if not present.all():
            raise ValueError('Incomplete canonical prediction day')
        processed = apply_postprocess(canonical_day(pred, np.arange(cache.num_sites)),
                                     cache, day, dataset.normalizer, training['postprocess'])
        evaluator.add_day(day, processed)
        if grid is not None:
            grid[mapping[day]] = processed

    with torch.inference_mode():
        for batch in loader(dataset, False, training, torch.Generator().manual_seed(12345)):
            with amp_context(device, training['precision']):
                values = model(model_inputs(portable_batch(batch, device))).float().cpu().numpy()
            for j, day in enumerate(batch['day_index'].numpy()):
                day, site = int(day), int(batch['site_pos'][j])
                if current is not None and day != current:
                    finish(current)
                    present[:] = False
                current = day
                if present[site]:
                    raise ValueError('Duplicate site prediction')
                pred[site], present[site] = values[j], True
        if current is not None:
            finish(current)
    if grid is not None:
        grid.flush()
    return evaluator.finish()
