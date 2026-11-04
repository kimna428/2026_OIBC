import numpy as np
from analysis.tsmixer.comparison.multi_ensemble import mix,weight_grid,MODELS


def candidates(step=.1):
    result=weight_grid(step)
    for weights in [dict(zip(MODELS,[1/3]*3)),dict(zip(MODELS,[.1,.8,.1]))]:
        if not any(all(np.isclose(w[n],weights[n]) for n in MODELS) for w in result):result.append(weights)
    return result


def ablations(weights):
    result={}
    for drop in MODELS:
        kept={n:w for n,w in weights.items() if n!=drop};mass=sum(kept.values())
        if mass>0:result[drop]={**{n:w/mass for n,w in kept.items()},drop:0.}
    return result
