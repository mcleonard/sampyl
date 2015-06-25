""" Sort of a temporary place to write down various priors I build
    Will make this more formal later. The plan is to offer a lot of prior
    distributions to build logps with.
"""

import numpy as np


def uniform(x, lower=0., upper=1.):
    if x < lower:
        return np.inf
    elif x > upper:
        return np.inf
    else:
        return -np.log(upper-lower)


def poisson(events, lam):
    if lam <= 0:
        return -np.inf
    else:
        return np.sum(events*np.log(lam)) - events.size*lam
