""" Sort of a temporary place to write down various priors I build
    Will make this more formal later. The plan is to offer a lot of prior
    distributions to build logps with.
"""
from .core import np

OUTOFBOUNDS = -np.inf


def uniform(x, lower=0., upper=1.):
    if x < lower:
        return OUTOFBOUNDS
    elif x > upper:
        return OUTOFBOUNDS
    else:
        return -np.log(upper-lower)


def poisson(events, lam):
    if lam <= 0:
        return OUTOFBOUNDS
    else:
        return np.sum(events*np.log(lam)) - events.size*lam
