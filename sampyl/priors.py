""" Sort of a temporary place to write down various priors I build
    Will make this more formal later. The plan is to offer a lot of prior
    distributions to build logps with.
"""
from .core import np
from functools import partial

OUTOFBOUNDS = -np.inf


def bound(f, *conditions):
    for each in conditions:
        if not np.all(each):
            return OUTOFBOUNDS
    else:
        return f


def prior_map(func, arr, **kwargs):
    f = partial(func, **kwargs)
    return np.apply_along_axis(f, axis=1, arr=arr[:, None])


def uniform(x, lower=0., upper=1.):
    logp = lambda lower, upper: -np.log(upper-lower)
    out = bound(logp(lower, upper), lower < x < upper)
    return out


def poisson(events, lam):
    logp = lambda events, lam: np.sum(events*np.log(lam)) - events.size*lam
    out = bound(logp(events, lam), lam > 0)
    return out
