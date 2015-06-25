import autograd.numpy as np
from autograd import grad


def grad_logp(dlogp, x):
    """ dlogp should be a list of gradient logps, respective to each
        paramter in x
    """
    return np.array([each(*x) for each in dlogp])


def default_start(start, logp):
    """ If start is None, return a zeros array with length equal to the number
        of arguments in logp
    """
    if start is None:
        default = np.ones(logp.__code__.co_argcount)
        return default
    else:
        return np.hstack([start])
