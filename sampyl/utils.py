from .core import np
from itertools import count


def grad_vec(grad_logp, x):
    """ grad_logp should be a list of gradient logps, respective to each
        paramter in x
    """
    try:
        return np.array([each(*x) for each in grad_logp])
    except TypeError:  # Happens when grad_logp isn't iterable
        grad_logp = [grad_logp]
        return grad_vec(grad_logp, x)


def default_start(start, logp):
    """ If start is None, return a zeros array with length equal to the number
        of arguments in logp
    """
    if start is None:
        default = np.ones(logp.__code__.co_argcount)
        return default
    else:
        return np.hstack([start])
