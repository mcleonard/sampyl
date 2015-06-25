import numpy as np


def default_start(start, logp):
    """ If start is None, return a zeros array with length equal to the number
        of arguments in logp
    """
    if start is None:
        default = np.ones(logp.__code__.co_argcount)
        return default
    else:
        return np.hstack([start])
