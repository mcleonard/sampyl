""" Core file used for things a bunch of other files need. """


# Basically, all other files will import numpy from here. I did this way
# so that if autograd is installed, the other files will use its numpy.
# Otherwise, just use the normal numpy.
try:
    import autograd.numpy as np
    from autograd import grad
except ImportError:
    import numpy as np


def auto_grad_logp(logp):
    """ Automatically builds gradient logps using autograd. Returns as list
        containing one grad logp with respect to each variable in logp.
    """
    n = logp.__code__.co_argcount
    return [grad(logp, i) for i in range(n)]
