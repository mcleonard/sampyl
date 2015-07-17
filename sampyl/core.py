""" Core file used for things a bunch of other files need. """

from .exceptions import AutogradError

# Basically, all other files will import numpy from here. I did this way
# so that if autograd is installed, the other files will use its numpy.
# Otherwise, just use the normal numpy.
try:
    AUTOGRAD = True
    import autograd.numpy as np
    from autograd import grad
except ImportError:
    AUTOGRAD = False
    import numpy as np


def auto_grad_logp(logp, names=None):
    """ Automatically builds gradient logps using autograd. Returns as list
        containing one grad logp with respect to each variable in logp.

        If logp has unknown number of arguments, you can set n to the desired
        number.
    """
    if AUTOGRAD is False:
        raise AutogradError("Install autograd to use automatic "
                            "gradient functionality.")
    if names is None:
        n = logp.__code__.co_argcount
        names = logp.__code__.co_varnames[:n]
    return {var: grad(logp, i) for i, var in enumerate(names)}
