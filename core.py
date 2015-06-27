
try:
    import autograd.numpy as np
    from autograd import grad
except ImportError:
    import numpy as np

from utils import logp_var_names, default_start, count
from trace import Trace


class Sampler(object):
    # When subclassing, set this to False if grad logp functions aren't needed
    _grad_logp_flag = True

    def __init__(self, logp, dlogp=None, start=None, scale=1.):
        self.logp = logp
        self.var_names = logp_var_names(logp)
        self.start = default_start(start, logp)
        self.state = self.start
        self.scale = scale*np.ones(len(self.var_names))
        self.sampler = None
        self._sampled = 0
        self._accepted = 0
        if self._grad_logp_flag and dlogp is None:
            self.dlogp = auto_grad_logp(logp)
        else:
            self.dlogp = dlogp

    def step(self):
        pass

    def sample(self, num, burn=0, thin=1):
        if self.sampler is None:
            self.sampler = (self.step() for _ in count(start=0, step=1))
        samples = np.array([next(self.sampler) for _ in range(num)])
        trace = samples[burn+1::thin].view(Trace)
        trace.var_names = self.var_names
        return trace

    def reset(self):
        self.state = self.start
        self.sampler = None
        self._accepted = 0
        self._sampled = 0
        return self

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


def auto_grad_logp(logp):
    """ Automatically builds gradient logps using autograd. Returns as list
        containing one grad logp with respect to each variable in logp.
    """
    n = logp.__code__.co_argcount
    return [grad(logp, i) for i in range(n)]
