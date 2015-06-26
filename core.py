import autograd.numpy as np
from autograd import grad
from utils import logp_var_names, default_start, count
from trace import Trace


class Sampler(object):

    def __init__(self, logp, start=None, scale=1.):
        self.logp = logp
        self.var_names = logp_var_names(logp)
        self.start = default_start(start, logp)
        self.dlogp = [grad(logp, i) for i in range(len(self.var_names))]
        self.state = self.start
        self.scale = scale*np.ones(len(self.var_names))
        self.sampler = None
        self._sampled = 0
        self._accepted = 0

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
