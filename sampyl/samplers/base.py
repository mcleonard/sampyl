from ..core import np, auto_grad_logp
from ..utils import default_start, count
from ..trace import Trace


class Sampler(object):
    # When subclassing, set this to False if grad logp functions aren't needed
    _grad_logp_flag = True

    def __init__(self, logp, grad_logp=None, start=None, scale=1.):
        self.logp = logp
        self.var_names = logp_var_names(logp)
        self.start = default_start(start, logp)
        self.state = self.start
        self.scale = scale*np.ones(len(self.var_names))
        self.sampler = None
        self._sampled = 0

        if self._grad_logp_flag and grad_logp is None:
            self.grad_logp = auto_grad_logp(logp)
        else:
            if len(self.var_names) > 1 and len(grad_logp) != len(var_names):
                raise TypeError("grad_logp must be iterable with length equal"
                                " to the number of parameters in logp.")
            else:
                self.grad_logp = grad_logp

    def step(self):
        pass

    def sample(self, num, burn=-1, thin=1):
        """ Sample from distribution defined by logp.

            Parameters
            ----------

            num: int
                Number of samples to return.
            burn: int
                Number of samples to burn through
            thin: thin
                Thin the samples by this factor
        """
        if self.sampler is None:
            self.sampler = (self.step() for _ in count(start=0, step=1))
        samples = np.array([next(self.sampler) for _ in range(num)])
        trace = samples[burn+1::thin].view(Trace)
        trace.var_names = self.var_names
        return trace


def logp_var_names(logp):
    """ Returns a list of the argument names in logp """
    # Putting underscores after the names so that variables names don't
    # conflict with built in attributes
    names = logp.__code__.co_varnames[:logp.__code__.co_argcount]
    names = [each + "_" for each in names]
    return names