import numpy as np


class Metropolis(object):
    # TODO: Documentation
    # TODO: Allow for sticking in different proposal distributions.
    def __init__(self, logp, start=None, scale=1., tune_interval=100):
        self.logp = logp
        self.start = _default_start(start, logp)
        self.scale = scale
        self.sampler = generate_samples(logp, start=self.start, scale=scale)
        self._n_samples = 0
        self._unique_samples = 0
        self.tune_interval = tune_interval

    def sample(self, num, burn=0, thin=1):
        # TODO: Check if I should allocate the samples array first, then fill
        #       it while sampling. It will likely be faster.
        samples = np.zeros(self.start.size)
        batches = [self.tune_interval]*(num//self.tune_interval)
        if num % self.tune_interval != 0:
            batches + [num % self.tune_interval]

        for each in batches:
            sample_batch = np.vstack([next(self.sampler) for _ in range(each)])
            samples = np.vstack([samples, sample_batch])
            self._n_samples += each
            self._unique_samples += len(np.unique(sample_batch))
            self.scale = tune(self.scale, self.acceptance)

        return samples[burn+1::thin]

    def reset(self):
        self.sampler = generate_samples(self.logp,
                                        start=self.start,
                                        scale=self.scale)
        self._n_samples = 0
        self._unique_samples = 0
        return self

    @property
    def acceptance(self):
        return self._unique_samples/self._n_samples

    def __repr__(self):
        return 'Metropolis-Hastings sampler'


def proposal(x, scale=1):
    """ Sample a proposal x from a multivariate normal distribution. """
    try:
        dim = x.size
    except AttributeError:
        x = np.hstack([x])
        dim = x.size
    cov = np.diagflat(np.ones(dim))*scale
    y = np.random.multivariate_normal(x, cov)
    return y


def accept(x, y, logp):
    """ Return a boolean indicating if the proposed sample should be accepted,
        given the logp ratio logp(y)/logp(x).
    """
    delp = logp(*y) - logp(*x)

    if np.isfinite(delp) and np.log(np.random.uniform()) < delp:
        return True
    else:
        return False


def _default_start(start, logp):
    """ If start is None, return a zeros array with lengthequal to the number
        of arguments in logp
    """
    if start is None:
        default = np.ones(logp.__code__.co_argcount)
        return default
    else:
        return np.hstack(start)


def generate_samples(logp, start=None, scale=1):
    """ Returns a generator the yields the next sample. """
    x = _default_start(start, logp)
    while True:
        y = proposal(x, scale=scale)
        if accept(x, y, logp):
            x = y
        yield x


def tune(scale, acceptance):
    # Switch statement
    if acceptance < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acceptance < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acceptance < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acceptance > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acceptance > 0.75:
        # increase by double
        scale *= 2.0
    elif acceptance > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale
