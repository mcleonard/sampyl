"""
sampyl.samplers.metropolis
~~~~~~~~~~~~~~~~~~~~

Module implementing Metropolis-Hastings MCMC sampler.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""

from __future__ import division

from ..core import np
from ..state import State
from .base import Sampler


class Metropolis(Sampler):
    # TODO: Allow for sticking in different proposal distributions.
    """ Metropolis-Hastings sampler for drawing from a distribution
        defined by a logp function.

        Has automatic scaling such that acceptance rate stays around 50%

        :param logp: function
            log P(X) function for sampling distribution.
        :param start: 
            Dictionary of starting state for the sampler. Should have one
            element for each argument of logp.
        :param scale: *scalar or 1D array-like.*
            initial scaling factor for proposal distribution.
        :param tune_interval: *int.*
        :param scale: **scalar or 1D array-like.**
            initial scaling factor for proposal distribution.
        :param tune_interval: *int.*
            number of samples between tunings of scale factor.

        Example::

            def logp(x, y):
                ...

            start = {'x': x_start, 'y': y_start}
            metro = sampyl.Metropolis(logp, start)
            chain = metro.sample(20000, burn=5000, thin=4)

    """

    def __init__(self, logp, start, tune_interval=100, **kwargs):
                
        super(Metropolis, self).__init__(logp, start, None, grad_logp_flag=False,
                                         **kwargs)
        self.tune_interval = tune_interval
        self._steps_until_tune = tune_interval
        self._accepted = 0

    def step(self):
        """ Perform a Metropolis-Hastings step. """
        x = self.state
        y = proposal(x, scale=self.scale)
        if accept(x, y, self.model.logp):
            self.state = y
            self._accepted += 1

        self._sampled += 1

        self._steps_until_tune -= 1
        if self._steps_until_tune == 0:
            self.scale = tune(self.scale, self.acceptance)
            self._steps_until_tune = self.tune_interval

        return self.state

    @property
    def acceptance(self):
        return self._accepted/self._sampled

    def __repr__(self):
        return 'Metropolis-Hastings sampler'


def proposal(state, scale):
    """ Sample a proposal x from a multivariate normal distribution. """
    proposed = State.fromkeys(state.keys())
    for i, var in enumerate(state):
        proposed.update({var: np.random.normal(state[var], scale[var])})
    return proposed


def accept(x, y, logp):
    """ Return a boolean indicating if the proposed sample should be accepted,
        given the logp ratio logp(y)/logp(x).
    """
    delp = logp(y) - logp(x)
    if np.isfinite(delp) and np.log(np.random.uniform()) < delp:
        return True
    else:
        return False


def tune(scale, acceptance):
    """ Borrowed from PyMC3 """

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
