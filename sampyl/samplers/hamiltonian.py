"""
sampyl.samplers.hamiltonian
~~~~~~~~~~~~~~~~~~~~

Module implementing Hamiltonian MCMC sampler.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""


from __future__ import division

from ..core import np
from ..state import State
from .base import Sampler
from ..model import Model


class Hamiltonian(Sampler):
    def __init__(self, logp, start, step_size=1, n_steps=5, **kwargs):

        """ Hamiltonian MCMC sampler. Uses the gradient of log P(theta) to
            make informed proposals.

            Arguments
            ----------
            logp: function
                log P(X) function for sampling distribution
            start: dict
                Dictionary of starting state for the sampler. Should have one
                element for each argument of logp. So, if logp = f(x, y), then
                start = {'x': x_start, 'y': y_start}

            Keyword Arguments
            -----------------
            grad_logp: function or list of functions
                Functions that calculate grad log P(theta). Pass functions
                here if you don't want to use autograd for the gradients. If
                logp has multiple parameters, grad_logp must be a list of
                gradient functions w.r.t. each parameter in logp.
            scale: dict
                Same format as start. Scaling for initial momentum in
                Hamiltonian step.
            step_size: float
                Step size for the deterministic proposals.
            n_steps: int 
                Number of deterministic steps to take for each proposal.
            """

        super(Hamiltonian, self).__init__(logp, start, **kwargs)

        self.step_size = step_size / (np.hstack(self.state.values()).size)**(1/4)
        self.n_steps = n_steps

    def step(self):

        x = self.state
        r0 = initial_momentum(x, self.scale)
        y, r = x, r0

        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.model.grad)

        if accept(x, y, r0, r, self.model.logp):
            x = y
            self._accepted += 1

        self.state = x
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


def leapfrog(x, r, step_size, grad):

    r1 = r + step_size/2*grad(x)
    x1 = x + step_size*r1
    r2 = r1 + step_size/2*grad(x1)
    return x1, r2


def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E_new - E]))
    return (np.log(np.random.rand()) < A)


def energy(logp, x, r):
    r1 = r.tovector()
    return logp(x) - 0.5*np.dot(r1, r1)


def initial_momentum(state, scale):
    new = State.fromkeys(state.keys())
    for var in state:
        mu = np.zeros(np.shape(state[var]))
        cov = np.diagflat(scale[var])
        try:
            new.update({var: np.random.multivariate_normal(mu, cov)})
        except ValueError:
            # If the var is a single float
            new.update({var: np.random.normal(0, scale[var])})

    return new