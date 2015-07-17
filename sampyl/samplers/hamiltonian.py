from __future__ import division

from ..core import np
from ..state import State
from .base import Sampler


class Hamiltonian(Sampler):
    def __init__(self, logp, start, step_size=1, n_steps=5, **kwargs):

        try:
            super().__init__(logp, start, **kwargs)
        except TypeError:
            super(Hamiltonian, self).__init__(logp, start, **kwargs)

        self.step_size = step_size / (np.hstack(self.state.values()).size)**(1/4)
        self.n_steps = n_steps

    def step(self):

        x = self.state
        r0 = initial_momentum(x, self.scale)
        y, r = x, r0

        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.grad_logp)

        if accept(x, y, r0, r, self.logp):
            x = y
            self._accepted += 1

        self.state = x
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


def grad_vec(grad_logp, state):
    """ grad_logp should be a list of gradient logps, respective to each
        parameter in x
    """
    if hasattr(grad_logp, '__call__'):
        return np.array([grad_logp(*state.values())])
    else:
        return np.array([grad_logp[each](*state.values()) for each in state])


def leapfrog(x, r, step_size, grad_logp):
    r_new = r + step_size/2*grad_vec(grad_logp, x)
    x_new = x + step_size*r_new
    r_new = r_new + step_size/2*grad_vec(grad_logp, x_new)
    return x_new, r_new


def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E_new - E]))
    return (np.log(np.random.rand()) < A)


def energy(logp, x, r):
    r1 = r.tovector()
    return logp(*x.values()) - 0.5*np.dot(r1, r1)


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

    
    # for i, var in enumerate(state):
    #     mu = np.zeros(np.shape(state[var]))
    #     r.update({var: np.random.normal(mu, scale[i])})
    # return r
    
    return new