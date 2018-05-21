"""
sampyl.samplers.NUTS
~~~~~~~~~~~~~~~~~~~~

This module implements No-U-Turn Sampler (NUTS).

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""


from __future__ import division

import collections

from ..core import np
from .base import Sampler
from .hamiltonian import energy, leapfrog, initial_momentum


class NUTS(Sampler):
    """ No-U-Turn sampler (Hoffman & Gelman, 2014) for sampling from a
        probability distribution defined by a log P(theta) function.

        For technical details, see the paper:
        http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

        :param logp: log P(X) function for sampling distribution
        :param start: 
            Dictionary of starting state for the sampler. Should have one
            element for each argument of logp.
        :param grad_logp: (optional)
            Function or list of functions that calculate grad log P(theta). 
            Pass functions here if you don't want to use autograd for the 
            gradients. If logp has multiple parameters, grad_logp must be 
            a list of gradient functions w.r.t. each parameter in logp.

            If you wish to use a logp function that returns both the logp
            value and the gradient, set grad_logp = True.
        :param scale: (optional) 
            Dictionary with same format as start. Scaling for initial 
            momentum in Hamiltonian step.
        :param step_size: (optional) *float.*
            Initial step size for the deterministic proposals.
        :param adapt_steps: (optional) *int.*
            Integer number of steps used for adapting the step size to 
            achieve a target acceptance rate.
        :param Emax: (optional) *float.* Maximum energy.
        :param target_accept: (optional) *float.* Target acceptance rate.
        :param gamma: (optional) *float.*
        :param k: (optional) *float.* Scales the speed of step size 
            adaptation.
        :param t0: (optional) *float.* Slows initial step size adaptation.

        Example ::

            def logp(x, y):
                ...

            start = {'x': x_start, 'y': y_start}
            nuts = sampyl.NUTS(logp, start)
            chain = nuts.sample(1000)

    """

    def __init__(self, logp, start,
                 step_size=0.25,
                 adapt_steps=100,
                 Emax=1000.,
                 target_accept=0.65,
                 gamma=0.05,
                 k=0.75,
                 t0=10.,
                 **kwargs):

        super(NUTS, self).__init__(logp, start, **kwargs)

        self.step_size = step_size / len(self.state.tovector())**(1/4.)
        self.adapt_steps = adapt_steps
        self.Emax = Emax
        self.target_accept = target_accept
        self.gamma = gamma
        self.k = k
        self.t0 = t0

        self.Hbar = 0.
        self.ebar = 1.
        self.mu = np.log(self.step_size*10)

    def step(self):
        """ Perform one NUTS step."""

        H = self.model.logp
        dH = self.model.grad
        x = self.state
        r0 = initial_momentum(x, self.scale)
        u = np.random.uniform()
        e = self.step_size
        xn, xp, rn, rp, y = x, x, r0, r0, x
        j, n, s = 0, 1, 1

        while s == 1:
            v = bern(0.5)*2 - 1
            if v == -1:
                xn, rn, _, _, x1, n1, s1, a, na = buildtree(xn, rn, u, v, j, e, x, r0,
                                                            H, dH, self.Emax)
            else:
                _, _, xp, rp, x1, n1, s1, a, na = buildtree(xp, rp, u, v, j, e, x, r0,
                                                            H, dH, self.Emax)

            if s1 == 1 and bern(np.min(np.array([1, n1/n]))):
                y = x1

            dx = (xp - xn).tovector()
            s = s1 * (np.dot(dx, rn.tovector()) >= 0) * \
                     (np.dot(dx, rp.tovector()) >= 0)
            n = n + n1
            j = j + 1

        if self._sampled >= self.adapt_steps:
            self.step_size = self.ebar
        else:
            # Adapt step size
            m = self._sampled + 1
            w = 1./(m + self.t0)
            self.Hbar = (1 - w)*self.Hbar + w*(self.target_accept - a/na)
            log_e = self.mu - (m**.5/self.gamma)*self.Hbar
            self.step_size = np.exp(log_e)
            z = m**(-self.k)
            self.ebar = np.exp(z*log_e + (1 - z)*np.log(self.ebar))

        self.state = y
        self._sampled += 1

        return y


def bern(p):
    return np.random.uniform() < p


def buildtree(x, r, u, v, j, e, x0, r0, H, dH, Emax):
    if j == 0:
        x1, r1 = leapfrog(x, r, v*e, dH)
        E = energy(H, x1, r1)
        E0 = energy(H, x0, r0)
        dE = E - E0

        n1 = (np.log(u) - dE <= 0)
        s1 = (np.log(u) - dE < Emax)
        return x1, r1, x1, r1, x1, n1, s1, np.min(np.array([1, np.exp(dE)])), 1
    else:
        xn, rn, xp, rp, x1, n1, s1, a1, na1 = \
            buildtree(x, r, u, v, j-1, e, x0, r0, H, dH, Emax)
        if s1 == 1:
            if v == -1:
                xn, rn, _, _, x2, n2, s2, a2, na2 = \
                    buildtree(xn, rn, u, v, j-1, e, x0, r0, H, dH, Emax)
            else:
                _, _, xp, rp, x2, n2, s2, a2, na2 = \
                    buildtree(xp, rp, u, v, j-1, e, x0, r0, H, dH, Emax)
            if bern(n2/max(n1 + n2, 1.)):
                x1 = x2

            a1 = a1 + a2
            na1 = na1 + na2

            dx = (xp - xn).tovector()
            s1 = s2 * (np.dot(dx, rn.tovector()) >= 0) * \
                      (np.dot(dx, rp.tovector()) >= 0)
            n1 = n1 + n2
        return xn, rn, xp, rp, x1, n1, s1, a1, na1
        