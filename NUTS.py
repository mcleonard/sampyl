import autograd.numpy as np

from core import Sampler
from hamiltonian import leapfrog, energy, initial_momentum


class NUTS(Sampler):
    def __init__(self, logp, start=None, scale=1., step_size=0.25,
                 Emax=1000., target_accept=0.65, gamma=0.05,
                 k=0.75, t0=10.):
        super().__init__(logp, start, scale)
        self.step_size = step_size
        self.Emax = Emax
        self.target_accept = target_accept
        self.gamma = gamma
        self.k = k
        self.t0 = t0

        self.Hbar = 0
        self.mu = np.log(self.step_size*10)

    def step(self):

        H = self.logp
        dH = self.dlogp
        x = self.state
        r0 = initial_momentum(x, self.scale)
        u = np.random.uniform(0., np.exp(energy(H, x, r0)))
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

            dx = xp - xn
            n = n + n1
            s = s1 * (np.dot(dx, rn) >= 0)* (np.dot(dx, rp) >= 0)
            j = j + 1

        m = self._sampled
        w = 1./(m + self.t0)
        self.Hbar = (1 - w)*self.Hbar + w*(self.target_accept - a*1./na)
        self.step_size = np.exp(self.mu - (m**.5/self.gamma)*self.Hbar)

        if (y != self.state).any():
            self.state = y
            self._accepted += 1
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

        n1 = int(np.log(u) <= E)
        s1 = int(np.log(u) < Emax + E)

        return x1, r1, x1, r1, x1, n1, s1, np.min(np.array([1, np.exp(dE)])), 1.
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

            dx = xp - xn
            s1 = s2 * (np.dot(dx, rn) >= 0) * (np.dot(dx, rp) >= 0)
            n1 = n1 + n2
        return xn, rn, xp, rp, x1, n1, s1, a1, na1
        