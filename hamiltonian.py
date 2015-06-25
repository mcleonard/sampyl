import numpy as np
from utils import default_start


class Hamiltonian(object):
    def __init__(self, logp, dlogp, start=None, eta=0.1, L=100):
        self.logp = logp
        self.dlogp = dlogp
        self.start = default_start(start, logp)
        self.eta = eta
        self.L = L
        self.sampler = generate_samples(logp, dlogp,
                                        start=self.start,
                                        eta=eta,
                                        L=L)

    def sample(self, num, burn=0, thin=1):
        samples = np.zeros((num, self.start.size))
        for i in range(num):
            samples[i, :] = next(self.sampler)
        return samples[burn+1::thin]


def leapfrog(x, r, eta, dlogp):
    r_new = r + eta/2*dlogp(*x)
    x_new = x + eta*r_new
    r_new = r_new + (eta/2)*dlogp(*x)
    return x_new, r_new


def accept(x, y, r_0, r, logp):
    numer = np.exp(logp(*y) - 0.5*np.dot(r, r))
    denom = np.exp(logp(*x) - 0.5*np.dot(r_0, r_0))
    A = np.min(np.array([1, numer/denom]))
    return np.random.rand() < A


def generate_samples(logp, dlogp, start=None, eta=0.1, L=100):
    x = default_start(start, logp)
    while True:
        r_0 = np.random.randn()
        y, r = x, r_0
        for i in range(L):
            y, r = leapfrog(y, r, eta, dlogp)
        if accept(x, y, r_0, r, logp):
            x = y
        yield x
