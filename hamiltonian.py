import autograd.numpy as np
from autograd import grad
from utils import default_start, grad_logp


class Hamiltonian(object):
    def __init__(self, logp, start=None, eta=0.1, L=20):
        self.logp = logp
        self.start = default_start(start, logp)
        self.dlogp = [grad(logp, i) for i in range(len(self.start))]
        self.eta = eta
        self.L = L
        self.sampler = None
        self.state = self.start

    def step(self):
        x = step(self.logp, self.dlogp, start=self.state,
                 eta=self.eta, L=self.L)
        self.state = x
        return x

    def sample(self, num, burn=0, thin=1):
        if self.sampler is None:
            self.sampler = generate_samples(self.logp, self.dlogp,
                                            start=self.state,
                                            eta=self.eta, L=self.L)
        samples = np.vstack(np.array([next(self.sampler)
                                      for _ in range(num)]))
        self.state = samples[-1]
        return samples[burn+1::thin]


def leapfrog(x, r, eta, dlogp):
    r_new = r + eta/2*grad_logp(dlogp, x)
    x_new = x + eta*r_new
    r_new = r_new + (eta/2)*grad_logp(dlogp, x_new)
    return x_new, r_new


def accept(x, y, r_0, r, logp):
    numer = np.exp(logp(*y) - 0.5*np.dot(r, r))
    denom = np.exp(logp(*x) - 0.5*np.dot(r_0, r_0))
    A = np.min(np.array([1, numer/denom]))
    return np.random.rand() < A


def step(logp, dlogp, start=None, eta=0.001, L=10):
    x = default_start(start, logp)
    cov = np.diagflat(np.ones(x.shape))
    r_0 = np.random.multivariate_normal(np.zeros(x.shape), cov)
    y, r = x, r_0
    for i in range(L):
        y, r = leapfrog(y, r, eta, dlogp)
    if accept(x, y, r_0, r, logp):
        x = y
    return x


def generate_samples(logp, dlogp, start=None, eta=0.001, L=10):
    x = default_start(start, logp)
    while True:
        x = step(logp, dlogp, start=x, eta=eta, L=L)
        yield x
