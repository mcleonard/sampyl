from core import np
from utils import grad_logp
from core import Sampler


class Hamiltonian(Sampler):
    def __init__(self, logp, start=None, scale=1., step_size=1, n_steps=5):
        super().__init__(logp, start=start, scale=scale)
        self.step_size = step_size / len(self.state)**(1/4)
        self.n_steps = n_steps

    def step(self):

        x = self.state
        r0 = initial_momentum(x, self.scale)
        y, r = x, r0

        for i in range(self.n_steps):
            y, r = leapfrog(y, r, self.step_size, self.dlogp)

        if accept(x, y, r0, r, self.logp):
            x = y
            self._accepted += 1

        self.state = x
        self._sampled += 1
        return x

    @property
    def acceptance_rate(self):
        return self._accepted/self._sampled


def leapfrog(x, r, step_size, dlogp):
    r_new = r + step_size/2*grad_logp(dlogp, x)
    x_new = x + step_size*r_new
    r_new = r_new + (step_size/2)*grad_logp(dlogp, x_new)
    return x_new, r_new


def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E_new - E]))
    return (np.log(np.random.rand()) < A)


def energy(logp, x, r):
    return logp(*x) - 0.5*np.dot(r, r)


def initial_momentum(state, scale):
    cov = np.diagflat(scale*np.ones(len(state)))
    r = np.random.multivariate_normal(np.zeros(state.shape), cov)
    return r
