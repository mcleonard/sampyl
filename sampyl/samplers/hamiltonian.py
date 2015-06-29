from ..core import np
from ..utils import grad_vec
from .base import Sampler


class Hamiltonian(Sampler):
    def __init__(self, logp, grad_logp=None, start=None, scale=1.,
                 step_size=1, n_steps=5):

        super().__init__(logp, grad_logp, start=start, scale=scale)
        self.step_size = step_size / sum(self.var_sizes.values())**(1/4)
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


def leapfrog(x, r, step_size, grad_logp):
    r_new = r + step_size/2*grad_vec(grad_logp, x)
    x_new = x + step_size*r_new
    r_new = r_new + (step_size/2)*grad_vec(grad_logp, x_new)
    return x_new, r_new


def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E_new - E]))
    return (np.log(np.random.rand()) < A)


def energy(logp, x, r):
    # Need to stack r into a 1D vector before taking inner product
    r1 = np.hstack(r)
    return logp(*x) - 0.5*np.dot(r1, r1)


def initial_momentum(state, scale):
    r = []
    for i, var in enumerate(state):
        r.append(np.random.normal(np.zeros(var.shape), scale[i]))
    return np.array(r)
