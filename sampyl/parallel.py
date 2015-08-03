import copy
from functools import partial
from multiprocessing import Pool

from .core import np, grad
from .samplers import *
from .state import State


def f(n_samples, sampler, **kwargs):
        trace = sampler.sample(n_samples, **kwargs)
        return trace

def parallel(sampler, n_chains, n_samples, **kwargs):

    samplers = [copy.deepcopy(sampler) for _ in range(n_chains)]
    # Randomize start
    for sampler in samplers:
        sampler.state.update({var: val + np.random.randn(*np.shape(val))
                              for var, val in sampler.state.items()})
        sampler.seed = np.random.randint(0, 2**16)
        sampler.grad_logp = None

    func = partial(f, n_samples, **kwargs)
    pool = Pool(processes=n_chains)
    chains = pool.map(func, samplers)
    return chains
