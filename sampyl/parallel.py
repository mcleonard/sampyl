import copy
from functools import partial
from multiprocessing import Pool

from numpy.lib.recfunctions import stack_arrays

from .core import np, grad
from .samplers import *
from .state import State
from .progressbar import update_progress


def f(n_samples, sampler, **kwargs):
        kwargs.update({'progress_bar': False})
        trace = sampler.sample(n_samples, **kwargs)
        return trace


def parallel(sampler, n_chains, n_samples, progress_bar=True, **kwargs):

    samplers = init_samplers(sampler, n_chains)

    batches = [n_samples//10]*10
    batches.append(n_samples%10)

    pool = Pool(processes=n_chains)
    chains = None
    for i, N in enumerate(batches):
        func = partial(f, N, **kwargs)
        if chains is None:
            chains = pool.map(func, samplers)
        else:
            # Reinitialize samplers where the previous batch left off so that
            # the new batch starts at the correct state
            samplers = init_samplers(sampler, n_chains, chains=chains)
            new_chains = pool.map(func, samplers)
            chains = [stack_arrays([new, old], usemask=False, asrecarray=True)
                      for new, old in zip(new_chains, chains)]

        if progress_bar:
                update_progress(N*(i+1), n_samples)

    if progress_bar:
        update_progress(n_samples, n_samples, end=True)

    return chains


def init_samplers(sampler, n_chains, chains=None):

    samplers = [copy.deepcopy(sampler) for _ in range(n_chains)]
    for sampler in samplers:
        # Randomize start and seed
        sampler.state.update({var: val + np.random.randn(*np.shape(val))
                              for var, val in sampler.state.items()})
        sampler.seed = np.random.randint(0, 2**16)

        # Can't pickle autograd grad functions, so clear it here, then
        # build them when sample method is called.
        sampler.grad_logp = None

    if chains is not None:
        # This way we can update the samplers' states to the last state
        # in each chain. This ensures that each batch starts off where
        # the previous one left off.
        for sampler, chain in zip(samplers, chains):
            sampler.state.update({var: val for var, val
                                  in zip(sampler.state, chain[-1])})

    return samplers
