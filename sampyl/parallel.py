"""
sampyl.parallel
~~~~~~~~~~~~~~~~~~~~

This module implements generating multiple Markov chains in parallel.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""


from __future__ import division

import copy
from functools import partial
from multiprocessing import Pool

from .core import np, AUTOGRAD
from .samplers import *
from .distributions import *
from .state import State
from .progressbar import update_progress


def f(n_samples, sampler):
        trace = sampler.sample(n_samples, progress_bar=False)
        return trace


def parallel(sampler, n_chains, samples, progress_bar=True, **kwargs):

    samplers = init_samplers(sampler, n_chains)
    chains = [copy.copy(samples) for _ in range(n_chains)]

    N_batches = 10
    n_samples = len(samples)
    batches = [n_samples//N_batches]*N_batches
    batches.append(n_samples % N_batches)

    pool = Pool(processes=n_chains)
    for i, N in enumerate(batches):
        func = partial(f, N)

        if i != 0:
            # Reinitialize samplers where the previous batch left off so that
            # the new batch starts at the correct state
            samplers = init_samplers(sampler, n_chains, chains=new_chains)
        new_chains = pool.map(func, samplers)

        for new, chain in zip(new_chains, chains):
            for j in range(N):
                chain[i*N + j] = new[j]

        if progress_bar:
                update_progress(N*(i+1), n_samples)

    if progress_bar:
        update_progress(n_samples, n_samples, end=True)

    burn, thin = kwargs.get('burn'), kwargs.get('thin')
    chains = [chain[burn::thin] for chain in chains]
    return chains


def init_samplers(sampler, n_chains, chains=None):

    samplers = [copy.deepcopy(sampler) for _ in range(n_chains)]
    for sampler in samplers:
        # Randomize start and seed
        sampler.state.update({var: val + np.random.randn(*np.shape(val))*val/5
                              for var, val in sampler.state.items()})
        sampler.seed = np.random.randint(0, 2**16)

        # Can't pickle autograd grad functions, so clear it here, then
        # build them when sample method is called.
        if AUTOGRAD and hasattr(sampler.model, 'grad_func'):
            sampler.model.grad_func = None

    if chains is not None:
        # This way we can update the samplers' states to the last state
        # in each chain. This ensures that each batch starts off where
        # the previous one left off.
        for sampler, chain in zip(samplers, chains):
            sampler.state.update({var: val for var, val
                                  in zip(sampler.state, chain[-1])})

    return samplers
