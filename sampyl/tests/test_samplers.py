from ..core import np
from ..exceptions import *
from .logps import *
import sampyl as smp
import pytest

np_source = np.__package__

n_samples = 100


def test_parallel_lin_model():

    logp = linear_model_logp
    start = {'b':np.zeros(5), 'sig': 1.}
    metro = smp.Metropolis(logp, start)
    nuts = smp.NUTS(logp, start)

    metro_chain = metro.sample(n_samples, n_chains=4)
    nuts_chain = nuts.sample(n_samples, n_chains=4)

    assert(len(metro_chain) == 4)
    assert(len(nuts_chain) == 4)

def test_parallel_2D():

    start = {'lam1': 1., 'lam2': 1.}
    metro = smp.Metropolis(poisson_logp, start)
    nuts = smp.NUTS(poisson_logp, start)

    metro_chain = metro.sample(n_samples, n_chains=4)
    nuts_chain = nuts.sample(n_samples, n_chains=4)

    assert(len(metro_chain) == 4)
    assert(len(nuts_chain) == 4)

def test_sample_chain():
    start = {'lam1': 1., 'lam2': 1.}
    step1 = smp.Metropolis(poisson_logp, start, condition=['lam2'])
    step2 = smp.NUTS(poisson_logp, start, condition=['lam1'])

    chain = smp.Chain([step1, step2], start)
    trace = chain.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_conditional_chain():

    logp = poisson_logp
    start = {'lam1': 1., 'lam2': 2.}
    metro = smp.Metropolis(logp, start, condition=['lam2'])
    nuts = smp.NUTS(logp, start, condition=['lam1'])

    state = metro._conditional_step()
    assert(state['lam2'] == 2.)
    nuts.state.update(state)
    state = nuts._conditional_step()
    assert(len(state) == 2)


def test_conditional():
    logp  = poisson_logp
    start = {'lam1': 1., 'lam2': 2.}
    metro = smp.Metropolis(logp, start, condition=['lam2'])
    state = metro._conditional_step()
    assert(len(state) == 2)
    assert(state['lam2'] == 2.)

def test_metropolis_linear_model():
    logp = linear_model_logp
    start = {'b':np.zeros(5), 'sig': 1.}
    metro = smp.Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_hamiltonian_linear_model():
    logp = linear_model_logp
    start = {'b': np.zeros(5), 'sig': 1.}
    hmc = smp.Hamiltonian(logp, start)
    trace = hmc.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_nuts_linear_model():
    logp = linear_model_logp
    start = {'b': np.zeros(5), 'sig': 1.}
    nuts = smp.NUTS(logp, start)
    trace = nuts.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_metropolis():
    logp  = normal_1D_logp
    start = {'x': 1.}
    metro = smp.Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_hmc_autograd():
    logp = normal_1D_logp
    start = {'x': 1.}
    if np_source == 'autograd.numpy':
        hmc = smp.Hamiltonian(logp, start)
        trace = hmc.sample(n_samples)
        assert(trace.shape == (n_samples,))
    elif np_source == 'numpy':
        with pytest.raises(AutogradError):
            hmc = smp.Hamiltonian(logp, start)


def test_hmc_pass_grad_logp():
    logp, grad_logp = normal_1D_logp, normal_1D_grad_logp
    start = {'x': 1.}
    hmc = smp.Hamiltonian(logp, start, grad_logp=grad_logp)
    trace = hmc.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_NUTS_autograd():
    logp = normal_1D_logp
    start = {'x': 1.}
    if np_source == 'autograd.numpy':
        nuts = smp.NUTS(logp, start)
        trace = nuts.sample(n_samples)
        assert(trace.shape == (n_samples,))
    elif np_source == 'numpy':
        with pytest.raises(AutogradError):
            nuts = smp.NUTS(logp, start)


def test_NUTS_pass_grad_logp():
    logp, grad_logp = normal_1D_logp, normal_1D_grad_logp
    start = {'x': 1.}
    nuts = smp.NUTS(logp, start, grad_logp=grad_logp)
    trace = nuts.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_sampler_num_logp():
    logp = 1.
    start = {'x': None}
    with pytest.raises(TypeError):
        metro = smp.Metropolis(logp, start)


def test_sampler_no_args_logp():
    def logp():
        return x
    start = {'x': None}
    with pytest.raises(ValueError):
        metro = smp.Metropolis(logp, start)


def test_metropolis_two_vars():
    logp = poisson_logp
    start = {'lam1':1., 'lam2':1.}
    metro = smp.Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_metropolis_two_vars_start():
    logp = poisson_logp
    start = {'lam1':1., 'lam2':1.}
    metro = smp.Metropolis(logp, start)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))

def test_slice():
    logp = normal_1D_logp
    start = {'x': 1.}
    slice = smp.Slice(logp, start)
    trace = slice.sample(n_samples)
    assert(trace.shape == (n_samples,))

def test_slice_two_vars():
    logp = poisson_logp
    start = {'lam1': 1., 'lam2': 1.}
    slice = smp.Slice(logp, start)
    trace = slice.sample(n_samples)
    assert(trace.shape == (n_samples,))



