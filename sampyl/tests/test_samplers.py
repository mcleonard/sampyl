from ..core import np
from ..exceptions import *
from .logps import *
import sampyl as smp
import pytest

np_source = np.__package__

n_samples = 100


def test_sample_chain():
    logp, _ = poisson_delta()

    step1 = smp.Metropolis(logp, condition=['lam2'])
    step2 = smp.NUTS(logp, condition=['lam1'])

    chain = smp.Chain([step1, step2])
    trace = chain.sample(n_samples)
    assert(trace.shape == (n_samples,))

def test_conditional_chain():

    logp, _ = poisson_delta()

    metro = smp.Metropolis(logp, condition=['lam2'])
    nuts = smp.NUTS(logp, condition=['lam1'])

    start = np.array([1., 1.])
    metro.state = start
    state = metro._conditional_step()
    assert(state[1] == 1.)
    nuts.state = state
    state = nuts._conditional_step()
    assert(len(state) == 2)


def test_conditional():

    logp, _ = poisson_delta()
    start = np.array([1., 1.])
    metro = smp.Metropolis(logp, start=start, condition=['lam2'])
    state = metro._conditional_step()
    assert(len(state) == 2)
    assert(state[1] == 1.)

def test_metropolis_linear_model():
    logp, _ = linear_model_5features()
    metro = smp.Metropolis(logp)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_hamiltonian_linear_model():
    logp, _ = linear_model_5features()
    hmc = smp.Hamiltonian(logp)
    trace = hmc.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_nuts_linear_model():
    logp, _ = linear_model_5features()
    nuts = smp.NUTS(logp)
    trace = nuts.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_metropolis():
    logp, _ = normal_1D()
    metro = smp.Metropolis(logp)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_hmc_autograd():
    logp, _ = normal_1D()
    if np_source == 'autograd.numpy':
        hmc = smp.Hamiltonian(logp)
        trace = hmc.sample(n_samples)
        assert(trace.shape == (n_samples,))
    elif np_source == 'numpy':
        with pytest.raises(AutogradError):
            hmc = smp.Hamiltonian(logp)


def test_hmc_pass_grad_logp():
    logp, grad_logp = normal_1D()
    hmc = smp.Hamiltonian(logp, grad_logp=grad_logp)
    trace = hmc.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_NUTS_autograd():
    logp, _ = normal_1D()
    if np_source == 'autograd.numpy':
        nuts = smp.NUTS(logp)
        trace = nuts.sample(n_samples)
        assert(trace.shape == (n_samples,))
    elif np_source == 'numpy':
        with pytest.raises(AutogradError):
            nuts = smp.NUTS(logp)


def test_NUTS_pass_grad_logp():
    logp, grad_logp = normal_1D()
    nuts = smp.NUTS(logp, grad_logp=grad_logp)
    trace = nuts.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_sampler_num_logp():
    logp = 1.
    with pytest.raises(TypeError):
        metro = smp.Metropolis(logp)


def test_sampler_no_args_logp():
    def logp():
        return x
    with pytest.raises(ValueError):
        metro = smp.Metropolis(logp)


def test_metropolis_two_vars():
    logp, _ = poisson_delta()
    metro = smp.Metropolis(logp)
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))


def test_metropolis_two_vars_start():
    logp, _ = poisson_delta()
    metro = smp.Metropolis(logp, start=(1., 1.))
    trace = metro.sample(n_samples)
    assert(trace.shape == (n_samples,))
