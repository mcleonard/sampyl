from ..core import np

import sampyl as smp

n_samples = 100

temps = np.array([66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67,
                  75, 70, 81, 76, 79, 75, 76, 58])

failures = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 
                     0, 0, 0, 0, 1, 0, 1])

def logistic(t):
    return 1./(1 + np.exp(-t))

def logp(b):
    p_hat = logistic(b[0] + b[1]*temps)
    llh = smp.bernoulli(failures, p_hat)
    b_prior = smp.normal(b, mu=0, sig=10)
    return llh + b_prior

start = smp.find_MAP(logp, {'b':np.zeros(2)})

def test_metropolis_scaling():
    metro = smp.Metropolis(logp, start, scale=start)
    chain = metro.sample(n_samples)
    assert(chain.shape == (n_samples,))


# def test_nuts_scaling():
#     nuts = smp.NUTS(logp, start, scale=start)
#     chain = nuts.sample(n_samples)
#     assert(chain.shape == (n_samples,))


def logp(a, b):
    p_hat = logistic(a + b*temps)
    llh = smp.bernoulli(failures, p_hat)
    a_prior = smp.normal(a, mu=0, sig=10)
    b_prior = smp.normal(b, mu=0, sig=10)
    return llh + a_prior + b_prior

start = smp.find_MAP(logp, {'a':0., 'b':0.})

def test_metropolis_scaling_2_params():
    metro = smp.Metropolis(logp, start, scale=start)
    chain = metro.sample(n_samples)
    assert(chain.shape == (n_samples,))