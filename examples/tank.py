"""
German tank problem from
http://matatat.org/sampyl/examples/german_tank_problem.html
"""

import sampyl as smp
from sampyl import np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Data
serials = np.array([10, 256, 202, 97])
m = np.max(serials)

# log P(N | D)
def logp(N):
    # Samplers will pass in floats, we need to make them integers
    N = np.floor(N).astype(int)

    # Log-likelihood
    llh = smp.discrete_uniform(serials, lower=1, upper=N)

    prior = smp.discrete_uniform(N, lower=m, upper=10000)

    return llh + prior

# Slice sampler for drawing from the posterior
sampler = smp.Slice(logp, {'N':300})
chain = sampler.sample(20000, burn=4000, thin=4)

posterior = np.floor(chain.N)
plt.hist(posterior, range=(0, 1000), bins=100,
         histtype='stepfilled', normed=True)
plt.xlabel("Total number of tanks")
plt.ylabel("Posterior probability mass")

plt.show()
pass