import sys
sys.path.append('.')
import sampyl as smp
from sampyl import np
from sampyl.diagnostics import diagnostics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# correlated gaussian
def logp(x, y):
    icov = np.linalg.inv(np.array([[1., .8], [.8, 1.]]))
    d = np.array([x, y])
    return -.5 * np.dot(np.dot(d, icov), d)
#logp_xy = lambda(th): logp(th[0], th[1])

# NUTS sample
nuts = smp.NUTS(logp)
nuts_trace = nuts.sample(1000)

# Metropolis Hastings
met = smp.Metropolis(logp)
met_trace = met.sample(1000)

# gibbs sampler (NUTS-within-gibbs)
step1 = smp.Metropolis(logp, condition=['y']) # samples x & y, given z
step2 = smp.NUTS(logp, condition=['x']) # samples z, given x & y
chain = smp.Chain([step1, step2])
gibbs_trace = chain.sample(1000)

# compute effective sample size based on autocorrelation
nuts_eff = diagnostics.compute_n_eff_acf(nuts_trace.x)
met_eff = diagnostics.compute_n_eff_acf(met_trace.x)
gibbs_eff = diagnostics.compute_n_eff_acf(gibbs_trace.x)
print "NUTS effective sample size: %2.2f"%nuts_eff
print "MH   effective sample size: %2.2f"%met_eff
print "Gibbs effective sample size: %2.2f"%gibbs_eff

# graphically compare samples
fig, axarr = plt.subplots(1, 3, figsize=(15, 6))
axarr[0].scatter(nuts_trace.x, nuts_trace.y)
axarr[0].set_title("NUTS samples")
axarr[1].scatter(met_trace.x, met_trace.y)
axarr[1].set_title("MH samples")
axarr[2].scatter(gibbs_trace.x, gibbs_trace.y)
axarr[2].set_title("Gibbs samples")
plt.show()

