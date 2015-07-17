import sys
sys.path.append('.')
import sampyl as smp
from sampyl.state import State
from sampyl import np
from sampyl.diagnostics import diagnostics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# correlated gaussian log likelihood
def logp(x, y):
    icov = np.linalg.inv(np.array([[1., .8], [.8, 1.]]))
    d = np.array([x, y])
    return -.5 * np.dot(np.dot(d, icov), d)
logp_xy = lambda(th): logp(th[0], th[1])

# compare slice samplers, metropolis hastings, and the two variable 
# slice sampler
ssamp = smp.Slice(logp, start={'x': 4., 'y': 4.} )
slice_trace = ssamp.sample(1000)

met = smp.Metropolis(logp, start={'x': 4., 'y': 4.})
met_trace = met.sample(1000)

bslice = smp.Slice(logp_xy, start={'th': np.array([4., 4.])})
btrace = bslice.sample(1000)

# compute effective sample size based on autocorrelation
slice_eff = diagnostics.compute_n_eff_acf(slice_trace.x)
met_eff   = diagnostics.compute_n_eff_acf(met_trace.x)
b_eff     = diagnostics.compute_n_eff_acf(btrace.th[:,0])
print "Slice         effective sample size: %2.2f"%slice_eff
print "MH            effective sample size: %2.2f"%met_eff
print "two var slice effective sample size: %2.2f"%b_eff

print " ----- "
print "Slice sampler evals per sample: ", ssamp.evals_per_sample

# graphically compare samples
fig, axarr = plt.subplots(1, 3, figsize=(12,4))
axarr[0].scatter(slice_trace.x, slice_trace.y)
axarr[0].set_title("Slice samples")
axarr[1].scatter(met_trace.x, met_trace.y)
axarr[1].set_title("MH samples")
axarr[2].scatter(btrace.th[:,0], btrace.th[:,1])
axarr[2].set_title("Two var Slice samples")
for ax in axarr:
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
plt.show()

