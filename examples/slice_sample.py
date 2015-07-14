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

# compare the performance of NUTS and Metropolis by effective sample size
ssamp = smp.Slice(logp)
slice_trace = ssamp.sample(1000)

met = smp.Metropolis(logp)
met_trace = met.sample(1000)

# compute effective sample size based on autocorrelation
slice_eff = diagnostics.compute_n_eff_acf(slice_trace.x)
met_eff = diagnostics.compute_n_eff_acf(met_trace.x)
print "Slice effective sample size: %2.2f"%slice_eff
print "MH    effective sample size: %2.2f"%met_eff

# graphically compare samples
fig, axarr = plt.subplots(1, 2)
axarr[0].scatter(slice_trace.x, slice_trace.y)
axarr[0].set_title("Slice samples")
axarr[1].scatter(met_trace.x, met_trace.y)
axarr[1].set_title("MH samples")

print "Slice: evals per sample: ", ssamp.evals_per_sample

for ax in axarr:
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))

sns.jointplot(slice_trace.x, slice_trace.y, kind='kde')

plt.show()

