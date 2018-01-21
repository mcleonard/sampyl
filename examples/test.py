import sampyl as smp
from sampyl import np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import seaborn as sns
N = 200

# True parameters
sigma = 1
true_b = np.array([2, 1, 4])

# Features, including a constant
X = np.ones((N, len(true_b)))
X[:,1:] = np.random.rand(N, len(true_b)-1)

# Outcomes
y = np.dot(X, true_b) + np.random.randn(N)*sigma
pass

plt.plot(y,X[:,1],'r.')
plt.plot(y,X[:,2],'g.')

# Here, b is a length 3 array of coefficients
def logp(b, sig):
    if smp.outofbounds(sig > 0):
        return -np.inf

    # Predicted value
    y_hat = np.dot(X, b)

    # Log-likelihood
    llh = smp.normal(y, mu=y_hat, sig=sig)

    # log-priors
    prior_sig = smp.exponential(sig)
    prior_b = smp.normal(b, mu=0, sig=100)

    return llh + prior_sig + prior_b

start = smp.find_MAP(logp, {'b': np.ones(3), 'sig': 1.})

nuts = smp.NUTS(logp, start)
chain = nuts.sample(2100, burn=100)

plt.ion()
plt.subplots()
for i in range(3):
    plt.hist(chain.b[:,i])
plt.show()

# slice traces
plt.subplots()
plt.plot(chain.b)
plt.show()
pass

slice = smp.Slice(logp,start)
trace = slice.sample(2100)

# histogram for SLice results
plt.subplots()
for i in range(3):
    plt.hist(trace.b[:,i])
plt.show()

# slice traces
plt.subplots()
plt.plot(trace.b)
plt.show()


pass

