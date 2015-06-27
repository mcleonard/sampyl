from ..core import np
import sampyl as smp


def normal_1D():
    mu, sig = 3, 2
    logp = lambda x: -0.5*np.log(2*np.pi) - \
                      0.5*np.log(sig**2) - \
                      np.sum((x - mu)**2)/(2*sig**2)
    grad_logp = lambda x: -2*(x - mu)/(2*sig**2)
    return logp, grad_logp


def normal_posterior():
    mu, sig = 10, 3
    data = (np.random.randn(20)*sig + mu)
    n = len(data)

    def logp(mu, sig):
        likelihood = -n*0.5*np.log(2*np.pi) - \
                      n*0.5*np.log(sig**2) - \
                      np.sum((data - mu)**2)/(2*sig**2)
        mu_prior = smp.priors.uniform(mu, 5, 15)
        sig_prior = -np.log(np.abs(sig))
        return likelihood + mu_prior + sig_prior

    def grad_logp(mu, sig):
        pass

    return logp, grad_logp
