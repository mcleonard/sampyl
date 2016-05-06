""" Module for statistical calculations """


from .core import np
import scipy.optimize as opt

__all__ = ['hpd', 'percentile']


def fit_hpd(data, alpha):
    cost = lambda q: np.diff(np.percentile(data, q=(q[0] + 0, q[0] + alpha), axis=0).T)
    res = opt.minimize(cost, (100 - alpha)/2., bounds=[(0.00001, 99.9999-alpha)])
    q_hpd = res.x[0]
    hpd = np.percentile(data, q=(q_hpd, q_hpd + alpha), axis=0).T
    return hpd

def hpd(chain, alpha=0.95):
    """ Return the Highest Posterior Density (HPD) interval 

        Note: This only works for uni-modal distributions!
    """
    hpds = {}
    alpha = 100*alpha
    fields = chain.dtype.fields.keys()
    for field in fields:
        param_chain = chain.field(field)
        if param_chain.ndim == 1:
            hpd = fit_hpd(param_chain, alpha)
        else:
            hpd = np.array([fit_hpd(each, alpha) for each in param_chain.T])
        hpds[field] = hpd
                
    return hpds

def percentile(chain, alpha=0.95):
    q = 100*(1 - alpha)/2.
    f = lambda x: np.percentile(x, q=(q, 100 - q), axis=0).T
    fields = chain.dtype.fields.keys()
    return {field: f(chain.field(field)) for field in fields}

def autocorr(chain):
    pass

def R_hat(chain):
    pass

def effective_samples(chain):
    pass