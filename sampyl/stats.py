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


def calc_R_hat(split_chains):
    """ Calculate R_hat from splitting chains """
    psi_j = split_chains.mean(axis=1)
    psi = psi_j.mean()
    
    B = n/(m-1)*np.sum((psi_j - psi)**2)
    
    sj2 = np.sum((split_chains - np.vstack(psi_j))**2, axis=1)/(n-1)
    W = sj2.mean()
    
    var_hat = (n-1)*W/n + B/n
    R_hat = np.sqrt(var_hat/W)
    
    return R_hat


def R_hat(chain):
    n = int(len(chains[0])/2.)
    m = int(len(chains)*2)
    fields = chains[0].dtype.fields.keys()
    R_hats = {}

    for field in fields:
        concat_chain = np.concatenate([each[field] for each in chains])
        
        # The code is a lot simpler if the chains are split evenly. 
        # To split the chains evenly, we need to trim them down by the remainder
        remainder = len(concat_chain) % m
        if remainder != 0:
            concat_chain = concat_chain[:-remainder]
        
        if chains[0].field(field).ndim == 1:
            split_chains = np.array(np.split(concat_chain, m))
            R_hat = calc_R_hat(split_chains)
            R_hats[field] = R_hat
        else:
            R_hats[field] = []
            for each in concat_chain.T:
                split_chains = np.array(np.split(each, m))
                R_hat = calc_R_hat(split_chains)
                R_hats[field].append(R_hat)
            R_hats[field] = np.array(R_hats[field])

    return R_hats

def effective_samples(chain):
    pass