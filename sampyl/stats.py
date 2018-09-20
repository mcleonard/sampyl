""" Module for statistical calculations """

from __future__ import division as _division

from .core import np
import scipy.optimize as _opt

__all__ = ['hpd', 'percentile', 'mean', 'median', 'summary', 'calc_R_hat', 'calc_n_eff']


def _fit_hpd(data, alpha):
    cost = lambda q: np.diff(np.percentile(data, q=(q[0] + 0, q[0] + alpha), axis=0).T).sum()
    res = _opt.minimize(cost, (100 - alpha)/2., bounds=[(0.00001, 99.9999-alpha)])
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
            hpd = _fit_hpd(param_chain, alpha)
        else:
            hpd = np.array([_fit_hpd(each, alpha) for each in param_chain.T])
        hpds[field] = hpd
                
    return hpds


def percentile(chain, alpha=0.95):
    q = 100*(1 - alpha)/2.
    f = lambda x: np.percentile(x, q=(q, 100 - q), axis=0).T
    fields = chain.dtype.fields.keys()
    return {field: f(chain.field(field)) for field in fields}


##### This part is for calculating R_hat, the potential scale reduction #####
#####                            See Page 284 of BDA3                   #####


def _calc_var_hat(split_chains):
    """ Calculate var_hat from split chains"""
    m, n = split_chains.shape
    chain_means = split_chains.mean(axis=1)
    grand_mean = chain_means.mean()
    
    B = n/(m-1)*np.sum((chain_means - grand_mean)**2)
    
    sj2 = np.sum((split_chains - np.vstack(chain_means))**2, axis=1)/(n-1)
    W = sj2.mean()
    
    var_hat = (n-1)*W/n + B/n
    
    return W, var_hat


def _calc_R_hat(split_chains):
    """ Calculate R_hat from split chains """
    
    W, var_hat = _calc_var_hat(split_chains)
    R_hat = np.sqrt(var_hat/W)
    
    return R_hat


def R_hat(chains):
    
    m = len(chains)*2
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
            R_hat = _calc_R_hat(split_chains)
            R_hats[field] = R_hat
        else:
            R_hats[field] = []
            for each in concat_chain.T:
                split_chains = np.array(np.split(each, m))
                R_hat = _calc_R_hat(split_chains)
                R_hats[field].append(R_hat)
            R_hats[field] = np.array(R_hats[field])

    return R_hats


##### Next part is for calculating effective samples #####
#####                See Page 286 of BDA3            #####

def _variogram(split_chains, t):
    if t == 0:
        return 0
    m, n = split_chains.shape
    return np.sum((split_chains[:,t:] - split_chains[:,:-t])**2)/(m*(n-t))


def _rho_hat(split_chains, t):
    W, var_hat = _calc_var_hat(split_chains)
    vario = _variogram(split_chains, t)
    return 1 - vario/2/var_hat


def calc_n_eff(split_chains):
    m, n = split_chains.shape
    t = 1
    rho_sum = 0
    rhos = []
    while t < n:
        rhos.append(_rho_hat(split_chains, t))
        rho_sum = _rho_hat(split_chains, t+1) + _rho_hat(split_chains, t+2)
        t += 1
        if rho_sum < 0 and t%2 == 1:
            break
    return m*n/(1+2*sum(rhos))


##### Summary statistics #####

def median(chain):
    fields = chain.dtype.fields.keys()
    return {field: np.median(chain.field(field), axis=0) for field in fields}


def mean(chain):
    fields = chain.dtype.fields.keys()
    return {field: np.mean(chain.field(field), axis=0) for field in fields}


def summary(chain):
    means = mean(chain)
    medians = median(chain)
    hpds = hpd(chain)
    
    fields = chain.dtype.fields.keys()

    print('    \tmean\tmedian\t95% HPDI\n')
    for field in fields:
        if chain.field(field).ndim == 1:
            output = '{}\t{:.3f}\t{:.3f}\t[{:.3f}, {:.3f}]'
            print(output.format(field,
                                means[field],
                                medians[field],
                                hpds[field][0],
                                hpds[field][1]))
        else:
            output = '{}.{}\t{:.3f}\t{:.3f}\t[{:.3f}, {:.3f}]'
            zipped_stats = zip(means[field], medians[field], hpds[field])
            for i, (mn, md, hpdi) in enumerate(zipped_stats):
                print(output.format(field, i, mn, md, hpdi[0], hpdi[1]))