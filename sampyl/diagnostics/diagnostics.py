import numpy as np
import statsmodels.tsa.stattools as stattools

def compute_r_hat(theta_chains):
    """ given a Nchains x Nsamps array, compute the potential scale reduction 
        factor, gelman-rubin convergence diagnostic.

        Input:
          theta_chains: m x n matrix of MCMC chains
                        (m = number of chains, n = number of samples per chain)

        Output:
          R_hat: potential scale reduction factor (page 297 in BDA 2)
    """
    m, n = theta_chains.shape
    chain_means = theta_chains.mean(axis=1)   # Nchains mean values
    grand_mean  = theta_chains.mean()

    # compute between chain variance
    B_over_n = 1. / (m - 1) * np.sum( (chain_means - grand_mean)**2 )

    # compute within sequence variance
    W = 1./(m*(n-1)) * np.sum( (theta_chains.T - chain_means)**2 )

    # compute pooled esitmate
    var_plus = (n-1.)/n * W + B_over_n

    # compute estimate scale reduction factor
    R_hat = (m+1.)/m * var_plus / W - (n-1.) / (m*n)
    return np.sqrt(R_hat)

def compute_n_eff(theta_chains):
    """ Compute n_effective from BDA """
    m, n = theta_chains.shape
    chain_means = theta_chains.mean(axis=1)   # Nchains mean values
    grand_mean  = theta_chains.mean()

    # compute between chain variance
    B_over_n = 1. / (m - 1) * np.sum( (chain_means - grand_mean)**2 )
    # compute within sequence variance
    W = 1./(m*(n-1)) * np.sum( (theta_chains.T - chain_means)**2 )
    # compute pooled esitmate
    var_plus = (n-1.)/n * W + B_over_n
    n_eff = m * n * var_plus / (n*B_over_n)
    return n_eff

def compute_n_eff_acf(theta_chain):
    """ computes autocorrelation based effective sample size"""
    n = theta_chain.shape[0]
    return n / (1. + 2 * stattools.acf(theta_chain)[1:].sum())


