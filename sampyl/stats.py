""" Module for statistical calculations """

from .core import np


def hpd(chain):
    pass

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