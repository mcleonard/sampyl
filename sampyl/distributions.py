""" Distribution log likelihoods for building Bayesian models. 

    These should all automatically sum the log likelihoods if `x` is 
    a numpy array.

"""

from .core import np


def outofbounds(*conditions):
    """ Utility function for catching out of bound parameters. Returns True if 
        any of the conditions aren't met. Typically you'll use this at the
        beginning of defining the log P(X) functions. Example ::

            def logp(x, y):
                # Bound x and y to be greater than 0
                if outofbounds(x > 0, y > 0):
                    return np.negative(np.inf)
    
    """

    for each in conditions:
        if not np.all(each):
            return True
    else:
        return False


def normal(x, mu=0, sig=1):
    """ Log likelihood of a normal distribution.

        :param x:  *int, float, np.array.*
        :param mu: (optional) *int, float, np.array.* 
            Location parameter of the normal distribution. Defaults to 0.
        :param sig: (optional) *int, float.* 
            Standard deviation of the normal distribution, sqrt(variance).
            Defaults to 1.

        .. math::
            \log{P(x; \mu, \sigma)} \propto -\log{\sigma} \
             - \\frac{(x - \mu)^2}{2 \sigma^2}

    """

    if np.size(mu) != 1 and len(x) != len(mu):
        raise ValueError('If mu is a vector, x must be the same size as mu.'
                         ' We got x={}, mu={}'.format(x, mu))
    return np.sum(-np.log(sig) - (x - mu)**2/(2*sig**2))


def uniform(x, lower=0, upper=1):
    """ Log likelihood of uniform distribution. 

        :param x:  *int, float, np.array.*
        :param lower: (optional) *int, float.* Lower bound, default is 0.
        :param upper: (optional) *int, float.* Upper bound, default is 1.

        .. math ::

            \log{P(x; a, b)} = -n\log(b-a)

    """

    if outofbounds(x > lower, x < upper):
        return np.negative(np.inf)
    
    return - np.size(x) * np.log(upper-lower)


def exponential(x, rate=1):
    """ Log likelihood of the exponential distribution. 

        :param x:  *int, float, np.array.*
        :param rate: (optional) *int, float, np.array.* Rate parameter, defaults to 1, 
            must be greater than 0.

        .. math ::
            
            \log{P(x; \lambda)} \propto \log{\lambda} - \lambda x
    """

    if outofbounds(rate > 0):
        return np.negative(np.inf)

    if np.size(rate) != 1 and len(x) != len(rate):
        raise ValueError('If rate is a vector, x must be the same size as rate.'
                         ' We got x={}, rate={}'.format(x, rate))
    return np.sum(np.log(rate) - rate*x)


def poisson(x, rate=1):
    """ Log likelihood of the poisson distribution.

        :param x:  *int, float, np.array.* Event count.
        :param rate: (optional) *int, float, np.array.* Rate parameter, defaults to 1, 
            must be greater than 0.

        .. math ::
            \log{P(x; \lambda)} \propto x*\log{\lambda} - \lambda

    """

    if outofbounds(rate > 0):
        return np.negative(np.inf)
    
    if np.size(rate) != 1 and len(x) != len(rate):
        raise ValueError('If rate is a vector, x must be the same size as rate.'
                         ' We got x={}, rate={}'.format(x, rate))
    return np.sum(x*np.log(rate)) - np.size(x)*rate


def binomial(k, n, p):
    """ Log likelihood of the binomial distribution.

        :param k: *int, np.array.* Number of successes.
        :param n: *int, np.array.* Number of trials.
        :param p: *int, float, np.array.* Success probability.
        
        .. math::
            \log{P(k; n, p)} \propto k \log(p) + (n-k)\log(1-p)
    """

    if outofbounds(0 < p, p < 1):
        return np.negative(np.inf)
    return np.sum(k*np.log(p) + (n-k)*np.log(1-p))


def bernoulli(k, p):
    """ Log likelihood for the bernoulli distribution. 

        :param k: *int, np.array.* Number of successes.
        :param p: *int, float, np.array.* Success probability.

        Special case of binomial distribution, with n set to 1.
    """

    return binomial(k, 1, p)


def beta(x, alpha=1, beta=1):
    """ Log likelihood of beta distribution.

        :param x: *float, np.array.* Must be between 0 and 1.
        :param alpha: (optional) *int, float.* Shape parameter, must be greater than 0.
        :param beta: (optional) *int, float.* Shape parameter, must be greater than 0.

        .. math ::
            \log{P(x; \\alpha, \\beta)} \propto (\\alpha - 1)\log(x) + \
                                            (\\beta - 1) \log(1 - x)
    """

    if outofbounds(0 < x, x < 1, alpha > 0, beta > 0):
        return np.negative(np.inf)
    return np.sum((alpha - 1)*np.log(x) + (beta - 1)*np.log(1-x))

def student_t():
    pass


