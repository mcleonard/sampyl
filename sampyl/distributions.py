""" 

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

Distribution log likelihoods for building Bayesian models. 

These should all automatically sum the log likelihoods if `x` is a numpy array.

"""

import numbers
from sampyl.core import np
from scipy.special import gamma


def fails_constraints(*conditions):
    """ Utility function for catching out of bound parameters. Returns True if 
        any of the conditions aren't met. Typically you'll use this at the
        beginning of defining the log P(X) functions. Example ::

            def logp(x, y):
                # Bound x and y to be greater than 0
                if outofbounds(x > 0, y > 0):
                    return -np.inf
    
    """

    for each in conditions:
        if not np.all(each):
            return True
    else:
        return False


def normal(x, mu=0, sig=1):
    """ Normal distribution log-likelihood.

        :param x:  *int, float, np.array.*
        :param mu: (optional) *int, float, np.array.* 
            Location parameter of the normal distribution. Defaults to 0.
        :param sig: (optional) *int, float.* 
            Standard deviation of the normal distribution, :math:`\sigma > 0`.
            Defaults to 1.

        .. math::
            \log{P(x; \mu, \sigma)} \propto -\log{\sigma} \
             - \\frac{(x - \mu)^2}{2 \sigma^2}

    """

    if np.size(mu) != 1 and len(x) != len(mu):
        raise ValueError('If mu is a vector, x must be the same size as mu.'
                         ' We got x={}, mu={}'.format(x, mu))

    if fails_constraints(sig >= 0):
        return -np.inf

    return np.sum(-np.log(sig) - (x - mu)**2/(2*sig**2))


def half_normal(x, mu=0, sig=1):
    if fails_constraints(x >= 0):
        return -np.inf

    return normal(x, mu=mu, sig=sig)


def uniform(x, lower=0, upper=1):
    """ Uniform distribution log-likelihood. Bounds are inclusive.

        :param x:  *int, float, np.array.*
        :param lower: (optional) *int, float.* Lower bound, default is 0.
        :param upper: (optional) *int, float.* Upper bound, default is 1.

        .. math ::

            \log{P(x; a, b)} = -n\log(b-a)

    """

    if fails_constraints(x >= lower, x <= upper):
        return -np.inf
    
    return -np.size(x) * np.log(upper-lower)


def discrete_uniform(x, lower=0, upper=1):
    """ Discrete Uniform distribution log-likelihood.

        :param x:  *int, np.array[int].* 
        :param lower: (optional) *int, float.* Lower bound, default is 0.
        :param upper: (optional) *int, float.* Upper bound, default is 1.

        .. math ::

            \log{P(x; a, b)} = -n\log(b-a)
    """

    if fails_constraints(x >= lower, x <= upper):
        return -np.inf

    if isinstance(x, np.ndarray):
        if x.dtype != np.int_:
            raise ValueError('x must be integers, function received {}'.format(x))
        else:
            return -np.size(x) * np.log(upper-lower)
    elif isinstance(x, numbers.Integral):
        return -np.log(upper-lower)
    else:
        return -np.inf



def exponential(x, rate=1):
    """ Log likelihood of the exponential distribution. 

        :param x:  *int, float, np.array.*
        :param rate: (optional) *int, float, np.array.* Rate parameter, :math:`\lambda > 0`. Defaults to 1.

        .. math ::
            
            \log{P(x; \lambda)} \propto \log{\lambda} - \lambda x
    """

    if fails_constraints(x > 0, rate > 0):
        return -np.inf

    if np.size(rate) != 1 and len(x) != len(rate):
        raise ValueError('If rate is a vector, x must be the same size as rate.'
                         ' We got x={}, rate={}'.format(x, rate))
    return np.sum(np.log(rate) - rate*x)


def poisson(x, rate=1):
    """ Poisson distribution log-likelihood.

        :param x:  *int, float, np.array.* Event count.
        :param rate: (optional) *int, float, np.array.* Rate parameter, :math:`\lambda > 0`. Defaults to 1.
            

        .. math ::
            \log{P(x; \lambda)} \propto x \log{\lambda} - \lambda

    """

    if fails_constraints(rate > 0):
        return -np.inf
    
    if np.size(rate) != 1 and len(x) != len(rate):
        raise ValueError('If rate is a vector, x must be the same size as rate.'
                         ' We got x={}, rate={}'.format(x, rate))
    return np.sum(x*np.log(rate)) - np.size(x)*rate


def binomial(k, n, p):
    """ Binomial distribution log-likelihood.

        :param k: *int, np.array.* Number of successes. :math:`k <= n`
        :param n: *int, np.array.* Number of trials. :math:`n > 0`
        :param p: *int, float, np.array.* Success probability. :math:`0<= p <= 1`
        
        .. math::
            \log{P(k; n, p)} \propto k \log{p} + (n-k)\log{(1-p)}
    """
    if k > n:
        raise ValueError("k must be less than or equal to n")
    if fails_constraints(0 < p, p < 1):
        return -np.inf
    return np.sum(k*np.log(p) + (n-k)*np.log(1-p))


def bernoulli(k, p):
    """ Bernoulli distribution log-likelihood. 

        :param k: *int, np.array.* Number of successes.
        :param p: *int, float, np.array.* Success probability.

        Special case of binomial distribution, with n set to 1.
    """

    return binomial(k, 1, p)


def beta(x, alpha=1, beta=1):
    """ Beta distribution log-likelihood.

        :param x: *float, np.array.* :math:`0 < x < 1`
        :param alpha: (optional) *int, float.* Shape parameter, :math:`\\alpha > 0`
        :param beta: (optional) *int, float.* Shape parameter, :math:`\\beta > 0`

        .. math ::
            \log{P(x; \\alpha, \\beta)} \propto (\\alpha - 1)\log{x} + \
                                            (\\beta - 1) \log{(1 - x)}
    """

    if fails_constraints(0 < x, x < 1, alpha > 0, beta > 0):
        return -np.inf
    return np.sum((alpha - 1)*np.log(x) + (beta - 1)*np.log(1-x))


def student_t(x, nu=1):
    """ Student's t log-likelihood

        :param x: *int, float, np.array.*
        :param nu: (optional) *int.* Degress of freedom.
    
        .. math ::
            \log{P(x; \\nu)} \propto \log{\Gamma \\left(\\frac{\\nu+1}{2} \\right)} - \
                                     \log{\Gamma \left( \\frac{\\nu}{2} \\right) } - \
                                     \\frac{1}{2}\log{\\nu} - \
                                     \\frac{\\nu+1}{2}\log{\left(1 + \\frac{x^2}{\\nu} \\right)}
    """
    
    if fails_constraints(nu >= 1):
        return -np.inf

    return np.sum(np.log(gamma(0.5*(nu + 1))) - np.log(gamma(nu/2.)) - \
            0.5*np.log(nu) - (nu+1)/2*np.log(1+x**2/nu))


def laplace(x, mu, tau):
    """ Laplace distribution log-likelihood 

        :param x: *int, float, np.array.* :math:`-\infty < \mu < \infty`
        :param mu: *int, float, np.array.* Location parameter. :math:`-\infty < \mu < \infty`
        :param tau: *int, float.* Scale parameter, :math:`\\tau > 0`

        .. math ::
            \log{P(x; \\mu, \\tau)} \propto \log{\\tau/2} - \\tau \\left|x - \mu \\right|

    """
    if fails_constraints(tau > 0):
        return -np.inf
    
    return np.sum(np.log(tau) - tau*np.abs(x - mu))


def cauchy(x, alpha=0, beta=1):
    """ Cauchy distribution log-likelihood.

        :param x: *int, float, np.array.* :math:`-\infty < x < \infty`
        :param alpha: *int, float, nparray.* Location parameter, :math:`-\infty < \\alpha < \infty`
        :param beta: *int, float.* Scale parameter, :math:`\\beta > 0`

        .. math::
            \log{P(x; \\alpha, \\beta)} \propto -\log{\\beta} - \
                                                \log{\left[1 + \left(\\frac{x - \\alpha}{\\beta}\\right)^2\\right]} 


    """
    if fails_constraints(beta > 0):
        return -np.inf

    return np.sum(-np.log(beta) - np.log(1 + ((x - alpha)/beta)**2))


def half_cauchy(x, alpha=0, beta=1):
    """ Half-Cauchy distribution log-likelihood (positive half).

        :param x: *int, float, np.array.* :math:`-\infty < x < \infty`
        :param alpha: *int, float, nparray.* Location parameter, :math:`-\infty < \\alpha < \infty`
        :param beta: *int, float.* Scale parameter, :math:`\\beta > 0`

        .. math::
            \log{P(x; \\alpha, \\beta)} \propto -\log{\\beta} - \
                                                \log{\left[1 + \left(\\frac{x - \\alpha}{\\beta}\\right)^2\\right]} 


    """
    if fails_constraints(x > 0):
        return -np.inf

    return cauchy(x, alpha=alpha, beta=beta)


def weibull(x, l, k):
    """ Weibull distribution log-likelihood. 

        :param x: *int, float, np.array.* :math:`x > 0`
        :param l: *float.* Scale parameter. :math:`\\lambda > 0`
        :param k: *float.* Shape parameter. :math:`k > 0`

    """

    if fails_constraints(l > 0, k > 0, x > 0):
        return -np.inf

    return np.sum(np.log(k/l) + (k-1)*np.log(x/l) - (x/l)**k)








