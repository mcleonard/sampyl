"""
sampyl.starting
~~~~~~~~~~~~~~~~~~~~

Module for calculating the maximum a posteriori for use in a starting
value for the samplers.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""

from .core import np, AUTOGRAD, auto_grad_logp
from scipy.optimize import minimize
from .state import State


def find_MAP(logp, start, grad_logp=None,
             method=None, bounds=None, verbose=False, **kwargs):

    """ Find the maximum a posteriori of logp. Requires a starting state.
        Optimizing is done with scipy.optimize.minimize. Documentation can be
        found here:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Arguments
        ---------
        logp: function
            log P(X) function for sampling distribution
        start: dict
            Dictionary of starting state for the optimizer. Should have one
            element for each argument of logp. So, if logp = f(x, y), then
            start = {'x': x_start, 'y': y_start}

        Keyword Arguments
        -----------------
        grad_logp: function
            grad log P(X) function for calculating the gradient. Uses autograd
            automatically if it is installed. Otherwise, you can pass in a
            gradient function to the optimizer.
        method: string
            Optimizing method, one of these or a callable function:
                'Nelder-Mead'
                'Powell'
                'CG'
                'BFGS'
                'Newton-CG'
                'Anneal (deprecated as of scipy version 0.14.0)'
                'L-BFGS-B'
                'TNC'
                'COBYLA'
                'SLSQP'
                'dogleg'
                'trust-ncg'
                custom - a callable object (added in version 0.14.0)
            If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending 
            if the problem has constraints or bounds.
        bounds: dict of tuples
            Tuples of bounding values for each parameter, example:
            bounds = {'x': (low, high), 'y': (low, high)}
            Use None if not bounded in a direction, ex: {'x': (0, None)}
        verbose: boolean
            Set to True to print out optimization information

    """

    # Making sure to get the keys from logp function so that arguments are
    # ordered correctly, then update from starting state.
    state = State.fromfunc(logp)
    state.update(start)

    # We find the MAP by minimizing, we need to negate the logp function
    def neg_logp(x):
        # Because the minimize function passes a single array, we need to put
        # it back into a state form so we can pass each variable to logp if
        # there are multiple variables
        args = state.fromvector(x)
        return -1*logp(*args.values())

    if AUTOGRAD and grad_logp is None:
        jac = auto_grad_logp(neg_logp)['x']
    else:
        jac = grad_logp

    # Formatting bounds correctly to be passed to the minimize function
    # If a variable is an array, you would have to manually write out tuples
    # for each element in that array. Here, you can define one bound, and it
    # is expanded out to the size of the variable.
    if bounds is not None:
        bnds = []
        for var in bounds:
            bnds.extend([bounds[var]]*state.size()[var])
    else:
        bnds = None

    results = minimize(neg_logp, state.tovector(),
                       jac=jac, method=method, bounds=bnds, **kwargs)

    if verbose:
        print(results)

    return state.fromvector(results.x)
