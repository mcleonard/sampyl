""" Module for calculating the maximum a posteriori for use in a starting
    value for the samplers. """

from .core import np, AUTOGRAD
from scipy.optimize import minimize
from .state import State


def find_MAP(logp, start, grad_logp=None,
             method=None, bounds=None, verbose=False):

    """ Find the maximum a posteriori of logp. Requires a starting state. """

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
        jac = grad(neg_logp, 0)
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
                       jac=jac, method=method, bounds=bnds)

    if verbose:
        print(results)

    return state.fromvector(results.x)
