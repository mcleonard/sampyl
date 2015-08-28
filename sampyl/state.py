"""
sampyl.state
~~~~~~~~~~~~~~~~~~~~

Module for State object which stores sampler states in a dictionary.

:copyright: (c) 2015 by Mat Leonard.
:license: Apache2, see LICENSE for more details.

"""

from __future__ import division

import sampyl
from sampyl.core import np
import collections


class State(collections.OrderedDict):
    """ State object for storing parameter values.
        
        Inherits from OrderedDict.

    """

    def tovector(self):
        """ Return the parameter values as a flat vector. """
        return np.hstack(self.values())

    def fromvector(self, vec):
        """ Update the state using a numpy array. 

            :param vec: np.array for updating the state.
        """
        var_sizes = self.size()
        i = 0
        for var in self:
            self[var] = np.squeeze(vec[i:(i+var_sizes[var])])
            i += var_sizes[var]
        return self

    def freeze(self):
        """ Return a immutable tuple of the state values."""
        return tuple(self.tovector())

    @staticmethod
    def init_fromvector(vec, state):
        ""
        vals = []
        var_sizes = state.size()
        i = 0
        for var in state:
            vals.append(np.squeeze(vec[i:(i+var_sizes[var])]))
            i += var_sizes[var]
        return State(zip(state.keys(), vals))

    @staticmethod
    def fromfunc(func):
        """ Initialize a State from the arguments of a function """
        var_names = func_var_names(func)
        return State.fromkeys(var_names)

    def size(self):
        return State([(var, np.size(self[var])) for var in self])

    def __add__(self, other):
        return handle_special(self, other, '__add__')

    def __sub__(self, other):
        return handle_special(self, other, '__sub__')

    def __mul__(self, other):
        return handle_special(self, other, '__mul__')

    def __truediv__(self, other):
        return handle_special(self, other, '__truediv__')

    def __radd__(self, other):
        return handle_special(self, other, '__radd__')

    def __rmul__(self, other):
        return handle_special(self, other, '__rmul__')

    def __rsub__(self, other):
        return handle_special(self, other, '__rsub__')

    def __rtruediv__(self, other):
       return handle_special(self, other, '__rtruediv__')


def handle_special(state, other, operator):
    if isinstance(other, int) or isinstance(other, float):
        return handle_number(state, other, operator)
    elif isinstance(other, dict):
        return handle_iterable(state, other, operator)
    else:
        raise TypeError("{} not supported for State and {}".format(operator, other))

def handle_number(state, other, operator):
    # Here, other is a float or integer
    vals = []
    for var in state:
        val = state[var]
        if isinstance(val, int) or isinstance(val, float):
            vals.append(getattr(float(val), operator)(other))
        elif isinstance(val, np.ndarray):
            vals.append(getattr(val.astype(float), operator)(other))
        else:
            raise TypeError('States can only contain Numpy arrays and scalars.'
                            ' We got {}'.format(state))
    if NotImplemented in vals:
        raise ValueError('Got NotImplemented for {}.{}({})'.format(state, operator, other))
    return State([(var, val) for var, val in zip(state, vals)])


def handle_iterable(state, other, operator):
    vals = [getattr(state[var], operator)(other[var]) for var in state]

    return State([(var, val) for var, val in zip(state, vals)])

def special_math_func(state, other, operator):
    """ A function for special math functions used in the State class.
        So, we need to handle state + 1, state + np.array(),
        state1 + state2, etc. basically we want to do the same thing
        every time but with different operators.
    """
    new = State([(var, vals) for var, each in zip(state, vals)])

    return new


def func_var_names(func):
    """ Returns a list of the argument names in func """
    names = func.__code__.co_varnames[:func.__code__.co_argcount]
    return names