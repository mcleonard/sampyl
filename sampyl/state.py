"""
sampyl.state
~~~~~~~~~~~~~~~~~~~~

Module for State object which stores sampler states in a dictionary.

:copyright: (c) 2015 by Mat Leonard.
:license: MIT, see LICENSE for more details.

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
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__add__')
        elif isinstance(other, collections.Iterable):
            return handle_iterable(self, other, '__add__')
        else:
            raise TypeError("Addition not supported for State and {}".format(other))

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__sub__')
        elif isinstance(other, collections.Iterable):
            return handle_iterable(self, other, '__sub__')
        else:
            raise TypeError("Subtraction not supported for State and {}".format(other))

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__mul__')
        else:
            raise TypeError("Multiplication not supported for State and {}".format(other))

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__truediv__')
        else:
            raise TypeError("Division not supported for State and {}".format(other))

    def __radd__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            # Commutative, so nothing changes
            return self + other
        else:
            raise TypeError("Can only broadcast from the left.")

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            # Commutative, so nothing changes
            return self * other
        else:
            raise TypeError("Can only broadcast from the left.")

    def __rsub__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__rsub__')
        elif isinstance(other, collections.Iterable):
            return handle_iterable(self, other, '__rsub__')
        else:
            raise TypeError("Subtraction not supported for State and {}".format(other))

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return handle_number(self, other, '__truediv__')
        else:
            raise TypeError("Division not supported for State and {}".format(other))


def handle_number(state, other, operator):
    vals = [getattr(state[var], operator)(other) for var in state]
    try:
        if NotImplemented in vals:
            vals = [getattr(other, operator)(state[var]) for var in state]
    except ValueError:
        pass
    return State([(var, val) for var, val in zip(state, vals)])


def handle_iterable(state, other, operator):
    if len(other) != len(state):
        # This might be the case:
        #        State({'x': np.array(1, 2, 3)}) + np.array([2,3,4])
        # So check if both are numpy arrays, then add
        # But first, we can only do this is len(state) is 1.
        if len(state) != 1:
            raise ValueError("Can't broadcast with sizes state: {},"
                             " other: {}".format(len(state), len(other)))
        var = list(state.keys())[0]
        val = state[var]
        if type(val) == np.ndarray and type(other) == np.ndarray:
            return State([(var, getattr(val, operator)(other))])
        else:
            raise ValueError("Can only operate on numpy arrays.")
    if isinstance(other, dict):
        vals = [getattr(state[var], operator)(other[var]) for var in state]
    else:
        # Otherwise, we have cases like
        # State({'x': foo, 'y': bar}) + [foo2, bar2]
        vals = [getattr(state[var], operator)(each) for var, each in zip(state, other)]
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