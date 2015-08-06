from __future__ import division

from .core import np
from collections import OrderedDict


class State(OrderedDict):
    """ State object for storing parameter values.
        Inherits from OrderedDict.
    """

    def tovector(self):
        """ Return the parameter values as a flat vector. """
        return np.hstack(self.values())

    def fromvector(self, vec):
        var_sizes = self.size()
        i = 0
        for var in self:
            self[var] = np.squeeze(vec[i:(i+var_sizes[var])])
            i += var_sizes[var]
        return self

    def freeze(self):
        return tuple(self.tovector())

    @staticmethod
    def init_fromvector(vec, state):
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

        return special_math_func(self, other, '__add__')

    def __sub__(self, other):
        return special_math_func(self, other, '__sub__')

    def __mul__(self, other):
        return special_math_func(self, other, '__mul__')

    def __truediv__(self, other):
        return special_math_func(self, other, '__truediv__')

    def __radd__(self, other):
        # Commutative, so nothing changes
        return self.__add__(other)

    def __rmul__(self, other):
        # Commutative, so nothing changes
        return self.__mul__(other)

    def __rsub__(self, other):
        return special_math_func(self, other, '__rsub__')

    def __rtruediv__(self, other):
        return special_math_func(self, other, '__rtruediv__')


def special_math_func(state, other, operator):
    """ A function for special math functions used in the State class.
        So, we need to handle state + 1, state + np.array(),
        state1 + state2, etc. basically we want to do the same thing
        every time but with different operators.
    """
    if not hasattr(other, '__iter__'):
        # other is just a number
        results = [getattr(state[each], operator)(other)
                   for each in state.keys()]
    else:
        try:
            # Both are dictionaries
            results = [getattr(state[each], operator)(other[each])
                       for each in state]
        except IndexError:
            # Both are iterables, but other is not a dictionary
            results = [getattr(state[i], operator)(j)
                       for i, j in zip(state, other)]
    out = State(zip(state.keys(), results))
    return out


def func_var_names(func):
    """ Returns a list of the argument names in func """
    names = func.__code__.co_varnames[:func.__code__.co_argcount]
    return names