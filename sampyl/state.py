from __future__ import division

from .core import np
from collections import OrderedDict


class State(OrderedDict):

    def tovector(self):
        return np.hstack(self.values())

    def size(self):
        return State([(var, np.size(self[var])) for var in self])

    def __add__(self, other):

        if not hasattr(other, '__iter__'):
            sums = [self[each] + other for each in self.keys()]
        else:
            try:
                sums = [self[each] + other[each] for each in self.keys()]
            except IndexError:
                sums = [self[i] + j for i, j in zip(self, other)]
        out = State(zip(self.keys(), sums))
        return out

    def __sub__(self, other):
        if not hasattr(other, '__iter__'):
            subs = [self[each] - other for each in self.keys()]
        else:
            try:
                subs = [self[each] - other[each] for each in self.keys()]
            except IndexError:
                subs = [self[i] - j for i, j in zip(self, other)]
        out = State(zip(self.keys(), subs))
        return out

    def __mul__(self, other):
        if not hasattr(other, '__iter__'):
            muls = [self[each] * other for each in self.keys()]
        else:
            try:
                muls = [self[each] * other[each] for each in self.keys()]
            except IndexError:
                muls = [self[i] * j for i, j in zip(self, other)]
        out = State(zip(self.keys(), muls))
        return out

    def __truediv__(self, other):
        if not hasattr(other, '__iter__'):
            divs = [self[each] / other for each in self.keys()]
        else:
            try:
                divs = [self[each] / other[each] for each in self.keys()]
            except IndexError:
                divs = [self[i] / j for i, j in zip(self, other)]
        out = State(zip(self.keys(), divs))
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        if not hasattr(other, '__iter__'):
            subs = [other - self[each] for each in self.keys()]
        else:
            try:
                subs = [other[each] - self[each] for each in self.keys()]
            except IndexError:
                subs = [j - self[i] for i, j in zip(self, other)]
        out = State(zip(self.keys(), subs))
        return out

    def __rtruediv__(self, other):
        if not hasattr(other, '__iter__'):
            divs = [other / self[each] for each in self.keys()]
        else:
            try:
                divs = [other[each] / self[each] for each in self.keys()]
            except IndexError:
                divs = [j / self[i] for i, j in zip(self, other)]
        out = State(zip(self.keys(), divs))
        return out
