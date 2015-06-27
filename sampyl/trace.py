""" Subclassing numpy.ndarray to add some utility things to the sample arrays.
"""

from .core import np


class Trace(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, var_names=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer,
                                 offset, strides, order)
        obj.var_names = var_names
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.var_names = getattr(obj, 'var_names', None)

    def __getattr__(self, name):
        if self.var_names is not None and name in self.var_names:
            return self[:, self.var_names.index(name)]
        else:
            return self.__getattribute__(name)
