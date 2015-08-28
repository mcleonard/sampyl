"""
sampyl.model
~~~~~~~~~~~~~~~~~~~~

Model for access to logp and grad functions.

:copyright: (c) 2015 by Mat Leonard.
:license: Apache2, see LICENSE for more details.

"""


import collections

from sampyl.core import auto_grad_logp, np
from sampyl.state import State, func_var_names


__all__ = ['init_model', 'Model', 'SingleModel']


class BaseModel(object):
    """ Base model for subclassing. """
    def __init__(self):
        self._logp_cache = {}
        self._grad_cache = {}

    def logp(self, state):
        """ Return log P(X) given a :ref:`state <state>` X"""
        pass

    def grad(self, state):
        pass

    def __call__(self, state):
        """ Return log P(X) and grad log P(X) given a :ref:`state <state>` X"""
        return self.logp(state), self.grad(state)

    def clear_cache(self):
        """ Clear caches. """
        del self._logp_cache
        del self._grad_cache
        self._logp_cache = {}
        self._grad_cache = {}


class SingleModel(BaseModel):
    """ A model for a logp function that returns both the cost function and
        the gradient. Caches values to improve performance. 

        :param logp_func: Function that returns log P(X) and its gradient. 
    """

    def __init__(self, logp_func):
        super(SingleModel, self).__init__()
        self.logp_func = logp_func
        self.var_names = func_var_names(logp_func)

    def logp(self, state):
        """ Return log P(X) given a :ref:`state <state>` X"""
        frozen_state = state.freeze()

        if frozen_state in self._logp_cache:
            logp_value = self._logp_cache[frozen_state]
        else:
            logp_value, grad_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value
            self._grad_cache[frozen_state] = grad_value

        return logp_value

    def grad(self, state):
        """ Return grad log P(X) given a :ref:`state <state>` X """
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()

        if frozen_state in self._grad_cache:
            grad_value = self._grad_cache[frozen_state]
        else:
            logp_value, grad_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value
            self._grad_cache[frozen_state] = grad_value

        return grad_state(grad_value, state)


class Model(BaseModel):
    """ A model for separate logp and grad_logp functions. 

        :param logp: 
            log P(X) function for sampling distribution.
        :param grad_logp: (optional) *function or list of functions.*
            Gradient log P(X) function. If left as None, then `grad_logp_flag`
            is checked. If the flag is `True`, then the gradient will be 
            automatically calculated with autograd.
        :param grad_logp_flag: (optional) *boolean.*
            Flag indicating if the gradient is needed or not.
    

    """
    def __init__(self, logp_func, grad_func=None, grad_logp_flag=False):
        super(Model, self).__init__()
        self.logp_func = check_logp(logp_func)
        self.grad_func = check_grad_logp(logp_func, grad_func, grad_logp_flag)
        self.var_names = func_var_names(logp_func)

    def logp(self, state):
        """ Return log P(X) given a :ref:`state <state>` X"""
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()

        if frozen_state in self._logp_cache:
            logp_value = self._logp_cache[frozen_state]
        else:
            logp_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value

        return logp_value

    def grad(self, state):
        """ Return grad log P(X) given a :ref:`state <state>` X """
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()
        if frozen_state in self._grad_cache:
            grad_value = self._grad_cache[frozen_state]
        else:
            grad_value = grad_vec(self.grad_func, state)
            self._grad_cache[frozen_state] = grad_value

        return grad_state(grad_value, state)


def init_model(logp, grad_logp=None, grad_logp_flag=False):
    """ Initialize a model and return it.

        :param logp: 
            log P(X) function for sampling distribution.
        :param grad_logp: (optional) *function, list of functions, or boolean.*
            Gradient log P(X) function. If left as None, then `grad_logp_flag`
            is checked. If the flag is `True`, then the gradient will be 
            automatically calculated with autograd.

            If `grad_logp` is set to True, then a SingleModel is returned.
        :param grad_logp_flag: (optional) *boolean.*
            Flag indicating if the gradient is needed or not.
    """

    if grad_logp is True:
        return SingleModel(logp)
    else:
        return Model(logp, grad_logp, grad_logp_flag)


def grad_vec(grad_logp, state):
    """ grad_logp should be a function, or a dictionary of gradient functions, 
        respective to each parameter in logp
    """
    
    if hasattr(grad_logp, '__call__'):
        # grad_logp is a single function
        grad_val = grad_logp(*state.values())
        return grad_val
    else:
        # got a dictionary instead
        grad_val = np.array([grad_logp[val](*state.values()) for val in state])
        return grad_val


def grad_state(grad_val, state):
    g_state = State.fromkeys(state.keys())
    if len(state) == 1:
        g_state.update({var: grad_val for var in state})
    else:
        g_state.update({var: val for var, val in zip(state, grad_val)})
    return g_state


def check_logp(logp):
    if not hasattr(logp, '__call__'):
        raise TypeError("logp must be a function")
    elif logp.__code__.co_argcount == 0:
        raise ValueError("logp must have arguments")
    else:
        return logp


def check_grad_logp(logp, grad_logp, grad_logp_flag):
    var_names = func_var_names(logp)
    if grad_logp_flag and grad_logp is None:
        return auto_grad_logp(logp)
    elif grad_logp_flag and grad_logp != 'logp':
        # User defined grad_logp function
        if len(var_names) > 1 and len(grad_logp) != len(var_names):
            raise TypeError("grad_logp must be iterable with length equal"
                                " to the number of parameters in logp.")
        else:
            return grad_logp
    else:
        return grad_logp