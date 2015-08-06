""" Model for access to logp and grad functions """


import collections

from .core import auto_grad_logp, np
from .state import func_var_names


class BaseModel(object):
    def __init__(self):
        self._logp_cache = {}
        self._grad_cache = {}

    def logp(self, state):
        pass

    def grad(self, state):
        pass

    def __call__(self, state):
        return self.logp(state), self.grad(state)

    def clear_cache(self):
        self._logp_cache = {}
        self._grad_cache = {}


class SingleModel(BaseModel):
    """ A model for a logp function that returns both the cost function and
        the gradient. Caches values to improve performance. """

    def __init__(self, logp_func):
        super(SingleModel, self).__init__()
        self.logp_func = logp_func

    def logp(self, state):
        frozen_state = state.freeze()
        if not isinstance(frozen_state, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            logp_value, _ = self.logp_func(*state.values())
            return logp_value

        if frozen_state in self._logp_cache:
            logp_value = self._logp_cache[frozen_state]
        else:
            logp_value, grad_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value
            self._grad_cache[frozen_state] = grad_value

        return logp_value

    def grad(self, state):
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()
        if not isinstance(frozen_state, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            _, grad_value = self.logp_func(*state.values())
            return grad_value

        if frozen_state in self._grad_cache:
            grad_value = self._grad_cache[frozen_state]
        else:
            logp_value, grad_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value
            self._grad_cache[frozen_state] = grad_value

        return grad_value


class Model(BaseModel):
    """ A model for separate logp and grad_logp functions. """
    def __init__(self, logp_func, grad_func=None, grad_logp_flag=False):
        super(Model, self).__init__()
        self.logp_func = check_logp(logp_func)
        self.grad_func = check_grad_logp(logp_func, grad_func, grad_logp_flag)

    def logp(self, state):
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()
        if not isinstance(frozen_state, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            logp_value = self.logp_func(*state.values())
            return logp_value

        if frozen_state in self._logp_cache:
            logp_value = self._logp_cache[frozen_state]
        else:
            logp_value = self.logp_func(*state.values())
            self._logp_cache[frozen_state] = logp_value

        return logp_value

    def grad(self, state):
        # Freeze the state as a tuple so we can use it as a dictionary key
        frozen_state = state.freeze()
        if not isinstance(frozen_state, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            grad_value = grad_vec(self.grad_func, state)
            return grad_value

        #print(type(frozen_state), frozen_state)
        if frozen_state in self._grad_cache:
            grad_value = self._grad_cache[frozen_state]
        else:
            grad_value = grad_vec(self.grad_func, state)
            self._grad_cache[frozen_state] = grad_value

        return grad_value


def init_model(logp, grad_logp=None, grad_logp_flag=False):
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
        return np.array([grad_logp(*state.values())])
    else:
        # got a dictionary instead
        return np.array([grad_logp[each](*state.values()) for each in state])


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