from ..core import np, auto_grad_logp
from ..utils import count


class Sampler(object):
    # When subclassing, set this to False if grad logp functions aren't needed

    def __init__(self, logp, grad_logp=None, start=None, scale=1.,
                 condition=None, grad_logp_flag=True):
        self.logp = check_logp(logp)
        self.var_names = logp_var_names(logp)
        self.var_sizes = logp.__annotations__
        self.state = default_start(start, logp)
        self.scale = scale*np.ones(len(self.var_names))
        self.sampler = None
        self._sampled = 0
        self._accepted = 0
        self.conditional = condition
        self._grad_logp_flag = grad_logp_flag

        if self._grad_logp_flag and grad_logp is None:
            self.grad_logp = auto_grad_logp(logp)
        elif self._grad_logp_flag:
            if len(self.var_names) > 1 and len(grad_logp) != len(var_names):
                raise TypeError("grad_logp must be iterable with length equal"
                                " to the number of parameters in logp.")
            else:
                self.grad_logp = grad_logp

        if condition is not None:
            self.joint_logp = self.logp

    def _conditional_step(self):
        """ Build a conditional logp and sample from it. """
        if self.conditional is None:
            return self.step()

        frozen_ind = [self.var_names.index(each) for each in self.conditional]
        frozen_state = self.state
        free_ind = [self.var_names.index(i) for i in self.var_names
                    if i not in self.conditional]

        def conditional_logp(*args):
            conditional_state = list(args)
            # Insert conditional values here, then pass to full logp
            for i in frozen_ind:
                conditional_state.insert(i, frozen_state[i])
            return self.joint_logp(*conditional_state)

        self.state = frozen_state[free_ind]
        self.logp = conditional_logp
        if self._grad_logp_flag:
            self.grad_logp = auto_grad_logp(conditional_logp, n=len(self.state))
        state = self.step()

        # Add the frozen variables back into the state
        new_state = state.tolist()
        for i in frozen_ind:
                new_state.insert(i, frozen_state[i])
        self.state = np.array(new_state)

        return np.array(new_state)


    def step(self):
        pass

    def sample(self, num, burn=-1, thin=1):
        """ Sample from distribution defined by logp.

            Parameters
            ----------

            num: int
                Number of samples to return.
            burn: int
                Number of samples to burn through
            thin: thin
                Thin the samples by this factor
        """
        if self.sampler is None:
            self.sampler = (self.step() for _ in count(start=0, step=1))

        dtypes = [(field, 'f8', self.var_sizes[field]) for field in self.var_names]
        samples = np.zeros(num, dtype=dtypes).view(np.recarray)
        for i in range(num):
            samples[i] = np.hstack(next(self.sampler))

        return samples[burn+1::thin]


def check_logp(logp):
    if not hasattr(logp, '__call__'):
        raise TypeError("logp must be a function")
    elif logp.__code__.co_argcount == 0:
        raise ValueError("logp must have arguments")
    else:
        return logp



def default_start(start, logp):
    """ If start is None, return a zeros array with length equal to the number
        of arguments in logp
    """
    var_sizes = logp.__annotations__
    var_names = logp.__code__.co_varnames[:logp.__code__.co_argcount]
    for each in var_names:
        if each not in var_sizes:
            var_sizes.update({each: 1})

    if start is None:
        start = [np.ones(var_sizes[each]) for each in var_names]
        return np.array(start)
    else:
        # Check that start has the correct sizes
        for i, var in enumerate(start):
            var_size = var_sizes[var_names[i]]
            if var_size > 1 and len(var) != var_size:
                raise ValueError("start must match sizes defined in logp")
        else:
            return np.array(start)


def logp_var_names(logp):
    """ Returns a list of the argument names in logp """
    # Putting underscores after the names so that variables names don't
    # conflict with built in attributes
    names = logp.__code__.co_varnames[:logp.__code__.co_argcount]
    return names
