from itertools import count
import time

from ..core import np, auto_grad_logp
from ..parallel import parallel
from ..progressbar import update_progress
from ..state import State, func_var_names


class Sampler(object):
    def __init__(self, logp, start,
                 grad_logp=None,
                 scale=None,
                 condition=None,
                 grad_logp_flag=True,
                 random_seed=None):
        self.logp = check_logp(logp)
        self.var_names = func_var_names(logp)
        self.state = State.fromfunc(logp)
        self.state.update(start)

        self.scale = default_scale(scale, self.state)
        self.sampler = None
        self._sampled = 0
        self._accepted = 0
        self.conditional = condition
        self._grad_logp_flag = grad_logp_flag
        self.seed = random_seed

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

        frozen_vars = self.conditional
        frozen_state = self.state
        free_vars = [var for var in self.state if var not in frozen_vars]

        def conditional_logp(*args):
            conditional_state = State([each for each in zip(free_vars, args)])
            # Insert conditional values here, then pass to full logp
            for i in frozen_vars:
                conditional_state.update({i: frozen_state[i]})
            return self.joint_logp(**conditional_state)

        self.state = State([(var, frozen_state[var]) for var in free_vars])
        self.logp = conditional_logp
        if self._grad_logp_flag:
            self.grad_logp = auto_grad_logp(conditional_logp, names=self.state.keys())
        state = self.step()

        # Add the frozen variables back into the state
        new_state = State([(name, None) for name in self.var_names])
        for var in state:
            new_state.update({var: state[var]})
        for var in frozen_vars:
            new_state.update({var: frozen_state[var]})

        self.state = new_state

        return self.state

    def step(self):
        """ This is what you define to create the sampler. """
        pass

    def sample(self, num, burn=-1, thin=1, n_chains=1, progress_bar=True):
        """ Sample from distribution defined by logp.

            Parameters
            ----------

            num: int
                Number of samples to return.
            burn: int
                Number of samples to burn through
            thin: thin
                Thin the samples by this factor
            n_chains: int
                Number of chains to return. Each chain is given its own
                process and the OS decides how to distribute the processes.
            progress_bar: boolean
                Show the progress bar, default = True
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if self._grad_logp_flag and self.grad_logp is None:
            self.grad_logp = auto_grad_logp(self.logp)

        # Constructing a recarray to store samples
        dtypes = [(var, 'f8', np.shape(self.state[var])) for var in self.state]
        samples = np.zeros(num, dtype=dtypes).view(np.recarray)

        if n_chains != 1:
            return parallel(self, n_chains, samples,
                            burn=burn, thin=thin,
                            progress_bar=progress_bar)

        if self.sampler is None:
            self.sampler = (self.step() for _ in count(start=0, step=1))

        start_time = time.time()
        for i in range(num):
            samples[i] = next(self.sampler).tovector()

            if progress_bar and time.time() - start_time > 1:
                update_progress(i+1, num)
                start_time = time.time()

        if progress_bar:
            update_progress(i+1, num, end=True)

        return samples[burn+1::thin]


def check_logp(logp):
    if not hasattr(logp, '__call__'):
        raise TypeError("logp must be a function")
    elif logp.__code__.co_argcount == 0:
        raise ValueError("logp must have arguments")
    else:
        return logp


def default_scale(scale, state):
    """ If scale is None, return a State object with arrays of ones matching
        the shape of values in state.
    """

    if scale is None:
        new_scale = State.fromkeys(state.keys())
        for var in state:
            new_scale.update({var: np.ones(np.shape(state[var]))})
        return new_scale
    else:
        return scale
