from itertools import count
import time
import unicodedata

from ..core import np, auto_grad_logp, AUTOGRAD
from ..parallel import parallel
from ..progressbar import update_progress
from ..state import State, func_var_names
from ..posterior import init_posterior


class Sampler(object):
    def __init__(self, logp, start,
                 grad_logp=None,
                 scale=None,
                 condition=None,
                 grad_logp_flag=True,
                 random_seed=None):

        self.model = init_posterior(logp, grad_logp, grad_logp_flag)

        self._logp_func = logp
        self._grad_func = grad_logp
        self.var_names = func_var_names(logp)

        self.state = State.fromkeys(self.var_names)

        # Making sure we normalize here because if some parameters use unicode 
        # symbols, they are normalized through the func_var_names function. Then, we
        # need to normalize them here as well or the keys in start won't match the
        # keys from var_names
        start = {unicodedata.normalize('NFKC', key): val for key, val in start.items()}

        self.state.update(start)

        self.scale = default_scale(scale, self.state)
        self.sampler = None
        self._sampled = 0
        self._accepted = 0
        self.conditional = condition
        self._grad_logp_flag = grad_logp_flag
        self.seed = random_seed

        if random_seed:
            np.random.seed(random_seed)

        if condition is not None:
            self._joint_logp = self._logp_func

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
            return self._joint_logp(**conditional_state)

        self.state = State([(var, frozen_state[var]) for var in free_vars])
        self._logp_func = conditional_logp
        if self._grad_logp_flag and AUTOGRAD:
            self.model.grad_func = auto_grad_logp(conditional_logp, names=self.state.keys())
        self.model.logp_func = self._logp_func
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
        """ This is what you define to create the sampler. Requires that a
            :ref:`state <state>` object is returned."""
        pass

    def sample(self, num, burn=0, thin=1, n_chains=1, progress_bar=True):
        
        """ 
            Sample from :math:`P(X)`

            :param num: *int.* Number of samples to draw from :math:`P(X)`.
            :param burn: (optional) *int.*
                Number of samples to discard from the beginning of the chain.
            :param thin: (optional) *float.*
                Thin the samples by this factor.
            :param n_chains: (optional) *int.*
                Number of chains to return. Each chain is given its own
                process and the OS decides how to distribute the processes.
            :param progress_bar: (optional) *boolean.*
                Show the progress bar, default = True.
            :return: Record array with fields taken from arguments of 
                logp function.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        if AUTOGRAD and hasattr(self.model, 'grad_func') \
                    and self.model.grad_func is None:
            self.model.grad_func = auto_grad_logp(self._logp_func)

        # Constructing a recarray to store samples
        dtypes = [(var, 'f8', np.shape(self.state[var])) for var in self.state]
        samples = np.zeros(num, dtype=dtypes).view(np.recarray)

        if n_chains != 1:
            return parallel(self, n_chains, samples,
                            burn=burn, thin=thin,
                            progress_bar=progress_bar)

        if self.sampler is None:
            self.sampler = (self.step() for _ in count(start=0, step=1))

        start_time = time.time() # For progress bar
        
        # Start sampling, add each 
        for i in range(num):
            samples[i] = tuple(next(self.sampler).values())

            if progress_bar and time.time() - start_time > 1:
                update_progress(i+1, num)
                start_time = time.time()

        if progress_bar:
            update_progress(i+1, num, end=True)

        # Clearing the cache after a run to save on memory.
        self.model.clear_cache()

        return samples[burn::thin]


    def __call__(self, num, burn=0, thin=1, n_chains=1, progress_bar=True):

        return self.sample(num, burn=burn, thin=thin, n_chains=n_chains, 
                           progress_bar=progress_bar)


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
