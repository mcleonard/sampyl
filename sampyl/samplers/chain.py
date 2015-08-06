from ..core import np
from .base import Sampler


class Chain(Sampler):
    def __init__(self, steps, start, **kwargs):
        """ Sampler for chaining together multiple other samplers.

            Arguments
            ---------
            steps: list or tuple
                List of sampler objects conditioned on parameters. 
                This sampler iterates through each sampler in steps, using the
                step method of each to update the state.
            start: dict
                Dictionary of starting state for the sampler. Should have one
                element for each argument of logp. So, if logp = f(x, y), then
                start = {'x': x_start, 'y': y_start}

        """
        # Find the logp function with all the parameters
        logps = [each._logp_func for each in steps]
        logp_index = np.argmax([each.__code__.co_argcount for each in logps])

        super(Chain, self).__init__(logps[logp_index], start, **kwargs)

        self.steps = steps

    def step(self):

        for sampler in self.steps:
            sampler.state = self.state
            state = sampler._conditional_step()
            self.state = state

        self._sampled += 1
        return self.state
