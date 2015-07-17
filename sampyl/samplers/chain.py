from ..core import np
from .base import Sampler


class Chain(Sampler):
    def __init__(self, steps, start, **kwargs):
        # Find the logp function with all the parameters
        logps = [each.logp for each in steps]
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
