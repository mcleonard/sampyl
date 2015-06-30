from ..core import np
from .base import Sampler


class Chain(Sampler):
    def __init__(self, steps, **kwargs):
        super().__init__(steps[0].logp, **kwargs)
        self.steps = steps


    def step(self):

        for sampler in self.steps:
            sampler.state = self.state
            state = sampler._conditional_step()
            self.state = state

        self._sampled += 1
        return self.state
