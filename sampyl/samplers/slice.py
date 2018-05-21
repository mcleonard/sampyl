"""
sampyl.samplers.slice
~~~~~~~~~~~~~~~~~~~~

This module implements the slice sampler.

:copyright: (c) 2015 by Andrew Miller.
:license: MIT, see LICENSE for more details.

"""


from __future__ import division

from ..core import np
from ..state import State
from .base import Sampler


class Slice(Sampler):
    """ Slice sampler (Neal, 2003) for creating a Markov chain that 
        leaves the the distribution defined by logp invariant

        For technical details, see Neal's paper:
            http://projecteuclid.org/euclid.aos/1056562461

        Andrew Miller (acm@seas.harvard.edu) 7-13-15

        Adapted from code written by Ryan Adams (rpa@seas.harvard.edu)

        :param logp: *function.* :math:`\log{P(X)}` function for sampling
                                 distribution.
        :param start: *scalar or 1D array-like.* Starting state for sampler.
        :param compwise: (optional) *boolean.* Component-wise univariate 
                         slice sample
                         (or random direction)
        :param width: (optional) *int, float.* (Initial) width of the slice
        :param step_out: (optional) *boolean.* Perform step-out procedure
        :param doubling_step: (optional) *boolean.* If stepping out, double
                              slice width?
        :param max_steps_out: (optional) *int.* Max number of steps out to perform
        :param verbose: (optional) *boolean.* Print steps out
    """

    def __init__(self, logp,
                       start,
                       compwise      = False, 
                       width         = 1.,
                       step_out      = True,
                       doubling_step = True,
                       max_steps_out = 10,
                       verbose       = False,
                       **kwargs):
        
        
        super(Slice, self).__init__(logp, start, None, grad_logp_flag=False,
                                             **kwargs)
        self._num_evals = 0

        # sampler  this is either a random direction or component-wise slice sampler
        self.compwise      = compwise
        self.width         = width
        self.step_out      = step_out
        self.doubling_step = doubling_step
        self.max_steps_out = max_steps_out
        self.verbose       = verbose

    def step(self):
        """ Perform a slice sample step """
        dims = self.state.tovector().shape[0]
        if self.compwise:
            ordering = range(dims)
            np.random.shuffle(ordering)
            new_x = self.state.tovector.copy()
            for d in ordering:
                direction    = np.zeros((dims))
                direction[d] = 1.0
                new_x        = self.direction_slice(direction, new_x)
        else:
            direction = np.random.randn(dims)
            direction = direction / np.sqrt(np.sum(direction**2))
            new_x = self.direction_slice(direction, self.state.tovector())

        self.state = self.state.fromvector(new_x)
        self._sampled += 1
        return self.state

    def direction_slice(self, direction, init_x):
        """ one dimensional directional slice sample along direction specified
            Implements the stepping out procedure from Neal
        """
        def dir_logprob(z):
            self._num_evals += 1
            cstate = State.init_fromvector(direction*z + init_x, self.state)
            return self.model.logp(cstate)

        def acceptable(z, llh_s, L, U):
            while (U-L) > 1.1*self.width:
                middle = 0.5*(L+U)
                splits = (middle > 0 and z >= middle) or (middle <= 0 and z < middle)
                if z < middle:
                    U = middle
                else:
                    L = middle
                # Probably these could be cached from the stepping out.
                if splits and llh_s >= dir_logprob(U) and llh_s >= dir_logprob(L):
                    return False
            return True

        upper = self.width*np.random.rand()
        lower = upper - self.width
        llh_s = np.log(np.random.rand()) + dir_logprob(0.0)

        l_steps_out = 0
        u_steps_out = 0
        if self.step_out:
            if self.doubling_step:
                while (dir_logprob(lower) > llh_s or
                       dir_logprob(upper) > llh_s) and \
                       (l_steps_out + u_steps_out) < self.max_steps_out:
                    if np.random.rand() < 0.5:
                        l_steps_out += 1
                        lower       -= (upper-lower)
                    else:
                        u_steps_out += 1
                        upper       += (upper-lower)
            else:
                while dir_logprob(lower) > llh_s and \
                        l_steps_out < max_steps_out:
                    l_steps_out += 1
                    lower       -= self.width
                while dir_logprob(upper) > llh_s and \
                        u_steps_out < max_steps_out:
                    u_steps_out += 1
                    upper       += self.width

        start_upper = upper
        start_lower = lower

        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*np.random.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                print(new_z, direction*new_z + init_x, new_llh,
                      llh_s, init_x, dir_logprob(init_x))
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s and \
                    acceptable(new_z, llh_s, start_lower, start_upper):
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if self.verbose:
            print("Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in)

        return new_z*direction + init_x

    @property
    def evals_per_sample(self):
        return self._num_evals/float(self._sampled)

    def __repr__(self):
        return 'Slice sampler'


