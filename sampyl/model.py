"""
sampyl.model
~~~~~~~~~~~~~~~~~~~~

Model for building posterior distributions from

:copyright: (c) 2018 by Mat Leonard.
:license: MIT, see LICENSE for more details.

"""


class Model():
    """ Convenience class for building models from log-priors and 
        log-likelihood.

        Example::
            
            # Linear regression model
            def logp(b, sig):
                model = Model()
                
                # Estimate from data and coefficients 
                y_hat = np.dot(X, b)
                
                # Add log-priors for coefficients and model error
                model.add(smp.uniform(b, lower=-100, upper=100),
                          smp.half_normal(sig))

                # Add log-likelihood
                model.add(smp.normal(y, mu=y_hat, sig=sig))

                return model()

    """
    def __init__(self):

        self._logps = []

    def logp(self):

        return sum(self._logps)

    def add(self, *args):
        self._logps.extend(args)

    def __call__(self):
        return self.logp()
