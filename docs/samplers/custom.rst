.. _custom:

Custom Samplers
===============

You can build your own sampler by subclassing Sampler. A :ref:`model <model>`
is automatically generated from `logp`. The sampler is also initialized with
a :ref:`state <state>` generated from `start` and the arguments of `logp`. With these,
you define the `step` method, which should generate one sample and return a 
:ref:`state <state>`.

As an example, here's snippet from the :ref:`Metropolis <metropolis>` sampler. ::

    from sampyl import Sampler
    from sampyl import np
    
    class Metropolis(Sampler):

        def __init__(self, logp, start, **kwargs):
            # No gradient is needed, so set it to None, and the flag to False
            super(Metropolis, self).__init__(logp, start, None, grad_logp_flag=False, **kwargs)

        def step(self):
            """ Perform a Metropolis-Hastings step. """
            x = self.state
            y = proposal(x, self.scale)
            if accept(x, y, self.model.logp):
                self.state = y

            return self.state

    def proposal(state, scale):
        proposed = State.fromkeys(state.keys())
        for i, var in enumerate(state):
            proposed.update({var: np.random.normal(state[var], scale[var])})
        return proposed

    def accept(x, y, logp):
        delp = logp(y) - logp(x)
        if np.isfinite(delp) and np.log(np.random.uniform()) < delp:
            return True
        else:
            return False


.. module:: sampyl


.. autoclass:: Sampler
    :members:

    **Attributes**

    .. attribute:: model
        
        :ref:`Model <model>` with logp and grad functions.

    .. attribute:: state

        The current :ref:`state <state>` of the model.

    .. attribute:: sampler

        Calling the sample method creates an infinite generator which returns 
        samples as :ref:`states <state>`.

    **Methods**
    

    
    



Custom Samplers
===============

.. module:: sampyl

.. autoclass:: Sampler
