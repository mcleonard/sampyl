.. _model:

Model
=====

The model is completely defined by log P(X). You can use this object to get 
log P(X) and/or the gradient. Models contain caches for both log P(X) and the
gradient. This is intended to be used when building new models as users won't
typically need this.

There are two models currently. :ref:`Model <model_class>` expects separate
log P(X) and gradient functions. :ref:`SingleModel <single_model_class>`
expects one function that returns both log P(x) and the gradient.

Example usage::
    
    def logp(X):
        ...
    
    model = init_model(logp)
    x = some_state
    logp_val = model.logp(x)
    grad_val = model.grad(x)
    logp_val, grad_val = model(x)


.. _model_class:

.. module:: model

.. autofunction:: init_model

.. autoclass:: Model
    :members:
    :inherited-members:

    .. method:: __call__(state)

        Return log P(X) and grad log P(X) given a :ref:`state <state>` X 


.. _single_model_class:

.. autoclass:: SingleModel
    :members:
    :inherited-members:

    .. method:: __call__(state)

        Return log P(X) and grad log P(X) given a :ref:`state <state>` X