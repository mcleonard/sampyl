.. _samplers:

Samplers
========

Each sampler has the same API. First you create a function calculating log P(X),
then pass it to a sampler. To generate a chain, call the sample method.

Example::
    
    import sampyl as smp
    def logp(x, y):
        ...

    start = {'x': x_start, 'y': y_start}
    nuts = smp.NUTS(logp, start)
    chain = nuts.sample(1000)

Creating your own :ref:`custom samplers <custom>` is possible and straightfoward.

.. toctree::

    samplers/nuts
    samplers/metropolis
    samplers/slice
    samplers/hamiltonian
    samplers/custom