Introduction
============

What are we doing here?
-----------------------

Sampyl provides `Markov Chain Monte Carlo <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ (MCMC) samplers for drawing from probability distributions. Typically, this is used to sample from the posterior distribution of a Bayesian model. Other MCMC packages such as `PyMC <https://github.com/pymc-devs/pymc3>`_ and `PyStan <https://pystan.readthedocs.org/en/latest/>`_, while great and you should check them out, require you to create models using non-Pythonic syntax and semantics. Sampyl allows you to create models completely with Python and Numpy. All that is required is a function that calculates :math:`\log{P(X)}` for the sampling distribution. You can create this function however you want.


Installation
------------

You can install Sampyl from PyPI with ::

	pip install sampyl-mcmc

Sampyl depends on Numpy, Scipy, and `autograd`_. You'll also need matplotlib for the examples notebooks.

.. _autograd: https://github.com/HIPS/autograd/