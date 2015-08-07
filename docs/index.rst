.. Sampyl documentation master file, created by
   sphinx-quickstart on Thu Aug  6 23:09:13 2015.


Sampyl: MCMC samplers in Python  
===============================

Release v\ |version|

Sampyl is a Python library implementing Markov Chain Monte Carlo (MCMC) samplers in Python. It's designed for use in Bayesian parameter estimation. Currently, only the
samplers are implemented. However, we are working on providing an API for building
Bayesian models.

Sampyl allows the user to define a model any way they want, all that is required is
a function that calculates log P(X). This function can be written completely in Python,
or written in C/C++ and wrapped with Python. For samplers that require the gradient 
of P(X), NUTS for example, Sampyl can calculate the gradients automatically
with autograd_.

.. _autograd: https://github.com/HIPS/autograd/

Start here
----------

.. toctree::
   :maxdepth: 2

   introduction
   tutorial


Samplers
--------

.. toctree::
	:maxdepth: 1

	samplers/nuts
	samplers/metropolis
	samplers/slice
	samplers/hamiltonian
	samplers/custom




Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

