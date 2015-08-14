Introduction
============

What are we doing here?
-----------------------

Sampyl provides `Markov Chain Monte Carlo <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ (MCMC) samplers for drawing from probability distributions. Typically, this is used to sample from the posterior distribution of a Bayesian model. Other MCMC packages such as `PyMC <https://github.com/pymc-devs/pymc3>`_ and `PyStan <https://pystan.readthedocs.org/en/latest/>`_, while great and you should check them out, require you to create models using non-Pythonic syntax and semantics. Sampyl allows you to create models completely with Python and Numpy. All that is required is a function that calculates :math:`\log{P(X)}` for the sampling distribution. You can create this function however you want.


Installation
------------

Currently Sampyl is not ready for prime-time, so I haven't put it on PyPI yet. However, you can install it with pip from the GitHub repository: ::

	pip install git+https://github.com/mcleonard/sampyl

Sampyl depends on Numpy and Scipy. For samplers such as Hamiltonian MC and NUTS, Sampyl uses `autograd`_ for automatic gradient calculations, but it is optional.

.. _autograd: https://github.com/HIPS/autograd/


Sampyl License
--------------

Copyright 2015 Mat Leonard

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this file except in compliance with the License. You may obtain a copy of the License at

`http://www.apache.org/licenses/LICENSE-2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
