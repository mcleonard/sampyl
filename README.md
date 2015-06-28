Sampyl
=======
June 27th, 2015: version 0.1

Sampyl is a package for sampling from probability distributions using MCMC methods. Similar to PyMC3 using theano to compute gradients, Sampyl uses [autograd](https://github.com/HIPS/autograd) to compute gradients. However, you are free to write your own gradient functions, autograd is not necessary. This project was started as a way to use MCMC samplers by defining models purely with Python and numpy.

Sampyl includes three samplers currently:

* Metropolis-Hastings
* Hamiltonian
* NUTS

For each sampler, you pass in a function that calculates the log probability of the distribution you wish to sample from. For the Hamiltonian and NUTS samplers, gradient log probability functions are also required. If autograd is installed, the gradients are calculated automatically. Otherwise, the samplers accept gradient log-p functions which can be used whether autograd is installed or not.

It is still under active development with more features coming soon!

Dependencies
-----------

Requires Python 3.

Currently, numpy is the only dependency. To use the automatic gradient log-P capabilities, you will need to install [autograd](https://github.com/HIPS/autograd).

Documentation
------------
Basically none exist right now, will work on that soon. Check out the [example notebook](http://nbviewer.ipython.org/github/mcleonard/sampyl/blob/master/Examples.ipynb) though for some guidance.

Tests
-----------
Tests are written for use with pytest, in the tests folder.


License
-------
[Apache License, version 2.0](https://github.com/mcleonard/sampyl/blob/master/LICENSE)