# Sampyl

Auguest 11, 2015: version 0.2.2


Sampyl is a package for sampling from probability distributions using MCMC methods. Similar to PyMC3 using theano to compute gradients, Sampyl uses [autograd](https://github.com/HIPS/autograd) to compute gradients. However, you are free to write your own gradient functions, autograd is not necessary. This project was started as a way to use MCMC samplers by defining models purely with Python and numpy.

Sampyl includes these samplers currently:

* Metropolis-Hastings
* Hamiltonian
* NUTS
* Slice

For each sampler, you pass in a function that calculates the log probability of the distribution you wish to sample from. For the Hamiltonian and NUTS samplers, gradient log probability functions are also required. If autograd is installed, the gradients are calculated automatically. Otherwise, the samplers accept gradient log-p functions which can be used whether autograd is installed or not.

It is still under active development with more features coming soon!

### Dependencies

Works for Python 2 or 3.

Currently, [numpy](http://www.numpy.org/) and [scipy](http://www.scipy.org/) are the only dependencies. To use the automatic gradient log-P capabilities, you will need to install [autograd](https://github.com/HIPS/autograd).


### Installation
Since this is a very alpha stage package, I'm not willing to put it up on PyPI yet. For now you can install it with pip like so:

`pip install git+https://github.com/mcleonard/sampyl`


### Documentation

You can find the documentation at http://sampyl.readthedocs.org/en/latest/. It is still a work in progress, of course, but we're working on it.


### Tests

Tests are written for use with pytest, in the tests folder.


### License

[Apache License, version 2.0](https://github.com/mcleonard/sampyl/blob/master/LICENSE)