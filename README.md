Sampyl
=======
June 27th, 2015: version 0.1

Sampyl is a package for sampling from probability distributions using MCMC methods. Similar to PyMC3 using theano to compute gradients, Sampyl uses [autograd](https://github.com/HIPS/autograd) to compute gradients. This project was started as a way to use MCMC samplers by defining models purely with Python and numpy.

Sampyl includes three samplers currently:

* Metropolis-Hastings
* Hamiltonian
* NUTS

The first two seem to be working fine. However, NUTS sometimes works, mostly doesn't. Still working on it.

Cheers,

Mat Leonard

Dependences
-----------

Currently, numpy and autograd are the only dependencies.

Documentation
------------
Basically none exist right now, will work on that soon. Check out the [example notebook](http://nbviewer.ipython.org/github/mcleonard/sampyl/blob/master/Examples.ipynb) though for some guidance.

License
-------
[Apache License, version 2.0](https://github.com/mcleonard/sampyl/blob/master/LICENSE)