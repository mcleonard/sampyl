from distutils.core import setup

DISTNAME = 'sampyl'
DESCRIPTION = 'MCMC Samplers'
LONG_DESCRIPTION = """Sampyl contains a set of Markov Chain Monte Carlo (MCMC) samplers used to sample from arbitrary probability distributions. Typically, this is used to sample from the posterior distribution of a Bayesian estimation model. With Sampyl, all models are contained in normal Python functions. As such, models can be created in typical, Pythonic fashion."""
AUTHOR = 'Mat Leonard'
AUTHOR_EMAIL = 'leonard.mat@gmail.com'
URL = 'https://github.com/mcleonard/sampyl'
LICENSE = 'Apache License, Version 2.0'
VERSION = '0.1'

classifiers = ['Development Status :: 1 - Pre-alpha',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.4',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

if __name__ == '__main__':
    setup(
        name = DISTNAME,
        packages = ['sampyl', 'sampyl.samplers',
                    'sampyl.tests'],
        version = VERSION,
        description = DESCRIPTION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        url = URL,
        classifiers = classifiers)