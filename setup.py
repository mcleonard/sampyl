#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.6',
               'License :: OSI Approved :: MIT License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']


setuptools.setup(name='sampyl-mcmc',
                 version='0.3',
                 description='MCMC Samplers in Python & Numpy',
                 author='Mat Leonard',
                 author_email='leonard.mat@gmail.com',
                 url='http://matatat.org/sampyl/',
                 packages=['sampyl', 'sampyl.samplers'],
                 classifiers=classifiers,
                 install_requires=['numpy', 'scipy', 'autograd'])
