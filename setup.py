#!/usr/bin/env python

from distutils.core import setup


classifiers = ['Development Status :: 1 - Pre-Alpha',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.4',
               'License :: OSI Approved :: Apache Software License',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Operating System :: OS Independent']

if __name__ == "__main__":
    setup(name='sampyl',
          version='0.2',
          description='MCMC Samplers',
          author='Mat Leonard',
          author_email='leonard.mat@gmail.com',
          url='https://github.com/mcleonard/sampyl.git',
          packages=['sampyl', 'sampyl.samplers', 'sampyl.tests'],
          classifiers=classifiers)