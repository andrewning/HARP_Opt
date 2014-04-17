#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup
# from numpy.distutils.core import setup, Extension



setup(
    name='HARP_Opt',
    version='0.1.0',
    description='Horizontal Axis Rotor Performance Optimization',
    author='S. Andrew Ning',
    author_email='andrew.ning@nrel.gov',
    package_dir={'': 'src'},
    py_modules=['harpopt'],
    packages=['rotorse', 'commonse'],  # TODO: these should come directly from GitHub once released
    install_requires=['CCBlade>=1.1.1', 'akima>=1.0'],
    license='Apache License, Version 2.0',
    dependency_links=['https://github.com/WISDEM/CCBlade/tarball/master#egg=CCBlade-1.1.1',
        'https://github.com/andrewning/akima/tarball/master#egg=akima-1.0.0'],
    zip_safe=False
)
