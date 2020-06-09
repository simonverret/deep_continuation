#!/usr/bin/env python
from distutils.core import setup

setup(name='deep_continuation',
    version='0.2',
    description='Analytical continuation with neural networks',
    author='Simon Verret',
    author_email='simon.verret@mila.quebec',
    url='https://github.com/simonverret/deep_continuation',
    packages=['deep_continuation'],
    install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'torch',
        ]
    )