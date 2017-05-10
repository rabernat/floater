from __future__ import print_function

from setuptools import setup
import warnings
import numpy as np
import os
import tempfile, subprocess, shutil

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='floater',
      version='0.1',
      description='Utilities for processing lagrangian float data',
      url='http://github.com/rabernat/floater',
      author='Ryan Abernathey',
      author_email='rpa@ldeo.columbia.edu',
      license='MIT',
      packages=['floater'],
      scripts=['scripts/floater_convert'],
      install_requires=['numpy', 'scipy', 'future', 'scikit-image'],
      zip_safe=False)
