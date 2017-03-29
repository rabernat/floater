from __future__ import print_function

from setuptools import setup, Extension
from Cython.Build import cythonize
import warnings
import numpy as np
import os
import tempfile, subprocess, shutil

def readme():
    with open('README.md') as f:
        return f.read()

extra_compile_args = ["-std=c++11"]
extra_link_args = ["-std=c++11"]
#extra_compile_args = ["-std=gnu++11"]
#extra_link_args = ["-std=gnu++11"]

runtime_library_dirs = []#['/usr/local/gcc-4.8/lib']

# check for openmp following
# http://stackoverflow.com/questions/16549893/programatically-testing-for-openmp-support-from-a-python-setup-script
# see http://openmp.org/wp/openmp-compilers/
omp_test = \
r"""
#include <omp.h>
#include <stdio.h>
int main() {
#pragma omp parallel
printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(), omp_get_num_threads());
}
"""

def check_for_openmp():
    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)
    filename = r'test.c'
    try:
        cc = os.environ['CC']
    except KeyError:
        cc = 'gcc'
    with open(filename, 'w', 1) as file:
        file.write(omp_test)
    with open(os.devnull, 'w') as fnull:
        result = subprocess.call([cc, '-fopenmp', filename],
                                 stdout=fnull, stderr=fnull)
    print('check_for_openmp() result: ', result)
    os.chdir(curdir)
    #clean up
    shutil.rmtree(tmpdir)

    return result==0

use_openmp = True
if check_for_openmp() and use_openmp:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:
    warnings.warn('Could not link with openmp. Model will be slow.')

ext_module = Extension(
    "floater.hexgrid",
    ["floater/hexgrid.pyx"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    runtime_library_dirs=runtime_library_dirs
)

setup(name='floater',
      version='0.1',
      description='Utilities for processing lagrangian float data',
      url='http://github.com/rabernat/floater',
      author='Ryan Abernathey',
      author_email='rpa@ldeo.columbia.edu',
      license='MIT',
      packages=['floater'],
      scripts=['scripts/floater_convert'],
      install_requires=['numpy', 'cython'],
      ext_modules = cythonize(ext_module),
      include_dirs = [np.get_include()],
      test_suite = 'nose.collector',
      zip_safe=False)
