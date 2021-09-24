#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-14
# @Filename: build.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# isort:skip_file

# Order matters here. Keep this import at the top.
from setuptools import setup

import glob
import sys
from distutils.core import Extension

from shutil import copyfile
import os

LIBSOFA_PATH = 'cextern/sofa'
LIBCOORDIO_PATH = 'cextern/conv.cpp'


extra_compile_args = ['-c', '-pedantic', '-Wall', '-W', '-O']
extra_link_args = []

class getPybindInclude(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    https://github.com/pybind/python_example/blob/master/setup.py
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


sofa_sources = glob.glob(LIBSOFA_PATH + '/*.c')
includes = [
    'include',
    # '/usr/local/include',
    # '/usr/local/include/eigen3',
    # '/usr/include/eigen3',
    # '/usr/include',
    getPybindInclude(),
    getPybindInclude(user=True)
]


extra_compile_args2 = ["--std=c++11", "-fPIC", "-v", "-O3"]
extra_link_args2 = None
if sys.platform == 'darwin':
    extra_compile_args2 += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    extra_link_args2 = ["-v", '-mmacosx-version-min=10.9']

ext_modules = [
    Extension(
        'coordio.libsofa',
        sources=sofa_sources,
        include_dirs=[LIBSOFA_PATH],
        libraries=[],
        define_macros=[],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c',
        optional=False),
    Extension(
        'coordio.libcoordio',
        sources=[LIBCOORDIO_PATH],
        include_dirs=includes,
        extra_compile_args=extra_compile_args2,
        extra_link_args=extra_link_args2),
]

setup(ext_modules=ext_modules)

buildDir = glob.glob("build/lib*")[0]
soFiles = glob.glob(buildDir + "/coordio/lib*.so")
for soFile in soFiles:
    base, filename = os.path.split(soFile)
    dest = "coordio/%s"%filename
    copyfile(soFile, dest)
    mode = os.stat(dest).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(dest, mode)