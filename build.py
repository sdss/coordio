#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2020-08-14
# @Filename: build.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import glob
import os
import shutil
from distutils.command.build_ext import build_ext
from distutils.core import Distribution, Extension


LIBSOFA_PATH = os.path.join(os.path.dirname(__file__),
                            'cextern/sofa')


def get_sources():

    dirs = [LIBSOFA_PATH]

    sources = []
    for dir_ in dirs:
        sources += glob.glob(dir_ + '/*.c')

    return sources


extra_compile_args = ['-c', '-pedantic', '-Wall', '-W', '-O']
extra_link_args = []


ext_modules = [
    Extension(
        'coordio.libsofa',
        sources=get_sources(),
        include_dirs=[LIBSOFA_PATH],
        libraries=[],
        define_macros=[],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c',
        optional=False),
]


# The *args is needed because the autogenerated setup.py sends arguments
# (this is a leftover from the old Poetry build system).
def build(*args):

    distribution = Distribution({'name': 'extended',
                                 'ext_modules': ext_modules})
    distribution.package_dir = 'extended'

    cmd = build_ext(distribution)
    cmd.ensure_finalized()
    cmd.run()

    # Copy built extensions back to the project
    for output in cmd.get_outputs():
        relative_extension = os.path.relpath(output, cmd.build_lib)
        shutil.copyfile(output, relative_extension)
        mode = os.stat(relative_extension).st_mode
        mode |= (mode & 0o444) >> 2
        os.chmod(relative_extension, mode)


if __name__ == '__main__':
    build()
