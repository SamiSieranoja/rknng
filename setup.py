#!/usr/bin/env python

from distutils.core import setup, Extension

module1 = Extension('rpdivknng', sources=['py_interf.c','rknng_lib.cpp','options.c'],
                        include_dirs=['/usr/local/lib'])

setup(name = 'rpdivknng',
        version='1.0',
        description='kNN graph via random point division',
        ext_modules = [module1])

