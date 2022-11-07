#!/usr/bin/env python

from distutils.core import setup, Extension

module1 = Extension('spam', sources=['py_interf.c','rknng_lib.cpp','options.c'],
                        include_dirs=['/usr/local/lib'])

setup(name = 'spam',
        version='1.0',
        description='This is my spam package',
        ext_modules = [module1])

