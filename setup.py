#!/usr/bin/env python
import os
import re
from io import open
from setuptools import setup, find_packages

from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, 'README.rst'), encoding='utf8') as f:
    long_description = f.read()

# Scripts
scripts = []
for dirname, dirnames, filenames in os.walk('scripts'):
    for filename in filenames:
        if filename.endswith('.py') or filename.endswith('.sh') or filename.endswith('.awk'):
            scripts.append(os.path.join(dirname, filename))

setup(
    name='singlet_fission',
    version='0.0.1',
    description='Semiempirical models for estimating singlet fission rates in cofacially stacked homodimers',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/humeniuka/singlet_fission',
    author='Alexander Humeniuk',
    author_email='alexander.humeniuk@gmail.com',
    license='LICENSE.txt',
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'scipy', 'tdqm'],
    scripts=scripts,
    include_package_data=True,
    zip_safe=False,
)
