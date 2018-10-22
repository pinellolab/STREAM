#!/usr/bin/env python

from setuptools import setup
import sys

if sys.version_info.major != 3:
    raise RuntimeError('STREAM requires Python 3')


setup(name='stream',
      version="0.2.6",
      description='Single-cell Trajectories Reconstruction, Exploration And Mapping of single-cell data',
      url='https://github.com/pinellolab/stream',
      author='Huidong Chen',
      author_email='huidong.chen AT mgh DOT harvard DOT edu',
      license='Affero',
      packages=['stream'],
      package_dir={'stream': 'stream'},
      install_requires=[''],)

