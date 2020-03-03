#!/usr/bin/env python

from setuptools import setup
import sys
from pathlib import Path

if sys.version_info.major != 3:
    raise RuntimeError('STREAM requires Python 3')


#No dependency packages are specified in setup.py file
#STREAM is built based on bioconda recipe. All the required packages can be found in the recipe file 
#https://github.com/bioconda/bioconda-recipes/tree/master/recipes/stream

setup(name='stream',
      version="0.4.2",
      description='Single-cell Trajectories Reconstruction, Exploration And Mapping of single-cell data',
      long_description=Path('README.md').read_text('utf-8'),
      url='https://github.com/pinellolab/stream',
      author='Huidong Chen',
      author_email='huidong.chen AT mgh DOT harvard DOT edu',
      license='AGPL-3',
      packages=['stream'],
      package_dir={'stream':'stream'},
      package_data={'stream': ['tests/*']},
      include_package_data = True,
      install_requires=[''],
      entry_points = {'console_scripts': ['stream=stream.command_line:main',
      'stream_run_test=stream.tests.stream_run_test:main']})

