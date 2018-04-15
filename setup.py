#!/usr/bin/env python
"""Description:
Setup script for STREAM -- Single-cell Trajectories Reconstruction, Exploration And Mapping of single-cell data
@status:  beta
@version: $Revision$
@author:  Luca Pinello
@contact: lpinello AT mgh DOT harvard DOT edu
"""
from setuptools import setup
import subprocess as sb
import sys

version="0.1.0"


def main():
      setup(name='stream',
      version=version,
      description='Single-cell Trajectories Reconstruction, Exploration And Mapping of single-cell data',
      url='https://github.com/pinellolab/stream',
      author='Luca Pinello',
      author_email='lpinello AT mgh DOT harvard DOT edu',
      license='Affero',
      packages=['STREAM'],
      package_dir={'STREAM': 'STREAM'},
      package_data={'STREAM': ['*','templates/*','static/*','precomputed/*']},
      install_requires=[''],
      zip_safe=False,
      entry_points = {'console_scripts': ['STREAM=STREAM.STREAM:main',
					 'STREAM_webapp=STREAM.app:main'],
    }
)




if __name__ == '__main__':
    main()
    if sys.argv[1]=='install':
   
	sb.call('unzip -o upload-button.zip && cd upload-button && python setup.py install && cd .. && rm -Rf upload-button',shell=True)
    	sys.stdout.write ('\nPython package installed')
