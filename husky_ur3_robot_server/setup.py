#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup()
d['packages'] = ['husky_ur3_robot_server']
d['package_dir'] = {'': 'src'}

setup(**d)