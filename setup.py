#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(
    name='python-hqsom',
    version='0.1',
    description='A hqsom implementation in python',
    author='Joseph Lynch',
    author_email='jolynch@mit.edu',
    url='https://github.com/jolynch/python-hqsom.git',
    packages=find_packages(exclude=['tests', 'demos']),
    include_package_data=True,
    setup_requires=['setuptools'],
    install_requires=[
        'numpy',
        'pillow',
    ],
    license='MIT License'
)
