# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 20:16:30 2018

@author: kalvi
"""

from setuptools import find_packages
from setuptools import setup

#REQUIRED_PACKAGES = ['some_PyPI_package>=1.0']
REQUIRED_PACKAGES = [
        'numpy==1.13.3',
        'tensorflow==1.8.0',
        'pandas==0.22.0',
        'nltk==3.2.5'
        ]

setup(
    name='trainer',
    version='0.2.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='baseline attack training package'
)
