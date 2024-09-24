from setuptools import setup, find_packages
import numpy as np

packages = find_packages(where='src')

setup(
    name='eclipse_mappyr',
    version='0.1.0',
    description='Functions to simulate the effect of eclipse mapping in binary systems',
    author='Cole Johnston, Timothy van Reeth',
    author_email='colej@mpa-garching.mpg.de',
    url='https://github.com/colej/eclipse_mappyr',
    packages=packages,
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.25.0',
        'matplotlib>=3.7.0',
        'pandas>=2.2.0',
    ],
    classifiers=[
        # Choose appropriate classifiers from:
        # https://pypi.org/classifiers/
    ],
    python_requires='>=3.7',
)
