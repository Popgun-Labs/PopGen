#!/usr/bin/env python
from setuptools import setup, find_packages
from popgen import __version__

readme = open('README.md').read()
requirements_txt = open('requirements.txt').read().split('\n')
requirements = list(filter(lambda x: '--extra' not in x and x is not '', requirements_txt))

dependency_links = list(filter(lambda x: '--extra' in x, requirements_txt))
dependency_links = list(map(lambda x: x.split(' ')[-1], dependency_links))

setup(
    # Metadata
    name='popgen',
    version=__version__,
    author='Popgun Research Team',
    author_email='angus@wearepopgun.com',
    url='https://github.com/Popgun-Labs/popgen',
    description='Generative Modelling Research Toolkit.',
    long_description=readme,
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    install_requires=requirements,
    dependency_links=dependency_links,
    include_package_data=True
)
