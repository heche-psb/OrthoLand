#!/usr/bin/python3
# contact: heche@psb.vib-ugent.be

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='orthomason',
    version='0.0.0.1',
    packages=['orthomason'],
    url='http://github.com/heche-psb/OrthoMason',
    license='GPL',
    author='Hengchi Chen',
    author_email='heche@psb.vib-ugent.be',
    description='Python package and CLI for finding orthologues',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['command'],
    include_package_data=True,
    install_requires=[
       'biopython==1.76',
       'click==7.1.2',
       'pandas<=1.4.4',
       'scipy<=1.5.4'
    ],
    entry_points='''
        [console_scripts]
        orthomason=command:cli
    ''',
)
