#!/usr/bin/python3
# contact: heche@psb.vib-ugent.be

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cognate',
    version='0.0.0.1',
    packages=['cognate'],
    url='http://github.com/heche-psb/cognate',
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
       'scipy<=1.5.4',
       'rich==12.5.1',
       'numpy>=1.19.0',
       'joblib==0.11',
       'networkx==3.1',
       'tqdm==4.65.0',
    ],
    entry_points='''
        [console_scripts]
        cognate=command:cli
    ''',
)
