import requests
from setuptools import setup, find_packages

setup(
    name="SESAMI",
    version="2.8",
    packages=find_packages(),
    description="Characterization Tools for Porous Materials Using Nitrogen/Argon Adsorption",
    author="Guobin Zhao",
    author_email="sxmzhaogb@gmai.com",
    url="https://github.com/hjkgrp/SESAMI_web/",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={
        'SESAMI': ['mplstyle.txt','lasso_model.sav']
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "scipy",
        "matplotlib",
        "pandas",
        "numpy",
        "scikit-learn"
    ],
)
