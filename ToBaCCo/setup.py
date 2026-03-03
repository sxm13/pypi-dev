from setuptools import setup, find_packages

setup(
    name="tobacco_mof",
    version="4.0.3",
    packages=find_packages(),
    description="Topologically Based Crystal Constructor (ToBaCCo)",
    author="Guobin Zhao",
    author_email="sxmzhaogb@gmai.com",
    url="https://github.com/sxm13/pypi-dev/tree/main/ToBaCCo",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy<2.0.0",
        "scipy<1.13.0",
        "ase",
        "networkx",
        "gemmi",
        "tqdm"
    ],
)
