from setuptools import setup, find_packages

setup(
    name="tobacco-mof",
    version="4.0.0",
    packages=find_packages(),
    description="Topologically Based Crystal Constructor (ToBaCCo)",
    author="Guobin Zhao",
    author_email="sxmzhaogb@gmai.com",
    url="https://github.com/sxm13/pypi-dev/ToBaCCo/",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "numpy",
        "ase",
        "networkx",
        "scipy",
        "gemmi"
    ],
)
