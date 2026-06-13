from pathlib import Path
import shutil
import subprocess

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


ROOT = Path(__file__).resolve().parent


def install_conda_dependency():
    conda_executable = shutil.which("conda")
    if not conda_executable:
        print(
            "Warning: conda was not found. Please install "
            "'conda-forge::zeopp-lsmo' manually."
        )
        return

    subprocess.check_call(
        [conda_executable, "install", "-y", "conda-forge::zeopp-lsmo"]
    )


class InstallWithConda(install):
    def run(self):
        install_conda_dependency()
        super().run()


class DevelopWithConda(develop):
    def run(self):
        install_conda_dependency()
        super().run()


setup(
    name="gtsr",
    version="0.0.2",
    description="Graph neural network tool for solvent removal from MOF structures",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Xiao-Yan Li Group",
    license="MIT",
    python_requires=">=3.9",
    packages=["gtsr", "gtsr.src", "gtsr.ckpt"],
    py_modules=["GTsRunner"],
    package_dir={
        "gtsr": "gtsr",
        "gtsr.src": "src",
        "gtsr.ckpt": "ckpt",
    },
    package_data={
        "gtsr.ckpt": [
            "free_best.pth",
            "all_best.pth",
            "stability_best.pkl",
        ],
    },
    include_package_data=True,
    cmdclass={
        "install": InstallWithConda,
        "develop": DevelopWithConda,
    },
    install_requires=[
        "ase>=3.19",
        "numpy>=1.21",
        "pymatgen>=2018.6.11",
        "scikit-learn>=1.0",
        "torch>=1.12",
        "molSimplify==1.8.0",
        "rdkit",
        "networkx"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
