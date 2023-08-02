import os
from setuptools import find_packages, setup

with open(os.path.join("UR_gym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="UR_gym",
    description="Set of UR5 environments based on PyBullet physics engine and gymnasium.",
    author="Wanqing Xia",
    author_email="wxia612@aucklanduni.ac.nz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WanqingXia/UR-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={"UR_gym": ["version.txt"]},
    version=__version__,
    install_requires=["gymnasium~=0.26", "pybullet", "numpy", "scipy", "pyquaternion", "tqdm", "wandb"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
