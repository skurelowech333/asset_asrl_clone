#!/usr/bin/env python
"""
Setup script for ASSET
Note: This project uses CMake for building the C++ extension.
Install using: pip install -e . (for development) or pip install . (for installation)
Requires: CMake, C++17 compiler, Intel MKL, Python development headers
"""

from setuptools import setup, find_packages

setup(
    name="asset-asrl",
    version="0.5.2",
    description="Astrodynamics Software and Science Enabling Toolkit",
    long_description="High-speed, extensible, object-oriented trajectory design and optimization",
    author="James B. Pezent et al.",
    url="https://github.com/AlabamaASRL/asset_asrl",
    license="Apache 2.0",
    packages=find_packages(include=["asset_asrl", "asset_asrl.*"]),
    package_data={
        "asset_asrl": [
            "asset-apache2.txt",
            "eigen-mpl2.txt",
            "fmt-mit.txt",
            "pybind11-bsd.txt",
            "autodiff-mit.txt",
            "boost-threads.txt",
            "lambert.txt",
            "keplerprop.txt",
        ]
    },
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "matplotlib",
        "spiceypy",
        "seaborn",
        "basemap",
        "intel-openmp"
    ],
    zip_safe=False,
)
