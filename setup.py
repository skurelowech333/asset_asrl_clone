#!/usr/bin/env python
"""
Setup script for ASSET with Cython compilation
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Get all .pyx files from asset_asrl/Forces
cython_extensions = []
forces_dir = "asset_asrl/Forces"

if os.path.exists(forces_dir):
    for pyx_file in os.listdir(forces_dir):
        if pyx_file.endswith(".pyx"):
            module_name = f"asset_asrl.Forces.{pyx_file[:-4]}"
            ext_path = os.path.join(forces_dir, pyx_file)
            cython_extensions.append(
                Extension(
                    module_name,
                    [ext_path],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3"],
                )
            )

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
    ext_modules=cythonize(
        cython_extensions,
        language_level="3",
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "matplotlib",
        "spiceypy",
        "seaborn",
        "basemap",
        "intel-openmp",
        "Cython>=0.29.0",
    ],
    zip_safe=False,
)
