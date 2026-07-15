#!/usr/bin/env python
"""
Setup script for ASSET with Cython compilation
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

import numpy as np
import os
import sysconfig



# ============================================================
# Include directories
# ============================================================

PYTHON_INCLUDE = sysconfig.get_paths()["include"]


INCLUDE_DIRS = [
    PYTHON_INCLUDE,
    np.get_include(),
]



# ============================================================
# Find all Cython modules
# ============================================================

cython_extensions = []

root_dir = "asset_asrl"


for root, dirs, files in os.walk(root_dir):

    for pyx_file in files:

        if pyx_file.endswith(".pyx"):

            pyx_path = os.path.join(
                root,
                pyx_file
            )


            #
            # Convert:
            #
            # asset_asrl\GravityPines.pyx
            #
            # into:
            #
            # asset_asrl.GravityPines
            #

            module_name = (
                pyx_path
                .replace(os.sep, ".")
                .replace(".pyx", "")
            )


            cython_extensions.append(

                Extension(
                    module_name,
                    [pyx_path],

                    include_dirs=INCLUDE_DIRS,

                    #
                    # MSVC optimization
                    #
                    extra_compile_args=[
                        "/O2"
                    ],
                )
            )



# ============================================================
# Print build modules
# ============================================================

print("\nCython modules:")

for ext in cython_extensions:
    print("  ", ext.name)



# ============================================================
# Setup
# ============================================================

setup(

    name="asset-asrl",

    version="0.5.2",

    description=(
        "Astrodynamics Software and Science "
        "Enabling Toolkit"
    ),

    long_description=(
        "High-speed, extensible, "
        "object-oriented trajectory "
        "design and optimization"
    ),

    author="James B. Pezent et al.",

    url=(
        "https://github.com/"
        "AlabamaASRL/asset_asrl"
    ),

    license="Apache 2.0",


    packages=find_packages(
        include=[
            "asset_asrl",
            "asset_asrl.*"
        ]
    ),


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