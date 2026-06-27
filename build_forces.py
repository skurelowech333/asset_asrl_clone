
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            "C:\Users\Sarah\Documents\Research\new\ASSET\asset_asrl\Forces\Gravity.pyx",
            "C:\Users\Sarah\Documents\Research\new\ASSET\asset_asrl\Forces\Drag.pyx",
            "C:\Users\Sarah\Documents\Research\new\ASSET\asset_asrl\Forces\J2Kernel.pyx",
        ],
        compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()]
)
