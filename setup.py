# Mostly copied from pybind/cmake_example

import os
import re
import sys
import platform
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import multiprocessing

ROOT = Path(__file__).resolve().parent


def read_version():
    # CMakeLists.txt is the single source of truth for the project version.
    text = (ROOT / "CMakeLists.txt").read_text()
    match = re.search(r"project\(ASSET VERSION ([\d.]+)", text)
    if match is None:
        raise RuntimeError("Could not find ASSET version in CMakeLists.txt")
    return match.group(1)


def version_tuple(version):
    return tuple(int(part) for part in re.findall(r"\d+", version))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = version_tuple(
                re.search(r"version\s*([\d.]+)", out.decode()).group(1)
            )
            if cmake_version < (3, 1, 0):
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DPIP_INSTALL=True",
        ]
        build_args = ["--target", "asset"]

        cfg = "Debug" if self.debug else "Release"
        build_args += ["--config", cfg]

        # Limit core count if we're on a github runner
        if "GITHUB_WORKSPACE" in os.environ:
            build_args += ["--", "-j1"]
        elif platform.system() != "Windows":
            # build_args += ["--", "-j{}".format(int(0.5 * multiprocessing.cpu_count()))]
            build_args += ["--", "-j4"]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), extdir)
            ]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    version=read_version(),
    long_description="",
    ext_modules=[CMakeExtension("asset")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
