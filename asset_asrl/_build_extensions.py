from pathlib import Path
from setuptools import Extension
from Cython.Build import cythonize

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent


def find_extensions():

    extensions = []

    # =====================================================
    # CYTHON MODULES
    # =====================================================
    for pyx in (BASE_DIR / "Forces").rglob("*.pyx"):

        module = ".".join(
            pyx.relative_to(REPO_ROOT).with_suffix("").parts
        )

        sources = [
            str(pyx.relative_to(REPO_ROOT))  # MUST be relative
        ]

        # =================================================
        # OPTIONAL: C++ KERNEL SUPPORT
        # =================================================
        kernel_dir = pyx.parent / "kernels"

        if kernel_dir.exists():
            for cpp in kernel_dir.rglob("*.cpp"):
                sources.append(str(cpp.relative_to(REPO_ROOT)))

        extensions.append(
            Extension(
                module,
                sources=sources,
                language="c++"
            )
        )

    return extensions


def build_extensions():

    return cythonize(
        find_extensions(),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False
        }
    )
