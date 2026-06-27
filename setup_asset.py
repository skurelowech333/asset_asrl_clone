#!/usr/bin/env python3
import subprocess
import sys
import shutil
import time
from pathlib import Path

PY_VERSION = "3.12"

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / "env"


# -------------------------
# COLORS
# -------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def ok(msg): print(f"{C.GREEN}[OK]{C.RESET} {msg}")
def warn(msg): print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")
def err(msg): print(f"{C.RED}[ERROR]{C.RESET} {msg}")
def info(msg): print(f"{C.CYAN}[INFO]{C.RESET} {msg}")
def title(msg): print(f"\n{C.BOLD}{C.BLUE}{msg}{C.RESET}")


# -------------------------
# RUNNER
# -------------------------
def run(cmd, desc=None):
    if desc:
        info(desc)

    start = time.time()

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in p.stdout:
        print(line, end="")

    success = (p.wait() == 0)
    dt = time.time() - start

    if success:
        ok(f"done in {dt:.1f}s")
    else:
        err(f"failed in {dt:.1f}s")

    return success


def run_strict(cmd, desc=None):
    if not run(cmd, desc):
        sys.exit(1)


# -------------------------
# CHECKS
# -------------------------
def conda_exists():
    return shutil.which("conda") is not None


def env_exists():
    return ENV_PATH.exists()


# -------------------------
# ENV
# -------------------------
def setup_env():
    title("ENV SETUP")

    if not env_exists():
        run_strict(
            f'conda create -y --prefix "{ENV_PATH}" python={PY_VERSION}',
            "Creating env"
        )
    else:
        ok("env exists")


# -------------------------
# SUBMODULES (CRITICAL FIX)
# -------------------------
def init_submodules():
    title("SUBMODULES")

    run_strict("git submodule sync --recursive", "sync submodules")
    run_strict("git submodule update --init --recursive", "update submodules")

    ok("submodules ready")


# -------------------------
# CORE DEPENDENCIES
# -------------------------
def install_deps():
    title("DEPENDENCIES")

    run_strict(
        f'conda run -p "{ENV_PATH}" python -m pip install --upgrade pip setuptools wheel',
        "upgrading pip tooling"
    )

    run_strict(
        f'conda run -p "{ENV_PATH}" python -m pip install '
        "cmake ninja cython scikit-build-core",
        "installing build tools"
    )

    # MKL optional but OK
    run_strict(
        f'conda install -y -p "{ENV_PATH}" -c conda-forge mkl intel-openmp',
        "installing MKL"
    )


def build_asset():
    title("BUILDING ASSET (CYTHON + C++)")

    run_strict(
        f'conda run -p "{ENV_PATH}" python -m pip install '
        f'-e "{REPO_ROOT}" --no-build-isolation',
        "compiling ASSET (pyx + C++ extensions)"
    )

    ok("ASSET build complete")


# -------------------------
# SPYDER KERNEL
# -------------------------
def install_kernel():
    title("SPYDER KERNEL")

    run_strict(
        f'conda run -p "{ENV_PATH}" python -m pip install spyder-kernels==3.1.*',
        "installing kernel"
    )
    
def build_cython_extensions():
    """Compile Cython extensions (.pyx files to .pyd/.so)"""
    title("BUILDING CYTHON EXTENSIONS")
    
    info("Compiling Cython modules in asset_asrl/Forces/")
    
    # Run setup.py build_ext in-place to compile .pyx files
    build_cmd = f'conda run -p "{ENV_PATH}" python setup.py build_ext --inplace'
    
    if not run(build_cmd, "Building Cython extensions"):
        warn("Cython build had issues, but continuing with pure Python fallback...")
        return False
    
    ok("Cython extensions built successfully")
    return True


# -------------------------
# OVERLAY
# -------------------------
def setup_overlay():
    title("OVERLAY")

    site_cmd = "import site; print(site.getsitepackages()[0])"

    result = subprocess.run(
        f'conda run -p "{ENV_PATH}" python -c "{site_cmd}"',
        shell=True,
        capture_output=True,
        text=True
    )

    site_packages = Path(result.stdout.strip())

    with open(site_packages / "asset_asrl_local.pth", "w") as f:
        f.write(str(REPO_ROOT))

    ok("overlay injected")


# -------------------------
# VERIFY
# -------------------------
def verify():
    title("VERIFY")

    cmd = """
import asset_asrl
import asset_asrl.Forces
from asset_asrl.Forces.Gravity import GravityVF
print("FORCES OK")
"""

    return run(
        f'conda run -p "{ENV_PATH}" python -c "{cmd}"',
        "final verification"
    )


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    print(f"{C.BOLD}{C.BLUE}ASSET SETUP (FIXED BUILD PIPELINE){C.RESET}")

    if not conda_exists():
        err("conda missing")
        sys.exit(1)

    setup_env()
    init_submodules()
    install_deps()
    build_cython_extensions()
    build_asset()

    install_kernel()
    setup_overlay()

    if verify():
        title("SUCCESS")
        ok("ASSET READY")
    else:
        title("FAILED")
        warn("check build logs")