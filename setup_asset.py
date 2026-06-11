#!/usr/bin/env python3
import subprocess
import sys
import shutil
import time
from pathlib import Path

PY_VERSION = "3.12"

REPO_ROOT = Path(__file__).resolve().parent
ENV_PATH = REPO_ROOT / "env"


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def ok(msg): 
    print(f"{C.GREEN}[OK]{C.RESET} {msg}")
    
def warn(msg): 
    print(f"{C.YELLOW}[WARN]{C.RESET} {msg}")
    
def err(msg): 
    print(f"{C.RED}[ERROR]{C.RESET} {msg}")
    
def info(msg): 
    print(f"{C.CYAN}[INFO]{C.RESET} {msg}")
    
def title(msg): 
    print(f"\n{C.BOLD}{C.BLUE}{msg}{C.RESET}")


def run(cmd, desc=None):
    if desc:
        info(desc)

    start = time.time()
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in p.stdout:
        print(line, end="")  # live stream

    success = p.wait() == 0
    elapsed = time.time() - start
    
    if success and desc:
        ok(f"Completed in {elapsed:.1f}s")
    elif not success:
        err(f"Failed after {elapsed:.1f}s")
    
    return success


def conda_exists():
    """Check if conda is available"""
    try:
        result = subprocess.run(
            "conda --version",
            shell=True,
            capture_output=True,
            timeout=5,
            text=True
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def env_exists():
    return ENV_PATH.exists()


def delete_env():
    warn("Deleting existing environment")

    run(f'conda env remove -p "{ENV_PATH}" -y', "Removing conda env")
    shutil.rmtree(ENV_PATH, ignore_errors=True)

    ok("Environment deleted")
    

def create_env():
    return run(
        f'conda create -y --prefix "{ENV_PATH}" python={PY_VERSION}',
        "Creating fresh environment"
    )


def rebuild_env_if_needed():
    """Only rebuild if the environment doesn't exist or is corrupted"""
    title("ENV SETUP")
    
    if env_exists():
        ok("Environment already exists, skipping rebuild")
        return True
    
    info("Creating fresh environment")
    return create_env()
    

def pip(cmd, desc=None):
    return run(
        f'conda run -p "{ENV_PATH}" python -m pip {cmd}',
        desc
    )


def pip_install_batch(packages, desc=None):
    """Install multiple packages in a single pip call for better performance"""
    cmd = "install " + " ".join(packages)
    return pip(cmd, desc)


def init_submodules():
    title("INITIALIZING GIT SUBMODULES")
    
    # Check if submodules are configured
    result = subprocess.run(
        "git config --file .gitmodules --get-regexp path",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if not result.stdout.strip():
        ok("No submodules configured")
        return True
    
    # Initialize and update submodules in one call
    if not run("git submodule update --init --recursive", 
               "Initializing and updating submodules"):
        err("Failed to update submodules")
        return False
    
    ok("Submodules initialized and updated successfully")
    return True


def verify_installation():
    """Verify that asset_asrl.Forces can be imported"""
    title("VERIFYING INSTALLATION")
    
    verify_cmd = (
        f'conda run -p "{ENV_PATH}" python -c '
        '"from asset_asrl.Forces.Gravity import GravityVF; print(\'Forces module loaded successfully\')"'
    )
    
    result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        ok("asset_asrl.Forces module verified")
        return True
    else:
        warn("Asset Forces module verification failed")
        warn("This may be due to missing build dependencies")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False


def print_interpreter_instructions():
    title("PYTHON INTERPRETER SETUP (IMPORTANT)")

    python_path = ENV_PATH / "python.exe"

    print(f"""
        {C.CYAN}STEP 1:{C.RESET}
            Open Spyder
        
        {C.CYAN}STEP 2:{C.RESET}
            Go to:
                Tools → Preferences → Python Interpreter
        
        {C.CYAN}STEP 3:{C.RESET}
            Select:
                {python_path}
        
        {C.CYAN}STEP 4:{C.RESET}
            Restart kernel
        
        {C.CYAN}STEP 5:{C.RESET}
            Verify inside Spyder:
                import sys
                print(sys.executable)
        
            It MUST show:
                {python_path}
        """)

if __name__ == "__main__":

    print(f"{C.BOLD}{C.BLUE}\nASSET Setup\n{C.RESET}")

    if not conda_exists():
        err("Conda not found")
        sys.exit(1)

    ok("Conda detected")

    rebuild_env_if_needed()
    
    if not init_submodules():
        err("Submodule initialization failed")
        sys.exit(1)

    title("INSTALLING DEPENDENCIES")

    # Batch install all dependencies in one call for speed
    pip_install_batch([
        "--upgrade",
        "pip",
        "setuptools>=62.0",
        "wheel",
        "ninja",
        "cmake>=3.16",
        "cython>=0.29.0",
        "spyder",
        "spyder-kernels",
        "asset-asrl",
    ], "Installing core dependencies")
    
    title("BUILDING AND INSTALLING LOCAL PACKAGE")
    
    # Install local package in editable mode with verbose output to catch build errors
    if not pip("install -e . --verbose --no-cache-dir", "Installing repo (editable mode with build)"):
        warn("Local package installation had issues, but continuing...")
    
    title("VERIFYING INSTALLATION")
    
    # Verify the installation
    if verify_installation():
        title("DONE")
        ok("Setup complete - asset_asrl.Forces verified")
    else:
        title("SETUP COMPLETE (WITH WARNINGS)")
        warn("Setup completed but asset_asrl.Forces could not be verified")
        warn("This may require manual build steps or additional system dependencies")
        print(f"\n{C.YELLOW}Troubleshooting steps:{C.RESET}")
        print("1. Ensure you have a C++17 compiler installed (MSVC, GCC, or Clang)")
        print("2. Verify CMake is installed: conda list cmake")
        print("3. Check Intel MKL: conda list mkl")
        print("4. Try manual rebuild: conda run -p <env_path> pip install -e . --verbose\n")
    
    print_interpreter_instructions()

    print(f"\nInterpreter:\n{ENV_PATH / 'python.exe'}\n")
