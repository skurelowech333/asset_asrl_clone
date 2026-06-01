#!/usr/bin/env python3
import subprocess
import sys
import shutil
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

    return p.wait() == 0


def conda_exists():
    return run("conda --version")


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


def rebuild_env():
    title("ENV REBUILD")

    if env_exists():
        delete_env()

    create_env()
    

def pip(cmd, desc=None):
    return run(
        f'conda run -p "{ENV_PATH}" python -m pip {cmd}',
        desc
    )


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

    rebuild_env()

    title("INSTALLING DEPENDENCIES")

    pip("install --upgrade pip", "Upgrading pip")
    
    # Spyder support
    pip(
        "install spyder spyder-kernels",
        "Installing Spyder and spyder-kernels"
    )
    
    # ASSET
    pip("install asset-asrl", "Installing compiled ASSET wheel")
    pip("install -e .", "Installing repo (editable)")

    title("DONE")

    ok("Setup complete")
    
    print_interpreter_instructions()

    print(f"\nInterpreter:\n{ENV_PATH / 'python.exe'}\n")
    