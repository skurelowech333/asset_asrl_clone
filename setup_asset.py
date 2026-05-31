#!/usr/bin/env python3
"""
ASSET Conda Setup Script (Spyder AUTO-FIX VERSION)
- Works from Spyder, CMD, PowerShell
- No activation required
- Auto-rebuilds broken envs
- Auto-installs Spyder if missing
"""

import sys
import subprocess
import os
import shutil

ENV_NAME = "upd_asset_env"
PY_VERSION = "3.12"


# =========================================================
# Console
# =========================================================
class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def success(msg): print(f"{Color.GREEN}[OK] {msg}{Color.RESET}")
def warning(msg): print(f"{Color.YELLOW}[WARN] {msg}{Color.RESET}")
def error(msg): print(f"{Color.RED}[ERROR] {msg}{Color.RESET}")
def info(msg): print(f"{Color.CYAN}[INFO] {msg}{Color.RESET}")
def header(msg): print(f"\n{Color.BOLD}{Color.BLUE}{msg}{Color.RESET}")


# =========================================================
# RUN (LIVE OUTPUT)
# =========================================================
def run(cmd, desc=""):
    if desc:
        info(desc)

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in p.stdout:
        print(line, end="")

    return p.wait() == 0


# =========================================================
# CONDA HELPERS
# =========================================================
def conda_exists():
    return subprocess.run(
        "conda --version",
        shell=True,
        capture_output=True
    ).returncode == 0


def env_exists():
    try:
        out = subprocess.check_output("conda env list", shell=True, text=True)
        return ENV_NAME in out
    except:
        return False


def env_works():
    try:
        r = subprocess.run(
            f'conda run -n {ENV_NAME} python -c "print(1)"',
            shell=True,
            capture_output=True,
            text=True
        )
        return r.returncode == 0
    except:
        return False


# =========================================================
# ENV RESET
# =========================================================
def delete_env():
    warning("Deleting broken environment")

    run(f"conda env remove -n {ENV_NAME} -y", "Conda remove")
    run("conda clean -a -y", "Cleaning conda cache")

    base = os.path.expanduser("~\\anaconda3")
    path = os.path.join(base, "envs", ENV_NAME)

    if os.path.exists(path):
        warning(f"Force removing: {path}")
        shutil.rmtree(path, ignore_errors=True)

    success("Environment deleted")


def create_env():
    header("Creating environment")
    return run(
        f"conda create -y -n {ENV_NAME} python={PY_VERSION}",
        "Creating conda env"
    )


def ensure_env():
    header("Checking environment")

    if env_exists() and env_works():
        success("Environment OK")
        return

    warning("Broken or missing env → rebuilding")
    delete_env()
    create_env()


# =========================================================
# SPYDER HANDLING (NEW FIX)
# =========================================================
def ensure_spyder():
    header("Checking Spyder installation")

    # test if spyder exists
    test = subprocess.run(
        f'conda run -n {ENV_NAME} python -c "import spyder"',
        shell=True,
        capture_output=True,
        text=True
    )

    if test.returncode == 0:
        success("Spyder already installed")
        return

    warning("Spyder missing → installing")

    run(
        f'conda run -n {ENV_NAME} pip install spyder',
        "Installing Spyder"
    )


def launch_spyder():
    header("Launching Spyder")

    # MOST RELIABLE WAY ON WINDOWS
    run(
        f'conda run -n {ENV_NAME} python -m spyder',
        "Starting Spyder"
    )


# =========================================================
# MODULE INSPECTION
# =========================================================
def print_installed_modules():
    header("Installed Modules")

    cmd = (
        f'conda run -n {ENV_NAME} python -c "'
        "import pkgutil;"
        "mods = sorted([m.name for m in pkgutil.iter_modules()]);"
        "print('\\n'.join(mods))"
        '"'
    )

    run(cmd, "Listing modules")


def print_asset_structure():
    header("ASSET Structure")

    cmd = (
        f'conda run -n {ENV_NAME} python -c "'
        "import asset_asrl;"
        "import pkgutil;"
        "mods = [m.name for m in pkgutil.iter_modules(asset_asrl.__path__)];"
        "print('ASSET PATH:', asset_asrl.__file__);"
        "print('\\n'.join(sorted(mods)))"
        '"'
    )

    run(cmd, "Inspecting ASSET")


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"\n{Color.BOLD}{Color.BLUE}")
    print("=" * 60)
    print("ASSET Setup (Spyder AUTO-FIX)")
    print("=" * 60)
    print(Color.RESET)

    if not conda_exists():
        error("Conda not found")
        sys.exit(1)

    success("Conda detected")

    ensure_env()

    header("Installing dependencies")

    run(f'conda run -n {ENV_NAME} python -m pip install --upgrade pip',
        "Upgrading pip")

    run(f'conda run -n {ENV_NAME} pip install numpy scipy',
        "Installing core deps")

    run(f'conda run -n {ENV_NAME} pip install asset-asrl',
        "Installing ASSET")

    run(f'conda run -n {ENV_NAME} pip install -e .',
        "Editable install")

    # IMPORTANT NEW STEP
    ensure_spyder()

    header("Verification")

    run(f'conda run -n {ENV_NAME} python -c "import asset_asrl; print(asset_asrl.__file__)"',
        "Import check")

    run(f'conda run -n {ENV_NAME} python -c "import asset_asrl; print(\"ASSET OK\")"',
        "Final test")

    print_asset_structure()
    print_installed_modules()

    header("DONE")

    print("\nNext step:")
    print(f"{Color.CYAN}conda run -n {ENV_NAME} python -m spyder{Color.RESET}")

    if "--spyder" in sys.argv:
        launch_spyder()


if __name__ == "__main__":
    main()