#!/usr/bin/env python3
"""
ASSET Setup Script
Run this after cloning: python setup_asset.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    if description:
        print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("ASSET Installation Setup")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]}")

    # Check OS
    system = platform.system()
    print(f"✓ OS: {system}")

    # Get repo root
    repo_root = Path(__file__).parent.absolute()
    venv_path = repo_root / "venv"

    print(f"✓ Repository: {repo_root}")

    # Create virtual environment
    print(f"\n📦 Creating virtual environment...")
    if venv_path.exists():
        print(f"  (venv already exists at {venv_path})")
    else:
        if not run_command(f"{sys.executable} -m venv venv", "Creating venv"):
            print("❌ Failed to create virtual environment")
            sys.exit(1)

    # Determine activate script path
    if system == "Windows":
        activate_cmd = str(venv_path / "Scripts" / "activate.bat")
        pip_cmd = str(venv_path / "Scripts" / "pip")
        python_cmd = str(venv_path / "Scripts" / "python")
    else:
        activate_cmd = str(venv_path / "bin" / "activate")
        pip_cmd = str(venv_path / "bin" / "pip")
        python_cmd = str(venv_path / "bin" / "python")

    print(f"✓ Virtual environment ready at: {venv_path}")

    # Upgrade pip
    if not run_command(f"{python_cmd} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: pip upgrade failed, continuing...")

    # Install ASSET
    if not run_command(f"{pip_cmd} install -e .", "Installing ASSET"):
        print("❌ Failed to install ASSET")
        sys.exit(1)

    # Verify installation
    print(f"\n✓ Verifying installation...")
    verify_cmd = f"{python_cmd} -c \"import asset_asrl; print('ASSET imported successfully!')\""
    if not run_command(verify_cmd):
        print("⚠️  Warning: Could not verify ASSET import (may be due to missing Intel MKL)")
    else:
        print("✓ ASSET verified!")

    # Print next steps
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print(f"\nTo activate the virtual environment, run:")
    if system == "Windows":
        print(f"  venv\\Scripts\\activate")
    else:
        print(f"  source venv/bin/activate")

    print(f"\nThen verify with:")
    print(f"  python -c \"import asset_asrl; print('Success!')\"")

    print(f"\n📖 For Spyder users:")
    print(f"  Tools → Preferences → Python Interpreter")
    print(f"  Select: {python_cmd}")
    print(f"  Then restart Spyder")

    print(f"\n📚 See INSTALL.md for more details")
    print("=" * 60)


if __name__ == "__main__":
    main()
