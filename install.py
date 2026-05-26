#!/usr/bin/env python
"""
One-command install script for ASSET
Run with: python install.py
"""
import subprocess
import sys
import platform

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("ASSET: Installing All Dependencies")
    print("=" * 50)
    
    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install asset-asrl and dev dependencies
    if not run_command(f"{sys.executable} -m pip install asset-asrl", "Installing asset-asrl"):
        print("Error installing asset-asrl...")

    
    # Install local package in editable mode with dev dependencies
    if not run_command(f"{sys.executable} -m pip install -e .", "Installing local package"):
        print("Error installing local package...")
    
    print("\n" + "=" * 50)
    print("Installation complete!")
    print("=" * 50)
    print("\nVerify installation:")
    print("  python -c \"from asset_asrl.Forces import GravityVF; print('Works')\"")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
