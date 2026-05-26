#!/bin/bash
# One-command install for ASSET and all dependencies
set -e

echo "============================================"
echo "ASSET: Installing All Dependencies"
echo "============================================"
echo ""

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing ASSET and all dependencies..."
pip install asset-asrl -e ".[dev]"

echo ""
echo "============================================"
echo "✓ Installation complete!"
echo "============================================"
echo ""
echo "Verify:"
echo "  python -c \"from asset_asrl.Forces import GravityVF; print('✓ Works')\""
