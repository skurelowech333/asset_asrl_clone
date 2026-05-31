# Installation Guide for ASSET

## Quick Start

### Prerequisites
- Python 3.7+
- CMake >= 3.16
- C++17 compiler (GCC, Clang)
- Intel MKL

### Install

```bash
# Clone repository
git clone https://github.com/skurelowech333/ASSET.git
cd ASSET

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install
pip install -e .
```

### Verify
```bash
python -c "import asset_asrl; print('Success!')"
```

## Spyder Configuration

After installation, set Spyder to use your environment:
1. Tools → Preferences → Python Interpreter
2. Select: `/path/to/ASSET/venv/bin/python`
3. Restart Spyder

## Troubleshooting

**Module not found?** Make sure virtual environment is activated and Spyder uses correct interpreter.

**Build errors?** Install: `sudo apt-get install cmake build-essential` (Ubuntu) or `brew install cmake` (macOS)

**MKL not found?** Download from https://software.intel.com/oneapi/toolkit/mkl

For more details, see the original repo: https://github.com/AlabamaASRL/asset_asrl
