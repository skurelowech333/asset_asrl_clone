<img src="https://github.com/AlabamaASRL/asset_asrl/assets/40646929/ebd03b27-df85-41bb-9cd0-6bfa06c14f26" width="300" height="300" align="center">

# ASSET: Astrodynamics Software and Science Enabling Toolkit

ASET (Astrodynamics Software and Science Enabling Toolkit) is a modular, extensible library for trajectory design and optimal control. It uses a custom implementation of vector math formalisms to enable rapid implementation of dynamical systems and automatic differentiation. The phase object is the core of the optimal control functionality, and by linking multiple phases together, the user can construct scenarios of arbitrary complexity. A newly developed high performance interior-point optimizer (PSIOPT) is included with the library, which enables quick turnaround from concept to solution.

Development funded by NASA under Grant No. 80NSSC19K1643.

## Installation

### From PyPI (Recommended)
The easiest way to install ASSET is from PyPI:

```bash
pip install asset-asrl
```

This installs a precompiled binary wheel with all dependencies.

### From Source
For development or to build custom extensions:

**Requirements:**
- CMake >= 3.16
- C++17 compiler (GCC, Clang, or MSVC)
- Python 3.7+
- Intel MKL (Math Kernel Library)
- Dependencies: numpy, matplotlib, spiceypy, seaborn, basemap, intel-openmp

**Build:**
```bash
git clone https://github.com/skurelowech333/ASSET.git
cd ASSET
git submodule update --init --recursive
pip install -e .
```

## Features

- **Modular Design**: Extensible architecture for custom trajectory design
- **Automatic Differentiation**: Built-in AD for optimization
- **Optimal Control**: Phase-based trajectory optimization
- **High Performance**: PSIOPT interior-point optimizer
- **Force Models**: Comprehensive gravity and drag force implementations
  - Central gravity
  - J2 perturbations  
  - Drag modeling
  - N-body gravity
  - Enhanced gravity models

## Quick Start

```python
import asset_asrl as asset

# Access force models
from asset_asrl.Forces import Gravity, Drag

# Create trajectory optimization problem
# ... your code here
```

## Documentation

Full documentation available at: https://alabamaasrl.github.io/asset_asrl/

## This Repository

This is a personal development clone of the official [ASSET repository](https://github.com/AlabamaASRL/asset_asrl) with custom modifications, particularly enhancements to the Forces module.

### Custom Changes
- Enhanced Forces module with additional force model implementations
- Production-ready package structure

## Citation

If you use ASSET in a published work, please cite:

```bibtex
@inproceedings{Pezent2022,
  author = {James B. Pezent and Jared Sikes and William Ledbetter and Rohan Sood and Kathleen C. Howell and Jeffrey R. Stuart},
  title = {ASSET: Astrodynamics Software and Science Enabling Toolkit},
  booktitle = {AIAA SCITECH 2022 Forum},
  pages = {AIAA 2022-1131},
  year = {2022},
  doi = {10.2514/6.2022-1131}
}
```

## License

Apache License 2.0 - See LICENSE.txt for details
