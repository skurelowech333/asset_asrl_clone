# Building ASSET with native SPICE frames and Pines gravity

This branch adds native SPICE frame transforms (`SpicePxform`,
`SpicePxformApply`), a runtime force interface (`ForceField3`), a fused
body-frame node (`SpiceForceFrame`), and the vendored Pines gravity model
(`PinesGravity`).

## Not in the repository

These are needed locally but not committed:

| Path | Contents | How to get it |
|------|----------|---------------|
| `dep/eigen`, `dep/autodiff`, `dep/pybind11`, `dep/fmt` | Third-party C++ headers | Use the same versions as the upstream ASSET `dep/` tree, or copy them from the existing checkout |
| `native/data/egm96.nor` | EGM96 spherical-harmonic coefficients (about 5 MB) | Copy from the existing checkout, or get the file from the Pines/EGM96 source used upstream |
| CSPICE | `libcspice.so` and headers | NAIF CSPICE toolkit, Linux 64-bit build; this build used `/root/cspice` |
| MKL | Intel MKL libraries | Install oneAPI MKL; this build used `/root/mkl` |
| SPICE kernels | `pck00010.tpc` (lunar and planetary constants) | NAIF generic kernels; default path `/root/spice_kernels/pck00010.tpc` |
| Orekit data | Orekit 13.1 data directory | Default `/root/orekit-data` (override with `OREKIT_DATA`) |

## Toolchain used

- clang-15 and lld-15
- CMake, Ninja
- Python 3.12 virtualenv with `numpy`, `spiceypy`, `orekit-jpype` (for the benchmarks)

## Build

```bash
export PATH=/root/venv-asset/bin:$PATH
cd <build dir>
cmake <source dir> -DCMAKE_C_COMPILER=clang-15 -DCMAKE_CXX_COMPILER=clang++-15 \
      -DCMAKE_LINKER=lld-15
ninja
```

Copy the built module next to the Python package:

```bash
cp src/asset.cpython-312-x86_64-linux-gnu.so <package dir>/
```

## Run

```bash
export LD_LIBRARY_PATH=/root/venv-asset/lib:/root/cspice:/root/mkl/lib
export PYTHONPATH=<package dir>:<source dir>
```

Tests:

```bash
python -m unittest tests.test_spice_pxform
```

Benchmarks (run from Spyder with F5, or from the command line):

```bash
export SPICE_PCK=/root/spice_kernels/pck00010.tpc
export OREKIT_DATA=/root/orekit-data
python benchmarks/benchmark_moon_iau.py          # IAU_MOON transforms, then propagation
python benchmarks/benchmark_transform_isolation.py
python benchmarks/benchmark_pines_propagation_orekit.py
```

Do not overwrite the `.so` in the package directory while a benchmark is
running. It is memory-mapped.

## Notes

- `SpiceForceFrame` works with any kernel derived from `ForceField3`.
  `PinesGravity` is the only current implementation.
- Lunar gravity: no GRAIL coefficients are included. The GRAIL SHADR
  reader lives on the `grail_coefficients` branch of
  `AlabamaASRL/asset_asrl` and must be merged before using GRAIL fields.
