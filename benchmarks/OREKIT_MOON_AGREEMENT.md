# ASSET vs Orekit: Moon body-fixed frame agreement

Script: `benchmarks/benchmark_orekit_pxform.py`

## Frame pair

| Side | Frame | Source |
|---|---|---|
| ASSET | `SpicePxformApply("J2000", "IAU_MOON")` | NAIF `pck00010.tpc` (official, from GitHub mirror `alfonsogonzalez/AWP` at commit `f90a593`) |
| Orekit | `EME2000 -> CelestialBodyFactory.getMoon().getBodyOrientedFrame()` (MoonIAU) | `orekit-data` (`DirectoryCrawler`), IAU 2009 model |

`IAU_MOON` is the IAU body-fixed Moon frame, which Orekit's MoonIAU implements.

## Results (200 samples, ET 8.2e5 to 3.0e8 s, random vectors of 2e6 m)

- ASSET vs SPICE (`pxform`): agreement to 1e-10 at all tested epochs. ASSET is exact to SPICE.
- ASSET vs Orekit: |diff| max 0.318 m, median 0.167 m.
- Fitted small-angle frame rotation: 7.2 mas, axis close to the Moon spin axis (z).
- Residual after the rotation fit: max 0.30 m. A single fixed rotation does not explain the disagreement.
- Best-fit epoch shift for Orekit: about -0.05 s, roughly constant over ET. Residual after shift still about 0.14 m.

## Discrepancies and reasons

1. **Base inertial frame: J2000 (SPICE) vs EME2000 (Orekit).** Both are mean equator and equinox of J2000, but they can differ by the ICRF/J2000 frame bias (~17 mas). Measured: Orekit's EME2000-to-GCRF displacement is 0.246 m at a 3.7e6 m vector, which is the expected size. Not verified directly, because SPICE's ICRF frame is not available in this environment.
2. **Fitted rotation ~7 mas about the spin axis.** Consistent with a frame-orientation or pole/phase convention difference between the two IAU implementations. Not confirmed.
3. **Epoch convention, ~-0.05 s.** Orekit `AbsoluteDate.J2000_EPOCH` is TT-based, and ET is TDB. TDB-TT is ~1.6 ms, which contributes only ~4 mm here, so it does not explain -0.05 s. Source of the 0.05 s not identified.
4. **Unexplained residual ~0.14 m after rotation and time shift.** Not attributed. Likely candidates are Orekit's internal IAU/nutation/libration implementation and time-scale handling, but I have not verified either.
5. **`MOON_PA` not used.** The principal-axes frame needs a Moon frame kernel (`moon_pa`), which is not available offline. `IAU_MOON` is the closest IAU equivalent to Orekit's MoonIAU.
6. **Timing asymmetry.** Orekit timing includes `getTransformTo` per call. ASSET timing is `compute()` on a prebuilt VectorFunction. Both are Python loops, so speedups are approximate.
7. **Earlier attempts.** A hand-written libration kernel (`moon_iau2009.tpc`) failed to load in SPICE. It has been removed. The official `pck00010.tpc` is used instead.

## Status

Agreement is at the 0.3 m level on a 2e6 m vector, which is about 1.5e-7 rad. Items 1-2 are plausible explanations and item 3 is not explained. Item 4 remains open.
