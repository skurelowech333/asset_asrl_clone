# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:23:51 2026

@author: Sarah
"""

# =========================================================
# FILE:
# tests/physics/test_gravity_reference_values.py
# =========================================================

import numpy as np
import pytest

from asset_asrl.Forces.Gravity import GravityVF


# =========================================================
# OFFICIAL JPL / IAU VALUE
#
# https://ssd.jpl.nasa.gov/astro_par.html
#
# Earth GM:
# 398600.435507 km^3/s^2
# =========================================================

MU_EARTH = 398600.435507


# =========================================================
# WGS84 EARTH EQUATORIAL RADIUS
# =========================================================

R_EARTH = 6378.137  # km


# =========================================================
# STANDARD GRAVITY
#
# 9.80665 m/s^2
#
# Convert to km/s^2
# =========================================================

G0 = 9.80665 / 1000.0


# =========================================================
# MODEL FACTORY
# =========================================================

def make_model():

    return GravityVF(MU_EARTH)


def accel(model, r):

    return np.asarray(model.compute(r))


# =========================================================
# TEST 1
# SURFACE GRAVITY CHECK
#
# Expected:
#
# g = mu / r^2
#
# Near Earth surface:
# 9.798 m/s^2 at equator
# =========================================================

def test_surface_gravity_matches_reference():

    model = make_model()

    r = np.array([R_EARTH, 0.0, 0.0])

    a = accel(model, r)

    amag = np.linalg.norm(a)

    expected = MU_EARTH / (R_EARTH**2)

    # km/s^2
    assert np.isclose(amag, expected, rtol=1e-12)

    # compare against standard gravity
    # convert to m/s^2

    amag_mps2 = amag * 1000.0

    assert np.isclose(
        amag_mps2,
        9.798285479,
        rtol=1e-6
    )


# =========================================================
# TEST 2
# GEO ORBIT GRAVITY
#
# GEO radius:
# 42164 km
#
# Expected acceleration:
# 0.224 m/s^2
# =========================================================

def test_geostationary_gravity():

    model = make_model()

    r_geo = 42164.0

    r = np.array([r_geo, 0.0, 0.0])

    a = accel(model, r)

    amag = np.linalg.norm(a)

    expected = MU_EARTH / (r_geo**2)

    assert np.isclose(
        amag,
        expected,
        rtol=1e-12
    )

    # Convert to m/s^2

    amag_mps2 = amag * 1000.0

    assert np.isclose(
        amag_mps2,
        0.224207,
        rtol=1e-5
    )


# =========================================================
# TEST 3
# ISS ORBITAL VELOCITY CONSISTENCY
#
# ISS altitude ≈ 420 km
#
# Circular orbit:
#
# v = sqrt(mu / r)
#
# Expected ≈ 7.66 km/s
# =========================================================

def test_iss_circular_velocity():

    altitude = 420.0

    r = R_EARTH + altitude

    v = np.sqrt(MU_EARTH / r)

    assert np.isclose(
        v,
        7.657,
        rtol=1e-3
    )


# =========================================================
# TEST 4
# ESCAPE VELOCITY AT SURFACE
#
# v_escape = sqrt(2mu/r)
#
# Expected ≈ 11.186 km/s
# =========================================================

def test_escape_velocity():

    vesc = np.sqrt(
        2.0 * MU_EARTH / R_EARTH
    )

    assert np.isclose(
        vesc,
        11.186,
        rtol=1e-3
    )


# =========================================================
# TEST 5
# GEO ORBIT PERIOD
#
# Expected sidereal day:
#
# 86164.0905 seconds
# =========================================================

def test_geostationary_period():

    r_geo = 42164.0

    T = 2.0 * np.pi * np.sqrt(
        r_geo**3 / MU_EARTH
    )

    assert np.isclose(
        T,
        86164.0905,
        rtol=1e-5
    )


# =========================================================
# TEST 6
# RADIAL DIRECTION
# =========================================================

def test_gravity_is_radial():

    model = make_model()

    r = np.array([
        7000.0,
        2000.0,
        -1000.0
    ])

    a = accel(model, r)

    cross = np.cross(r, a)

    assert np.linalg.norm(cross) < 1e-12


# =========================================================
# TEST 7
# INVERSE SQUARE LAW
# =========================================================

def test_inverse_square_law():

    model = make_model()

    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([14000.0, 0.0, 0.0])

    a1 = np.linalg.norm(accel(model, r1))
    a2 = np.linalg.norm(accel(model, r2))

    ratio = a1 / a2

    assert np.isclose(
        ratio,
        4.0,
        rtol=1e-12
    )


# =========================================================
# TEST 8
# SINGULARITY PROTECTION
# =========================================================

def test_zero_radius_behavior():

    model = make_model()

    r = np.array([0.0, 0.0, 0.0])

    with pytest.raises(Exception):

        accel(model, r)