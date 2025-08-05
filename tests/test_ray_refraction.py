import numpy as np
from relativistic_ece.utils import refraction
from relativistic_ece.ray_refraction import ray_refraction


# --- Test Cases ---

# Define a set of plasma parameters to use in the tests
# Frequencies are in rad/s
WAVE_FREQ = 1.5e11  # Wave frequency (24 GHz)
PLASMA_FREQ = 0.8e11  # Plasma frequency
CYCLOTRON_FREQ = 1.0e11  # Electron cyclotron frequency


def test_vacuum():
    """
    Test Case 1: Vacuum
    In a vacuum (wpe=0, wce=0), the refractive index n should be 1,
    and the ray refractive index nr should also be 1.
    """
    wpe_vac = 0
    wce_vac = 0
    theta = np.pi / 4  # Angle shouldn't matter

    # Calculate phase refractive index components
    n = refraction(WAVE_FREQ, wpe_vac, wce_vac, theta)
    nr = ray_refraction(n, WAVE_FREQ, wpe_vac, wce_vac, theta)
    assert np.isclose(nr, 1.0), f"Expected 1.0, got {nr}"


def test_isotropic_medium():
    """
    Test Case 2: Isotropic Medium
    In an isotropic medium (wce = 0), the refractive index n should be
    the same as the ray refractive index nr.
    """
    wce_iso = 0
    theta = np.pi / 4  # Angle shouldn't matter

    # Calculate phase refractive index components
    n = refraction(WAVE_FREQ, PLASMA_FREQ, wce_iso, theta)

    # Calculate ray refractive index squared (should also be 1)
    nr = ray_refraction(n, WAVE_FREQ, PLASMA_FREQ, wce_iso, theta)
    assert np.isclose(nr, n), f"Expected {n}, got {nr}"


def test_bekefi_point_8():
    """
    Test Bekefi point 8.
    """
    w = WAVE_FREQ
    wpe = (1 - 1e-5) * w  # Slightly below w to avoid singularity
    wce = 3 / 2 * w
    theta = np.pi / 2  # Perpendicular propagation

    no = refraction(w, wpe, wce, theta, x_mode=False)
    nx = refraction(w, wpe, wce, theta, x_mode=True)
    nro = ray_refraction(no, w, wpe, wce, theta)
    nrx = ray_refraction(nx, w, wpe, wce, theta)

    assert np.isclose(nro, 1), f"Expected 1 for O mode, got {nro}"
    assert np.isclose(nrx, 1), f"Expected 0 for X mode, got {nrx}"


test_bekefi_point_8()
