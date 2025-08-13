from relece.cold_plasma import refraction, dielectric_coefs
import numpy as np


def _refraction(w, wpe, wce, x_mode=False):
    """
    Calculates the refractive index for perpendicular propagation.
    """
    theta = np.pi / 2  # Perpendicular propagation
    R, L, S, _, P = dielectric_coefs(w, wpe, wce)
    n = refraction(w, wpe, wce, theta, x_mode=x_mode)
    n2 = n**2
    if x_mode:
        n2_expected = (R * L) / S
    else:
        n2_expected = P

    return n2, n2_expected


def test_x_mode_refraction_a():
    """
    Test X mode refraction for w < wce.
    """
    w = 2 * np.pi * 40e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce, x_mode=True)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_x_mode_refraction_b():
    """
    Test X mode refraction for w > wce.
    """
    w = 2 * np.pi * 60e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce, x_mode=True)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_x_mode_refraction_c():
    """
    Test X mode refraction for w = wce.
    """
    w = 2 * np.pi * 50e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce, x_mode=True)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_x_mode_refraction_d():
    """
    Test X mode refraction for w < wpe.
    """
    w = 2 * np.pi * 30e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce, x_mode=True)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_o_mode_refraction_a():
    """
    Test O mode refraction for w > wce.
    """
    w = 2 * np.pi * 100e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_o_mode_refraction_b():
    """
    Test O mode refraction for w < wce.
    """
    w = 2 * np.pi * 40e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_o_mode_refraction_c():
    """
    Test O mode refraction for w = wce.
    """
    w = 2 * np.pi * 50e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_o_mode_refraction_d():
    """
    Test O mode refraction for w = wpe.
    """
    w = 2 * np.pi * 40e9  # Example frequency in rad/s
    wpe = 2 * np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2 * np.pi * 50e9  # Example cyclotron frequency in rad/s

    n2, n2_expected = _refraction(w, wpe, wce)

    assert np.isclose(n2, n2_expected), f"Expected {n2_expected}, got {n2}"


def test_bekefi_point_8():
    """
    Test O and X mode refraction at point 8 in Bekefi's figure 1.10.
    """
    w = 2 * np.pi * 40e9  # Example frequency in rad/s
    wpe = w
    wce = 3 / 2 * w

    no2, _ = _refraction(w, wpe, wce, x_mode=False)
    nx2, _ = _refraction(w, wpe, wce, x_mode=True)

    assert np.isclose(no2, 0), f"Expected 0, got {no2}"
    assert np.isclose(nx2, 1), f"Expected 1, got {nx2}"
