from relativistic_ece.utils import refraction, refraction_coefs
import numpy as np


def _refraction(w, wpe, wce, x_mode=False):
    """
    Calculates the refractive index for perpendicular propagation.
    """
    theta = np.pi / 2  # Perpendicular propagation
    R, L, S, _, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)
    n_perp, n_par = refraction(w, wpe, wce, theta, x_mode=True)
    n2 = n_perp**2 + n_par**2
    if x_mode:
        n2_expected = (R * L) / S
    else:
        n2_expected = P
    n2_expected = R * L / S

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
    n2_expected = wpe**2 * (2*w**2 - wpe**2) / (wpe**2 * w**2)

    n2, _ = _refraction(w, wpe, wce, x_mode=True)

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
    n2_expected = wpe**2 * (2*w**2 - wpe**2) / (wpe**2 * w**2)

    n2, _ = _refraction(w, wpe, wce)

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
