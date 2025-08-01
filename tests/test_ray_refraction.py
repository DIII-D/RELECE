import numpy as np
from relativistic_ece.utils import refraction
from relativistic_ece.ray_refraction import ray_refraction


def test_isotropic_nr():
    w = 2*np.pi * 100e9  # Example frequency in rad/s
    wpe = 2*np.pi * 40e9  # Example plasma frequency in rad/s
    wce = 2*np.pi * 1e5  # Example cyclotron frequency in rad/s
    theta = np.pi / 2  # Arbitrary angle, can be set to 0 for isotropic case

    n_perp, n_par = refraction(w, wpe, wce, theta)
    n = np.sqrt(n_perp**2 + n_par**2)
    nr = np.sqrt(ray_refraction(n, w, wpe, wce, theta))
    print(f"Refractive index: {nr}")
    print(f"Expected refractive index for isotropic case: {n}")

    assert np.isclose(nr, n, rtol=1e-3), f"Expected {n}, got {nr}"
