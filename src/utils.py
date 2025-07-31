"""
ECE signature modeling

Electron cyclotron emission (ECE) is the phenomenon of radiation from
gyrating electrons. In devices like DIII-D, ECE is exploited to measure
the temperature profile of the plasma.
"""
import numpy as np
from scipy import linalg
from scipy.constants import c


def refraction_coefs(w, wpe, wce, theta, eps_h=False):
    """Return coefficients needed to determine refraction index."""
    P = 1 - wpe**2 / w**2
    R = 1 - wpe**2 / (w * (w-wce))
    L = 1 - wpe**2 / (w * (w+wce))
    S = (R + L) / 2
    D = (R - L) / 2
    if eps_h:
        return S, D, P
    A = S * np.sin(theta)**2 + P * np.cos(theta)**2
    B = R * L * np.sin(theta)**2 + P * S * (1 + np.cos(theta)**2)
    F2 = (R*L - P*S)**2 * np.sin(theta)**4 + 4 * P**2 * D**2 * np.cos(theta)**2
    F = np.sqrt(F2)
    return A, B, F


def refraction(w, wpe, wce, theta, x_mode=False):
    """Calculates parallel and perpendicular refractive indices.

    This index depends on the wave mode and is calculated using the
    Appleton-Hartree equation. This equation yields a quadratic for nr2,
    which provides two roots. The perpendicular case (theta = pi/2)
    tells us which root should be assigned to the O vs X mode. If the
    wave frequency is greater than the cyclotron frequency, the "plus"
    root is the O mode and the "minus" root is the X mode, and vice-
    versa for the opposite case.

    Parameters
    ----------
    w : scalar
        Incident wave frequency (rad/s).
    wpe : scalar
        Plasma frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    theta : scalar
        Wave propagation angle.
    x_mode : bool
        Whether the X mode is selected.

    Returns
    -------
    n_perp : scalar
        Perpendicular refractive index.
    n_par : scalar
        Parallel refractive index.

    References
    ----------
    .. [1] Stix, T. H., 1962, *The Theory of Plasma Waves*,
           McGraw-Hill, New York.
    """
    A, B, F = refraction_coefs(w, wpe, wce, theta)
    n2_plus = (B + F) / (2 * A)
    n2_minus = (B - F) / (2 * A)

    # Select the wpe cutoff for the O mode
    if w >= wce:
        n2_O = n2_plus
        n2_X = n2_minus
    else:
        n2_O = n2_minus
        n2_X = n2_plus

    if x_mode:
        n2 = n2_X
    else:
        n2 = n2_O
    n = np.sqrt(n2)
    n_perp = np.sin(theta) * n
    n_par = np.cos(theta) * n
    return n_perp, n_par


def wavevector(nr, w, theta):
    """Calculates the wave vector from the refraction index.

    The wave is assumed to propagate in the x-z plane, with the
    magnetic field along the z-axis.
    """
    k = nr * w / c
    kx = k * np.sin(theta)
    ky = 0
    kz = k * np.cos(theta)
    return np.array([kx, ky, kz])


def dispersion(w, wpe, wce, theta, x_mode=False):
    """Calculates the dispersion tensor for cold plasma.

    The determinant of this tensor yields a quadratic expression for
    the cold plasma dispersion relation. It can also be used to
    determine the Stix frame as well as the hermitian dielectric
    tensor. It is referred to as Lambda in Ref. [1], although the
    implementation follows Ref. [2].

    Parameters
    ----------
    w : scalar
        Incident wave frequency (rad/s).
    wpe : scalar
        Plasma frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    theta : scalar
        Wave propagation angle.
    x_mode : bool
        Whether the X mode is selected.

    Returns
    -------
    complex ndarray
       Cold plasma dispersion tensor.

    References
    ----------
    .. [1] Bornatici, M., et al, 1983, "Electron cyclotron emission and
           absorption in fusion plasmas," *Nucl. Fusion*, 23(9),
           1153-1257.
    .. [2] Stix, T. H., 1962, *The Theory of Plasma Waves*,
           McGraw-Hill, New York.
    """
    S, D, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)
    n_perp, n_par = refraction(w, wpe, wce, theta, x_mode)
    n2 = n_perp**2 + n_par**2

    D00 = -n2 * np.cos(theta)**2 + S
    D01 = -1j * D
    D02 = n2 * np.sin(theta) * np.cos(theta)
    D10 = 1j * D
    D11 = -n2 + S
    D12 = 0
    D20 = n2 * np.sin(theta) * np.cos(theta)
    D21 = 0
    D22 = -n2 * np.sin(theta)**2 + P

    Lambda = np.array([[D00, D01, D02],
                       [D10, D11, D12],
                       [D20, D21, D22]])
    return Lambda


def polarization(Lambda, k, w):
    """
    Calculates the electric field (polarization) from the dispersion
    tensor. Since the dispersion tensor depends on frequency, the
    result is in the Fourier domain.

    Parameters
    ----------
    Lambda : complex ndarray
        Cold plasma dispersion tensor.
    k : complex ndarray
        Wave vector (1/m).
    w : scalar
        Wave frequency (rad/s).

    Returns
    -------
    complex ndarray
        Electric and magnetic fields.

    Raises
    ------
    ValueError
        If the dispersion tensor does not have nullity 1.

    References
    ----------
    .. [1] Stix, T. H., 1962, *The Theory of Plasma Waves*,
           McGraw-Hill, New York.
    """
    E = linalg.null_space(Lambda, rcond=1e-15)
    if np.shape(E)[1] > 1:
        raise ValueError("Dispersion tensor must have nullity 1.")

    B = c / w * np.cross(k, E)

    return E, B


def cold_plasma_eps_h(w, wpe, wce, theta):
    """
    Calculates the cold plasma Hermitian dielectric tensor.

    Parameters
    ----------
    w : scalar
        Incident wave frequency (rad/s).
    wpe : scalar
        Plasma frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    theta : scalar
        Wave propagation angle.

    Returns
    -------
    complex ndarray
        Cold plasma Hermitian dielectric tensor.
    """
    S, D, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)

    eps00 = S
    eps01 = -1j * D
    eps02 = 0
    eps10 = 1j * D
    eps11 = S
    eps12 = 0
    eps20 = 0
    eps21 = 0
    eps22 = P

    eps_h = np.array([[eps00, eps01, eps02],
                      [eps10, eps11, eps12],
                      [eps20, eps21, eps22]])
    return eps_h


def _refraction_derivs(w, wpe, wce):
    """Derivatives of refraction coefs with respect to w."""
    dSdw = 2 * w * wpe**2 / (w**2 - wce**2)**2
    dDdw = wce * wpe**2 * (wce**2 - 3*w**2) / (w**2 * (wce**2 - w**2)**2)
    dPdw = 2 * wpe**2 / w**3

    return dSdw, dDdw, dPdw


def cold_plasma_dwepshdw(w, wpe, wce, eps_h):
    """
    Calculates the derivative of w times eps_h. This quantity is needed
    in order to calculate the spectral energy density.
    """
    dSdw, dDdw, dPdw = _refraction_derivs(w, wpe, wce)

    deps00 = dSdw
    deps01 = -1j * dDdw
    deps02 = 0
    deps10 = 1j * dDdw
    deps11 = dSdw
    deps12 = 0
    deps20 = 0
    deps21 = 0
    deps22 = dPdw

    deps_hdw = np.array([[deps00, deps01, deps02],
                         [deps10, deps11, deps12],
                         [deps20, deps21, deps22]])
    dweps_hdw = eps_h + w * deps_hdw
    return dweps_hdw
