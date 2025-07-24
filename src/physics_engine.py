"""
ECE signature modeling

Electron cyclotron emission (ECE) is the phenomenon of radiation from
gyrating electrons. In devices like DIII-D, ECE is exploited to measure
the temperature profile of the plasma.
"""
import numpy as np
from scipy.constants import c



def _calculate_refraction_coefs(w, wpe, wce, theta, eps_h=False):
    """Return coefficients needed to determine refraction index."""
    P = 1 - wpe**2 / w**2
    R = 1 - wpe**2 / (w * (w+wce))
    L = 1 - wpe**2 / (w * (w-wce))
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
    """Calculates the cold plasma refraction index squared.

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
    scalar
        Cold plasma refraction index squared.

    References
    ----------
    .. [1] Hutchinson, Ian, “Electromagnetic waves in plasmas,” in
           Introduction to Plasma Physics I, MIT OpenCourseWare, 2003,
           pp. 96-1
    """
    A, B, F = _calculate_refraction_coefs(w, wpe, wce, theta)
    nr2_plus = (B + F) / (2 * A)
    nr2_minus = (B - F) / (2 * A)

    # Select the wpe cutoff for the O mode
    if w >= wce:
        nr2_O = nr2_plus
        nr2_X = nr2_minus
    else:
        nr2_O = nr2_minus
        nr2_X = nr2_plus

    if x_mode:
        return nr2_X
    return nr2_O


def wavenumber(n, w):
    return n * w / c


# def dij(w, wpe, wce, theta, x_mode=False):
#     """Calculates equation 5.75 from the reference.

#     The determinant of this tensor yields a quadratic expression for
#     the cold plasma dispersion relation. It can also be used to
#     determine the Stix frame as well as the hermitian dielectric
#     tensor.

#     Reference: R. Parker, “Electromagnetic waves in plasmas,” in
#     Introduction to Plasma Physics I, MIT OpenCourseWare, 2006,
#     pp. 96-144

#     :param w: wave frequency
#     :type w: float
#     :param wpe: plasma frequency
#     :type wpe: float
#     :param wce: cyclotron frequency
#     :type wce: float
#     :param theta: propagation angle
#     :type theta: float
#     :param x_mode: whether the X mode is selected
#     :type x_mode: bool
#     :returns: cold plasma dispersion tensor
#     :rtype: np.ndarray
#     """
#     S, D, P = _calculate_refraction_coefs(w, wpe, wce, theta, eps_h=True)
#     nr2 = refraction(w, wpe, wce, theta, x_mode)

#     D00 = -nr2 * np.cos(theta)**2 + S
#     D01 = -1j * D
#     D02 = nr2 * np.sin(theta) * np.cos(theta)
#     D10 = 1j * D
#     D11 = -nr2 + S
#     D12 = 0
#     D20 = nr2 * np.sin(theta) * np.cos(theta)
#     D21 = 0
#     D22 = -nr2 * np.sin(theta)**2 + P

#     Dij = np.array([[D00, D01, D02],
#                     [D10, D11, D12],
#                     [D20, D21, D22]])
#     return Dij


def cold_plasma_eps_h(w, wpe, wce, theta):
    """
    TODO: write docstring
    """
    S, D, P = _calculate_refraction_coefs(w, wpe, wce, theta, eps_h=True)

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


def cold_plasma_dweps_hdw(w, wpe, wce, theta):
    """
    Calculates the derivative of w times eps_h. This quantity is needed
    in order to calculate the spectral energy density.
    """
    eps_h = cold_plasma_eps_h(w, wpe, wce, theta)
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
