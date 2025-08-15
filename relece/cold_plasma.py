"""Calculates wave propagation quantities for cold plasma.

These include:
- Hermitian dielectric tensor
- (Ray) refractive index
- Polarization
"""
import numpy as np
from scipy.constants import c
from scipy.differentiate import derivative


def dielectric_coefs(w, wpe, wce, nu=1e-6):
    """Calculate the Stix coefficients for the dielectric tensor.

    To account for singularities at the positive or negative cyclotron
    frequency or 0, an infinitesimally small causality term is added to
    the wave frequency.
    """
    w += np.where(w == 0 or np.abs(w) == np.abs(wce), 1j * nu, 0)
    P = np.real(1 - wpe**2 / w**2)
    R = np.real(1 - wpe**2 / (w * (w - wce)))
    L = np.real(1 - wpe**2 / (w * (w + wce)))
    S = (R + L) / 2
    D = (R - L) / 2
    return R, L, S, D, P


def refraction_coefs(w, wpe, wce, theta):
    """Return coefficients used to determine refraction index."""
    R, L, S, D, P = dielectric_coefs(w, wpe, wce)
    A = S * np.sin(theta)**2 + P * np.cos(theta)**2
    B = R * L * np.sin(theta)**2 + P * S * (1 + np.cos(theta)**2)
    F2 = (R * L - P * S)**2 * np.sin(theta)**4 + 4 * P**2 * D**2 * np.cos(theta)**2
    F = np.sqrt(F2)
    return A, B, F, R, L, S, D, P


def refraction(w, wpe, wce, theta, x_mode=False):
    """Calculates parallel and perpendicular refractive indices.

    This index depends on the wave mode and is calculated using the
    Appleton-Hartree equation. This equation yields a quadratic for nr2,
    which provides two roots. The perpendicular case (theta = pi/2)
    tells us which root should be assigned to the O vs X mode. If the
    wave frequency is greater than the cyclotron frequency, the "plus"
    root is the O mode and the "minus" root is the X mode, and vice-
    versa for the opposite case.

    It is important to note that cold plasma dispersion is invalid near
    the fundamental cyclotron frequency, where a thermal dispersion
    relation is more appropriate [2]_.

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
    n : scalar
        Refractive index.

    Raises
    ------
    RuntimeWarning
        Avoid cold plasma dispersion near the fundamental cyclotron
        frequency.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).
    .. [2] H. Weitzner and D. B. Batchelor, Phys. Fluids **23**, 1359
           (1980).

    """
    # if np.any(np.isclose(w, wce) or np.isclose(w, -wce)):
    #     raise RuntimeWarning(
    #         "Cold plasma approximation may be invalid near the "
    #         "fundamental cyclotron frequency."
    #     )
    # A, B, F, *_ = refraction_coefs(w, wpe, wce, theta)
    # n2_plus = (B + F) / (2 * A)
    # n2_minus = (B - F) / (2 * A)

    # # Select the wpe cutoff for the O mode
    # if w > wce:
    #     n2_O = n2_plus
    #     n2_X = n2_minus
    # else:
    #     n2_O = n2_minus
    #     n2_X = n2_plus

    # if x_mode:
    #     n2 = n2_X
    # else:
    #     n2 = n2_O
    # n2 = np.real(n2).astype(np.complex128)

    # n = np.sqrt(n2)
    # return n
    delta = np.sqrt(wce**2 * np.sin(theta)**4 + 4 * (w**2 - wpe**2)**2
                    * np.cos(theta)**2 / w**2)
    if x_mode:
        delta *= -1
    p = 2 * wpe**2 * (w**2 - wpe**2) / w**2
    q = 2 * (w**2 - wpe**2) - wce**2 * np.sin(theta)**2 + wce * delta
    n2 = 1 - p / q
    return np.sqrt(n2)


def dispersion(w, wpe, wce, theta, x_mode=False):
    """
    Calculates the dispersion tensor for cold plasma.

    The determinant of this tensor yields a quadratic expression for
    the cold plasma dispersion relation. It can also be used to
    determine the Stix frame as well as the hermitian dielectric
    tensor.

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
    .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).
    .. [2] M. Bornatici *et al.*, Nucl. Fusion **23**, 1153 (1983).

    """
    *_, S, D, P = dielectric_coefs(w, wpe, wce)
    n = refraction(w, wpe, wce, theta, x_mode)
    n2 = n**2
    n_perp = n * np.sin(theta)
    n_par = n * np.cos(theta)
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


def polarization(w, wpe, wce, theta, k, n):
    """Calculates the field polarization.

    The result is given in terms of the electric and magnetic fields,
    normalized to the radial (x) component of the electric field.

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

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).

    """
    *_, S, D, P = dielectric_coefs(w, wpe, wce)
    Ex = 1
    Ey = 1j * D / (n**2 - S)
    Ez = -n**2 * np.cos(theta) * np.sin(theta) / (P - n**2 * np.sin(theta)**2)
    E = np.array([Ex, Ey, Ez])
    B = n * np.cross([np.sin(theta), 0, np.cos(theta)], E)

    return E, B


def cold_plasma_eps_h(w, wpe, wce, theta):
    """Calculates the cold plasma Hermitian dielectric tensor.

    The toroidal magnetic field is assumed to lie along the z-axis.

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

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).

    """
    *_, S, D, P = refraction_coefs(w, wpe, wce, theta)

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


def group_velocity_magnitude(w, wpe, wce, n):
    """Calculates the group velocity magnitude in a cold plasma.

    This routine is given by equation (13) in [1]_, where mu is the
    refractive index.

    Parameters
    ----------
    w : scalar
        Wave frequency (rad/s).
    wpe : scalar
        Plasma frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    n : scalar
        Refractive index.

    Returns
    -------
    vg : scalar
        Group velocity magnitude (m/s).

    References
    ----------
    .. [1] R. F. Mullaly, J. Atmos. Terr. Phys. **9**, 322 (1956).

    """
    X = (wpe / w)**2
    Y = wce / w
    eta = 1 - X
    lambda_ = 1 + X / (n**2 - 1)
    p = X * lambda_ * (lambda_**2 - eta**2 * lambda_ - X * Y**2)
    q = eta * (lambda_ - 1)**2 * (lambda_**2 - 2 * eta * lambda_ + Y**2)
    mup = np.sqrt((lambda_ - 1) / (lambda_ - eta)) * (1 - p / q)
    return c / mup


def _refraction_derivs(w, wpe, wce, nu=1e-6):
    """Derivatives of refraction coefs with respect to `w`."""
    w = w + 1j * nu
    dSdw = 2 * w * wpe**2 / (w**2 - wce**2)**2
    dDdw = wce * wpe**2 * (wce**2 - 3*w**2) / (w**2 * (wce**2 - w**2)**2)
    dPdw = 2 * wpe**2 / w**3

    return dSdw, dDdw, dPdw


def _dwepshdw(w, wpe, wce, eps_h):
    """Calculates the derivative of `w` times `eps_h`."""
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


def spectral_energy_density(w, wpe, wce, eps_h, E, B):
    """Equation (5) from Harvey et al. (1993)."""
    dwepshdw = _dwepshdw(w, wpe, wce, eps_h)
    B2 = np.vdot(B, B)
    EwepsE = np.vdot(E, dwepshdw @ E)
    return (B2 + EwepsE) / (8 * np.pi)


# def _nr_coefs(w, wpe, wce, theta):
#     """Calculates `A` and `B` as well as their `theta` derivatives."""
#     A, B, _, R, L, S, _, P = refraction_coefs(w, wpe, wce, theta)
#     Ap = (S - P) * np.sin(2*theta)
#     Bp = (R * L - P * S) * np.sin(2*theta)
#     App = 2 * (S - P) * np.cos(2*theta)
#     Bpp = 2 * (R * L - P * S) * np.cos(2*theta)
#     return A, B, Ap, Bp, App, Bpp


# def _dndtheta(n, A, B, Ap, Bp):
#     """Calculates the derivative of `n` with respect to `theta`."""
#     n2 = n**2
#     np1 = (n / 2) * (Bp - Ap * n2) / (2 * A * n2 - B)
#     return np1


# def _d2ndtheta2(n, np1, A, B, Ap, Bp, App, Bpp):
#     """Calculates the second derivative of `n`."""
#     p = (np1*n**2 * (3*Ap*B-2*A*Bp) - 2*A*Ap*np1*n**4 + n**3 * (App*B - 3*Ap*Bp
#                                                                 + 2*A*Bpp)
#          + 2*n**5 * (Ap**2 - A*App) - B*Bp*np1 + n * (Bp**2 - B*Bpp))
#     q = 2 * (B - 2*A*n**2)**2
#     return p / q


# def _nr_xyz(n, np1, npp, theta):
#     """Calculates more important coefficients for `nr`."""
#     x = np1 / n
#     y = np.sqrt(1 + x**2)
#     xp = (npp - x) / n
#     yp = x * xp / y
#     z = (np.cos(theta) + x * np.sin(theta)) / y
#     zp = ((xp*np.sin(theta) + x*np.cos(theta) - np.sin(theta)) - yp / y * z
#           - (yp * (x*np.sin(theta) + np.cos(theta))) / y**2)
#     return y, zp


# def ray_refraction(n, w, wpe, wce, theta, nu=1e-6):
#     """
#     Calculates the ray refractive index in a magnetized cold plasma as
#     defined in [1].

#     Parameters
#     ----------
#     n : scalar
#         Magnetized cold plasma refractive index.
#     w : scalar
#         Wave frequency (rad/s).
#     wpe : scalar
#         Plasma frequency (rad/s).
#     wce : scalar
#         Cyclotron frequency (rad/s).
#     theta : scalar
#         Wave propagation angle.
#     eps_h : complex ndarray
#         Cold plasma permittivity tensor.

#     Returns
#     -------
#     scalar : nr
#         Ray refractive index.

#     References
#     ----------
#     .. [1] G. Bekefi, *Radiation Processes in Plasmas* (Wiley, New York,
#            1966).
#     .. [2] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
#            1992).
#     """
#     A, B, Ap, Bp, App, Bpp = _nr_coefs(w, wpe, wce, theta)
#     n2 = n**2
#     # F = 2 * A * n2 - B
#     # degenerate_mask = np.isclose(F, 0)
#     # if np.isclose(F.all(), 0):
#     #     return n  # If F is zero, the modes are degenerate
#     np1 = _dndtheta(n, A, B, Ap, Bp)
#     npp = _d2ndtheta2(n, np1, A, B, Ap, Bp, App, Bpp)
#     y, zp = _nr_xyz(n, np1, npp, theta)

#     nr2 = np.abs(n2 * np.sin(theta) * y / zp)
#     # nr2 = np.where(degenerate_mask, n2, nr2)  # Handle degenerate case
#     return np.sqrt(nr2)


def _Z(w, wpe, wce, theta, x_mode):
    n = np.real(refraction(w, wpe, wce, theta, x_mode=x_mode))
    dndtheta = derivative(
        lambda theta: np.real(refraction(w, wpe, wce, theta, x_mode=x_mode)),
        theta
    ).df
    Z = (np.cos(theta) + dndtheta / n * np.sin(theta)) / np.sqrt(1 + (dndtheta / n)**2)
    return Z


def ray_refraction(w, wpe, wce, theta, x_mode=False):
    n = np.real(refraction(w, wpe, wce, theta, x_mode=x_mode))
    dndtheta = derivative(
        lambda t: np.real(refraction(w, wpe, wce, t, x_mode=x_mode)),
        theta
    ).df
    dZdtheta = derivative(
        lambda t: _Z(w, wpe, wce, t, x_mode),
        theta
    ).df
    nr2 = n * np.abs(np.sin(theta) * np.sqrt(1 + (dndtheta / n)**2) / dZdtheta)
    return np.sqrt(nr2)
