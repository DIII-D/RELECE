"""
ECE signature modeling

Electron cyclotron emission (ECE) is the phenomenon of radiation from
gyrating electrons. In devices like DIII-D, ECE is exploited to measure
the temperature profile of the plasma.
"""
import numpy as np
from scipy import linalg
from scipy.constants import m_e, c
from scipy.special import jv, jvp
from scipy.interpolate import interpn


def refraction_coefs(w, wpe, wce, theta, eps_h=False, nu=1e-6):
    """
    Return coefficients needed to determine refraction index. Note that
    an infinitesimally small collision term is added to the wave
    frequency to account for Stix's causality prescription and handle
    singularities.
    """
    w = w + 1j * nu
    P = 1 - wpe**2 / w**2
    R = 1 - wpe**2 / (w * (w - wce))
    L = 1 - wpe**2 / (w * (w + wce))
    S = (R + L) / 2
    D = (R - L) / 2
    if eps_h:
        return R, L, S, D, P
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
    n : scalar
        Refractive index.

    References
    ----------
    .. [1] Stix, T. H., 1962, *The Theory of Plasma Waves*,
           McGraw-Hill, New York.
    """
    A, B, F = refraction_coefs(w, wpe, wce, theta)
    n2_plus = (B + F) / (2 * A)
    n2_minus = (B - F) / (2 * A)

    # Select the wpe cutoff for the O mode
    if w > wce:
        n2_O = n2_plus
        n2_X = n2_minus
    else:
        n2_O = n2_minus
        n2_X = n2_plus

    if x_mode:
        n2 = n2_X
    else:
        n2 = n2_O
    return np.sqrt(n2)


def wavevector(n, w, theta):
    """Calculates the wave vector from the refraction index.

    The wave is assumed to propagate in the x-z plane, with the
    magnetic field along the z-axis.
    """
    k = n * w / c
    kx = k * np.sin(theta)
    ky = 0
    kz = k * np.cos(theta)
    return np.array([kx, ky, kz])


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
    .. [1] Stix, T. H., 1962, *The Theory of Plasma Waves*,
           McGraw-Hill, New York.
    .. [2] Bornatici, M., et al, 1983, "Electron cyclotron emission and
           absorption in fusion plasmas," *Nucl. Fusion*, 23(9),
           1153-1257.
    """
    _, _, S, D, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)
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
    _, _, S, D, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)

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


def get_Sn_bar(u_par, u_perp, n_perp, n, Y):
    """This Sn_bar is a factor of 1/u_perp from that of Smirnov and
    Harvey.
    """
    b = np.abs(n_perp * u_perp / Y)
    Jn = jv(n, b)
    Jnp = jvp(n, b)

    Sn00 = u_perp * (n * Jn / b)**2
    Sn01 = -1j * u_perp * (n * Jn * Jnp / b)
    Sn02 = u_par * (n * Jn**2 / b)
    Sn10 = np.conj(Sn01)
    Sn11 = u_perp * Jnp**2
    Sn12 = 1j * u_par * Jn * Jnp
    Sn20 = np.conj(Sn02)
    Sn21 = np.conj(Sn12)
    Sn22 = u_par**2 / u_perp * Jn**2

    Sn = np.array([[Sn00, Sn01, Sn02],
                   [Sn10, Sn11, Sn12],
                   [Sn20, Sn21, Sn22]])
    Sn_bar = Sn / (m_e * c)
    return Sn_bar


def get_U_bar(v, theta, f, n, Y, n_par, u_perp, u_par):
    """Normalized U as defined by Smirnov and Harvey."""
    gamma = np.sqrt(1 + u_perp**2 + u_par**2)
    u = v / c
    dfdu, dfdtheta = np.gradient(f, u, theta)
    dfdu_perp = dfdu * np.cos(theta) - dfdtheta / u * np.sin(theta)
    dfdu_par = dfdu * np.sin(theta) + dfdtheta / u * np.cos(theta)

    xi_u = np.sqrt(u_perp**2 + u_par**2)
    xi_theta = np.arctan2(u_perp, u_par)
    dfdu_perp = interpn((u, theta), dfdu_perp, (xi_u, xi_theta), bounds_error=False,
                        fill_value=0)
    dfdu_par = interpn((u, theta), dfdu_par, (xi_u, xi_theta), bounds_error=False,
                       fill_value=0)

    U_bar = 1 / gamma * (n * Y * dfdu_perp + n_par * u_perp * dfdu_par)
    return U_bar


def integral_n(n, w, wce, n_perp, n_par, v, theta, f, tensor='eps_a'):
    """Calculates the integral to sum for both coefficients.

    This integral may be one of two, each specified within equations (6)
    and (7) in [1]. The implementation largely follows the notation in
    [2], with u being the normalized momentum (p/mc).

    Parameters
    ----------
    n : scalar
        Harmonic number.
    w : scalar
        Wave frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    n_perp : scalar
        Perpendicular refractive index.
    n_par : scalar
        Parallel refractive index.
    v : ndarray
        Momentum per mass coordinate (from CQL3D).
    theta : ndarray
        Angle from magnetic field coordinate (from CQL3D). This should
        extend from 0 to pi.
    f : ndarray
        Relativistic Maxwellian on the polar grid (normalized).
    tensor : str, optional
        Type of tensor to be calculated. Options are 'eps_a' or 'G'.

    Returns
    -------
    integral : complex ndarray
        The integral needed to calculate the specified tensor.

    Raises
    ------
    ValueError
        If the tensor type is not recognized.

    References
    ----------
    [1] R. W. Harvey *et al.*, Phys. Fluids B **5**, 446 (1993).

    [2] A. P. Smirnov and R. W. Harvey, The GENRAY Ray Tracing Code,
    CompX (2003), https://compxco.com/Genray_manual.pdf.
    """
    Y = wce / w
    A2 = 1 - n_par**2  # n_par < 1 so A^2 > 0
    R = np.sqrt(np.abs(n**2 * Y**2 - A2) / A2)
    A = np.sqrt(A2)
    a0 = n_par * n * Y / (1 - n_par**2)
    u_perp = R * np.sin(theta)
    u_par = a0 + (R / A) * np.cos(theta)

    Sn_bar = get_Sn_bar(u_par, u_perp, n_perp, n, Y)
    if tensor == 'eps_a':
        U_bar = get_U_bar(v, theta, f, n, Y, n_par, u_perp, u_par)
        integrand = U_bar * Sn_bar
    elif tensor == 'G':
        # TODO
        return
    else:
        raise ValueError("Tensor must be 'eps_a' or 'G'.")

    jacobian = 2 * np.pi * R * np.sin(theta) * R**2 / A
    integral = np.trapezoid(integrand * jacobian, theta, axis=0)

    return integral
