"""Calculates the emission and absorption coefficients.

Notes
-----
This calculation uses the relativistic emission and anti-Hermitian
dielectric tensors. This uses the electron momentum distribution to
obtain the contributions from the most significant harmonics and sum
them together.
"""
import numpy as np
from scipy.constants import c, m_e
from scipy.special import jv, jvp
from scipy.interpolate import interpn


def get_Sn_bar(u_par, u_perp, n_perp, n, Y):
    """A factor of `1 / u_perp` from `Sn_bar` of Smirnov and Harvey.

    `n` is the harmonic number here.
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

    Sn_bar = np.array([[Sn00, Sn01, Sn02],
                       [Sn10, Sn11, Sn12],
                       [Sn20, Sn21, Sn22]])
    return Sn_bar


def distpoints(u_perp, u_par, u, theta, f):
    """Gets the interpolated points along the resonance ellipse."""
    xi_u = np.hypot(u_perp, u_par)
    xi_theta = np.arctan2(u_perp, u_par)
    f_interp = interpn((u, theta), f, (xi_u, xi_theta), bounds_error=False,
                       fill_value=0)
    return f_interp


def get_U_bar(v, theta, f, n, Y, n_par, u_perp, u_par):
    """Normalized U as defined by Smirnov and Harvey."""
    gamma = np.sqrt(1 + u_perp**2 + u_par**2)
    u = v / c
    dfdu, dfdtheta = np.gradient(f, u, theta)
    dfdu_perp = dfdu * np.cos(theta) - dfdtheta / u * np.sin(theta)
    dfdu_par = dfdu * np.sin(theta) + dfdtheta / u * np.cos(theta)

    dfdu_perp = distpoints(u_perp, u_par, u, theta, dfdu_perp)
    dfdu_par = distpoints(u_perp, u_par, u, theta, dfdu_par)

    U_bar = 1 / gamma * (n * Y * dfdu_perp + n_par * u_perp * dfdu_par)
    return U_bar


def integral_n(n, w, wpe, wce, n_perp, n_par, v, theta, f, tensor):
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
    tensor : str
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
    .. [1] R. W. Harvey *et al.*, Phys. Fluids B **5**, 446 (1993).
    .. [2] A. P. Smirnov and R. W. Harvey, The GENRAY Ray Tracing Code,
           CompX (2003), https://compxco.com/Genray_manual.pdf.
    """
    X = (wpe / w)**2
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
        integrand = -np.pi * X * (m_e * c)**3 * U_bar * Sn_bar
    elif tensor == 'G':
        f_interp = distpoints(u_perp, u_par, v, theta, f)
        gamma = np.sqrt(1 + u_perp**2 + u_par**2)
        integrand = (np.pi / (2 * np.pi)**5 * X / m_e * (m_e * c)**5 * f_interp * u_perp
                     * Sn_bar / gamma)
    else:
        raise ValueError("Tensor must be 'eps_a' or 'G'.")

    jacobian = 2 * np.pi * R * np.sin(theta) * R**2 / A
    integral = np.trapezoid(integrand * jacobian, theta, axis=0)

    return integral


def alpha_n(n, w, wpe, wce, n_perp, n_par, v, theta, f, E, S):
    """Calculates the `n`th contribution to the absorption."""
    eps_a = integral_n(n, w, wpe, wce, n_perp, n_par, v, theta, f, tensor='eps_a')
    alpha_n = w / 4 * np.pi * np.vdot(E, eps_a @ E) / S
    return alpha_n


def j_n(n, w, wpe, wce, n_perp, n_par, nr, v, theta, f, E, S):
    """Calculates the `n`th contribution to the emission."""
    G = integral_n(n, w, wpe, wce, n_perp, n_par, v, theta, f, tensor='G')
    j_n = np.pi * nr**2 * (w / c)**2 * np.vdot(E, G @ E) / S
    return j_n


def sum_harmonics(cn, initial, tolerance=1e-6, *args):
    """Sums over all harmonics within `tolerance`.

    The function starts at `n = initial` and sums outward from there.
    """
    c = 0
    cnext = cn(initial, *args)
    n = 1
    while np.abs(cnext) > tolerance:
        c += cnext
        cnext = cn(initial + n, *args)
        n *= -1
        if n > 0:
            n += 1
    return c
