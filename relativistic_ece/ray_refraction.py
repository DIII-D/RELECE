"""
Calculates the ray refractive index in a cold plasma using equation
1.121 from Ref. [1].

References
----------
.. [1] Bekefi, G., 1966, *Radiation Processes in Plasmas*,
       Wiley, New York.
"""
import numpy as np

from relativistic_ece.utils import refraction_coefs


def _nr_coefs(w, wpe, wce, theta):
    """
    Calculates the coefficients A, B, and F in the Appleton-Hartree
    equation for the refractive index.
    """
    R, L, S, _, P = refraction_coefs(w, wpe, wce, theta, eps_h=True)
    A, B, _ = refraction_coefs(w, wpe, wce, theta)
    Ap = (S - P) * np.sin(2*theta)
    Bp = (R*L - P*S) * np.sin(2*theta)
    App = 2 * (S - P) * np.cos(2*theta)
    Bpp = 2 * (R*L - P*S) * np.cos(2*theta)
    return A, B, Ap, Bp, App, Bpp


def _dndtheta(n, A, B, Ap, Bp):
    """
    Calculates the derivative of the refractive index with respect to
    the propagation angle.
    """
    n2 = n**2
    np1 = (n / 2) * (Bp - Ap * n2) / (2 * A * n2 - B)
    return np1


def _d2ndtheta2(n, np1, A, B, Ap, Bp, App, Bpp):
    """
    Calculates the second derivative of the refractive index with
    respect to the propagation angle.
    """
    q = (np1*n**2 * (3*Ap*B-2*A*Bp) - 2*A*Ap*np1*n**4 + n**3 * (App*B - 3*Ap*Bp
                                                                + 2*A*Bpp)
         + 2*n**5 * (Ap**2 - A*App) - B*Bp*np1 + n * (Bp**2 - B*Bpp))
    d = 2 * (B - 2*A*n**2)**2
    return q / d


def _nr_xyz(n, np1, npp, theta):
    """
    Calculates more coefficients needed to calculate the ray refractive
    index.
    """
    x = np1 / n
    y = np.sqrt(1 + x**2)
    xp = (npp - x) / n
    yp = x * xp / y
    z = (np.cos(theta) + x * np.sin(theta)) / y
    zp = ((xp*np.sin(theta) + x*np.cos(theta) - np.sin(theta)) - yp / y * z
          - (yp * (x*np.sin(theta) + np.cos(theta))) / y**2)
    return y, zp


def ray_refraction(n, w, wpe, wce, theta, nu=1e-6):
    """
    Calculates the ray refractive index in a magnetized cold plasma as
    defined in Ref. [1].

    Parameters
    ----------
    n : scalar
        Magnetized cold plasma refractive index.
    w : scalar
        Wave frequency (rad/s).
    wpe : scalar
        Plasma frequency (rad/s).
    wce : scalar
        Cyclotron frequency (rad/s).
    theta : scalar
        Wave propagation angle.
    eps_h : complex ndarray
        Cold plasma permittivity tensor.

    Returns
    -------
    scalar : nr
        Ray refractive index.

    References
    ----------
    .. [1] Bekefi, G., 1966, *Radiation Processes in Plasmas*, Wiley,
           New York.
    .. [2] Stix, T. H., 1992, *Waves in Plasmas*, AIP Press, New York.
    """
    A, B, Ap, Bp, App, Bpp = _nr_coefs(w, wpe, wce, theta)
    n2 = n**2
    F = 2 * A * n2 - B
    degenerate_mask = np.isclose(F, 0)
    if np.isclose(F.all(), 0):
        return n  # If F is zero, the modes are degenerate
    np1 = _dndtheta(n, A, B, Ap, Bp)
    npp = _d2ndtheta2(n, np1, A, B, Ap, Bp, App, Bpp)
    y, zp = _nr_xyz(n, np1, npp, theta)

    nr2 = np.abs(n2 * np.sin(theta) * y / zp)
    nr2 = np.where(degenerate_mask, n2, nr2)  # Handle degenerate case
    return np.sqrt(nr2)
