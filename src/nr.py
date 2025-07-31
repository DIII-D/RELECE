"""
Calculates the ray refractive index in a cold plasma using equation
1.121 from Ref. [1].

References
----------
.. [1] Bekefi, G., 1966, *Radiation Processes in Plasmas*,
       Wiley, New York.
"""
import numpy as np

import utils


def _dn_dtheta_n(S, D, P, theta, n2):
    """
    Calculates the derivative of the refractive index with respect to
    the propagation angle divided by the refractive index itself. Call
    this quantity x.
    """
    numerator = (D**2 + P*S - S**2 + n2 * (S - P)) * np.sin(2 * theta)
    denominator = (S**2 + 3*P*S - D**2 + (D**2 + P*S - S**2) * np.cos(2 * theta)
                   - 2 * (P + S + (P - S) * np.cos(2 * theta)) * n2)
    return numerator / denominator


def _dx_dtheta(S, D, P, theta, n2, x):
    """
    Derivative of (1/n)(dn/dtheta). The formula is calculated
    analytically.
    """
    numerator = (n2 * (x * (2 * np.sin(2*theta) * (D**2*(3*P + S) - S * (P - S)**2)
                            + np.sin(4*theta) * (P - S) * (D**2 + P*S - S**2))
                       - 6 * (P - S) * (D**2 + P*S - S**2)
                       - 2 * np.cos(2*theta) * (D**2 * (P + 3*S) + S * (P - S) * (5*P
                                                                                  + 3*S)
                                                )
                       + 4 * n2 * (P - S) * (np.cos(2*theta) * (P + S) + P - S))
                 - 2 * (D**2 + P*S - S**2) * (np.cos(2*theta) * (D**2 - 3*P*S - S**2)
                                              - D**2 - P*S + S**2))
    denominator = (np.cos(2*theta) * (D**2 + P*S - S**2) - D**2
                   - 2 * n2 * (np.cos(2*theta) * (P - S) + P + S) + 3*P*S + S**2)**2
    return numerator / denominator


def _nr_root(x, n2):
    """
    Calculates the radical quantity that appears frequently in the
    expression for nr. Call this quantity y.
    """
    return np.sqrt(1 + x**2 / n2)


def _dy_dtheta(x, dxdth, y):
    return x * dxdth / y


def ray_refraction(n_par, n_perp, w, wpe, wce, theta, eps_h):
    """
    Calculates the refraction of a ray in a cold plasma.

    Parameters
    ----------
    n_par : scalar
        Parallel refractive index.
    n_perp : scalar
        Perpendicular refractive index.
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
    complex ndarray
        Refracted wave vector.
    """
    S, D, P = utils.refraction_coefs(w, wpe, wce, theta, eps_h=True)
    n2 = n_par**2 + n_perp**2
    x = _dn_dtheta_n(S, D, P, theta, n2)
    # TODO: Finish implementation