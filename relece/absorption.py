"""
This code calculates the absorption coefficients using the fully
relativistic anti-Hermitian dielectric tensor.
"""
import numpy as np
from scipy.constants import c, m_e
from scipy.special import jv, jvp

"""
The following commented code derives the anti-Hermitian dielectric
tensor from the relativistic Maxwellian distribution function. It has
been discarded in favor of a more general approach.
"""
# def _get_eps_h_coefs(w, wpe, wce, n_par, Te):
#     mu = m_e * c**2 / Te
#     a = np.pi * (wpe/w)**2 / (4 * n_par**2 * special.kve(2, mu))
#     R = np.sqrt((wce/w)**2 - 1 + n_par**2)
#     q = R / (1 - n_par**2)
#     xi = mu * n_par * R / (1 - n_par**2)

#     s = (1, -1)
#     ps = np.zeros(2)
#     gamma_s = np.zeros(2)
#     F = np.zeros(2)
#     for i in range(2):
#         ps[i] = (n_par * (wce/w) + s[i] * R) / (1 - n_par**2)
#         gamma_s[i] = np.sqrt(1 + ps[i]**2)
#         F = np.exp(-mu * (gamma_s[i] - 1))

#     return mu, a, R, s, xi, F, ps, q


# def _aij(w, wce, a, R, s, xi, F, ps, q):
#     sum_11 = sum_13 = sum_33 = 0
#     for i in range(2):
#         sum_11 += (1 + s[i] / xi) * F[i]
#         sum_13 += (ps[i] + s[i] * (ps[i] + 2*s[i]*q) / xi
#                    + 3*s[i]*q / xi**2) * F[i]
#         sum_33 += (ps[i]**2 + s[i]*ps[i]*(ps[i] + 4*s[i]*q) / xi
#                    + 6*s[i]*q*(ps[i] + s[i]*q) / xi**2
#                    + 12*s[i]*q**2 / xi**3) * F[i]
#     a11 = a * R * sum_11
#     a13 = a * R * (w/wce) * sum_13
#     a33 = a * R * (w/wce)**2 * sum_33
#     return a11, a13, a33


# def _bij(w, wce, n_par, mu, a, R, s, xi, F, ps, q):
#     sum_11 = sum_13 = sum_33 = 0
#     for i in range(2):
#         sum_11 += s[i] * (1 + 3*s[i] / xi + 3 / xi**2) * F[i]
#         sum_13 += s[i] * (ps[i] + 3*s[i]*(ps[i] + s[i]*q) / xi
#                           + 3*(ps[i] + 4*s[i]*q) / xi**2
#                           + 15*q / xi**3) * F[i]
#         sum_33 += s[i] * (ps[i]**2 + 3*s[i]*ps[i]*(ps[i] + 2*s[i]*q) / xi
#                           + 3*(ps[i]**2 + 8*s[i]*q*ps[i] + 4*q**2) / xi**2
#                           + 30*q*(ps[i] + 2*s[i]*q) / xi**3
#                           + 90*q**2 / xi**4) * F[i]
#     b11 = a * R * (R / (n_par * mu)) * (w/wce)**2 * sum_11
#     b13 = a * R * (R / (n_par * mu)) * (w/wce)**3 * sum_13
#     b33 = a * R * (R / (n_par * mu)) * (w/wce)**4 * sum_33
#     return b11, b13, b33


# def _cij(w, wce, n_par, mu, a, R, s, xi, F, ps, q):
#     return -4 * _bij(w, 2*wce, n_par, mu, a, R, s, xi, F, ps, q)


# def _dij(w, wce, n_par, mu, a, R, s, xi, F, ps, q):
#     sum_11 = sum_13 = sum_33 = 0
#     for i in range(2):
#         sum_11 += (1 + 6*s[i] / xi + 15 / xi**2 + 15*s[i] / xi**3) * F[s]
#         sum_13 += (ps[i] * (1 + 6*s[i] / xi + 15 / xi**2 + 15*s / xi**3)
#                    + 2 * (q / xi) * (2 + 15*s[i] / xi + 45 / xi**2
#                                      + 105*s[i] / (2*xi**3))) * F[i]
#         sum_33 += (ps[i]**2 * (1 + 6*s[i] / xi + 15 / xi**2 + 15*s[i] / xi**3)
#                    + 4 * (q*ps[i] / xi) * (2 + 15*s / xi + 45 / xi**2
#                                            + 105*s[i] / (2*xi**3))
#                    + 10 * (q / xi)**2 * (2 + 18*s[i] / xi + 63 / xi**2
#                                          + 84*s / xi**3)) * F[i]
#     d11 = a * R * (R / (n_par * mu))**2 * (5/8) * (w/wce)**4 * sum_11
#     d13 = a * R * (R / (n_par * mu))**2 * (5/8) * (w/wce)**5 * sum_13
#     d33 = a * R * (R / (n_par * mu))**2 * (5/8) * (w/wce)**6 * sum_33
#     return d11, d13, d33


# def _fij(w, wce, n_par, mu, a, R, s, xi, F, ps, q):
#     return -(128/5) * _dij(w, 2*wce, n_par, mu, a, R, s, xi, F, ps, q)


# def _gij(w, wce, n_par, mu, a, R, s, xi, F, ps, q):
#     return (243/5) * _dij(w, 3*wce, n_par, mu, a, R, s, xi, F, ps, q)


# def eps_a(w, wpe, wce, n_par, n_perp, Te):
#     """
#     Computes the fully relativistic anti-Hermitian dielectric tensor,
#     as given by Ref. [1].

#     Parameters
#     ----------
#     w : scalar
#         Incident wave frequency (rad/s).
#     wpe : scalar
#         Plasma frequency (rad/s).
#     wce : scalar
#         Cyclotron frequency (rad/s).
#     n_par : scalar
#         Refractive index along the toroidal magnetic field.
#     n_perp : scalar
#         Refractive index perpendicular to the toroidal magnetic field.
#     Te : scalar
#         Background electron temperature (eV)

#     Returns
#     -------
#     complex ndarray
#         The full 3-by-3 dielectric tensor.

#     References
#     ----------
#     .. [1] E. Mazzucato, I. Fidone, and G. Granata, Phys. Fluids
#            **30**, 3745 (1987).
#     """
#     mu, a, R, s, xi, F, ps, q = _get_eps_h_coefs(w, wpe, wce, n_par, Te)
#     if np.real(R) <= 0:
#         return np.zeros((3, 3))  # Accounts for the 'S' in [1]
#     a11, a13, a33 = _aij(w, wce, a, R, s, xi, F, ps, q)
#     b11, b13, b33 = _bij(w, wce, n_par, mu, a, R, s, xi, F, ps, q)
#     c11, c13, c33 = _cij(w, wce, n_par, mu, a, R, s, xi, F, ps, q)
#     d11, d13, d33 = _dij(w, wce, n_par, mu, a, R, s, xi, F, ps, q)
#     f11, f13, f33 = _fij(w, wce, n_par, mu, a, R, s, xi, F, ps, q)
#     g11, g13, g33 = _gij(w, wce, n_par, mu, a, R, s, xi, F, ps, q)

#     e11 = a11 + n_perp**2 * (b11 + c11) + n_perp**4 * (d11 + f11 + g11)
#     e12 = -1j * (a11 + n_perp**2 * (2*b11 + c11)
#                  + n_perp**4 * (3*d11 + 3*f11/2 + g11))
#     e21 = -np.conj(e12)
#     e22 = (a11 + n_perp**2 * (3*b11 + c11)
#            + n_perp**4 * (37*d11/5 + 2*f11 + g11))
#     e13 = n_perp * (a13 + n_perp**2 * (b13 + c13)
#                     + n_perp**4 * (d13 + f13 + g13))
#     e31 = -np.conj(e13)
#     e23 = 1j * n_perp * (a13 + n_perp**2 * (2*b13 + c13)
#                          + n_perp**4 * (3*d13 + 3*f13/2 + g13))
#     e32 = -np.conj(e23)
#     e33 = n_perp**2 * (a33 + n_perp**2 * (b33 + c33)
#                        + n_perp**4 * (d33 + f33 + g33))

#     result = np.array([[e11, e12, e13],
#                        [e21, e22, e33],
#                        [e31, e32, e33]])
#     return result


def hermitian_Sn_bar(p_par, p_perp, k_perp, n, wce):
    b = np.abs(k_perp * p_perp / (m_e * wce))
    Jn = jv(n, b)
    Jnp = jvp(n, b)

    Sn00 = p_perp * (n * Jn / b)**2
    Sn01 = -1j * p_perp * (n * Jn * Jnp / b)
    Sn02 = p_par * (n * Jn**2 / b)
    Sn10 = np.conj(Sn01)
    Sn11 = p_perp * Jnp**2
    Sn12 = 1j * p_par * Jn * Jnp
    Sn20 = np.conj(Sn02)
    Sn21 = np.conj(Sn12)
    Sn22 = p_par**2 / p_perp * Jn**2

    Sn = np.array([[Sn00, Sn01, Sn02],
                   [Sn10, Sn11, Sn12],
                   [Sn20, Sn21, Sn22]])
    Sn_bar = p_perp * Sn / (m_e * c)**2
    return Sn_bar
