import numpy as np
from scipy.constants import c, m_e
from scipy.special import kve


# def relativistic_maxwellian(p_par, p_perp, w, wce, n, n_par, Te):
#     """
#     Compute the normalized relativistic Maxwellian distribution
#     function and normalized Un, as described in equations 5.43 and 5.45
#     of Ref. [1].

#     This is a special case in the calculation of the anti-Hermitian
#     dielectric tensor, as it is the only distribution of interest that
#     can be described analytically.

#     Parameters
#     ----------
#     p_par : scalar
#         Parallel momentum (kg m/s).
#     p_perp : scalar
#         Perpendicular momentum (kg m/s).
#     w : scalar
#         Wave frequency (rad/s).
#     wce : scalar
#         Electron cyclotron frequency (rad/s).
#     n : scalar
#         Harmonic number.
#     n_par : scalar
#         Parallel refractive index.
#     Te : scalar
#         Electron temperature (eV).

#     Returns
#     -------
#     scalar
#         The value of the relativistic Maxwellian distribution function.

#     References
#     ----------
#     .. [1] Smirnov, A. P., and R. W. Harvey, 2003, "The GENRAY ray
#            tracing code."
#     """
#     p_par_bar = p_par / (m_e * c)
#     p_perp_bar = p_perp / (m_e * c)

#     theta = m_e * c**2 / Te
#     gamma = np.sqrt(1 + (p_par_bar**2 + p_perp_bar**2))
#     fm_bar = theta / (4 * np.pi * kve(2, theta)) * np.exp(theta * (1 - gamma))

#     Y = wce / w
#     Un_bar = -p_perp_bar * theta * fm_bar / \
#         gamma**2 * (n * Y + n_par * p_par_bar)

#     return fm_bar, Un_bar


def relativistic_maxwellian(jx=300, )