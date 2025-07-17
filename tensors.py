import numpy as np
from scipy import constants, special

### ANTI-HERMITIAN DIELECTRIC TENSOR ###
def _get_eps_h_coefs(w, wpe, wce, n_par, Te):
    mu = constants.m_e * constants.c**2 / Te
    a = np.pi * (wpe/w)**2 / (4 * n_par**2 * special.kve(2, mu))
    R = np.sqrt((wce/w)**2 - 1 + n_par**2)
    q = R / (1 - n_par**2)
    xi = mu * n_par * R / (1 - n_par**2)

    s = (1, -1)
    ps = np.zeros(2)
    gamma_s = np.zeros(2)
    F = np.zeros(2)
    for i in range(2):
        ps[i] = (n_par * (wce/w) + s[i] * R) / (1 - n_par**2)
        gamma_s[i] = np.sqrt(1 + ps[i]**2)
        F = np.exp(-mu * (gamma_s[i] - 1))

    return a, R, s, xi, F, ps, q


def _aij(w, wpe, wce, n_par, Te):
    a, R, s, xi, F, ps, q = _get_eps_h_coefs(w, wpe, wce, n_par, Te)
    sum_11 = sum_13 = sum_33 = 0
    for i in range(2):
        sum_11 += (1 + s[i] / xi) * F[i]
        sum_13 += (ps[i] + s[i] * (ps[i] + 2*s[i]*q) / xi
                   + 3*s[i]*q / xi**2) * F[i]
        sum_33 += (ps[i]**2 + s[i]*ps[i] * (ps[i] + 4*s[i]*q) / xi
                   + 6*s[i]*q * (ps[i] + s[i]*q) / xi**2
                   + 12*s[i]*q**2 / xi**3) * F[i]
    a11 = a * R * sum_11
    a13 = a * R * (w/wce) * sum_13
    a33 = a * R * (w/wce)**2 * sum_33


def eps_a():
    pass