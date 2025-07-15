"""
ECE signature modeling

Electron cyclotron emission (ECE) is the phenomenon of radiation from
gyrating electrons. In devices like DIII-D, ECE is exploited to measure
the temperature profile of the plasma.
"""
import numpy as np
from scipy import constants, special
import matplotlib.pyplot as plt


# Parameters
f = 90e9                    # EM wave frequency (Hz)
ne = 2e19                   # Electron number density (per m^3)
fce = 50e9                  # Cyclotron frequency (Hz)
Te = 3000                   # Electron temperature (eV)
n = 2                       # ECE harmonic
x_mode = True               # X or O mode
theta = np.pi / 2           # Viewing angle
res_ellipse_N = 1000        # Number of points on resonant ellipse
tolerance = 0.001

# Generated values
w = 2*np.pi * f
wpe = np.sqrt(ne * constants.e**2
              / (constants.m_e * constants.epsilon_0))  # Plasma freq (rad/s)
wce = 2*np.pi * fce
wr = wce / 2 * (1 + np.sqrt(1 + 4 * wpe**2/wce**2))  # RH cutoff (rad/s)

# Constants
c = constants.speed_of_light * 1e2   # cm/s
m_e = constants.electron_mass * 1e3  # g


def _calculate_refraction_coefs(w=w, wpe=wpe, wce=wce, theta=theta):
    """Return coefficients needed to determine refraction index."""
    P = 1 - wpe**2 / w**2
    R = 1 - wpe**2 / (w * (w+wce))
    L = 1 - wpe**2 / (w * (w-wce))
    S = (R + L) / 2
    D = (R - L) / 2
    A = S * np.sin(theta)**2 + P * np.cos(theta)**2
    B = R * L * np.sin(theta)**2 + P * S * (1 + np.cos(theta)**2)
    F = (R*L - P*S)**2 * np.sin(theta)**4 + 4 * P**2 * D**2 * np.cos(theta)**2

    return A, B, F


def _refraction_modes(A, B, F):
    """Return the refraction indices for both modes."""
    N_plus = (B + F) / (2 * A)
    N_minus = (B - F) / (2 * A)
    return N_plus, N_minus


def refraction(w=w, wpe=wpe, wce=wce, theta=theta, x_mode=x_mode):
    """Calculates the cold plasma refraction index squared.

    This index depends on the wave mode. The O mode is cut off (i.e.
    N^2 <= 0) at the plasma frequency,  whereas the X mode still
    propagates. This is the physical difference between the modes which
    we shall use to select the correct one.

    Reference: R. Parker, “Electromagnetic waves in plasmas,” in
    Introduction to Plasma Physics I, MIT OpenCourseWare, 2006,
    pp. 96-144

    :param w: wave frequency
    :type w: float
    :param wpe: plasma frequency
    :type wpe: float
    :param wce: cyclotron frequency
    :type wce: float
    :param theta: propagation angle
    :type theta: float
    :param x_mode: whether the x_mode is selected
    :type x_mode: bool
    :returns: cold plasma refraction index squared
    :rtype: float
    """
    A, B, F = _calculate_refraction_coefs(w, wpe, wce)
    A_cutoff, B_cutoff, F_cutoff = _calculate_refraction_coefs(wpe, wpe, wce)
    N2_plus, N2_minus = _refraction_modes(A, B, F)
    N2_plus_cutoff, N2_minus_cutoff = _refraction_modes(A_cutoff, B_cutoff,
                                                        F_cutoff)

    if (N2_plus_cutoff < tolerance):
        N2_O = N2_plus
        N2_X = N2_minus
    else:
        N2_O = N2_minus
        N2_X = N2_plus

    if (x_mode):
        return N2_X
    return N2_O


def eps_h():
    """Return cold plasma Hermitian dielectric tensor.
    """
    pass