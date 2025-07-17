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

# Generated values
w = 2*np.pi * f
wpe = np.sqrt(ne * constants.e**2
              / (constants.m_e * constants.epsilon_0))  # Plasma freq (rad/s)
wce = 2*np.pi * fce
wr = wce / 2 * (1 + np.sqrt(1 + 4 * wpe**2/wce**2))  # RH cutoff (rad/s)

# Constants
c = constants.speed_of_light * 1e2   # cm/s
m_e = constants.electron_mass * 1e3  # g


def _calculate_refraction_coefs(w, wpe, wce, theta, tensor=False):
    """Return coefficients needed to determine refraction index."""
    P = 1 - wpe**2 / w**2
    R = 1 - wpe**2 / (w * (w+wce))
    L = 1 - wpe**2 / (w * (w-wce))
    S = (R + L) / 2
    D = (R - L) / 2
    A = S * np.sin(theta)**2 + P * np.cos(theta)**2
    B = R * L * np.sin(theta)**2 + P * S * (1 + np.cos(theta)**2)
    F2 = (R*L - P*S)**2 * np.sin(theta)**4 + 4 * P**2 * D**2 * np.cos(theta)**2
    F = np.sqrt(F2)
    if (tensor):
        return S, D, P
    return A, B, F


def refraction(w, wpe, wce, theta, x_mode=False):
    """Calculates the cold plasma refraction index squared.

    This index depends on the wave mode and is calculated using the
    Appleton-Hartree equation. This equation yields a quadratic for N2,
    which provides two roots. The perpendicular case (theta = pi/2)
    tells us which root should be assigned to the O vs X mode. If the
    wave frequency is greater than the cyclotron frequency, the "plus"
    root is the O mode and the "minus" root is the X mode, and vice-
    versa for the opposite case.

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
    :param x_mode: whether the X mode is selected
    :type x_mode: bool
    :returns: cold plasma refraction index squared
    :rtype: float
    """
    A, B, F = _calculate_refraction_coefs(w, wpe, wce, theta)
    N2_plus = (B + F) / (2 * A)
    N2_minus = (B - F) / (2 * A)

    # Select the wpe cutoff for the O mode
    if (w >= wce):
        N2_O = N2_plus
        N2_X = N2_minus
    else:
        N2_O = N2_minus
        N2_X = N2_plus

    if (x_mode):
        return N2_X
    return N2_O


def dij(w, wpe, wce, theta, x_mode=False):
    """Calculates equation 5.75 from the reference.

    The determinant of this tensor yields a quadratic expression for
    the cold plasma dispersion relation. It can also be used to
    determine the Stix frame as well as the hermitian dielectric
    tensor.

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
    :param x_mode: whether the X mode is selected
    :type x_mode: bool
    :returns: cold plasma dispersion tensor
    :rtype: np.ndarray
    """
    S, D, P = _calculate_refraction_coefs(w, wpe, wce, tensor=True)
    N2 = refraction(w, wpe, wce, theta, x_mode)

    D00 = -N2 * np.cos(theta)**2 + S
    D01 = -1j * D
    D02 = N2 * np.sin(theta) * np.cos(theta)
    D10 = 1j * D
    D11 = -N2 + S
    D12 = 0
    D20 = N2 * np.sin(theta) * np.cos(theta)
    D21 = 0
    D22 = -N2 * np.sin(theta)**2 + P

    Dij = np.array([[D00, D01, D02],
                    [D10, D11, D12],
                    [D20, D21, D22]])
    return Dij


def stix_frame(w, wpe, wce, theta, x_mode):
    """Calculates the normalized wave E field in the Stix frame."""
    pass