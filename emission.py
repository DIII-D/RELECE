"""
ECE signature modeling

Electron cyclotron emission (ECE) is the phenomenon of radiation from
gyrating electrons. In devices like DIII-D, ECE is exploited to measure
the temperature profile of the plasma.
"""
import numpy as np
from scipy import constants, special
import matplotlib.pyplot as plt


# Experimental parameters
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


def refraction(w=w, wpe=wpe, wce=wce, x_mode=x_mode):
    """Return the cold plasma refraction index squared.

    This index depends on the wave mode. The O mode has the same index
    as in the isotropic case, whereas the X mode is subject to the
    effects of the ambient magnetic field.

    Reference: R. Parker, “Electromagnetic waves in plasmas,” in
    Introduction to Plasma Physics I, MIT OpenCourseWare, 2006,
    pp. 96–144

    Keyword arguments:
    w -- the wave frequency (default: global)
    wpe -- the plasma frequency (default: global)
    wce -- the cyclotron frequency (default: global)
    x_mode -- whether the x_mode is selected (default: global)
    """

    if (x_mode):
        R = 1 - wpe**2 / (w * (w+wce))
        L = 1 - wpe**2 / (w * (w-wce))
        S = 0.5 * (R + L)

        N2 = R * L / S
    else:
        N2 = 1 - wpe**2 / w**2

    return N2
