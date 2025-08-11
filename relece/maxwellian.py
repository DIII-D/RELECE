import numpy as np
from scipy.constants import c, m_e
from scipy.special import kv


def _relativistic_maxwellian(v, Te):
    """Calculate the amplitude of the relativistic Maxwellian
    distribution at momentum per mass v.
    """
    Tnorm = Te / (m_e * c**2)
    gamma = np.sqrt(1 + (v / c)**2)
    return np.exp(-gamma / Tnorm) / (4 * np.pi * Tnorm * kv(2, 1/Tnorm))


def relativistic_maxwellian_distribution(Te, jx=300, iy=200, enorm=200):
    """Generates the relativistic Maxwellian distribution.

    This function imitates the output of CQL3D, i.e., its output is
    discretized over a polar grid in normalized momentum space
    (momentum / rest mass).

    Parameters
    ----------
    Te : scalar
        Electron temperature (eV).
    jx : scalar
        Momentum grid resolution.
    iy : scalar
        Angular grid resolution.
    enorm : scalar
        Maximum particle energy (keV).

    Returns
    -------
    v : ndarray
        Momentum per mass coordinate.
    theta : ndarray
        Angle from magnetic field coordinate.
    f : ndarray
        Relativistic maxwellian on the polar grid.

    """
    gammanorm = enorm * 1e3 / (m_e * c**2)
    vnorm = c * np.sqrt(gammanorm**2 - 1)
    v = np.linspace(0, vnorm, num=jx)
    theta = np.linspace(0, 2 * np.pi, num=iy)
    vgrid, tgrid = np.meshgrid(v, theta)
    f = _relativistic_maxwellian(vgrid, Te)
    return v, tgrid, f

