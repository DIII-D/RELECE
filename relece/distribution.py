import numpy as np
from scipy.constants import c, m_e, e
from scipy.special import kv


class Distribution:
    """Base class for arbitrary electron momentum distributions.

    This class is designed to accept arbitrary output from some radial
    coordinate in CQL3D. Therefore, it is expected that `u` will be
    treated as momentum per unit mass, whereas we treat is simply as
    momentum normalized to ``m_e * c``. In other words, `u` will be
    unitless.

    Parameters
    ----------
    f : ndarray
        The distribution function values. Must be broadcastable to the
        coordinate grid defined by `u` and `theta`.
    u : ndarray
        The normalized momentum per mass, defined as ``p / m_e``.
    theta : ndarray
        The polar angle in radians from 0 to pi.

    """
    def __init__(self, f, u, theta):
        self._validate_input(f, u, theta)
        self.f = f
        self.u = u / c  # Convert momentum per mass to normalized momentum
        self.theta = theta


    @staticmethod
    def _validate_input(f, u, theta):
        if u.ndim != 1 or theta.ndim != 1:
            raise ValueError("`u` and `theta` must be 1D arrays.")
        if f.shape != (len(u), len(theta)):
            raise ValueError("`f` must have shape ``(len(u), len(theta)).``")


class MaxwellJuttnerDistribution(Distribution):
    """Generate Maxwell-Juttner distribution.

    This is a `Distribution` subclass that generates a 2D polar
    distribution for relativistic thermal plasma.

    """
    def __init__(self, e_temp, jx=300, iy=200, enorm=200):
        self.e_temp = e_temp
        f, u, theta = self._define_distribution(e_temp, jx, iy, enorm)
        super().__init__(f, u, theta)


    @staticmethod
    def _relativistic_maxwellian(u, e_temp):
        """Calculate the amplitude of the Maxwell-Juttner distribution
        at normalized momentum `u`.
        
        """
        # Note that theta is the traditional parameter used to define
        # the distribution.
        theta = e_temp * e / (m_e * c**2)
        gamma = np.hypot(1, u)
        normalization = 1 / (4 * np.pi * theta * kv(2, 1 / theta))
        return normalization * np.exp(-gamma / theta)


    def _define_distribution(self, e_temp, jx, iy, enorm):
        """Define the Maxwell-Juttner distribution."""
        gammanorm = 1 + enorm * e * 1e3 / (m_e * c**2)
        unorm = np.sqrt(gammanorm**2 - 1)
        u = np.linspace(0, unorm, jx)
        theta = np.linspace(0, np.pi, iy)
        f_1D = self._relativistic_maxwellian(u, e_temp)
        f = np.tile(f_1D, (iy, 1)).T
        return f, u, theta

