import numpy as np
from scipy import constants
from scipy import special
from scipy import interpolate
from scipy import integrate


class Distribution:
    """
    Base class for arbitrary electron momentum distributions.

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
    normalize : bool
        Whether to normalize the distribution function.
    cql3d : bool
        Whether to use the CQL3D coordinate system. Essentially, this
        means `u` is assumed to be the normalized momentum as opposed
        to the momentum per mass.

    Attributes
    ----------
    f : ndarray
        The distribution function.
    u : ndarray
        The normalized momentum per mass, defined as ``p / m_e``.
    theta : ndarray
        The polar angle in radians from 0 to pi.

    Methods
    -------
    ev : float
        Gets the interpolated points along the resonance ellipse.

    """

    def __init__(self, f, u, theta, normalize=True, cql3d=True):
        self._validate_input(f, u, theta)
        if cql3d:
            u /= constants.c  # Convert momentum per mass to normalized momentum
        if normalize:
            self.f = self._normalize(f, u, theta)
        else:
            self.f = f
        self.u = u
        self.theta = theta

    @staticmethod
    def _validate_input(f, u, theta):
        if u.ndim != 1 or theta.ndim != 1:
            raise ValueError("`u` and `theta` must be 1D arrays.")
        if f.shape != (len(u), len(theta)):
            raise ValueError("`f` must have shape ``(len(u), len(theta)).``")

    @staticmethod
    def _normalize(f, u, theta):
        """`f` is integrated over 3D normalized momentum space."""
        jacobian = 2 * np.pi * u**2 * np.sin(theta)
        integral = integrate.simpson(integrate.simpson(f * jacobian, u), theta)
        return f / integral

    def ev(self, u_perp, u_par):
        """Gets the interpolated points along the resonance ellipse."""
        xi_u = np.hypot(u_perp, u_par)
        xi_theta = np.arctan2(u_perp, u_par)
        f_interp = interpolate.interpn(
            (self.u, self.theta),
            self.f,
            (xi_u, xi_theta),
            bounds_error=False,
            fill_value=0
        )
        return f_interp


class MaxwellJuttnerDistribution(Distribution):
    """
    Generate Maxwell-Juttner distribution.

    This is a `Distribution` subclass that generates a 2D polar
    distribution for relativistic thermal plasma.

    Parameters
    ----------
    temperature : scalar
        The temperature of the plasma (eV).
    jx : int
        The number of grid points in the momentum direction.
    iy : int
        The number of grid points in the polar angle direction.
    enorm : float
        The maximum energy on the momentum grid (keV).

    """

    def __init__(self, temperature, jx=300, iy=200, enorm=200):
        f, u, theta = self._define_distribution(temperature, jx, iy, enorm)
        super().__init__(f, u, theta, normalize=False)

    @staticmethod
    def _relativistic_maxwellian(u, temperature):
        """
        Calculate the amplitude of the Maxwell-Jüttner distribution
        at normalized momentum `u`.
        """
        normalized_t = temperature * constants.e / (constants.m_e * constants.c**2)
        gamma = np.hypot(1, u)
        normalization = 1 / (4 * np.pi * normalized_t * special.kv(2, 1 / normalized_t))
        return normalization * np.exp(-gamma / normalized_t)

    def _define_distribution(self, temperature, jx, iy, enorm):
        """Define the Maxwell-Jüttner distribution."""
        gammanorm = 1 + enorm * constants.e * 1e3 / (constants.m_e * constants.c**2)
        unorm = np.sqrt(gammanorm**2 - 1)
        u = np.linspace(0, unorm, jx)
        theta = np.linspace(0, np.pi, iy)
        f_1D = self._relativistic_maxwellian(u, temperature)
        f = np.tile(f_1D, (iy, 1)).T
        return f, u, theta
