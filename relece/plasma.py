"""
plasma - defines the various plasma classes.

At some point, this should define relativistic thermal plasmas as well
as cold plasmas.

"""
from abc import ABC, abstractmethod
from typing import override

import numpy as np
from scipy import constants
from scipy.differentiate import derivative
from scipy.linalg import null_space
from scipy.special import jv, jvp

from . import distribution as dist

# Convert SI units to Gaussian for plasma calculations
c = constants.speed_of_light * 1e2  # m/s to cm/s
e = constants.elementary_charge * c / 10  # C to statC
m_e = constants.electron_mass * 1e3  # kg to g


class Plasma(ABC):
    """
    Base class for plasmas.

    This class is not meant to be instantiated directly, but rather
    serves as a base for specific plasma types.

    """

    def __init__(
        self,
        density,
        magnetic_field,
        temperature=0,
        distribution=None,
        collision_rate=1e-3
    ):
        self.density = density * 1e6  # Convert to 1/cm^3
        self.magnetic_field = magnetic_field * 1e4  # Convert to G
        self.temperature = temperature
        if distribution is None:
            distribution = dist.MaxwellJuttnerDistribution(temperature)
        self.collision_rate = collision_rate
        self.wpe = self._get_plasma_frequency()
        self.wce = self._get_cyclotron_frequency()

    def _get_plasma_frequency(self):
        gamma_thermal = self._get_gamma_thermal()
        wpe = np.sqrt(4 * np.pi * self.density * e**2 / gamma_thermal * m_e)
        return wpe

    def _get_cyclotron_frequency(self):
        gamma_thermal = self._get_gamma_thermal()
        wce = e * self.magnetic_field / (gamma_thermal * m_e * c)
        return wce

    def _get_gamma_thermal(self):
        """Calculate the thermal Lorentz factor."""
        v_thermal = np.sqrt(self.temperature * e * 1e7 / m_e)
        gamma_thermal = 1 / np.sqrt(1 - (v_thermal / c)**2)
        return gamma_thermal

    @abstractmethod
    def refractive_index(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculate the refractive index for a given wave.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).
        mode : str, optional
            Mode of propagation ('O' or 'X'). Default is 'O'.
        angle : scalar, optional
            Angle of propagation with respect to magnetic field
            (radians). Default is pi/2.

        Returns
        -------
        n : scalar
            Refractive index.

        """
        pass

    def ray_refractive_index(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculates the ray refractive index for a given wave.

        This quantity is defined from the real part of the refractive
        index and is calculated numerically.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).
        mode : str, optional
            Mode of propagation ('O' or 'X'). Default is 'O'.
        angle : scalar, optional
            Angle of propagation with respect to magnetic field
            (rad). Default is pi/2.

        Returns
        -------
        nr : scalar
            Ray refractive index.

        Raises
        ------
        RuntimeWarning
            If the derivative calculation fails for one or more angles.

        References
        ----------
        .. [1] G. Bekefi, *Radiation Processes in Plasmas* (Wiley, New York,
           1966).

        """
        n = np.real(self.refractive_index(frequency, mode, angle))
        dn = derivative(
            lambda theta: np.real(
                self.refractive_index(frequency, mode, theta)),
            angle
        )
        denominator = derivative(
            lambda theta: self._ray_refraction_denom_helper(
                frequency, mode, theta),
            angle
        )
        if not np.any(dn.success) or not np.any(denominator.success):
            self._raise_differentiation_warning()
        numerator = np.sqrt(1 + (dn.df / n)**2)

        nr2 = np.abs(n**2 * np.sin(angle) * numerator / denominator.df)
        return np.sqrt(nr2)

    def _ray_refraction_denom_helper(self, frequency, mode, angle):
        """Important quantity for ray refractive index calculation"""
        n = np.real(self.refractive_index(
            frequency, mode, angle))
        dn = derivative(
            lambda theta: np.real(
                self.refractive_index(frequency, mode, theta)),
            angle
        )
        if np.any(not dn.success):
            self._raise_differentiation_warning()

        denom_helper = ((np.cos(angle) + (dn.df / n) * np.sin(angle))
                        / np.sqrt(1 + (dn.df / n)**2))
        return denom_helper

    @staticmethod
    def _raise_differentiation_warning():
        raise RuntimeWarning(
            "Derivative calculation failed for one or more angles."
        )

    def e_field_polarization(self, frequency, mode='O', angle=np.pi/2):
        """Solve the wave equation for the electric field polarization"""
        n = self.refractive_index(frequency, mode, angle)
        epsilon = self.get_dielectric_tensor(frequency)
        kk_minus_eye = np.array([
            [-np.cos(angle)**2,               0,
             np.cos(angle) * np.sin(angle)],
            [0,                               -1,  0],
            [np.cos(angle) * np.sin(angle),   0,   -np.sin(angle)**2]
        ])  # Outer product of wave direction - delta_ij
        dispersion_tensor = epsilon + n**2 * kk_minus_eye

        e_polarization = null_space(dispersion_tensor)
        return e_polarization

    @abstractmethod
    def get_dielectric_tensor(self, frequency):
        """
        Calculate the dielectric tensor for the plasma at a given
        frequency.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).

        Returns
        -------
        epsilon : ndarray
            Dielectric tensor (3x3 matrix).

        """
        pass

    @abstractmethod
    def get_spectral_energy_flux_density(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculate the spectral energy flux density for a given wave.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).
        distribution : Distribution
            Particle distribution function.
        mode : str, optional
            Mode of propagation ('O' or 'X'). Default is 'O'.
        angle : scalar, optional
            Angle of propagation with respect to magnetic field
            (radians). Default is pi/2.

        Returns
        -------
        S : scalar
            Spectral energy flux density (J/m^2/s/Hz).

        """
        pass

    @staticmethod
    def _get_resonance_ellipse(n_par, y, phi, harmonic):
        # Define convenient helper variables.
        a2 = 1 - n_par**2
        a = np.sqrt(a2)
        r = np.sqrt(np.abs(harmonic**2 * y**2 - a2) / a2)
        a0 = n_par * harmonic * y / (1 - n_par**2)

        u_perp = r * np.sin(phi)
        u_par = a0 + (r / a) * np.cos(phi)
        return u_par, u_perp

    @staticmethod
    def _get_sn_bar_tensor(u_perp, u_par, y, n_perp, harmonic):
        b = np.abs(n_perp * u_perp / y)
        jn = jv(harmonic, b)
        jnp = jvp(harmonic, b)

        sn_xx = u_perp * (harmonic * jn / b)**2
        sn_xy = -1j * u_perp * (harmonic * jn * jnp / b)
        sn_xz = u_par * (harmonic * jn**2 / b)
        sn_yx = np.conj(sn_xy)
        sn_yy = u_perp * jnp**2
        sn_yz = 1j * u_par * jn * jnp
        sn_zx = np.conj(sn_xz)
        sn_zy = np.conj(sn_yz)
        sn_zz = u_par**2 / u_perp * jn**2

        sn_bar = np.array([
            [sn_xx, sn_xy, sn_xz],
            [sn_yx, sn_yy, sn_yz],
            [sn_zx, sn_zy, sn_zz]
        ])
        return sn_bar

    @staticmethod
    def _get_u_bar(distribution, u_perp, u_par, y, n_par, harmonic):
        gamma = np.sqrt(1 + u_perp**2 + u_par**2)

        f = distribution.f
        u = distribution.u
        theta = distribution.theta
        dfdu, dfdtheta = np.gradient(f, u, theta)
        dfdu_perp = dfdu * np.cos(theta) - dfdtheta / u * np.sin(theta)
        dfdu_par = dfdu * np.sin(theta) + dfdtheta / u * np.cos(theta)

        # Shift gradient to 1D array over resonance ellipse.
        dperp_distribution = dist.Distribution(
            dfdu_perp, u, theta, normalize=False, cql3d=False
        )
        dpar_distribution = dist.Distribution(
            dfdu_par, u, theta, normalize=False, cql3d=False
        )
        dfdu_perp = dperp_distribution.ev(u_perp, u_par)
        dfdu_par = dpar_distribution.ev(u_perp, u_par)

        u_bar = (harmonic * y * dfdu_perp + n_par * u_perp * dfdu_par) / gamma
        return u_bar


class ColdPlasma(Plasma):
    """
    Magnetized cold electron plasma class. This is a good approximation
    for ray tracing away from the fundamental harmonic and is much
    computationally simpler.

    Attributes
    ----------
    density : scalar
        Plasma density (1/cm^3).
    magnetic_field : scalar
        Magnetic field strength (G).
    collision_rate : scalar, optional
        Collision rate (1/s).
    wpe : scalar
        Plasma frequency (radian/s).
    wce : scalar
        Cyclotron frequency (radian/s).

    Methods
    -------
    refractive_index(frequency, mode='O', angle=np.pi/2)
        Calculate the refractive index for a given wave.
    ray_refractive_index(frequency, mode='O', angle=np.pi/2)
        Calculate the ray refractive index for a given wave.
    e_field_polarization(frequency, mode='O', angle=np.pi/2)
        Calculate the electric field polarization for a given wave.
    get_spectral_energy_flux_density(frequency, mode='O', angle=np.pi/2)
        Calculate the energy flux density per frequency per unit volume
        `k` space.
    get_dielectric_tensor(frequency)
        Calculate the dielectric tensor for the plasma at a given
        frequency.

    References
    ----------
    .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
       1992).
    .. [2] G. Bekefi, *Radiation Processes in Plasmas* (Wiley, New York,
       1966).
    .. [3] R. F. Mullaly, J. Atmos. Terr. Phys. **9**, 322 (1956).

    """

    def __init__(self, density, magnetic_field, distribution=None, collision_rate=1e-3):
        """
        Initialize a `Plasma` object with zero temperature.

        Note that, although the distribution is a delta function in the
        cold plasma limit, an arbitrary distribution is accepted since
        the absorption and emission strengths are assumed to be
        determined by the tail electrons in all cases.[1]_

        .. [1] R. W. Harvey *et al.*, Phys. Fluids B **5**, 446 (1993).

        """
        super().__init__(
            density,
            magnetic_field,
            distribution=distribution,
            temperature=0,
            collision_rate=collision_rate
        )

    def refractive_index(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculate the refractive index for a given wave.

        In the high-frequency cold plasma regime, this quantity is
        given by the Altar-Appleton-Hartree dispersion relation [1]_.
        The X and O modes are defined as usual, with the X mode having
        polarization perpendicular to the magnetic field and the O mode
        having polarization parallel to the magnetic field at
        perpendicular propagation. At arbitrary angles of propagation,
        a mode is traced from its polarization at pi/2.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).
        mode : str, optional
            Mode of propagation ('O' or 'X'). Default is 'O'.
        angle : scalar, optional
            Angle of propagation with respect to magnetic field
            (radians). Default is pi/2.

        Returns
        -------
        n : scalar
            Refractive index.

        References
        ----------
        .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).

        """
        w = 2 * np.pi * frequency
        # Effective plasma and cyclotron frequencies squared
        wpe2 = self.wpe**2 * w / (w + 1j * self.collision_rate)
        wce2 = self.wce**2 * w / (w + 1j * self.collision_rate)

        delta = np.sqrt(wce2 * np.sin(angle)**4 + 4 * (w**2 - wpe2)**2
                        * np.cos(angle)**2 / w**2)
        if mode == 'X':
            delta *= -1
        elif mode != 'O':
            raise ValueError("Invalid mode. Choose 'O' or 'X'.")

        n2_numerator = 2 * wpe2 * (w**2 - wpe2) / w**2
        n2_denominator = (
            2 * (w**2 - wpe2)
            - wce2 * np.sin(angle)**2
            + np.sqrt(wce2) * delta
        )
        n2 = 1 - (n2_numerator / n2_denominator)
        return np.sqrt(n2)

    @override
    def e_field_polarization(self, frequency, mode='O', angle=np.pi/2):
        w = 2 * np.pi * frequency
        s, d, p = self._stix_coefficients(w)
        n = self.refractive_index(frequency, mode, angle)
        x_polarization = 1
        y_polarization = 1j * d / (n**2 - s)
        z_polarization = (
            n**2 * np.cos(angle) * np.sin(angle)
            / (n**2 * np.sin(angle)**2 - p)
        )
        e_polarization = np.array(
            [x_polarization, y_polarization, z_polarization])
        e_norm = e_polarization / np.linalg.norm(e_polarization)
        return e_norm

    def _stix_coefficients(self, w):
        """
        Calculate coefficients used to describe wave dispersion [1]_.

        References
        ----------
        .. [1] T. H. Stix, *Waves in Plasmas* (AIP Press, Melville, NY,
           1992).

        """
        if np.isclose(w, 0) or np.isclose(w, self.wce):
            raise ValueError(
                "Frequency must be non-zero and not equal to the cyclotron frequency."
            )
        r = 1 - self.wpe**2 / (w * (w - self.wce))
        l_ = 1 - self.wpe**2 / (w * (w + self.wce))

        s = (r + l_) / 2
        d = (r - l_) / 2
        p = 1 - self.wpe**2 / w**2
        return s, d, p

    def get_spectral_energy_flux_density(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculate the energy flux density per frequency per unit volume
        `k` space.

        Parameters
        ----------
        frequency : scalar
            Frequency of the wave (Hz).
        mode : str, optional
            Mode of propagation ('O' or 'X'). Default is 'O'.
        angle : scalar, optional
            Angle of propagation with respect to magnetic field
            (radians). Default is pi/2.

        Returns
        -------
        spectral_energy_density : scalar
            Spectral energy density (J/m^3).

        """
        w = 2 * np.pi * frequency
        n = self.refractive_index(frequency, mode, angle)
        vg_magnitude = self._get_vg_magnitude(w, n)
        spectral_energy_density = self._get_spectral_energy_density(
            frequency, n, mode, angle
        )
        return vg_magnitude * spectral_energy_density

    def _get_vg_magnitude(self, w, n):
        """
        Calculate the magnitude of the group velocity.

        The symbols and equation come from Mullaly's derivation.[1]_

        .. [1] R. F. Mullaly, J. Atmos. Terr. Phys. **9**, 322
          (1956).
        """
        x = (self.wpe / w)**2
        y = self.wce / w
        eta = 1 - x
        mu = n
        lambda_ = 1 + x / (mu**2 - 1)
        numerator = x * lambda_ * (lambda_**2 - eta**2 * lambda_ - x * y**2)
        denominator = (
            eta * (lambda_ - 1)**2
            * (lambda_**2 - 2 * eta * lambda_ + y**2)
        )

        mu_prime = (
            np.sqrt((lambda_ - 1) / (lambda_ - eta))
            * (1 + numerator / denominator)
        )
        return c / mu_prime

    def _get_spectral_energy_density(self, frequency, n, mode, angle):
        w = 2 * np.pi * frequency
        d_epsilon_h = self._get_dielectric_tensor_derivative(w)
        d_w_epsilon_h = self.get_dielectric_tensor(frequency) + w * d_epsilon_h
        k_hat = np.array([np.sin(angle), 0, np.cos(angle)])
        e_polarization = self.e_field_polarization(frequency, mode, angle)
        relative_b_field = n * np.cross(k_hat, e_polarization)

        spectral_energy_density = (
            np.vdot(relative_b_field, relative_b_field)
            + np.vdot(e_polarization, d_w_epsilon_h @ e_polarization)
        ) / (8 * np.pi)
        return spectral_energy_density

    def get_dielectric_tensor(self, frequency):
        w = 2 * np.pi * frequency
        s, d, p = self._stix_coefficients(w)
        epsilon_h = np.array([
            [s,     -1j*d,  0],
            [1j*d,  s,      0],
            [0,     0,      p]
        ])
        return epsilon_h

    def _get_dielectric_tensor_derivative(self, w):
        ds, dd, dp = self._get_stix_coefficient_derivatives(w)
        epsilon_h_derivative = np.array([
            [ds,     -1j*dd,  0],
            [1j*dd,  ds,      0],
            [0,      0,       dp]
        ])
        return epsilon_h_derivative

    def _get_stix_coefficient_derivatives(self, w):
        ds = 2 * w * self.wpe**2 / (w**2 - self.wce**2)**2
        dd = (self.wce * self.wpe**2 * (self.wce**2 - 3*w**2)
              / (w**2 * (self.wce**2 - w**2)**2))
        dp = 2 * self.wpe**2 / w**3
        return ds, dd, dp
