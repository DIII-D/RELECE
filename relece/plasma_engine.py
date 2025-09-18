"""
plasma - defines the various plasma classes.

At some point, this should define relativistic thermal plasmas as well
as cold plasmas.

"""
from abc import ABC, abstractmethod
from typing import override
import warnings

import numpy as np
from scipy import constants
from scipy import differentiate
from scipy import integrate
from scipy import linalg
from scipy import special

from relece import distributions as dist

# Convert SI units to Gaussian for plasma calculations
c = constants.speed_of_light * 1e2        # m/s to cm/s
e = constants.elementary_charge * c / 10  # C to statC
m_e = constants.electron_mass * 1e3       # kg to g


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
        temperature=0.0,
        distribution=None,
        collision_rate=0
    ):
        self.density = density * 1e-6  # Convert to 1/cm^3
        self.magnetic_field = magnetic_field * 1e4  # Convert to G
        self.temperature = temperature
        if distribution is None:
            distribution = dist.MaxwellJuttnerDistribution(temperature)
        self.distribution = distribution
        self.collision_rate = collision_rate
        self.wpe = self._get_plasma_frequency()
        self.wce = self._get_cyclotron_frequency()

    def _get_plasma_frequency(self):
        wpe = np.sqrt(4 * np.pi * self.density * e**2 / m_e)
        return wpe

    def _get_cyclotron_frequency(self):
        wce = e * self.magnetic_field / (m_e * c)
        return wce

    @abstractmethod
    def refractive_index(self, frequency, mode='O', angle=np.pi/2) -> np.ndarray:
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
        n = self.refractive_index(frequency, mode, angle)
        dn = differentiate.derivative(
            lambda theta: (self.refractive_index(frequency, mode, theta)),
            angle
        )
        denominator = differentiate.derivative(
            lambda theta: self._ray_refraction_denom_helper(
                frequency, mode, theta),
            angle
        )
        if not np.any(dn.success) or not np.any(denominator.success):
            self._raise_differentiation_warning()
        numerator = np.sqrt(1 + (dn.df / n)**2)

        nr2 = np.abs(n**2 * np.sin(angle) * numerator / denominator.df)
        nr = np.sqrt(nr2)
        return nr

    def _ray_refraction_denom_helper(self, frequency, mode, angle):
        """Important quantity for ray refractive index calculation"""
        n = self.refractive_index(frequency, mode, angle)
        dn = differentiate.derivative(
            lambda theta: self.refractive_index(frequency, mode, theta),
            angle
        )
        if not np.any(dn.success):
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
        ])  # Outer product of wave direction minus delta_ij
        dispersion_tensor = epsilon + n**2 * kk_minus_eye

        e_polarization = linalg.null_space(dispersion_tensor)
        return e_polarization

    @abstractmethod
    def get_dielectric_tensor(self, frequency) -> np.ndarray:
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
    def spectral_energy_flux_density(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculate the energy flux density per frequency per unit volume
        `k` space.
        """
        pass

    def emission(
        self,
        frequency,
        mode='O',
        angle=np.pi/2,
        tolerance=1e-6,
        maxterms=10
    ):
        n = self.refractive_index(frequency, mode, angle)
        if (n * np.cos(angle))**2 >= 1:
            warnings.warn(
                "Mode does not escape the plasma.",
                RuntimeWarning
            )
            return np.nan
        j = self._sum_harmonics(
            frequency, mode, angle, n, tolerance, maxterms, 'emission'
        )
        return j

    def absorption(
        self,
        frequency,
        mode='O',
        angle=np.pi/2,
        tolerance=1e-6,
        maxterms=10
    ):
        n = self.refractive_index(frequency, mode, angle)
        if (n * np.cos(angle))**2 >= 1:
            warnings.warn(
                "Mode does not escape the plasma.",
                RuntimeWarning
            )
            return np.nan
        alpha = self._sum_harmonics(
            frequency, mode, angle, n, tolerance, maxterms, 'absorption'
        )
        return alpha

    def _j_alpha_helper(self, frequency, mode, angle, n):
        w = 2 * np.pi * frequency
        x = (self.wpe / w)**2
        y = self.wce / w

        n_perp = n * np.sin(angle)
        n_par = n * np.cos(angle)
        initial = np.ceil(np.sqrt(1 - n_par**2) / y)  # Lower bound for real roots

        s = self.spectral_energy_flux_density(frequency, mode, angle)
        e = self.e_field_polarization(frequency, mode, angle)
        return w, x, y, n_perp, n_par, initial, s, e

    def _sum_harmonics(self, frequency, mode, angle, n, tolerance, maxterms, term):
        """
        Sums over all harmonics within `tolerance`.

        The function starts at ``n = initial`` and sums outward from there.

        """
        (w, x, y, n_perp, 
         n_par, initial, s, e) = self._j_alpha_helper(frequency, mode, angle, n)
        c = 0
        if term == 'absorption':
            cn = lambda harmonic: self._alpha_n(x, y, n_perp, n_par, harmonic, w, s, e)
        else:
            nr = self.ray_refractive_index(frequency, mode, angle)
            cn = lambda harmonic: self._j_n(x, y, n_perp, n_par, nr, harmonic, w, s, e)
        cnext = cn(initial)
        i = 1  # Principal harmonic offset
        counter = 1

        while np.abs(cnext) > tolerance and counter < maxterms:
            c += cnext
            counter += 1
            i *= -1
            if i > 0:
                i += 1
            print(cnext)

        if counter == maxterms:
            raise ValueError("Reached max terms without convergence.")
        return c

    def _alpha_n(self, x, y, n_perp, n_par, harmonic, w, s, e):
        epsilon_a = self._integral_n(x, y, n_perp, n_par, harmonic, 'epsilon_a')
        alpha_n = w / (4 * np.pi) * np.vdot(e, epsilon_a @ e) / s
        return np.real(alpha_n)

    def _j_n(self, x, y, n_perp, n_par, nr, harmonic, w, s, e):
        current_correlation_tensor = self._integral_n(
            x, y, n_perp, n_par, harmonic, 'G'
        )
        j_n = (
            np.pi * nr**2 * (w / c)**2
            * np.vdot(e, current_correlation_tensor @ e) / s
        )
        return np.real(j_n)

    def _integral_n(self, x, y, n_perp, n_par, harmonic, tensor):
        """
        Calculate the integral over the resonance ellipse.

        `tensor` can be either 'epsilon_a' or 'G'. [After Harvey *et
        al.* (1992).]

        """
        theta = self.distribution.theta
        a_n, b_n, p_perp, p_par = self._get_resonance_ellipse(
            n_par, y, theta, harmonic
        )

        sn = self._get_sn_tensor(p_perp, p_par, y, n_perp, harmonic)
        gamma = np.sqrt(1 + (p_perp / (m_e * c))**2 + (p_par / (m_e * c))**2)
        if tensor == 'epsilon_a':
            uf = self._get_uf(self.distribution, p_perp, p_par, y, n_par, harmonic)
            # Rebroadcast (N) -> (N, 1, 1)
            integrand = -np.pi * x * np.expand_dims(uf, axis=(1, 2)) * sn
        else:
            f = self.distribution.ev(p_perp, p_par)
            integrand = (
                np.pi / (2 * np.pi)**5 * x / m_e
                * np.expand_dims(f * p_perp / gamma, axis=(1, 2))
                * sn
            )

        # Calculate the gradient of the delta function argument, g
        g_p_perp = (
            p_perp / (gamma * (m_e * c)**2)
            + n_par * p_par * p_perp / (gamma**4 * (m_e * c)**3)
        )
        g_p_par = (
            p_par / (gamma * (m_e * c)**2)
            + n_par * p_par**2 / (gamma**4 * (m_e * c)**3)
            - n_par / (gamma * m_e * c)
        )
        grad_g = np.hypot(g_p_perp, g_p_par)

        # Jacobian incorporates the transformation of the delta function
        jacobian = (2 * np.pi * a_n**2 * b_n
                    * np.expand_dims(np.sin(theta), axis=(1, 2)) / grad_g)
        integral_n = integrate.simpson(integrand * jacobian, theta, axis=0)
        # print(integral_n)
        return integral_n

    @staticmethod
    def _get_resonance_ellipse(n_par, y, theta, harmonic):
        """
        Defines the integration region imposed by the Dirac delta.
        Assumes ``n_par**2 < 1``.

        [After Freund *et al.* (1984).]

        """
        u_bar_n = c * harmonic * y * n_par / (1 - n_par**2)
        a_n = c * np.sqrt(((harmonic * y)**2 + n_par**2 - 1) / (1 - n_par**2))
        b_n = a_n * 1 / np.sqrt(1 - n_par**2)

        u_par = np.linspace(u_bar_n - b_n, u_bar_n + b_n, theta.size)
        u_perp = a_n * np.sqrt(1 - (u_par - u_bar_n)**2 / b_n**2)  # eq. 6

        p_perp = m_e * u_perp
        p_par = m_e * u_par
        return a_n, b_n, p_perp, p_par

    @staticmethod
    def _get_uf(distribution, p_perp, p_par, y, n_par, harmonic):
        """Equation 8, R. W. Harvey *et al.* (1993)."""
        gamma = np.sqrt(1 + (p_perp**2 + p_par**2) / (m_e * c)**2)

        f = distribution.f
        p = distribution.p
        theta = distribution.theta
        dfdp, dfdtheta = np.gradient(f, p, theta)
        p_grid, theta_grid = np.meshgrid(p, theta, indexing='ij')
        dfdp_perp = (dfdp * np.sin(theta_grid)
                     + dfdtheta / p_grid * np.cos(theta_grid))
        dfdp_par  = (dfdp * np.cos(theta_grid)
                     - dfdtheta / p_grid * np.sin(theta_grid))

        # Shift gradient to 1D array over resonance ellipse.
        dperp_distribution = dist.Distribution(dfdp_perp, p, theta, normalize=False)
        dpar_distribution = dist.Distribution(dfdp_par, p, theta, normalize=False)
        dfdp_perp = dperp_distribution.ev(p_perp, p_par)
        dfdp_par = dpar_distribution.ev(p_perp, p_par)

        uf = (
            harmonic * y * dfdp_perp
            + n_par * p_perp * dfdp_par / (m_e * c)
        ) / gamma
        return uf

    @staticmethod
    def _get_sn_tensor(p_perp, p_par, y, n_perp, harmonic):
        """Equation 9, R. W. Harvey *et al.* (1992)."""
        b = np.abs(n_perp * p_perp / (m_e * c * y))
        jn = special.jv(harmonic, b)
        jnp = special.jvp(harmonic, b)

        sn_xx = p_perp * (harmonic * jn / b)**2
        sn_xy = -1j * p_perp * (harmonic * jn * jnp / b)
        sn_xz = p_par * (harmonic * jn**2 / b)
        sn_yx = np.conj(sn_xy)
        sn_yy = p_perp * jnp**2
        sn_yz = 1j * p_par * jn * jnp
        sn_zx = np.conj(sn_xz)
        sn_zy = np.conj(sn_yz)
        sn_zz = p_par**2 / p_perp * jn**2

        sn = np.array([
            [sn_xx, sn_xy, sn_xz],
            [sn_yx, sn_yy, sn_yz],
            [sn_zx, sn_zy, sn_zz]
        ])
        sn = np.moveaxis(sn, -1, 0)  # sn: (3, 3, N) -> (N, 3, 3)
        return sn


class ColdPlasma(Plasma):
    """
    Magnetized cold electron plasma class. This is a good approximation
    for ray tracing away from the fundamental harmonic and is
    computationally much simpler.

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
        w2 = (2 * np.pi * frequency)**2
        wpe2 = self.wpe**2
        wce2 = self.wce**2

        delta = np.sqrt(wce2 * np.sin(angle)**4 + 4 * (w2 - wpe2)**2
                        * np.cos(angle)**2 / w2)
        if mode == 'X':
            delta *= -1
        elif mode != 'O':
            warnings.warn(
                "Invalid mode. Defaulting to 'O'.",
                RuntimeWarning
            )

        n2_numerator = 2 * wpe2 * (w2 - wpe2) / w2
        n2_denominator = (
            2 * (w2 - wpe2)
            - wce2 * np.sin(angle)**2
            + np.sqrt(wce2) * delta
        )
        n2 = 1 - (n2_numerator / n2_denominator)
        return np.real(np.sqrt(n2))

    @override
    def e_field_polarization(self, frequency, mode='O', angle=np.pi/2):
        """
        Analytical field polarization in a cold plasma.

        [After Stix (1992).]

        """
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
        Calculate coefficients used to describe wave dispersion.

        [After Stix (1992).]
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

    def spectral_energy_flux_density(self, frequency, mode='O', angle=np.pi/2):
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
        vg_magnitude = self._get_vg_magnitude(w, mode, angle)
        spectral_energy_density = self._get_spectral_energy_density(
            frequency, n, mode, angle
        )
        spectral_energy_flux_density = spectral_energy_density * vg_magnitude
        print(spectral_energy_flux_density)
        return spectral_energy_flux_density

    def _get_vg_magnitude(self, w, mode, angle):
        """Calculate the magnitude of the group velocity."""
        dkdw = differentiate.derivative(
                lambda w_: w_ * self.refractive_index(w_ / (2 * np.pi), mode, angle) / c,
                w
        )
        return 1 / dkdw.df

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
