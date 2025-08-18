"""
plasma - defines the various plasma classes.

At some point, this should define relativistic thermal plasmas as well
as cold plasmas.

"""
from abc import ABC, abstractmethod
import numpy as np
from scipy import constants
from scipy.differentiate import derivative

# Convert SI units to CGS for plasma calculations
c = constants.speed_of_light * 1e2  # m/s to cm/s
e = constants.elementary_charge * c / 10  # C to statC
m_e = constants.electron_mass * 1e3  # kg to g


class Plasma(ABC):
    """
    Base class for plasmas.

    This class is not meant to be instantiated directly, but rather
    serves as a base for specific plasma types.

    """
    def __init__(self, density, magnetic_field, collision_rate):
        """
        Initialize the cold plasma with density and magnetic field.

        Parameters
        ----------
        density : scalar
            Plasma density (1/cm^3).
        magnetic_field : scalar
            Magnetic field strength (G).
        collision_rate : scalar, optional
            Collision rate (1/s). Default is 1e-3 (nearly collisionless).

        """
        self.density = density
        self.magnetic_field = magnetic_field
        self.collision_rate = collision_rate
        self.wpe = self._get_plasma_frequency()
        self.wce = self._get_cyclotron_frequency()


    def _get_plasma_frequency(self):
        """Calculate the plasma frequency (radian/s)."""
        return np.sqrt(4 * np.pi * self.density * e**2 / m_e)


    def _get_cyclotron_frequency(self):
        """Calculate the cyclotron frequency (radian/s)."""
        return e * self.magnetic_field / (m_e * c)


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
        raise NotImplementedError("This method should be implemented in subclasses.")


class ColdPlasma(Plasma):
    """
    Magnetized cold electron plasma.

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
    def __init__(self, density, magnetic_field, collision_rate=1e-3):
        super().__init__(density, magnetic_field, collision_rate)

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
        n2 = 1 - ((2 * wpe2 * (w**2 - wpe2) / w**2)
                  / (2 * (w**2 - wpe2) - wce2 * np.sin(angle)**2 + np.sqrt(wce2) * delta))
        return np.sqrt(n2)

    def ray_refractive_index(self, frequency, mode='O', angle=np.pi/2):
        """
        Calculates the ray refractive index for a given wave.

        This quantity is defined from the real part of the refractive
        index and is calculated numerically.

        TODO: This quantity can be solved analytically for cold plasma.
        A broken implementation exists here:

            https://github.com/DIII-D/relativistic-ece/blob/ae7b395a3c0349987e1fc23e81ea6d42ac2d56c4/relece/cold_plasma.py#L313-L395

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
            lambda theta: np.real(self.refractive_index(frequency, mode, theta)),
            angle
        )
        denominator = derivative(
            lambda theta: self._ray_refraction_denom_helper(frequency, mode, theta),
            angle
        )
        if not np.any(dn.success) or not np.any(denominator.success):
            self._raise_differentiation_warning()
        numerator = np.sqrt(1 + (dn.df / n)**2)

        nr2 = np.abs(n**2 * np.sin(angle) * numerator / denominator.df)
        return np.sqrt(nr2)


    def _ray_refraction_denom_helper(self, frequency, mode, angle):
        """Important quantity for ray refractive index calculation"""
        n = np.real(self.refractive_index(frequency, mode, angle))
        dn = derivative(
            lambda theta: np.real(self.refractive_index(frequency, mode, theta)),
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
        w = 2 * np.pi * frequency
        s, d, p = self._stix_coefficients(w)
        n = self.refractive_index(frequency, mode, angle)
        x_polarization = 1
        y_polarization = 1j * d / (n**2 - s)
        z_polarization = n**2 * np.cos(angle) * np.sin(angle) / (n**2 * np.sin(angle)**2
                                                                 - p)
        return x_polarization, y_polarization, z_polarization


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
        l = 1 - self.wpe**2 / (w * (w + self.wce))

        s = (r + l) / 2
        d = (r - l) / 2
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
        vg_magnitude = self._get_vg_magnitude(w, mode, angle)
        spectral_energy_density = self._get_spectral_energy_density(w, mode, angle)
        return vg_magnitude * spectral_energy_density


    def _get_vg_magnitude(self, w, mode, angle):
        """
        Calculate the magnitude of the group velocity.

        The symbols and equation come from Mullaly_'s derivation.

        .. _Mullaly: R. F. Mullaly, J. Atmos. Terr. Phys. **9**, 322
          (1956).
        """
        x = (self.wpe / w)**2
        y = self.wce / w
        eta = 1 - x
        mu = self.refractive_index(w, mode, angle)
        lambda_ = 1 + x / (mu**2 - 1)
        numerator = x * lambda_ * (lambda_**2 - eta**2 * lambda_ - x * y**2)
        denominator = eta * (lambda_ - 1)**2 * (lambda_**2 - 2 * eta * lambda_ + y**2)
        
        mu_prime = np.sqrt((lambda_ - 1) / (lambda_ - eta)) * (1 + numerator
                                                               / denominator)
        return c / mu_prime


    def _get_spectral_energy_density(self, w):
        pass


    def get_dielectric_tensor(self, frequency):
        s, d, p = self._stix_coefficients(self.wpe)
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


    def _get_spectral_energy_density(self, frequency, mode, angle):
        pass

