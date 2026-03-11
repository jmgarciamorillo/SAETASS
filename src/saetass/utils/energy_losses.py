"""
This module provides functions to compute energy losses rates and timescales avaliable to be used in cosmic ray transport simulations.
The current version of this module focuses on the most relevant processes for protons and electrons, both in neutral and ionised gas, and includes the following mechanisms:

- Ionization losses
- Coulomb scattering losses
- Pion production losses
- Synchrotron losses
- Bremsstrahlung losses
- Inverse Compton losses

Moreover, the module is designed to be extensible, allowing for future additions or user-defined loss processes. Other features included are:

- Storage of individual loss components for detailed analysis and debugging.
- Support for both energy-space (dE/dt) and momentum-space (dP/dt) loss rates, with consistent conversion between them.
- Timescale computation for each loss mechanism, enabling direct comparison of their relative importance across the energy and spatial grids.

.. warning::
    Currently, the module does not support temporal evolution of the environment (e.g., time-varying magnetic fields or gas densities). These features are planned for future versions.

________________
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.constants as const
import logging
from typing import Optional, List, Dict, Callable
from enum import StrEnum
import numpy.typing as npt

logger = logging.getLogger(__name__)


class EnergyLossCalculator:
    """
    Main class to compute energy loss rates for cosmic ray particles in a given environment.

    This class is initialized for a specific particle species with the energy grid, spatial grid,
    gas density profile and particle mass. It provides methods to compute the energy loss rates
    for various processes, which are stored internally for later retrieval or analysis.

    Parameters
    ----------
    E_grid : u.Quantity or np.ndarray
        Energy grid for particles (in GeV)
    r_grid : u.Quantity or np.ndarray
        Radial grid for spatial variation (in pc)
    n_gas : u.Quantity or np.ndarray
        Gas density profile (in cm^-3)
    particle : Particle or str
        Choosen between "proton" or "electron". This will determine the particle mass and
        the relevant loss processes. More species are work in progress.
    """

    def __init__(
        self,
        E_grid: u.Quantity | np.ndarray,
        r_grid: u.Quantity | np.ndarray,
        n_gas: u.Quantity | np.ndarray,
        particle: Particle | str,
    ):
        # Validate parameters
        self._check_parameters(E_grid, r_grid, n_gas, particle)
        self.E_grid = E_grid.to(u.GeV)
        self.r_grid = r_grid.to(u.pc)
        self.n_gas = n_gas.to(u.cm**-3)

        # Precompute kinematic quantities
        self._compute_kinematics()

        # Storage for loss rates
        self._E_dot_components = {}
        self._P_dot_components = {}
        self._E_dot_total = None
        self._P_dot_total = None

        # Constants not available in astropy:
        # Classical electron radius
        self.r_e = (
            const.e.si**2 / (4 * np.pi * const.eps0 * const.m_e * const.c**2)
        ).to(u.cm)

    def _compute_kinematics(self):
        """Precompute kinematic quantities for the energy grid."""
        # Lorentz factor and velocity
        self.gamma = 1 + self.E_grid / (self.particle_mass * const.c**2)
        self.beta = np.sqrt(1 - 1 / self.gamma**2)

        # Momentum
        self.p_grid = (
            np.sqrt(
                (self.E_grid**2 + 2 * self.E_grid * (self.particle_mass * const.c**2))
            )
            / const.c
        )

        # Total energy
        self.E_tot = self.E_grid + self.particle_mass * const.c**2

        # Conversion factor from dE/dt to dP/dt
        self.dp_dE = (self.E_grid + self.particle_mass * const.c**2) / (
            self.p_grid * const.c**2
        )
        self.dE_dp = 1 / self.dp_dE

    def _check_parameters(
        self,
        E_grid: u.Quantity | np.ndarray,
        r_grid: u.Quantity | np.ndarray,
        n_gas: u.Quantity | np.ndarray,
        particle: Particle | str,
    ):
        """Validate input parameters and convert to proper units if needed."""

        if isinstance(E_grid, u.Quantity) and not E_grid.unit.is_equivalent(u.GeV):
            raise ValueError("E_grid must have units equivalent to GeV")
        elif isinstance(E_grid, np.ndarray):
            E_grid = E_grid * u.GeV  # Assume GeV if no units provided
        else:
            raise TypeError("E_grid must be an astropy Quantity or numpy ndarray")

        if isinstance(r_grid, u.Quantity) and not r_grid.unit.is_equivalent(u.pc):
            raise ValueError("r_grid must have units equivalent to pc")
        elif isinstance(r_grid, np.ndarray):
            r_grid = r_grid * u.pc  # Assume pc if no units provided
        else:
            raise TypeError("r_grid must be an astropy Quantity or numpy ndarray")

        if isinstance(n_gas, u.Quantity) and not n_gas.unit.is_equivalent(u.cm**-3):
            raise ValueError("n_gas must have units equivalent to cm^-3")
        elif isinstance(n_gas, np.ndarray):
            n_gas = n_gas * u.cm**-3  # Assume cm^-3 if no units provided
        else:
            raise TypeError("n_gas must be an astropy Quantity or numpy ndarray")

        particle = particle.lower() if isinstance(particle, str) else particle
        self.particle = Particle(particle)
        self.particle_mass = self.particle.mass
        self.particle_species = self.particle.species

    def compute_ionization_losses(
        self,
    ) -> u.Quantity:
        """
        Compute ionization energy loss rate using standard expressions for protons
        (:cite:ct:`MannheimSchlickeiser1994`) and electrons (:cite:ct:`Ginzburg1979`).

        Returns
        -------
        E_dot_ion : u.Quantity
            Energy loss rate with shape (len(E_grid), len(r_grid)) in GeV/s.
        """

        if self.particle_species == "hadronic":
            IH = 19.0 * u.eV

            # Maximum energy transfer in single collision
            q_max = (
                2
                * const.m_e
                * const.c**2
                * self.beta**2
                * self.gamma**2
                / (1 + 2 * self.gamma * const.m_e / self.particle_mass)
            )

            # Bethe-Bloch formula coefficient
            A = (
                np.log(
                    2
                    * const.m_e
                    * const.c**2
                    * self.beta**2
                    * self.gamma**2
                    * q_max
                    / IH**2
                )
                - 2 * self.beta**2
            )
        elif self.particle_species == "leptonic":
            # Ionization potential for hydrogen
            IH = 13.6 * u.eV

            # Bethe-Bloch formula coefficient
            A = (
                np.log((self.gamma - 1) * self.beta**2 * self.E_grid**2 / (2 * IH**2))
                + 1 / 8
            )

        prefactor = -2 * np.pi * self.r_e**2 * const.c * const.m_e * const.c**2
        E_dot_ion = prefactor * A[:, None] * self.n_gas[None, :] / self.beta[:, None]

        self._E_dot_components["ionization"] = E_dot_ion.to(u.GeV / u.s)
        logger.debug("Ionization losses computed")

        return E_dot_ion.to(u.GeV / u.s)

    def compute_pion_production_losses(self) -> u.Quantity:
        """
        Compute pion production energy loss rate using standard expressions for hadrons
        (:cite:ct:`KrakauSchlickeiser2015`).

        Returns
        -------
        E_dot_pion : u.Quantity
            Energy loss rate with shape (len(E_grid), len(r_grid)) in GeV/s.
        """
        E_grid_norm = self.E_grid.to("GeV").value
        n_gas_norm = self.n_gas.to("cm**-3").value

        E_thr = 0.3  # GeV
        k = 6  # cutoff sharpness parameter

        E_eff_norm = E_grid_norm * np.exp(-((E_thr / E_grid_norm) ** k))

        # Pion production loss rate from pp collisions
        # Formula: -3.85e-16 * n * E^1.28 * (E + 200)^-0.2 [GeV/s]
        E_dot_pion = (
            (
                -3.85e-16
                * np.outer(n_gas_norm, E_eff_norm**1.28 * (E_eff_norm + 200) ** -0.2).T
            )
            * u.GeV
            / u.s
        )

        self._E_dot_components["pion"] = E_dot_pion.to(u.GeV / u.s)
        logger.debug("Pion production losses computed")

        return E_dot_pion.to(u.GeV / u.s)

    def compute_sychrotron_losses(
        self, B_field: u.Quantity = None, U_B: u.Quantity = None
    ) -> u.Quantity:
        """
        Compute synchrotron energy loss rate using standard expressions
        (:cite:ct:`Ginzburg1979`).

        Parameters
        ----------
        B_field : u.Quantity
            Magnetic field strength with shape (len(r_grid)).
        U_B : u.Quantity
            Magnetic energy density with shape (len(r_grid)). If B_field is provided,
            U_B is ignored.

        Returns
        -------
        E_dot_synchrotron : u.Quantity
            Energy loss rate with shape (len(E_grid), len(r_grid)) in GeV/s.
        """

        # Convert B_field to proper units
        if B_field is not None:
            prefactor = -2.53e-18
            E_dot_synchrotron = (
                prefactor
                * np.outer(
                    self.E_grid.to("GeV").value ** 2,
                    B_field.to(u.microGauss).value ** 2,
                )
                * u.GeV
                / u.s
            )
        elif U_B is not None:
            prefactor = -4 / 3 * const.sigma_T * const.c
            E_dot_synchrotron = (
                prefactor
                * U_B[None, :]
                * self.gamma[:, None] ** 2
                * self.beta[:, None] ** 2
            )
        else:
            raise ValueError("Either B_field or U_B must be provided.")

        self._E_dot_components["synchrotron"] = E_dot_synchrotron.to(u.GeV / u.s)
        logger.debug("Synchrotron losses computed")

        return E_dot_synchrotron.to(u.GeV / u.s)

    def compute_bremsstrahlung_losses(
        self, ionised_mask: npt.NDArray[np.bool_]
    ) -> u.Quantity:
        """
        Compute bremsstrahlung energy loss rate using standard expressions (:cite:ct:`Ginzburg1979`).

        Parameters
        ----------
        ionised_mask : np.ndarray of bool
            Boolean array with shape (num_r,) indicating which radial points correspond
            to ionised gas.

             - For ionised gas, ``True``, the loss rate is computed using the weak-shielded formula (:cite:ct:`Ginzburg1979`).

             - For neutral gas, ``False``, the loss rate is computed using a interpolation between strong-shielded and weak-shielded formula (:cite:ct:`Ginzburg1979`), depending on the particle energy.

        Returns
        -------
        E_dot_brems : u.Quantity
            Energy loss rate with shape (len(E_grid), len(r_grid)) in GeV/s.
        """

        n_gas_norm = self.n_gas.to(u.cm**-3).value
        E_grid_norm = self.E_grid.to(u.GeV).value
        E_tot_norm = self.E_tot.to(u.GeV).value

        numE, numR = len(E_grid_norm), len(n_gas_norm)
        E_dot_brems = np.zeros((numE, numR)) * (u.GeV / u.s)

        # --- 1. Weak-shielded + Ionised ---
        # make WS_term shape (num_E, num_R)
        WS_term = (
            # Ginzburg 1979 formula for bremsstrahlung losses in ionised gas
            -1.37e-16
            * np.outer((np.log(self.gamma) + 0.36) * E_tot_norm, n_gas_norm)
            * u.GeV
            / u.s
        )

        # --- 2. Strong-shielded (SS) ---
        # also (num_E, num_R)
        SS_term = -8e-16 * np.outer(E_grid_norm, n_gas_norm) * u.GeV / u.s

        # --- 3. Intermediate-shielded ---
        w = np.clip((self.gamma - 100) / 700, 0, 1)
        w2D = w.reshape(-1, 1)  # (num_E, 1)

        IS_term = (1 - w2D) * WS_term + w2D * SS_term

        # --- Asignación dependiendo de la máscara ---
        for j in range(numR):

            if ionised_mask[j]:
                # Ionised -> always WS formula
                E_dot_brems[:, j] = WS_term[:, j]
            else:
                # Neutral -> need gamma regions
                for i in range(numE):

                    if self.gamma[i] < 100:
                        E_dot_brems[i, j] = WS_term[i, j]

                    elif self.gamma[i] < 800:
                        E_dot_brems[i, j] = IS_term[i, j]

                    else:
                        E_dot_brems[i, j] = SS_term[i, j]

        self._E_dot_components["bremsstrahlung"] = E_dot_brems.to(u.GeV / u.s)
        return E_dot_brems.to(u.GeV / u.s)

    def compute_coulomb_losses(
        self,
        T_gas: u.Quantity | np.ndarray,
        n_e: u.Quantity | np.ndarray = None,
    ) -> u.Quantity:
        """
        Compute Coulomb scattering energy loss rate using standard expressions for protons
        (:cite:ct:`MannheimSchlickeiser1994`) and electrons (:cite:ct:`Ginzburg1979`).

        Returns
        -------
        E_dot_coulomb : u.Quantity
            Energy loss rate with shape (len(E_grid), len(r_grid)) in GeV/s.
        """
        if self.particle_species == "hadronic":

            if n_e is None:
                n_e = self.n_gas.to(u.cm**-3)

            x_m = (3 * np.sqrt(np.pi) / 4) ** (1 / 3) * np.sqrt(
                2 * const.k_B * T_gas / (const.m_e * const.c**2)
            ).decompose().value

            ln_Lambda = (
                1
                / 2
                * np.log(
                    const.m_e**2
                    * const.c**4
                    * self.gamma[:, None] ** 2
                    * self.beta[:, None] ** 4
                    * self.particle_mass
                    / (
                        np.pi
                        * self.r_e
                        * const.hbar**2
                        * const.c**2
                        * n_e[None, :]
                        * (self.particle_mass + 2 * self.gamma[:, None] * const.m_e)
                    )
                )
            )

            prefactor = -4 * np.pi * self.r_e**2 * const.c * const.m_e * const.c**2

            E_dot_coulomb = (
                prefactor
                * n_e[None, :]
                * ln_Lambda
                * (self.beta[:, None] ** 2 / (self.beta[:, None] ** 3 + x_m**3))
            )

        elif self.particle_species == "leptonic":
            # Logarithmic term
            ln_term = (
                np.log(
                    (
                        self.E_grid[:, None]
                        * const.m_e
                        * const.c**2
                        / (
                            4
                            * np.pi
                            * self.r_e
                            * const.hbar**2
                            * const.c**2
                            * self.n_gas[None, :]
                        )
                    ).decompose()
                )
                - 3 / 4
            )

            # Prefactor
            prefactor = -2 * np.pi * self.r_e**2 * const.c * const.m_e * const.c**2

            E_dot_coulomb = (
                prefactor * (self.n_gas[None, :] / self.beta[:, None]) * ln_term
            ).to(u.GeV / u.s)

        self._E_dot_components["coulomb"] = E_dot_coulomb.to(u.GeV / u.s)
        logger.debug("Coulomb losses computed")

        return E_dot_coulomb.to(u.GeV / u.s)

    def compute_inverse_compton_losses(
        self,
        eps_grid: u.Quantity,
        dn_deps: u.Quantity,
        num_q: int = 120,
    ) -> u.Quantity:
        """
        Compute inverse Compton energy loss rate using the full Klein-Nishina cross section (:cite:ct:`BlumenthalGould1970`).

        .. note::
            The correct physical input is the photon number density spectrum :math:`\\frac{dn}{d\\epsilon}` (number of photons per unit volume per unit energy) rather than the energy density.
            The integration is performed using a vectorized algorithm over the photon energy grid and the kinematic :math:`q`-variable grid.

        Parameters
        ----------
        eps_grid : u.Quantity
            Photon energy grid with energy units, shape (n_eps).
            Should be positive and log-spaced for accuracy, covering the expected photon fields (e.g., CMB, infrared, optical, UV).
        dn_deps : u.Quantity
            Photon spectral number density with shape (n_eps, n_r).
            Units must be compatible with, for example, cm^(-3) eV^(-1).
        num_q : int, optional
            Number of points for integration over the Klein-Nishina phase space parameter :math:`q`.
            Default is 120.

        Returns
        -------
        E_dot_IC : u.Quantity
            Energy loss rate with shape (n_E, n_r) in GeV/s.
        """
        # Validate shapes
        if eps_grid.ndim != 1:
            raise ValueError("eps_grid must be a 1D array.")
        if dn_deps.ndim != 2:
            raise ValueError("dn_deps must be a 2D array with shape (n_eps, n_r).")
        if eps_grid.shape[0] != dn_deps.shape[0]:
            raise ValueError(
                f"eps_grid shape ({eps_grid.shape[0]}) must match first axis of dn_deps ({dn_deps.shape[0]})."
            )

        N_E = len(self.E_grid)
        N_eps = len(eps_grid)
        N_r = len(self.r_grid)

        # Convert to numpy arrays to avoid astropy overhead in large intermediate arrays
        gamma_val = self.gamma.to_value(u.dimensionless_unscaled)[
            :, np.newaxis, np.newaxis
        ]  # (N_E, 1, 1)
        eps_val = eps_grid.to_value(u.eV)[np.newaxis, :, np.newaxis]  # (1, N_eps, 1)
        mec2_eV = (const.m_e * const.c**2).to_value(u.eV)

        # Calculate Gamma parameter
        # Gamma = 4 * gamma * eps / (m_e c^2)
        Gamma = 4.0 * gamma_val * eps_val / mec2_eV  # shape (N_E, N_eps, 1)

        # Construct q grid for integration: q goes from 1 / (4 gamma^2) to 1
        q_min = np.clip(1.0 / (4.0 * gamma_val**2), 0.0, 1.0)  # shape (N_E, 1, 1)
        x = np.linspace(0.0, 1.0, num_q).reshape(1, 1, num_q)  # shape (1, 1, N_q)
        q = q_min + x * (1.0 - q_min)  # shape (N_E, 1, N_q)
        q_safe = np.clip(q, 1e-30, 1.0)  # Avoid log(0) just in case

        Gamma_q = Gamma * q_safe

        # Compute Klein-Nishina kernel F(q, Gamma) and prefactor
        F = (
            1.0
            + 2.0 * q_safe * (np.log(q_safe) - q_safe + 0.5)
            + ((1.0 - q_safe) * Gamma_q**2) / (2.0 * (1.0 + Gamma_q))
        )
        prefactor = ((4.0 * gamma_val**2 - Gamma) * q_safe - 1.0) / (1.0 + Gamma_q) ** 3

        integrand_q = prefactor * F  # shape (N_E, N_eps, N_q)

        # Integrate over q
        K_val = np.trapezoid(integrand_q, x=q, axis=2)  # shape (N_E, N_eps)

        # Weight the kernel by the photon spectral density and energy
        # eps_weighted = eps * dn_deps -> shape (N_eps, N_r)
        eps_weighted_val = (eps_grid[:, np.newaxis] * dn_deps).to_value(u.cm**-3)

        # integrand_eps = K * eps_weighted -> shape (N_E, N_eps, N_r)
        integrand_eps_val = K_val[:, :, np.newaxis] * eps_weighted_val[np.newaxis, :, :]

        # Integrate over eps
        integral_eps_val = np.trapezoid(
            integrand_eps_val, x=eps_grid.to_value(u.eV), axis=1
        )  # shape (N_E, N_r)
        I = integral_eps_val * (u.eV / u.cm**3)

        # Final physical loss rate: -dE/dt = 3 * sigma_T * c * I
        E_dot_IC = (-3.0 * const.sigma_T * const.c * I).to(u.GeV / u.s)

        self._E_dot_components["inverse_compton"] = E_dot_IC
        logger.debug("Inverse Compton losses computed")

        return E_dot_IC

    def compute_total_losses(self) -> u.Quantity:
        """
        Compute total energy loss rate by adding up all loss mechanisms previously computed.

        Returns
        -------
        E_dot_total : u.Quantity
            Total energy loss rate with shape (num_E, num_r) in GeV/s.
        """
        if not self._E_dot_components:
            raise RuntimeError("No energy loss mechanisms have been computed.")

        E_dot_total = 0
        # Sum all components
        for mechanism, E_dot in self._E_dot_components.items():
            E_dot_total += E_dot.to(u.GeV / u.s).value

        self._E_dot_total = E_dot_total * u.GeV / u.s

        logger.info("Total energy losses computed")

        return E_dot_total * u.GeV / u.s

    def get_momentum_loss_rate(self) -> np.ndarray:
        """
        Convert total energy loss rate to momentum loss rate. This is the quantity that
        ``LossSolver`` will use to compute the momentum-space losses in the transport equation.

        Returns:
            P_dot_total: Momentum loss rate with shape (num_E, num_r).
        """
        # Compute total energy losses if not already done
        if self._E_dot_total is None:
            E_dot_total = self.compute_total_losses()
        else:
            E_dot_total = self._E_dot_total

        # Convert from dE/dt to dP/dt using chain rule
        P_dot = E_dot_total * self.dp_dE[:, np.newaxis]

        self._P_dot_total = P_dot

        return self._P_dot_total

    def get_loss_timescales(
        self,
        r_index: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute characteristic loss timescales for each loss mechanisms.

        Parameters
        ----------
        r_index : Optional[int]
            Radial index to compute timescales (if ``None``, returns 2D array).

        Returns
        -------
        timescales : Dict[str, np.ndarray]
            Dictionary with timescales in years for each loss mechanism and total,
            with shape (num_E,) if r_index is provided, or (num_E, num_r) if r_index is None.
        """
        timescales = {}

        # Compute all mechanisms
        self.compute_total_losses()

        for mechanism, E_dot in self._E_dot_components.items():
            if r_index is not None:
                # Timescale at specific radius: τ = E / |dE/dt|
                tau = self.E_grid / (-E_dot[:, r_index])
                timescales[mechanism] = tau.to(u.yr)
            else:
                # Full 2D array
                tau = self.E_grid[:, np.newaxis] / (-E_dot)
                timescales[mechanism] = tau.to(u.yr)
        # Total timescale
        if r_index is not None:
            tau_total = self.E_grid / (-self._E_dot_total[:, r_index])
            timescales["total"] = tau_total.to(u.yr)
        else:
            tau_total = self.E_grid[:, np.newaxis] / (-self._E_dot_total)
            timescales["total"] = tau_total.to(u.yr)

        return timescales


class Particle(StrEnum):
    """Auxiliary class for correct particle types handling.

    .. note::
        Currently, the supported splitting schemes are: "proton" and "electron".


    Parameters
    ----------
    particle_type : str
        String identifier for the particle type (e.g., "proton", "electron"). This
        will raise a ``ValueError`` if an unsupported particle type is provided.
    """

    PROTON = ("proton", const.m_p, "hadronic")
    ELECTRON = ("electron", const.m_e, "leptonic")

    def __new__(cls, particle_type: str, mass: float, species: str):
        obj = str.__new__(cls, particle_type)
        obj._value_ = particle_type
        obj.mass = mass
        obj.species = species
        return obj
