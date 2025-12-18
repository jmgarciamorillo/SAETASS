import numpy as np
import astropy.units as u
import astropy.constants as const
import logging
from typing import Optional, List, Dict, Callable

logger = logging.getLogger(__name__)


class EnergyLossCalculator:
    """
    A comprehensive class for computing energy loss rates for cosmic ray particles.

    This class implements various energy loss mechanisms and provides a unified
    interface for computing loss rates (dE/dt) and their momentum-space equivalents
    (dP/dt) for use in transport equations.

    Attributes:
        E_grid (u.Quantity or np.ndarray): Energy grid for particles (in GeV)
        r_grid (u.Quantity or np.ndarray): Radial grid for spatial variation (in pc)
        n_gas (u.Quantity or np.ndarray): Gas density profile (in cm^-3)
        particle_mass (u.Quantity): Mass of the particle
    """

    def __init__(
        self,
        E_grid: u.Quantity | np.ndarray,
        r_grid: u.Quantity | np.ndarray,
        n_gas: u.Quantity | np.ndarray,
        particle_mass: u.Quantity,
    ):
        """
        Initialize the EnergyLossCalculator.
        """
        # Validate parameters
        self._check_parameters(E_grid, r_grid, n_gas, particle_mass)
        self.E_grid = E_grid.to(u.GeV)
        self.r_grid = r_grid.to(u.pc)
        self.n_gas = n_gas.to(u.cm**-3)
        self.particle_mass = particle_mass

        # Precompute kinematic quantities
        self._compute_kinematics()

        # Storage for loss rates
        self._E_dot_components = {}
        self._P_dot_components = {}
        self._E_dot_total = None
        self._P_dot_total = None

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

        # Conversion factor from dE/dt to dP/dt
        self.dp_dE = (self.E_grid + self.particle_mass * const.c**2) / (
            self.p_grid * const.c**2
        )
        self.dE_dp = 1 / self.dp_dE

    def compute_ionization_losses(self, species: str = "hadronic") -> u.Quantity:
        """
        Compute ionization energy loss rate.

        Uses the Bethe-Bloch formula for ionization losses in hydrogen gas.

        Returns:
            E_dot_ion: Energy loss rate with shape (num_E, num_r)
        """
        if species not in ["hadronic", "leptonic"]:
            raise ValueError("species must be 'hadronic' or 'leptonic'")

        if species == "hadronic":
            # Ionization potential for hydrogen
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
        elif species == "leptonic":
            # Ionization potential for hydrogen
            IH = 13.6 * u.eV

            # Bethe-Bloch formula coefficient
            v = self.beta * const.c
            E_max = self.gamma**2 * const.m_e * v**2 / (1 + self.gamma)
            A = (
                np.log(
                    self.gamma
                    * const.m_e
                    * v**2
                    * E_max
                    / (2 * IH**2 * (1 + self.gamma))
                )
                - (2 / self.gamma - 1 / self.gamma**2) * np.log(2)
                + 1 / self.gamma**2
                + 1 / 8 * (1 - 1 / self.gamma) ** 2
            )

        # Energy loss rate: -7.64e-18 * n * A
        n_gas_norm = self.n_gas.to("cm**-3").value
        E_dot_ion = -7.64e-18 * np.outer(A, n_gas_norm) * u.GeV / u.s

        self._E_dot_components["ionization"] = E_dot_ion
        logger.debug("Ionization losses computed")

        return E_dot_ion

    def compute_pion_production_losses(self) -> u.Quantity:
        """
        Compute pion production (hadronic) energy loss rate.

        Uses the parametrization from Kelner et al. (2006) for proton-proton
        inelastic collisions.

        Returns:
            E_dot_pion: Energy loss rate [GeV/s] with shape (num_E, num_r)
        """
        E_grid_norm = self.E_grid.to("GeV").value
        n_gas_norm = self.n_gas.to("cm**-3").value

        # Pion production loss rate from pp collisions
        # Formula: -3.85e-16 * n * E^1.28 * (E + 200)^-0.2 [GeV/s]
        E_dot_pion = (
            (
                -3.85e-16
                * np.outer(
                    n_gas_norm, E_grid_norm**1.28 * (E_grid_norm + 200) ** -0.2
                ).T
            )
            * u.GeV
            / u.s
        )

        self._E_dot_components["pion"] = E_dot_pion
        logger.debug("Pion production losses computed")

        return E_dot_pion

    def compute_sychrotron_losses(self, B_field: u.Quantity) -> u.Quantity:
        """
        Compute synchrotron energy loss rate.

        Parameters:
            B_field: Magnetic field strength [Gauss] with shape (num_r,)

        Returns:
            E_dot_synchrotron: Energy loss rate [GeV/s] with shape (num_E, num_r)
        """
        # Convert B_field to proper units
        B_field_norm = B_field.to(u.uG).value
        E_grid_norm = self.E_grid.to("GeV").value
        # Synchrotron loss rate formula
        E_dot_synchrotron = (
            -2.53e-18 * np.outer(E_grid_norm**2, B_field_norm**2) * u.GeV / u.s
        )

        self._E_dot_components["synchrotron"] = E_dot_synchrotron
        logger.debug("Synchrotron losses computed")

        return E_dot_synchrotron

    def compute_bremsstrahlung_losses(
        self, n_gas: u.Quantity, ionised_mask
    ) -> u.Quantity:
        """
        Compute bremsstrahlung losses using DRAGON2 expressions.

        Parameters
        ----------
        n_gas : array (num_r)
            Gas density profile [cm^-3]
        ionised_mask : boolean array (num_r)
            True  -> fully ionised gas
            False -> neutral gas (WS/IS/SS depending on gamma)

        Returns
        -------
        E_dot_brems : array (num_E, num_r) in GeV/s
        """

        n_gas_norm = n_gas.to(u.cm**-3).value
        E_grid_norm = self.E_grid.to(u.GeV).value

        numE, numR = len(E_grid_norm), len(n_gas_norm)
        E_dot = np.zeros((numE, numR)) * (u.GeV / u.s)

        # --- 1. Weak-shielded + Ionised ---
        ln_term = np.log(2 * self.gamma) - 1 / 3

        # make WS_term shape (num_E, num_R)
        WS_term = (
            -3.55e-20 * np.outer(ln_term * E_grid_norm, 2.0 * n_gas_norm) * u.GeV / u.s
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
                E_dot[:, j] = WS_term[:, j]
            else:
                # Neutral -> need gamma regions
                for i in range(numE):

                    if self.gamma[i] < 100:
                        E_dot[i, j] = WS_term[i, j]

                    elif self.gamma[i] < 800:
                        E_dot[i, j] = IS_term[i, j]

                    else:
                        E_dot[i, j] = SS_term[i, j]

        self._E_dot_components["bremsstrahlung"] = E_dot
        return E_dot

    def compute_coulomb_losses(
        self, T_gas: u.Quantity | np.ndarray, species: str = "hadronic"
    ) -> u.Quantity:
        """
        Compute Coulomb scattering energy loss rate.

        Relevant at lower energies where elastic collisions dominate.
        """

        n_gas_norm = self.n_gas.to(u.cm**-3).value

        if species == "hadronic":
            num_E = len(self.E_grid)
            num_r = len(self.n_gas)
            E_dot_coulomb = np.zeros((num_E, num_r)) * u.GeV / u.s
            x_m = 0.0286 * (T_gas.to(u.K).value / 2e6) ** 0.5
            for i in range(num_E):
                E_dot_coulomb[i, :] = (
                    (
                        -3.1e-16
                        * n_gas_norm
                        * (self.beta[i] ** 2)
                        / (self.beta[i] ** 3 + x_m**3)
                    )
                    * u.GeV
                    / u.s
                )
        elif species == "leptonic":
            E_grid_norm = (self.E_grid / (const.m_e * const.c**2)).value
            num_E = len(self.E_grid)
            num_r = len(self.n_gas)
            E_dot_coulomb = np.zeros((num_E, num_r)) * u.GeV / u.s
            for i in range(num_E):
                E_dot_coulomb[i, :] = (
                    (
                        -7.64e-18
                        * n_gas_norm
                        * (np.log(E_grid_norm[i]) - np.log(n_gas_norm) + 73.57)
                    )
                    * u.GeV
                    / u.s
                )
        else:
            raise ValueError("species must be 'hadronic' or 'leptonic'")

        self._E_dot_components["coulomb"] = E_dot_coulomb
        logger.debug("Coulomb losses computed")

        return E_dot_coulomb

    # def compute_coulomb_losses_v2(
    #     self,
    #     T_gas: u.Quantity,
    #     A: float = 1.0,
    #     Z: float = 1.0,
    #     species: str = "hadronic",
    # ) -> u.Quantity:
    #     """
    #     Python reimplementation of the C++ TCoulombLoss constructor.

    #     Fully unit-safe version using Astropy.
    #     Reproduces the complete Coulomb loss physics including:
    #     - Coulomb logarithms,
    #     - relativistic beta & gamma,
    #     - spatial dependence,
    #     - nuclear mass A and charge Z,
    #     - Ginzburg electron limit (A = 0).
    #     """

    #     # -----------------------------
    #     # Constants with correct units
    #     # -----------------------------
    #     r0 = 2.817e-15 * u.m
    #     m_e_c2 = (const.m_e * const.c**2).to(u.MeV)  # electron mass energy, MeV
    #     m_p_c2 = (const.m_p * const.c**2).to(u.MeV)  # proton mass energy, MeV

    #     # factor = 1e-9 * Myr (C++ convention)
    #     factor = (1e-9 * u.Myr).to(u.s)

    #     # ħ c in MeV·cm
    #     hbar_c = (const.hbar * const.c).to(u.MeV * u.cm)

    #     # -----------------------------
    #     # Grid sizes
    #     # -----------------------------
    #     num_E = len(self.E_grid)
    #     num_r = len(self.n_gas)

    #     # Output
    #     dEdt = np.zeros((num_E, num_r)) * (u.MeV / u.s)

    #     # Local gas density
    #     n_g = self.n_gas.to(u.cm**-3)

    #     # relativistic factors (assumed dimensionless)
    #     beta = self.beta
    #     gamma = 1.0 / np.sqrt(1.0 - beta**2)

    #     # -----------------------------
    #     # Compute x_m (same formula.
    #     # dimensionless quantity)
    #     # -----------------------------
    #     x_m = 0.0286 * np.sqrt((T_gas.to(u.K) / (2e6 * u.K)))

    #     # -----------------------------
    #     # CASE 1: HADRONS (A != 0)
    #     # -----------------------------
    #     if species == "hadronic" and A != 0:

    #         # Nuclear mass M_A in MeV
    #         M_A = A * m_p_c2

    #         for k in range(num_E):
    #             bet = beta[k]
    #             gam = gamma[k]

    #             bet3 = bet**3
    #             w_e = bet3 / (x_m + bet3)

    #             for j in range(num_r):

    #                 n = n_g[j]

    #                 # ---- Coulomb logarithm ----
    #                 # numerator / denominator must be dimensionless.
    #                 numerator = 4 * np.pi * r0**2 * n * (M_A + 2 * gam * m_e_c2)
    #                 denominator = 4 * (gam**2) * bet**4 * (m_e_c2**2) * M_A

    #                 ratio = (numerator / denominator).decompose()  # make dimensionless
    #                 coullog = -0.5 * np.log(ratio.value)

    #                 # ---- dp/dt (MeV/s) ----
    #                 prefactor = 4 * np.pi * r0**2 * m_e_c2

    #                 dpdt = (
    #                     prefactor
    #                     * (abs(Z) ** 2)
    #                     * n
    #                     / bet
    #                     * coullog
    #                     * w_e
    #                     * (factor / bet)
    #                 )

    #                 # dEdt[k, j] = dpdt

    #         beta_e = np.sqrt(2 * const.k_B * T_gas / (const.m_e * const.c**2))

    #         dEdt = (
    #             -3
    #             / 2
    #             * const.sigma_T
    #             * const.c**2
    #             * self.n_gas
    #             * (
    #                 1
    #                 / 2
    #                 * np.log(
    #                     const.m_e**2
    #                     * const.c**2
    #                     / (np.pi * r0 * const.hbar**2 * self.n_gas)
    #                     * const.m_p
    #                     * self.gamma**2
    #                     * self.beta**4
    #                     / (const.m_p + 2 * self.gamma * const.m_e * const.c**2)
    #                 )
    #             )
    #             / self.beta
    #             * (
    #                 np.erf(beta / beta_e)
    #                 - (2 / np.sqrt(np.pi))
    #                 * (1 + const.m_e / const.m_p)
    #                 * (beta / beta_e)
    #                 * np.exp(-(beta**2) / (beta_e**2))
    #             )
    #         )

    #     # -----------------------------
    #     # CASE 2: ELECTRONS (A = 0)
    #     # -----------------------------
    #     elif species == "leptonic" or A == 0:

    #         for k in range(num_E):
    #             bet = beta[k]
    #             gam = gamma[k]

    #             for j in range(num_r):
    #                 n = n_g[j]

    #                 if n > 0 * u.cm**-3:
    #                     # Coulomb log must be dimensionless:
    #                     # argument = (γ m_e c²) / ( n m_e c² (4π r_e ħ c)² )
    #                     denom = n * (m_e_c2 * (4 * np.pi * r0 * hbar_c) ** 2)
    #                     arg = (gam * m_e_c2 / denom).decompose()  # dimensionless
    #                     coullog2 = np.log(arg.value) - 0.75
    #                 else:
    #                     coullog2 = 0.0

    #                 prefactor = 2 * np.pi * r0**2 * m_e_c2

    #                 dpdt = (prefactor * n / bet * coullog2 * (factor / bet)).to(
    #                     u.MeV / u.s
    #                 )

    #                 dEdt[k, j] = dpdt

    #     else:
    #         raise ValueError("species must be 'hadronic' or 'leptonic'.")

    #     # return in GeV/s like original python code
    #     return dEdt.to(u.GeV / u.s)

    def compute_inverse_compton_losses(
        self,
        u_rad: u.Quantity,
        eps_grid: u.Quantity | None = None,
        *,
        num_eps: int = 120,
        num_q: int = 120,
    ) -> u.Quantity:
        """
        Compute inverse Compton energy loss rate using full Klein-Nishina eq.
        """

        num_E = len(self.E_grid)

        num_r = len(self.r_grid)

        # -------------------- prepare eps grid --------------------
        if eps_grid is None:
            # default: 1e-9 eV -> 1e3 eV (covers CMB..UV)
            eps_min = 1e-9 * u.eV
            eps_max = 1e3 * u.eV
            eps_grid = (
                np.exp(
                    np.linspace(np.log(eps_min.value), np.log(eps_max.value), num_eps)
                )
                * eps_min.unit
            )
        else:
            if not isinstance(eps_grid, u.Quantity):
                raise TypeError(
                    "eps_grid must be an astropy Quantity with energy units."
                )
            num_eps = eps_grid.size

        # interpret u_rad
        # (num_eps, num_r) : full spectral density per radius (preferred)

        if u_rad.ndim == 2:
            if u_rad.shape[1] == num_r and u_rad.shape[0] == num_eps:
                # good: (num_eps, num_r)
                n_photon = u_rad  # shape (num_eps, num_r)
            else:
                raise ValueError("u_rad shape must be 2D (num_eps, num_r).")

        # -------------------- precompute kernel K(gamma, eps) = ∫_{q_min}^1 F(q) dq --------------------
        # K shape: (num_E, num_eps)
        K = np.zeros((num_E, num_eps), dtype=float)

        # q grid
        q_min = 1 / (4.0 * self.gamma**2)  # shape (num_E,)

        for i_e in range(num_E):
            gamma = self.gamma[i_e]
            q_min_e = q_min[i_e]
            # if q_min >= 1 no scattering allowed -> K zeros remain
            if q_min_e >= 1.0:
                continue
            # define q_sub vector from max(q_min,q_floor) to 1)
            q_sub = np.linspace(q_min_e, 1.0, num_q)

            # vectorize over eps
            # For numeric efficiency, compute Gamma for all eps first:
            Gamma_eps = (
                4.0 * gamma * eps_grid / (const.m_e * const.c**2)
            ).decompose()  # shape (num_eps,)

            # For each eps compute integral over q
            for i_eps in range(num_eps):
                Gamma = Gamma_eps[i_eps]
                q = q_sub  # shape (num_q,)
                # compute integrand
                integrand = (
                    ((4 * gamma**2 * -Gamma) * q - 1)
                    / (1 + Gamma * q) ** 3
                    * (
                        1
                        + 2 * q * (np.log(q) - q + 0.5)
                        + ((1 - q) * (Gamma * q) ** 2) / (2 * (1 + Gamma * q))
                    )
                )

                # integrate in q
                Kval = np.trapezoid(integrand, q)
                K[i_e, i_eps] = Kval

        # -------------------- integrate over eps for each gamma and radius --------------------
        dEdt = np.zeros((num_E, num_r), dtype=float) * u.GeV / u.s

        for ir in range(num_r):
            n_eps = n_photon[:, ir]  # shape (num_eps,)
            # integrand per gamma: integrand_eps = n_eps * K[ie,:]  -> shape (num_E, num_eps)
            integrand_eps = K * n_eps[np.newaxis, :]
            integral_eps = np.trapezoid(
                integrand_eps, eps_grid, axis=1
            )  # shape (num_E,), numeric
            dEdt[:, ir] = -3.0 * const.sigma_T * const.c * integral_eps  # numeric erg/s

        E_dot_IC = dEdt.to(u.GeV / u.s)

        self._E_dot_components["inverse_compton"] = E_dot_IC
        logger.debug("Inverse Compton losses computed")

        return E_dot_IC

    def compute_total_losses(self) -> u.Quantity:
        """
        Compute total energy loss rate by summing all mechanisms.

        Returns:
            E_dot_total: Total energy loss rate with shape (num_E, num_r)
        """
        if not self._E_dot_components:
            raise RuntimeError("No energy loss mechanisms have been computed.")

        E_dot_total = 0
        # Sum all components
        for mechanism, E_dot in self._E_dot_components.items():
            E_dot_total += E_dot.to(u.GeV / u.s).value

        self._E_dot_total = E_dot_total * u.GeV / u.s

        logger.info("Total energy losses computed")

        return E_dot_total

    def get_momentum_loss_rate(self) -> np.ndarray:
        """
        Get total momentum loss rate dP/dt.

        This is the quantity needed for the lossFV operator in momentum space.

        Returns:
            P_dot: Momentum loss rate with shape (num_E, num_r)
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
        Compute characteristic loss timescales for each mechanism.

        Parameters:
            r_index: Radial index to compute timescales (if None, returns 2D array)

        Returns:
            timescales: Dictionary with timescales for each mechanism
        """
        timescales = {}

        # Compute all mechanisms
        self.compute_total_losses()

        for mechanism, E_dot in self._E_dot_components.items():
            if r_index is not None:
                # Timescale at specific radius: τ = E / |dE/dt|
                tau = self.E_grid / (-E_dot[:, r_index])
                timescales[mechanism] = tau
            else:
                # Full 2D array
                tau = self.E_grid[:, np.newaxis] / (-E_dot)
                timescales[mechanism] = tau

        # Total timescale
        if r_index is not None:
            tau_total = self.E_grid / (-self._E_dot_total[:, r_index])
            timescales["total"] = tau_total
        else:
            tau_total = self.E_grid[:, np.newaxis] / (-self._E_dot_total)
            timescales["total"] = tau_total

        return timescales

    def plot_loss_timescales(
        self,
        r_index: int,
        save_path: Optional[str] = None,
    ):
        """
        Plot loss timescales vs energy at a specific radius.

        Parameters:
            r_index: Index of radius to plot
            save_path: Path to save figure (if None, just displays)
        """
        import matplotlib.pyplot as plt

        timescales = self.get_loss_timescales(r_index=r_index)

        plt.figure(figsize=(10, 7))

        colors = {
            "ionization": "red",
            "pion": "orange",
            "synchrotron": "yellow",
            "bremsstrahlung": "green",
            "coulomb": "blue",
            "inverse_compton": "purple",
            "total": "black",
        }

        for mechanism, tau in timescales.items():
            label = mechanism.capitalize()
            color = colors.get(mechanism, "gray")
            lw = 2.5 if mechanism == "total" else 1.5
            ls = "-" if mechanism == "total" else "--"

            plt.loglog(
                self.E_grid.to("GeV").value,
                tau.to("Myr").value,
                label=label,
                color=color,
                linewidth=lw,
                linestyle=ls,
            )

        plt.xlabel("Energy (GeV)", fontsize=12)
        plt.ylabel("Loss Timescale (Myr)", fontsize=12)
        plt.title(
            f"Energy Loss Timescales at r = {self.r_grid[r_index]:.1f} pc", fontsize=13
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved loss timescales plot to {save_path}")

        plt.show()

    def _check_parameters(
        self,
        E_grid: u.Quantity | np.ndarray,
        r_grid: u.Quantity | np.ndarray,
        n_gas: u.Quantity | np.ndarray,
        particle_mass: u.Quantity,
    ):
        """Validate input parameters."""

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

        if not isinstance(particle_mass, u.Quantity):
            raise TypeError("particle_mass must be an astropy Quantity")
