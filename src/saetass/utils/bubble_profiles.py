"""
This module provides functions and classes to compute the spatial profiles of plasma and magnetic field parameters inside a stellar wind bubble.
The models available are based on the theoretical frameworks by

- :cite:ct:`Weaver1977`
- :cite:ct:`Morlino2021`

The module provides an extensible framework using :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator` to easily extract density, velocity, magnetic field or transport parameters to be used in SAETASS simulations.
"""

import logging
import math
from enum import StrEnum
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz

from .. import units as su


class BubbleModel(StrEnum):
    """Auxiliary class for correct particle types handling.

    .. note::
        Currently, the supported models are: ``"Weaver77"`` and ``"Morlino21"``.

    Parameters
    ----------
    model_type : str
        The model type (e.g., "Weaver77" or "Morlino21").
    """

    WEAVER77 = "Weaver77"
    MORLINO21 = "Morlino21"

    def __new__(cls, model_type: str):
        obj = str.__new__(cls, model_type)
        obj._value_ = model_type
        return obj


logger = logging.getLogger(__name__)


class BubbleProfileCalculator:
    """
    Main class to compute stellar wind bubble spatial profiles.

    This class is initialized with the radial grid and the corresponding model.
    It manages the calculation of macroscopic kinematic properties depending on the selected model.
    Built-in models may require additional parameters to compute specific profiles.

    Parameters
    ----------
    r_grid : astropy.units.Quantity
        Radial grid for the profiles computation. Units compatible with :py:data:`~saetass.units.LENGTH`.
    model : BubbleModel or str
        The bubble description model (e.g., "Weaver77" or "Morlino21").
    **kwargs :
        Specific physical parameters required by the selected model. For 'Weaver77' and
        'Morlino21', the required kwargs are:

        - ``L_wind`` : astropy.units.Quantity. Wind mechanical luminosity. Units compatible with :py:data:`~saetass.units.LUMINOSITY`.
        - ``M_dot`` : astropy.units.Quantity. Mass loss rate. Units compatible with :py:data:`~saetass.units.MASS_LOSS_RATE`.
        - ``rho_0`` : astropy.units.Quantity. Ambient mass density. Units compatible with :py:data:`~saetass.units.MASS_DENSITY`.
        - ``t_b`` : astropy.units.Quantity. Bubble age. Units compatible with :py:data:`~saetass.units.TIME`.

        Other optional kwargs: ``R_c`` (core radius). Units compatible with :py:data:`~saetass.units.LENGTH`.
    """

    @u.quantity_input(r_grid=su.LENGTH)
    def __init__(
        self,
        r_grid: u.Quantity,
        model: BubbleModel | str = "Morlino21",
        **kwargs,
    ):
        self.r_grid = r_grid.to(su.LENGTH)

        if isinstance(model, str):
            for m in BubbleModel:
                if m.value.lower() == model.lower():
                    model = m
                    break
        self.model = BubbleModel(model)

        self.kwargs = kwargs

        self.R_TS = None
        self.R_b = None
        self.v_w = None
        self.rho_w = None
        self.R_cd = None
        self.masks = {}

        self._compute_kinematics()

    def _compute_kinematics(self):
        """Dispatch the kinematics computation based on the model."""
        if self.model in [BubbleModel.WEAVER77, BubbleModel.MORLINO21]:
            self._compute_Weaver77_kinematics()
        else:
            raise NotImplementedError(
                f"Kinematics for model {self.model} are not implemented."
            )

    def _compute_Weaver77_kinematics(self):
        """
        Compute baseline parameters for Weaver77/Morlino21 models from wind properties.
        Expected kwargs: L_wind, M_dot, rho_0, t_b
        """
        required = ["L_wind", "M_dot", "rho_0", "t_b"]
        for key in required:
            if key not in self.kwargs:
                raise ValueError(
                    f"Model '{self.model}' requires parameter '{key}' in kwargs."
                )

        L_wind = self.kwargs["L_wind"]
        M_dot = self.kwargs["M_dot"]
        rho_0 = self.kwargs["rho_0"]
        t_b = self.kwargs["t_b"]

        # Weaver77 parameters
        kappa = 0.762865

        self.v_w = np.sqrt(2 * L_wind / M_dot).to(u.km / u.s)

        self.R_b = (kappa * (L_wind / rho_0 * t_b**3) ** (1 / 5)).to(u.pc)

        self.R_TS = (
            ((25 / (28 * np.pi)) ** 0.5)
            * (kappa**-1)
            * (L_wind**-0.2)
            * (M_dot**0.5 * self.v_w**0.5)
            * rho_0 ** (-3 / 10)
            * t_b ** (2 / 5)
        ).to(u.pc)

        self.rho_w = (3 * M_dot / (4 * math.pi * (self.R_TS**2) * self.v_w)).to(
            u.g / u.cm**3
        )
        self.R_cd = 0.95 * self.R_b

        r_val = self.r_grid.to(u.pc).value
        # If R_c is provided, use it, else 0
        R_c = self.kwargs.get("R_c", 0 * u.pc)
        R_c_val = R_c.to(u.pc).value

        self.masks = {
            "r_core": r_val < R_c_val,
            "r_wind": (r_val >= R_c_val) & (r_val < self.R_TS.value),
            "r_hot": (r_val >= self.R_TS.value) & (r_val < self.R_cd.value),
            "r_shell": (r_val >= self.R_cd.value) & (r_val < self.R_b.value),
            "r_bubble": (r_val >= self.R_TS.value) & (r_val <= self.R_b.value),
            "r_ISM": r_val >= self.R_b.value,
        }
        # Wind also includes the core region
        self.masks["r_wind_overall"] = r_val < self.R_TS.value
        logger.debug("Weaver77 kinematics computed")

    def compute_density_profile(self) -> u.Quantity:
        """
        Compute the gas density profile structure.

        Returns
        -------
        astropy.units.Quantity
            Gas number density profile. Units compatible with :py:data:`~saetass.units.NUMBER_DENSITY`.
        """
        if self.model in [BubbleModel.WEAVER77, BubbleModel.MORLINO21]:
            M_dot = self.kwargs["M_dot"]
            rho_0 = self.kwargs["rho_0"]
            t_b = self.kwargs["t_b"]
            R_c = self.kwargs.get("R_c", 0 * u.pc)

            n0 = (rho_0 / const.m_p).to(u.cm**-3).value

            n_c = 0
            if R_c.to(u.pc).value > 0:
                n_c = (
                    (M_dot / (4 * np.pi * R_c**2 * self.v_w) / const.m_p)
                    .to(u.cm**-3)
                    .value
                )

            r_pc = self.r_grid.to(u.pc)

            # Avoid divide-by-zero for r=0
            r_safe = r_pc.copy()
            r_safe[r_safe == 0 * u.pc] = 1e-10 * u.pc

            n_w = (
                (M_dot / (4 * np.pi * r_safe**2 * self.v_w) / const.m_p)
                .to(u.cm**-3)
                .value
            )
            if R_c.to(u.pc).value == 0 and len(n_w) > 1:
                n_w[0] = n_w[1]

            M_bubble = M_dot * t_b
            n_b = (
                (
                    M_bubble
                    / (4 / 3 * np.pi * ((self.R_cd - self.R_TS) ** 3) * const.m_p)
                )
                .to(u.cm**-3)
                .value
            )

            n_shell = 7.02 * n0

            n_profile = np.zeros_like(r_pc.value)

            n_profile[self.masks["r_core"]] = n_c
            n_profile[self.masks["r_wind"]] = n_w[self.masks["r_wind"]]
            n_profile[self.masks["r_hot"]] = n_b
            n_profile[self.masks["r_shell"]] = n_shell
            n_profile[self.masks["r_ISM"]] = n0

            return n_profile * u.cm**-3
        else:
            raise NotImplementedError(
                f"Density profile for {self.model} not implemented."
            )

    def compute_temperature_profile(
        self,
        T_w: u.Quantity = 2e2 * u.K,
        T_ISM: u.Quantity = 10 * u.K,
        T_bubble: u.Quantity = 1e6 * u.K,
        T_shell: u.Quantity = 100 * u.K,
        T_core: u.Quantity = 1e4 * u.K,
    ) -> u.Quantity:
        r"""
        Compute the gas temperature profile.

        Parameters
        ----------
        T_w : astropy.units.Quantity, optional
            Temperature in the wind region. Units compatible with :math:`\mathrm{K}`.
        T_ISM : astropy.units.Quantity, optional
            Temperature in the interstellar medium. Units compatible with :math:`\mathrm{K}`.
        T_bubble : astropy.units.Quantity, optional
            Temperature in the hot bubble. Units compatible with :math:`\mathrm{K}`.
        T_shell : astropy.units.Quantity, optional
            Temperature in the shell. Units compatible with :math:`\mathrm{K}`.
        T_core : astropy.units.Quantity, optional
            Temperature in the core. Units compatible with :math:`\mathrm{K}`.

        Returns
        -------
        astropy.units.Quantity
            Temperature profile. Units compatible with :math:`\mathrm{K}`.
        """
        if self.model in [BubbleModel.WEAVER77, BubbleModel.MORLINO21]:
            T_profile = np.zeros_like(self.r_grid.value)

            T_profile[self.masks["r_core"]] = T_core.to(u.K).value
            T_profile[self.masks["r_wind"]] = T_w.to(u.K).value
            T_profile[self.masks["r_hot"]] = T_bubble.to(u.K).value
            T_profile[self.masks["r_shell"]] = T_shell.to(u.K).value
            T_profile[self.masks["r_ISM"]] = T_ISM.to(u.K).value

            return T_profile * u.K
        else:
            raise NotImplementedError(
                f"Temperature profile for {self.model} not implemented."
            )

    def compute_velocity_profile(self) -> u.Quantity:
        """
        Compute the plasma advection velocity profile.

        Returns
        -------
        astropy.units.Quantity
            Advection velocity profile. Units compatible with :py:data:`~saetass.units.VELOCITY`.
        """
        v_field = np.zeros_like(self.r_grid.value)

        if self.model in [BubbleModel.MORLINO21, BubbleModel.WEAVER77]:
            v_field[self.masks["r_wind_overall"]] = self.v_w.to(u.km / u.s).value

            # v inside bubble = v_w / 4 * (R_TS / r)^2
            r_b_mask = self.masks["r_bubble"]
            r_bubble_pc = self.r_grid[r_b_mask].to(u.pc).value

            if len(r_bubble_pc) > 0:
                v_field[r_b_mask] = (
                    self.v_w.to(u.km / u.s).value
                    / 4
                    * (self.R_TS.to(u.pc).value / r_bubble_pc) ** 2
                )

            return v_field * (u.km / u.s)
        else:
            raise NotImplementedError(
                f"Velocity profile for {self.model} not implemented."
            )

    def compute_magnetic_field_profile(self, eta_B: float = 0.1) -> u.Quantity:
        """
        Compute the magnetic field profile.

        Parameters
        ----------
        eta_B : float, optional
            Magnetic field efficiency parameter, used in the Morlino21 model. Default is ``0.1``.

        Returns
        -------
        astropy.units.Quantity
            Magnetic field profile. Units compatible with :py:data:`~saetass.units.MAGNETIC_FIELD`.
        """
        delta_B = np.zeros_like(self.r_grid.value) * u.G

        if self.model == BubbleModel.MORLINO21:
            M_dot = self.kwargs["M_dot"]

            r_win = self.masks["r_wind_overall"]
            r_bub = self.masks["r_bubble"]

            r_w_cm = self.r_grid[r_win].to(u.cm).value
            # handle r=0
            r_w_cm_safe = r_w_cm.copy()
            if len(r_w_cm_safe) > 0 and r_w_cm_safe[0] == 0:
                if len(r_w_cm_safe) > 1:
                    r_w_cm_safe[0] = r_w_cm_safe[1]
                else:
                    r_w_cm_safe[0] = 1e-10

            val = np.sqrt(
                0.5 * eta_B * M_dot.to(u.g / u.s).value * self.v_w.to(u.cm / u.s).value
            )

            delta_B[r_win] = (1 / r_w_cm_safe * val) * u.G

            delta_B[r_bub] = (np.sqrt(11) / self.R_TS.to(u.cm).value * val) * u.G

            return delta_B
        else:
            raise NotImplementedError(
                f"Magnetic field profile not natively supported for {self.model}."
            )

    def _get_larmor_radius(self, E_k: u.Quantity, B_field: u.Quantity) -> u.Quantity:
        """Calculate Larmor radius given kinetic energy and magnetic field."""
        # Calculate momentum
        E_tot = E_k.to(u.erg) + const.m_p * const.c**2
        p = np.sqrt(E_tot**2 - (const.m_p * const.c**2) ** 2) / const.c

        r_L = np.zeros_like(B_field.value) * u.cm
        mask = B_field.value > 0
        r_L[mask] = (p / (const.e.si * B_field[mask])).to(u.pc)
        return r_L

    def _get_particle_velocity(self, E_k: u.Quantity) -> u.Quantity:
        """Calculate particle velocity from kinetic energy."""
        E_tot = E_k.to(u.erg) + const.m_p * const.c**2
        p = np.sqrt(E_tot**2 - (const.m_p * const.c**2) ** 2) / const.c
        v_p = p * const.c**2 / E_tot
        return v_p

    def compute_diffusion_profile(
        self,
        E_k: u.Quantity,
        r_Inj: u.Quantity = 1.0 * u.pc,
        D_ISM: u.Quantity | None = None,
        diffusion_model: str = "kolmogorov",
        eta_B: float = 0.1,
    ) -> u.Quantity:
        """
        Compute diffusion coefficient profile.

        Parameters
        ----------
        E_k : astropy.units.Quantity
            Particle kinetic energy. Units compatible with :py:data:`~saetass.units.ENERGY`.
        r_Inj : astropy.units.Quantity, optional
            Injection scale for turbulence. Units compatible with :py:data:`~saetass.units.LENGTH`. Default is ``1.0 * u.pc``.
        D_ISM : astropy.units.Quantity, optional
            Diffusion coefficient in the ISM. If ``None``, a default scaling is computed. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`. Default is ``None``.
        diffusion_model : str, optional
            Diffusion model inside the bubble, ``'kolmogorov'``, ``'kraichnan'`` or ``'bohm'``. Default is ``'kolmogorov'``.
        eta_B : float, optional
            Magnetic field efficiency parameter to compute the magnetic field if needed. Default is ``0.1``.

        Returns
        -------
        astropy.units.Quantity
            Diffusion coefficient profile. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`.
        """
        if self.model == BubbleModel.MORLINO21:
            B_field = self.compute_magnetic_field_profile(eta_B=eta_B)
            r_L = self._get_larmor_radius(E_k, B_field)
            v_p = self._get_particle_velocity(E_k)

            match diffusion_model.lower():
                case "bohm":
                    D_values = 1 / 3 * v_p * r_L
                case "kraichnan":
                    D_values = 1 / 3 * v_p * r_L ** (1 / 2) * r_Inj ** (1 / 2)
                case "kolmogorov":
                    D_values = 1 / 3 * v_p * r_L ** (1 / 3) * r_Inj ** (2 / 3)
                case _:
                    raise ValueError(
                        "Invalid diffusion model. Choose 'kolmogorov', 'kraichnan' or 'bohm'."
                    )

            if D_ISM is None:
                D_ISM = (3e28 * (E_k / (1 * u.GeV)) ** (1 / 3)) * u.cm**2 / u.s

            D_values[self.masks["r_ISM"]] = D_ISM
            return D_values.to(u.cm**2 / u.s)
        else:
            raise NotImplementedError(
                f"Diffusion profile not implemented natively for {self.model}."
            )

    def compute_source_term(
        self, E_k: u.Quantity, eta_inj: float = 0.1, Q_amplitude: float = 1000.0
    ) -> np.ndarray:
        """
        Compute source term for cosmic ray injection at the termination shock.

        .. warning::
            This method is a work in progress and may be used with caution.

        Parameters
        ----------
        E_k : u.Quantity
            Particle kinetic energy
        eta_inj : float, optional
            Injection efficiency parameter
        Q_amplitude : float, optional
            Amplitude of the source term

        Returns
        -------
        np.ndarray
            Source term array
        """
        Q = np.zeros_like(self.r_grid.value)
        r_val = self.r_grid.to(u.pc).value
        R_TS_val = self.R_TS.to(u.pc).value

        # Injection at termination shock
        injection_mask = (r_val >= 0.99 * R_TS_val) & (r_val <= 1.01 * R_TS_val)
        Q[injection_mask] = Q_amplitude
        return Q

    def compute_analytical_CR_profile(
        self, D_values: u.Quantity, f_gal: float = 1.0, f_TS: float = 1.0
    ) -> np.ndarray:
        """
        Calculate the analytical steady-state cosmic ray profile
        following the theoretical Morlino21 model equations.

        Parameters
        ----------
        D_values : astropy.units.Quantity
            Diffusion coefficient profile. Units compatible with :py:data:`~saetass.units.DIFFUSION_COEFFICIENT`.
        f_gal : float, optional
            Galactic background level. Default is ``1.0``.
        f_TS : float, optional
            Termination shock level. Default is ``1.0``.

        Returns
        -------
        np.ndarray
            Analytical CR profile
        """
        r_val = self.r_grid.to(u.pc).value
        r_bubble = self.masks["r_bubble"]
        r_ISM = self.masks["r_ISM"]
        r_wind = self.masks["r_wind_overall"]

        v_b = self.v_w.to(u.pc / u.Myr) / 4

        # D inside bubble at mid-point
        if len(D_values[r_bubble]) > 0:
            mid_idx = len(D_values[r_bubble]) // 2
            D_b = D_values[r_bubble][mid_idx].to(u.pc**2 / u.Myr)
        else:
            D_b = D_values[0].to(u.pc**2 / u.Myr)  # Fallback

        if len(D_values[r_ISM]) > 0:
            D_out = D_values[r_ISM][0].to(u.pc**2 / u.Myr)
        else:
            D_out = D_values[-1].to(u.pc**2 / u.Myr)

        R_TS_pc = self.R_TS.to(u.pc).value
        R_b_pc = self.R_b.to(u.pc).value

        # α(r,p)
        alpha = (v_b * self.R_TS / D_b) * (
            1.0 - R_TS_pc / np.where(r_val[r_bubble] == 0, 1e-10, r_val[r_bubble])
        )
        alpha = alpha.decompose().value

        alpha_b = (v_b * self.R_TS / D_b) * (1.0 - R_TS_pc / R_b_pc)
        alpha_b = alpha_b.decompose().value

        # β(p)
        beta = (D_out * self.R_b) / (v_b * self.R_TS**2)
        beta = beta.decompose().value

        EXP_MAX = 700.0
        EXP_MIN = -700.0
        alpha_clip = np.clip(alpha, EXP_MIN, EXP_MAX)
        # alpha_b_clip = np.clip(alpha_b, EXP_MIN, EXP_MAX)

        if np.all(alpha == alpha_clip):  # Normal case
            numerator = (
                np.exp(alpha) + beta * (np.exp(alpha_b) - np.exp(alpha))
            ) + f_gal / f_TS * beta * (np.exp(alpha) - 1.0)
            denominator = 1.0 + beta * (np.exp(alpha_b) - 1.0)
            f_b_over_ts = numerator / denominator

            f_b_over_ts_RB = (
                np.exp(alpha_b) + f_gal / f_TS * beta * (np.exp(alpha_b) - 1.0)
            ) / denominator
        else:  # Extreme case to avoid overflow
            f_b_over_ts = 1 + (1 - beta) / beta * np.exp(alpha - alpha_b)
            f_b_over_ts_RB = 1 / beta

        # f_out(r,p) / f_TS
        f_out_over_ts = f_b_over_ts_RB * (
            R_b_pc / np.where(r_val[r_ISM] == 0, 1e-10, r_val[r_ISM])
        ) + f_gal / f_TS * (
            1.0 - R_TS_pc / np.where(r_val[r_ISM] == 0, 1e-10, r_val[r_ISM])
        )

        f_b = f_b_over_ts
        f_out = f_out_over_ts

        # Compute f_w(r,p) inside wind zone
        f_w = np.zeros_like(r_val[r_wind])
        r_wind_pc = r_val[r_wind]
        D_w = D_values[r_wind].to(u.pc**2 / u.Myr).value

        if len(r_wind_pc) > 1:
            integrand = self.v_w.to(u.pc / u.Myr).value / np.where(D_w == 0, 1e-10, D_w)
            # Use cumulative_trapezoid instead of cumtrapz
            I_r = cumtrapz(integrand, r_wind_pc, initial=0.0)
            I_r = I_r[-1] - I_r
            f_w = f_TS * np.exp(-I_r)
        else:
            f_w = np.array([f_TS]) * np.ones_like(r_wind_pc)

        return np.concatenate([f_w, f_b, f_out])

    def get_all_profiles(self, E_k: u.Quantity, **kwargs) -> dict[str, Any]:
        """
        Compute all profiles and return them in a dictionary.

        Parameters
        ----------
        E_k : astropy.units.Quantity
            Particle kinetic energy. Units compatible with :py:data:`~saetass.units.ENERGY`.
        **kwargs
            Additional parameters (e.g. ``eta_B``, ``diffusion_model``).

        Returns
        -------
        dict of str to astropy.units.Quantity
            Dictionary containing all relevant profiles and parameters.
        """
        res = {
            "r_grid": self.r_grid,
            "R_TS": self.R_TS,
            "R_b": self.R_b,
            "R_cd": self.R_cd,
            "v_w": self.v_w,
            "rho_w": self.rho_w,
            "masks": self.masks,
            "n_gas": self.compute_density_profile(),
            "T_gas": self.compute_temperature_profile(),
            "v_field": self.compute_velocity_profile(),
        }

        if self.model == BubbleModel.MORLINO21:
            eta_B = kwargs.get("eta_B", 0.1)
            eta_inj = kwargs.get("eta_inj", 0.1)
            diffusion_model = kwargs.get("diffusion_model", "kolmogorov")

            res["B_field"] = self.compute_magnetic_field_profile(eta_B=eta_B)
            res["D_values"] = self.compute_diffusion_profile(
                E_k, diffusion_model=diffusion_model, eta_B=eta_B
            )
            res["Q"] = self.compute_source_term(E_k, eta_inj=eta_inj)

        return res
