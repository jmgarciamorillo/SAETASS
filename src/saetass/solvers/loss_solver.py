import logging
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import units as su
from ..grid import Grid
from ..solver import ParamSpec
from .hyperbolic_solver import HyperbolicSolver

logger = logging.getLogger(__name__)


class LossSolver(HyperbolicSolver):
    """
    Finite volume solver for energy losses in momentum space, inheriting from :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

    Solves the momentum-loss equation in conservative form,

    .. math::

        \\frac{\\partial \psi}{\\partial t} + \\frac{\\partial}{\\partial p}\\bigl(\\dot{p}(t,p)\\,\psi\\bigr) = 0,

    where :math:`\\dot{p} = dp/dt \\leq 0` is the (signed) momentum loss rate. The solver uses the conservative variable :math:`U = p \psi` and, also, :math:`V(t,y) = \\frac{\\dot{p}}{p\\ln(10)}` and :math:`y = \\log_{10}(p)`.
    The finite volume update is delegated to the base class across the momentum (p) axis.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``p_centers`` and ``p_faces``; optionally ``r_centers`` and ``r_faces`` for 2D problems.
    t_grid : ndarray
        Subproblem time grid. In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        P_dot : Quantity or callable
            Momentum loss rate, :math:`\\dot{p}`, at cell centers (momentum per time). A callable must have signature ``P_dot(t: Quantity) -> Quantity``.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL number for the adaptive sub-step calculation.
        inflow_value_psi : Quantity, optional
            Differential density :math:`\\psi` at the high-momentum boundary, used as an inflow condition when :math:`\\dot{p} > 0` (i.e. momentum gain).
        inflow_value_U : Quantity, optional
            Same inflow condition given directly for the conservative variable :math:`U = p \\psi` (momentum times differential density). Mutually exclusive with ``inflow_value_psi``.
        order : ``{1, 2}``
            Order of the numerical scheme.
        adiabatic_losses : bool
            If ``True``, include adiabatic losses. The key ``v_centers_physical`` must also be supplied.
        v_centers_physical : Quantity, optional
            Physical advection velocity at cell centres (velocity); required when ``adiabatic_losses`` is ``True``.
    """

    PARAM_SPECS = MappingProxyType(
        {
            **HyperbolicSolver.PARAM_SPECS,
            "P_dot": ParamSpec(su.MOMENTUM_LOSS_RATE, dynamic=True),
            "inflow_value_psi": ParamSpec(su.PSI_P),
            "inflow_value_U": ParamSpec(su.MOMENTUM * su.PSI_P),
            "adiabatic_losses": ParamSpec(),
            "v_centers_physical": ParamSpec(su.VELOCITY),
        }
    )

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initialize the loss solver."""
        self._validate_unit_free_inputs(t_grid, params)
        if grid.p_centers_phys is None:
            raise ValueError("LossSolver requires a Grid with a momentum axis.")
        self.p_centers_phys = grid.p_centers_phys.to_value(su.MOMENTUM)

        # Convert momentum loss parameters to general hyperbolic solver format
        loss_params = params.copy()

        # Set momentum axis as the main axis for losses
        loss_params["axis"] = 0

        # Add adiabatic losses
        v_centers_physical = loss_params.pop("v_centers_physical", None)
        if loss_params.pop("adiabatic_losses", False):
            if v_centers_physical is None:
                raise ValueError(
                    "If adiabatic_losses is True, v_centers_physical must be provided."
                )
            self.P_dot_adiabatic = self._adiabatic_losses(grid, v_centers_physical)

        # Rename loss-specific parameters to match the base class
        if "P_dot" in loss_params:
            P_dot_input = loss_params.pop("P_dot")
            if callable(P_dot_input):

                def dynamic_V_centers(t):
                    return self._generalized_velocity(P_dot_input(t), grid)

                loss_params["V_centers"] = dynamic_V_centers
            else:
                loss_params["V_centers"] = self._generalized_velocity(P_dot_input, grid)

        if "inflow_value_psi" in loss_params:
            if "inflow_value_U" in loss_params:
                raise ValueError(
                    "Provide either inflow_value_psi or inflow_value_U, not both."
                )
            loss_params["inflow_value_U"] = self._generalized_variable(
                loss_params.pop("inflow_value_psi"), grid
            )[-1]

        # Initialize the base class
        super().__init__(grid, t_grid, loss_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the primitive differential density to the conservative variable.
        """
        p_src = getattr(grid, "_p_centers_phys", None)
        if p_src is None:
            p_src = self.p_centers_phys
        p_centers = np.asarray(p_src, dtype=float)
        return p_centers * f

    def _inverse_generalized_variable(self, U: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the conservative variable back to the primitive differential density.
        """
        p_src = getattr(grid, "_p_centers_phys", None)
        if p_src is None:
            p_src = self.p_centers_phys
        p = np.asarray(p_src, dtype=float).flatten()

        # 1D CASE
        if U.ndim == 1:
            if U.shape[0] != p.shape[0]:
                raise ValueError(f"Shape mismatch: U {U.shape}, p {p.shape}")
            mask = p > 0
            f = np.zeros_like(U)
            f[mask] = U[mask] / p[mask]
            if mask.sum() < len(p):
                raise ValueError(
                    "p contains non-positive values, cannot divide by zero."
                )
            return f

        # 2D CASE
        elif U.ndim == 2:
            # U: (n_r, n_p), p: (n_p,)
            if U.shape[1] != p.shape[0]:
                raise ValueError(
                    f"Expected U.shape[1] == p.shape[0], got U {U.shape}, p {p.shape}"
                )

            mask = p > 0.0
            f = np.zeros_like(U)
            # Only divide columns where p > 0
            f[:, mask] = U[:, mask] / p[mask]
            if mask.sum() < len(p):
                raise ValueError(
                    "p contains non-positive values, cannot divide by zero."
                )
            return f

        else:
            raise ValueError(
                "inverse_generalized_variable only supports 1D or 2D arrays."
            )

    def _generalized_velocity(self, P_dot: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert the physical momentum loss rate to the generalized velocity used by the base-class finite-volume update.
        """
        P_dot_arr = np.asarray(P_dot, dtype=float)
        p_src = getattr(grid, "_p_centers_phys", None)
        if p_src is None:
            p_src = self.p_centers_phys
        p_centers = np.asarray(p_src, dtype=float)
        ln10 = np.log(10)
        denom = p_centers * ln10
        if P_dot_arr.ndim == 2:
            len_x = grid.shape[1]
            denom = np.tile(denom[:, None], (1, len_x))  # Expand for 2D grids

        mask = denom > 0.0
        gen_vel = np.zeros_like(P_dot_arr)
        gen_vel[mask] = P_dot_arr[mask] / denom[mask]

        # For p=0, set generalized velocity to zero to avoid singularity
        if not np.all(mask) and np.any(mask):
            gen_vel[~mask] = 0.0

        return gen_vel

    def _adiabatic_losses(
        self, grid: Grid, v_centers_physical: np.ndarray
    ) -> np.ndarray:
        """
        Compute the adiabatic momentum loss rate due to spherical expansion.

        The adiabatic loss term arises from the divergence of the advection velocity field and is given by

        .. math::

            \\dot{p}_{\\text{ad}} = -\\frac{p}{3}\\,\\nabla \\cdot \\mathbf{v},

        evaluated cell-by-cell on the spatial grid via a finite-difference approximation of the radial flux divergence.
        """
        r_faces = grid.r_faces.to_value(su.LENGTH)
        A_face = 4.0 * np.pi * r_faces
        V = (4.0 / 3.0) * np.pi * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)

        # TEMPORARY SOLUTION
        N = len(grid.r_centers)
        v_faces = np.zeros((len(self.p_centers_phys), N + 1))
        v_faces[:, 1:N] = 0.5 * (v_centers_physical[:, :-1] - v_centers_physical[:, 1:])
        v_faces[:, 0] = v_centers_physical[:, 0]
        v_faces[:, -1] = v_centers_physical[:, -1]

        Phi = A_face * v_faces
        div = (Phi[:, 1:] - Phi[:, :-1]) / V

        return (-self.p_centers_phys * div.T).T / 3.0 * 0
