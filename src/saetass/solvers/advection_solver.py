import logging
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import units as su
from ..grid import Grid
from ..solver import ParamSpec
from .hyperbolic_solver import HyperbolicSolver

logger = logging.getLogger(__name__)


class AdvectionSolver(HyperbolicSolver):
    """
    Finite volume solver for spherical advection, inheriting from :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

    Solves the spherical advection equation in conservative form,

    .. math::

        \\frac{\\partial \psi}{\\partial t} + \\frac{1}{r^2}\\frac{\\partial}{\\partial r}\\bigl(v(t,r)\\,r^2 \psi\\bigr) = 0,

    by introducing the conservative variable :math:`U = r^2 \psi`, :math:`V(t,y) = v(t,r)` and :math:`y = r`, and delegating the finite volume update to the base class across the spatial (:math:`r`) axis.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``r_centers`` and ``r_faces``; optionally ``p_centers`` and ``p_faces`` for 2D problems.
    t_grid : ndarray
        Subproblem time grid. In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration, already converted to canonical floats by :py:meth:`~saetass.solver.SubSolver.convert_params`. Accepted keys (and the units required at the :py:class:`~saetass.solver.Solver` level) are:

        v_centers : Quantity or callable
            Advection velocity at cell centers (velocity). A callable must have signature ``v_centers(t: Quantity) -> Quantity``.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL number for the adaptive sub-step calculation.
        inflow_value_U : Quantity
            Value of the conservative variable :math:`U = r^2 \\psi` at the outer boundary when the flow is directed inward (inflow condition), in units of area times differential density.
        order : ``{1, 2}``
            Order of the numerical scheme.
    """

    PARAM_SPECS = MappingProxyType(
        {
            **HyperbolicSolver.PARAM_SPECS,
            "v_centers": ParamSpec(su.VELOCITY, dynamic=True),
            "inflow_value_U": ParamSpec(su.AREA * su.PSI_P),
        }
    )

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initialize the advection solver."""
        self._validate_unit_free_inputs(t_grid, params)
        # Convert advection-specific parameters to general hyperbolic solver format
        hyperbolic_params = params.copy()

        # Set spatial axis (r) as the main axis for advection
        hyperbolic_params["axis"] = 1

        if "v_centers" in hyperbolic_params:
            hyperbolic_params["V_centers"] = hyperbolic_params.pop("v_centers")

        # Initialize the base class
        super().__init__(grid, t_grid, hyperbolic_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the primitive differential density to the conservative variable.
        """
        self._check_grid_state_consistency(grid, f)

        r = self.r_centers
        return f * r**2  # broadcasting automatically handles ND arrays

    def _inverse_generalized_variable(
        self,
        U: np.ndarray,
        grid: Grid,
    ) -> np.ndarray:
        """
        Map the conservative variable back to the primitive differential density.
        """
        self._check_grid_state_consistency(grid, U)

        r = self.r_centers
        r_squared = r**2

        # Broadcast r² along all axes except the last (radial) axis
        shape = (1,) * (U.ndim - 1) + r_squared.shape
        r_sq_b = r_squared.reshape(shape)

        # Safe inverse transform for all cells with r > 0
        f = np.divide(U, r_sq_b, out=np.zeros_like(U), where=r_sq_b != 0)

        # Handle the singular origin cell
        if r_squared[0] == 0:
            # Reconstruct origin from FVM-updated neighbours
            f[..., 0] = 2.0 * f[..., 1] - f[..., 2]

        return f

    def _compute_slopes(self, U: np.ndarray) -> np.ndarray:
        """
        Compute limited slopes in conservative-variable space, accounting for the spherical-geometry transformation.

        Overrides parent method to avoid excessive limiter clipping on the quadratic r^2 profile.
        """
        # 1. Recover mathematically exact primitive profile
        f = self._inverse_generalized_variable(U, self.grid)

        # 2. Compute well-behaved limited slopes on the flat/linear primitive f
        slopes_f = super()._compute_slopes(f)

        # 3. Map back to conservative variable slopes: dU/dr = 2rf + r^2(df/dr)
        r = self.r_centers
        shape = (1,) * (U.ndim - 1) + r.shape
        r_b = r.reshape(shape)

        return 2.0 * r_b * f + (r_b**2) * slopes_f

    def _check_grid_state_consistency(
        self, grid: Grid, state_array: np.ndarray
    ) -> None:
        """
        Verify that state is dimensionally compatible with grid.
        """
        if not grid.is_compatible_array(state_array):
            raise ValueError(
                f"Grid of shape {grid.shape} is not compatible with state array of shape {state_array.shape}."
            )
