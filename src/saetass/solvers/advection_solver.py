import numpy as np
from typing import Dict, Any
from .hyperbolic_solver import HyperbolicSolver
from ..state import State
from ..grid import Grid

import logging

logger = logging.getLogger(__name__)


class AdvectionSolver(HyperbolicSolver):
    """
    FV solver for spherical advection using W = r^2 u (reduced variable).
    Inherits core logic from HyperbolicFVSolver.

    Supports both 1D (spatial-only) and 2D (spatial × momentum) grids.
    When p_centers is present, processes each momentum slice independently.

    Key features:
      - Handles spherical advection with proper geometric factors
      - Solves using conservative variable W = r^2 * f
      - Implements upwind fluxes with slope limiting for second order
      - Supports non-uniform grids
    """

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: Dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Initialize the advection solver.

        Parameters:
        -----------
        grid : Grid
            Grid object containing r_centers, r_faces, and optionally p_centers, p_faces
        t_grid : np.ndarray
            Time grid for integration
        params : dict
            Dictionary containing solver parameters:
            - v_centers: Velocities at cell centers (array or callable v_centers(t) -> array)
            - limiter: Slope limiter ('minmod', 'vanleer', or 'mc')
            - cfl: CFL number for timestep calculation
            - inflow_value_U: Value of U at the outer boundary for inflow
            - order: Order of the scheme (1 or 2)
        """
        # Convert advection-specific parameters to general hyperbolic solver format
        hyperbolic_params = params.copy()

        # Set spatial axis (r) as the main axis for advection
        hyperbolic_params["axis"] = 1

        # Rename advection-specific parameters to match the base class
        if "v_centers" in hyperbolic_params:
            hyperbolic_params["V_centers"] = hyperbolic_params.pop("v_centers")

        if "inflow_value_U" in hyperbolic_params:
            hyperbolic_params["inflow_value_U"] = hyperbolic_params.pop(
                "inflow_value_U"
            )

        # Initialize the base class
        super().__init__(grid, t_grid, hyperbolic_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert a primitive variable f to a conservative variable U for spherical geometry:

            U = r^2 * f

        Supports 1D or ND arrays. Broadcasting occurs along the last axis,
        which corresponds to the radial coordinate in the grid.

        Parameters
        ----------
        f : np.ndarray
            Primitive variable (density, distribution function, etc.)
            Shape (..., nr), where nr = len(grid.r_centers)
        grid : Grid
            Grid object containing `r_centers`.

        Returns
        -------
        np.ndarray
            Conservative variable of same shape as f.
        """
        self._check_grid_state_consistency(grid, f)

        r = np.asarray(grid.r_centers)
        return f * r**2  # broadcasting automatically handles ND arrays

    def _inverse_generalized_variable(
        self,
        U: np.ndarray,
        grid: Grid,
    ) -> np.ndarray:
        """
        Convert conservative variable U back to primitive variable f = U / r².
        Supports ND arrays. Last axis is radial axis.

        At the origin cell (r[0] = 0), U[0] = 0 universally.
        f[0] is reconstructed by a 2nd-order backward linear extrapolation
        from the FVM-updated neighbours:
            f[0] ≈ 2·f[1] − f[2]  +  O(Δr²)
        """
        self._check_grid_state_consistency(grid, U)

        r = np.asarray(grid.r_centers)
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
        Overrides HyperbolicSolver's slope computation to correctly handle the
        geometric U = r² f transformation. Evaluates slopes on primitive
        variable f to avoid massive O(1) clipping from standard limiters on
        the quadratic U=r² profile, mapping back via the Product Rule.
        """
        # 1. Recover mathematically exact primitive profile
        f = self._inverse_generalized_variable(U, self.grid)

        # 2. Compute well-behaved limited slopes on the flat/linear primitive f
        slopes_f = super()._compute_slopes(f)

        # 3. Map back to conservative variable slopes: dU/dr = 2rf + r^2(df/dr)
        r = np.asarray(self.grid.r_centers)
        shape = (1,) * (U.ndim - 1) + r.shape
        r_b = r.reshape(shape)

        return 2.0 * r_b * f + (r_b**2) * slopes_f

    def _check_grid_state_consistency(
        self, grid: Grid, state_array: np.ndarray
    ) -> None:
        """
        Check that the last dimension of state_array matches the radial grid length.
        """
        if not grid.is_compatible_array(state_array):
            raise ValueError(
                f"Grid of shape {grid.shape} is not compatible with state array of shape {state_array.shape}."
            )
