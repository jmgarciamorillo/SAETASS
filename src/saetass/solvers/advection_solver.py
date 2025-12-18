import numpy as np
from typing import Dict, Any
from hyperbolic_solver import HyperbolicFVSolver
from state import State
from grid import Grid

import logging

logger = logging.getLogger(__name__)


class AdvectionFVSolver(HyperbolicFVSolver):
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
            - v_centers: Velocities at cell centers
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

    def _inverse_generalized_variable(self, U: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert conservative variable U back to primitive variable f: f = U / r^2
        Supports ND arrays. Last axis is radial axis. Safe division for r=0.
        """
        self._check_grid_state_consistency(grid, U)

        r = np.asarray(grid.r_centers)
        r_squared = r**2

        # Broadcast along all axes except last
        shape = (1,) * (U.ndim - 1) + r_squared.shape
        r_squared_broadcast = r_squared.reshape(shape)

        # Safe division
        f = np.divide(
            U, r_squared_broadcast, out=np.zeros_like(U), where=r_squared_broadcast != 0
        )

        # Handle singularity at r=0
        if r_squared[0] == 0:
            nonzero_idx = np.argmax(r_squared != 0)
            f[..., 0] = f[..., nonzero_idx]

        return f

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
