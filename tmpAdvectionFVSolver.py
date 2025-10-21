import numpy as np
from typing import Dict, Any
from HyperbolicFVSolver import HyperbolicFVSolver
from State import State
from Grid import Grid

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
        hyperbolic_params["index"] = 0

        # Rename advection-specific parameters to match the base class
        if "v_centers" in hyperbolic_params:
            hyperbolic_params["V_centers"] = hyperbolic_params.pop("v_centers")

        if "inflow_value_U" in hyperbolic_params:
            hyperbolic_params["inflow_value_U"] = hyperbolic_params.pop(
                "inflow_value_U"
            )

        # Initialize the base class
        super().__init__(grid, t_grid, hyperbolic_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, r_centers: np.ndarray) -> np.ndarray:
        """
        Convert primitive variable f to conservative variable U = r^2 * f.

        This is specific to spherical advection where the conservative variable
        accounts for geometric factors.

        Parameters:
        -----------
        f : np.ndarray
            Primitive variable values (density/distribution function)
        r_centers : np.ndarray
            Cell center positions (radii)

        Returns:
        --------
        np.ndarray
            Conservative variable U = r^2 * f
        """
        return r_centers**2 * f

    def _inverse_generalized_variable(
        self, U: np.ndarray, r_centers: np.ndarray
    ) -> np.ndarray:
        """
        Convert conservative variable U back to primitive variable f = U / r^2.

        Parameters:
        -----------
        U : np.ndarray
            Conservative variable values
        r_centers : np.ndarray
            Cell center positions (radii)

        Returns:
        --------
        np.ndarray
            Primitive variable f = U / r^2
        """
        # Avoid division by zero at r=0
        denom = r_centers**2
        mask = denom > 0.0
        f = np.zeros_like(U)
        f[mask] = U[mask] / denom[mask]

        # For r=0, use the neighboring value to avoid singularity
        if not np.all(mask) and np.any(mask):
            first_nonzero = np.where(mask)[0][0]
            f[~mask] = f[first_nonzero]

        return f


# Example usage
if __name__ == "__main__":
    # Create a grid and state
    r_centers = np.linspace(0.0, 1.0, 100)
    r_faces = np.concatenate([[0.0], 0.5 * (r_centers[:-1] + r_centers[1:]), [1.0]])
    grid = Grid(r_centers=r_centers, r_faces=r_faces)

    # Initial condition
    f_init = np.zeros_like(r_centers)
    f_init[(r_centers > 0.4) & (r_centers < 0.6)] = 1.0
    state = State(f_init)

    # Velocity field (constant)
    v_centers = np.ones_like(r_centers)

    # Time grid
    t_grid = np.linspace(0.0, 0.5, 100)

    # Create solver
    params = {
        "v_centers": v_centers,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_U": 0.0,
        "order": 2,
    }

    solver = AdvectionFVSolver(grid=grid, t_grid=t_grid, params=params)

    # Run for 10 steps
    solver.advance(10, state)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(r_centers, f_init, "k--", label="Initial")
    plt.plot(r_centers, state.get_f(), "r-", label="Final")
    plt.xlabel("Radius")
    plt.ylabel("f")
    plt.legend()
    plt.title("Advection Test")
    plt.grid(True)
    plt.show()
