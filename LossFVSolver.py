import numpy as np
from typing import Dict, Any
from HyperbolicFVSolver import HyperbolicFVSolver
from State import State
from Grid import Grid

import logging

logger = logging.getLogger(__name__)


class LossFVSolver(HyperbolicFVSolver):
    """
    FV solver for momentum losses using the generalized hyperbolic solver framework.

    This solver handles the equation:
    ∂f/∂t + ∂(P_dot * f)/∂p = 0

    Where P_dot is the momentum loss rate (dp/dt < 0 for losses).

    Key features:
    - Handles momentum space as the main axis
    - Converts user-provided P_dot to generalized velocity
    - Works with both 1D (momentum-only) and 2D (spatial × momentum) grids
    - Supports boundary conditions appropriate for momentum space
    """

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: Dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Initialize the momentum loss solver.

        Parameters:
        -----------
        grid : Grid
            Grid object containing p_centers, p_faces, and optionally r_centers, r_faces
        t_grid : np.ndarray
            Time grid for integration
        params : dict
            Dictionary containing solver parameters:
            - P_dot: Momentum loss rate at cell centers (dp/dt)
            - limiter: Slope limiter ('minmod', 'vanleer', or 'mc')
            - cfl: CFL number for timestep calculation
            - inflow_value_f: Value of f at high-momentum boundary for inflow
            - order: Order of the scheme (1 or 2)
        """
        # Convert momentum loss parameters to general hyperbolic solver format
        loss_params = params.copy()

        # Set momentum axis as the main axis for losses
        loss_params["axis"] = 0

        # Rename loss-specific parameters to match the base class
        if "P_dot" in loss_params:
            loss_params["V_centers"] = self._generalized_velocity(
                loss_params.pop("P_dot"), grid
            )

        if "inflow_value_f" in loss_params:

            loss_params["inflow_value_U"] = self._generalized_variable(
                loss_params.pop("inflow_value_f"), grid
            )[-1]

        # Initialize the base class
        super().__init__(grid, t_grid, loss_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert primitive variable f to generalized variable for logarithmic momentum space.

        In momentum our setup, the generalized variable is defined as: U = p * f

        Parameters:
        -----------
        f : np.ndarray
            Primitive variable values (distribution function)
        grid : Grid
            Grid object containing p_centers

        Returns:
        --------
        np.ndarray
            Conservative variable
        """
        p_centers = grid._p_centers_phys
        return p_centers * f

    def _inverse_generalized_variable(self, U: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert conservative variable U back to primitive variable f = U / p,
        with proper handling of p = 0 singularity, supporting multidimensional U.
        """
        p = grid._p_centers_phys  # 1D array of p values (e.g. radial or angular grid)

        # Check alignment of p with U: find which axis matches length of p
        if p.shape[0] == U.shape[0]:
            # p varies along axis 0
            reshape = (U.shape[0],) + (1,) * (U.ndim - 1)
        elif p.shape[0] == U.shape[-1]:
            # p varies along last axis
            reshape = (1,) * (U.ndim - 1) + (U.shape[-1],)
        else:
            raise ValueError(
                f"Shape mismatch: p_centers has shape {p.shape}, "
                f"but doesn't align with any axis of U with shape {U.shape}"
            )

        p_broadcast = p.reshape(reshape)
        mask = p_broadcast > 0.0

        # Compute f safely
        f = np.zeros_like(U)
        f = np.where(mask, U / p_broadcast, 0.0)

        # Handle singularity by propagating first non-zero value
        if not np.all(mask) and np.any(mask):
            # Project mask along non-p axis to find a non-singular index
            nonzero_idx = (
                np.where(mask)[0][0] if reshape[0] != 1 else np.where(mask)[-1][0]
            )
            f = np.where(
                mask, f, np.take(f, nonzero_idx, axis=np.where(reshape != 1)[0][0])
            )

        return f

    def _generalized_velocity(self, P_dot: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert momentum loss rate P_dot to generalized velocity for the solver.

        Parameters:
        -----------
        P_dot : np.ndarray
            Momentum loss rate at cell centers (dp/dt)
        grid : Grid
            Grid object containing p_centers

        Returns:
        --------
        np.ndarray
            Generalized velocity for the hyperbolic solver
        """
        p_centers = grid._p_centers_phys
        ln10 = np.log(10)
        denom = p_centers * ln10
        if P_dot.ndim == 2:
            len_x = grid.shape[1]
            denom = np.tile(denom[:, None], (1, len_x))  # Expand for 2D grids

        mask = denom > 0.0
        gen_vel = np.zeros_like(P_dot)
        gen_vel[mask] = P_dot[mask] / denom[mask]

        # For p=0, set generalized velocity to zero to avoid singularity
        if not np.all(mask) and np.any(mask):
            gen_vel[~mask] = 0.0

        return gen_vel


# Example usage
if __name__ == "__main__":
    # Create a grid and state for momentum space
    p_centers = np.logspace(0, 3, 100)  # log-spaced momentum grid from 1 to 1000
    p_faces = np.concatenate(
        [
            [0.9 * p_centers[0]],
            np.sqrt(p_centers[:-1] * p_centers[1:]),
            [1.1 * p_centers[-1]],
        ]
    )
    grid = Grid(p_centers=p_centers, p_faces=p_faces)

    # Initial condition (power law distribution)
    f_init = p_centers ** (-2.0)  # p^-2 power law
    state = State(f_init)

    # Loss rate (e.g., synchrotron losses ~ p^2)
    P_dot = -0.01 * p_centers**2  # negative for losses

    # Time grid
    t_grid = np.linspace(0.0, 1.0, 100)

    # Create solver
    params = {
        "P_dot": P_dot,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_f": 0.0,
        "order": 2,
    }

    solver = LossFVSolver(grid=grid, t_grid=t_grid, params=params)

    # Run for 10 steps
    solver.advance(10, state)

    # Visualize results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.loglog(p_centers, f_init, "k--", label="Initial")
    plt.loglog(p_centers, state.f, "r-", label="Final")
    plt.xlabel("Momentum")
    plt.ylabel("f(p)")
    plt.legend()
    plt.title("Momentum Loss Test")
    plt.grid(True)
    plt.show()
