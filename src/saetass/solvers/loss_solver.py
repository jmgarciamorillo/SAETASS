import numpy as np
from typing import Dict, Any
from hyperbolic_solver import HyperbolicFVSolver
from state import State
from grid import Grid

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

        # Add adiabatic losses
        if loss_params["adiabatic_losses"] == True:
            try:
                v_centers_physical = loss_params.pop("v_centers_physical")
            except:
                raise ValueError(
                    "If adiabatic_losses is True, v_centers_physical must be provided."
                )
            self.P_dot_adiabatic = self._adiabatic_losses(grid, v_centers_physical)

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
        handling both 1D and 2D cases where p corresponds to the momentum coordinate.

        Parameters
        ----------
        U : np.ndarray
            Conservative variable array.
            - 1D case: shape (N,) or (N,1) or (1,N)
            - 2D case: shape (Np, Nx), where Np is number of momentum points,
            and Nx number of spatial points.
        p : np.ndarray
            1D array of momentum values of length Np.

        Returns
        -------
        f : np.ndarray
            Primitive variable array with same shape as U.
        """
        p = np.asarray(grid._p_centers_phys).flatten()

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
            # Solo dividimos columnas donde p > 0
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

    def _adiabatic_losses(
        self, grid: Grid, v_centers_physical: np.ndarray
    ) -> np.ndarray:

        r_faces = grid.r_faces
        A_face = 4.0 * np.pi * r_faces
        V = (4.0 / 3.0) * np.pi * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)

        # TEMPORARY SOLUTION
        N = len(grid.r_centers)
        v_faces = np.zeros((len(grid._p_centers_phys), N + 1))
        v_faces[:, 1:N] = 0.5 * (v_centers_physical[:, :-1] - v_centers_physical[:, 1:])
        v_faces[:, 0] = v_centers_physical[:, 0]
        v_faces[:, -1] = v_centers_physical[:, -1]

        Phi = A_face * v_faces
        div = (Phi[:, 1:] - Phi[:, :-1]) / V

        return (-grid._p_centers_phys * div.T).T / 3.0 * 0


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
        "cfl": 0.4,
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
