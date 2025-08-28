from SubproblemSolver import SubproblemSolver
import numpy as np

import logging

logger = logging.getLogger(__name__)


class DiffusionSolver(SubproblemSolver):
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        params: dict,
        **kwargs,
    ):
        """Initializes the diffusion solver with spatial and temporal grids."""

        # Extract parameters
        D_values = params.get("D_values", None)
        if D_values is None:
            raise ValueError("D_values parameter is required for DiffusionSolver.")
        Q_values = params.get("Q_values", None)
        logger.debug(
            f"Q_values: min={np.min(Q_values):.4g}, max={np.max(Q_values):.4g}"
        )
        if Q_values is None:
            raise ValueError("Q_values parameter is required for DiffusionSolver.")

        super().__init__(x_grid, t_grid, 1, f_values, Q_values, **kwargs)
        self.delta_r = self.delta_x
        self.r0 = self.domain_start
        if not np.isclose(self.r0, 0.0):
            raise ValueError(
                "DiffusionSolver requires the spatial domain to start at r=0."
            )

        self.f_end = f_values[
            -1
        ]  # Assuming f_values includes also the boundary condition and not only the interior points
        self.D_values = D_values if D_values is not None else np.zeros_like(x_grid)

        # Ensure f_values starts with a non-zero value
        if np.isclose(f_values[0], 0.0):
            f_values[0] = 1e-10  # or any other small value

    def _get_num_timesteps(self) -> int:
        """Returns the number of time steps in the temporal grid."""
        return len(self._t_grid) - 1

    def _compute_problem_specific_coefficients(self, **kwargs) -> tuple:
        """
        Calculates the problem-specific coefficients (q, s) for Crank-Nicolson.
        """

        q = np.zeros(self.num_points)
        s = np.zeros(self.num_points)

        # For r=0 (first point)
        q[0] = 3 * self.D_values[0] * self.delta_t / (self.delta_r**2)
        # print(f"q[0] = {q[0]}")  # Debugging output

        # For interior points
        for i in range(1, self.num_points):
            # D[i+1] and D[i-1] are safe because x_grid_calc excludes boundaries
            q[i] = self.D_values[i] * self.delta_t / (2 * self.delta_r**2)
            s[i] = (
                self.delta_t
                / (4 * self.delta_r)
                * (
                    2 * self.D_values[i] / (self.r0 + i * self.delta_r)
                    + (self.D_values[i + 1] - self.D_values[i - 1]) / (2 * self.delta_r)
                )
            )

            # print(f"q[{i}] = {q[i]}, s[{i}] = {s[i]}")  # Debugging output
        return q, s

    def _compute_problem_matrices(self, problem_coefficients: tuple) -> tuple:
        """
        Constructs the LHS (tildeA) and RHS (A) matrices for Crank-Nicolson.
        """
        q, s = problem_coefficients

        # Diagonals
        Adiag = np.ones(self.num_points) - 2 * q
        Adiag[0] = 1 - q[0]
        tildeAdiag = np.ones(self.num_points) + 2 * q
        tildeAdiag[0] = 1 + q[0]

        # Matrices
        A = np.diag(Adiag)
        tildeA = np.diag(tildeAdiag)

        for i in range(1, self.num_points - 1):
            A[i, i - 1] = q[i] - s[i]
            A[i, i + 1] = s[i] + q[i]
            tildeA[i, i - 1] = s[i] - q[i]
            tildeA[i, i + 1] = -s[i] - q[i]

        # Boundary points
        A[0, 1] = q[0]
        A[-1, -2] = q[-1] - s[-1]
        tildeA[0, 1] = -q[0]
        tildeA[-1, -2] = s[-1] - q[-1]

        # Print matrices for debugging
        # print("A matrix:\n", A)
        # print("tildeA matrix:\n", tildeA)

        return tildeA, A

    def _compute_rhs_vector(
        self,
        problem_coefficients: tuple,
        rhs_matrix: np.ndarray,
        current_f_values: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the right-hand side vector of the linear equation (Ax = b).
        """
        q, s = problem_coefficients

        # Ucc and tildeUcc vectors for boundary conditions
        Ucc = np.zeros(self.num_points)
        tildeUcc = np.zeros(self.num_points)
        Ucc[-1] = (s[-1] + q[-1]) * self.f_end
        tildeUcc[-1] = -(s[-1] + q[-1]) * self.f_end

        sol = (
            rhs_matrix @ current_f_values
            + Ucc
            - tildeUcc
            + self.delta_t * self.Q_values[:-1]
        )
        logger.debug(
            f"Q_values: min={np.min(self.Q_values):.4g}, max={np.max(self.Q_values):.4g}"
        )
        logger.debug(
            f"f_values (after computation): min={np.min(sol):.4g}, max={np.max(sol):.4g}"
        )

        return sol
