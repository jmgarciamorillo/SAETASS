from SubproblemSolver import SubproblemSolver
import numpy as np


class LossSolver(SubproblemSolver):
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        P_dot: np.ndarray,
        Q_values: np.ndarray = None,
        is_homogeneous: bool = False,
        **kwargs,
    ):
        """
        Initializes the loss solver with spatial and temporal grids.
        """
        super().__init__(
            x_grid, t_grid, 1, f_values, Q_values, is_homogeneous, **kwargs
        )
        self.P_dot = np.asarray(P_dot)

    def _get_num_timesteps(self) -> int:
        """Returns the number of time steps in the temporal grid."""
        return len(self._t_grid) - 1

    def _compute_problem_specific_coefficients(self, **kwargs) -> np.ndarray:
        """
        Calculates the problem-specific coefficients (d) for Crank-Nicolson.
        """
        # Remove last point to match calculation grid
        if self.is_homogeneous:
            d = -self.P_dot * self.delta_t / self.delta_x
        else:
            # For non-homogeneous grids, use delta_x_array
            d = -self.P_dot * self.delta_t / self.delta_x_array

        return d

    def _compute_problem_matrices(self, d: np.ndarray) -> tuple:
        """
        Constructs the LHS (tildeB) and RHS (B) matrices for Crank-Nicolson.
        """
        num_points = self.num_points

        # Diagonals for B
        main_diag_B = 1 - d
        upper_diag_B = 1 + d[1:]

        # Diagonals for tildeB
        main_diag_B_tilde = 1 + d
        upper_diag_B_tilde = 1 - d[1:]

        # Construct matrices
        B = np.diag(main_diag_B)
        tildeB = np.diag(main_diag_B_tilde)
        for i in range(num_points - 1):
            B[i, i + 1] = upper_diag_B[i]
            tildeB[i, i + 1] = upper_diag_B_tilde[i]

        return tildeB, B

    def _compute_rhs_vector(
        self,
        d: np.ndarray,
        rhs_matrix: np.ndarray,
        current_f_values: np.ndarray,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the right-hand side vector of the linear equation (Ax = b).
        """
        # Q_values must be sized to num_points
        rhs = rhs_matrix @ current_f_values + self.delta_t * (
            self.Q_values[1:] + self.Q_values[:-1]
        )
        return rhs
