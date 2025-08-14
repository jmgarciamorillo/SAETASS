from SubproblemSolver import SubproblemSolver
import numpy as np


class AdvectionSolver(SubproblemSolver):
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        v_field_n: np.ndarray = None,
        v_field_n1: np.ndarray = None,
        Q_values: np.ndarray = None,
        **kwargs,
    ):
        """Initializes the advection solver with spatial and temporal grids."""
        super().__init__(x_grid, t_grid, 1, f_values, Q_values, **kwargs)
        self.v_field_n = v_field_n if v_field_n is not None else np.ones_like(x_grid)
        self.v_field_n1 = v_field_n1 if v_field_n1 is not None else np.ones_like(x_grid)

    def _get_num_timesteps(self) -> int:
        """Returns the number of time steps in the temporal grid."""
        return len(self._t_grid) - 1

    def _compute_problem_specific_coefficients(self, **kwargs) -> tuple:
        """
        Calculates the problem-specific coefficients (a_n, b_n, a_n1, b_n1) for Crank-Nicolson.
        """
        delta_t = self.delta_t
        delta_r = self.delta_x
        num_points = self.num_points
        v_field_n = self.v_field_n
        v_field_n1 = self.v_field_n1

        a_n = np.zeros(num_points)
        b_n = np.zeros(num_points)
        a_n1 = np.zeros(num_points)
        b_n1 = np.zeros(num_points)

        for i in range(num_points):
            a_n[i] = 4 * i**2 * v_field_n[i] * delta_t / ((2 * i - 1) ** 2 * delta_r)
            b_n[i] = 4 * i**2 * v_field_n[i] * delta_t / ((2 * i + 1) ** 2 * delta_r)
            a_n1[i] = 4 * i**2 * v_field_n1[i] * delta_t / ((2 * i - 1) ** 2 * delta_r)
            b_n1[i] = 4 * i**2 * v_field_n1[i] * delta_t / ((2 * i + 1) ** 2 * delta_r)

        return a_n, b_n, a_n1, b_n1

    def _compute_problem_matrices(self, problem_coefficients: tuple) -> tuple:
        """
        Constructs the LHS (tildeC) and RHS (C) matrices for Crank-Nicolson.
        """
        a_n, b_n, a_n1, b_n1 = problem_coefficients

        # Diagonals for C
        main_diag_C = 1 - a_n
        lower_diag_C = 1 + b_n[:-1]

        # Diagonals for tildeC
        main_diag_C_tilde = 1 + a_n1
        lower_diag_C_tilde = 1 - b_n1[:-1]

        # Construct matrices
        C = np.diag(main_diag_C)
        tildeC = np.diag(main_diag_C_tilde)
        for i in range(1, self.num_points):
            C[i, i - 1] = lower_diag_C[i - 1]
            tildeC[i, i - 1] = lower_diag_C_tilde[i - 1]

        return tildeC, C

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
        # Q_values must be sized to num_points
        rhs = rhs_matrix @ current_f_values + self.delta_t * (
            self.Q_values[:-1] + np.append(np.zeros(1), self.Q_values[:-2])
        )
        return rhs

    # Override the parent method to adjust the first value
    def _solve_timestep(
        self, lhs_matrix: np.ndarray, rhs_vector: np.ndarray
    ) -> np.ndarray:
        """
        Solves the linear system Ax = b for the current time step.
        """
        sol = super()._solve_timestep(lhs_matrix, rhs_vector)

        # In-place adjust first value
        sol[0] = sol[1]

        return sol
