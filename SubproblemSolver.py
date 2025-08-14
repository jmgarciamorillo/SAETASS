import numpy as np
from scipy.linalg import solve
from abc import ABC, abstractmethod


class SubproblemSolver(ABC):

    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        _boundary_conditions_type: int,
        f_values: np.ndarray,
        Q_values: np.ndarray = None,
        is_homogeneous: bool = True,
        **kwargs,
    ):
        """Initializes the solver with spatial and temporal grids."""
        self._x_grid = np.asarray(x_grid)
        self._t_grid = np.asarray(t_grid)
        self._boundary_conditions_type = _boundary_conditions_type
        self.is_homogeneous = is_homogeneous

        # Debugging information
        # print("is_homogeneous:", is_homogeneous)

        if self.is_homogeneous:
            self.delta_x, self.domain_start, self.domain_end = self._get_grid_data(
                self._x_grid
            )
        else:
            self.delta_x_array, self.domain_start, self.domain_end = (
                self._get_grid_data_nonhomogeneous(self._x_grid)
            )

        self.delta_t, self.t_start, self.t_end = self._get_grid_data(self._t_grid)

        self._x_grid_calc, self.num_points = self._get_calculation_grid()
        self._num_timesteps = self._get_num_timesteps()

        self._f_values = f_values
        self.Q_values = self._initialize_source_term(Q_values)

    def _get_grid_data(self, grid) -> tuple:
        """Obtains grid parameters."""

        diff_values = np.diff(grid)
        if not np.allclose(diff_values, diff_values[0]):
            raise ValueError("Grid must be homogeneous (uniform spacing).")

        delta = diff_values[0]
        grid_start = grid[0]
        grid_end = grid[-1]

        return delta, grid_start, grid_end

    def _get_grid_data_nonhomogeneous(self, grid) -> tuple:
        """Obtains grid parameters."""

        diff_values = np.diff(grid)
        if np.allclose(diff_values, diff_values[0]):
            raise Warning(
                "Grids is homogeneous (uniform spacing). Change to homogeneous grid for performance."
            )

        delta = diff_values
        grid_start = grid[0]
        grid_end = grid[-1]

        return delta, grid_start, grid_end

    def _get_calculation_grid(self) -> tuple:
        """Gets the number of points in the spatial grid."""

        match self._boundary_conditions_type:
            case 1:  # Boundary condition only at the end of the domain
                x_grid_calc = self._x_grid[:-1]  # Excluding the last point
            case 2:  # Boundary condition at r=0 and at the end of the domain
                x_grid_calc = self._x_grid[1:-1]  # Excluding the first and last points
            case _:
                raise NotImplementedError("Boundary conditions type not implemented.")

        num_points = len(x_grid_calc)

        return x_grid_calc, num_points

    @abstractmethod
    def _get_num_timesteps(self) -> int:
        """Gets the number of points in the spatial grid."""
        pass

    def _initialize_source_term(self, Q_values) -> np.ndarray:
        """Defines the source/sink term Q."""

        if Q_values is None:
            Q = np.zeros(len(self._x_grid_calc))
        else:
            Q = np.asarray(Q_values)

        return Q

    @abstractmethod
    def _compute_problem_specific_coefficients(self, *args, **kwargs) -> tuple:
        """
        Calculates the problem-specific coefficients (e.g., a_n, b_n, s, q, d).
        These depend on the particular differential equation.
        """
        pass

    @abstractmethod
    def _compute_problem_matrices(self, problem_coefficients: tuple) -> tuple:
        """
        Constructs the LHS (left-hand side) and RHS (right-hand side) matrices
        for the Crank-Nicolson linear system.
        """
        pass

    @abstractmethod
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
        pass

    def _solve_timestep(
        self, lhs_matrix: np.ndarray, rhs_vector: np.ndarray
    ) -> np.ndarray:
        """Solves the linear system for the next time step."""
        # Using assume_a="tridiagonal" for performance if the matrix is tridiagonal
        return solve(lhs_matrix, rhs_vector, assume_a="tridiagonal")

    def run_simulation(self, num_timesteps: int) -> np.ndarray:
        """
        Runs the simulation for a specified number of time steps.
        This method handles the core simulation loop without any plotting or data saving.
        """
        f_calclulation = self._f_values[:-1]  # Exclude the last point for calculations

        for n in range(1, num_timesteps + 1):
            # print(f"Time step {n}\n")  # Including a print statement for debugging

            # These methods are specific to each problem and are abstract
            problem_coeffs = self._compute_problem_specific_coefficients()
            lhs_matrix, rhs_matrix = self._compute_problem_matrices(problem_coeffs)
            rhs_vector = self._compute_rhs_vector(
                problem_coeffs, rhs_matrix, f_calclulation
            )

            # This method is common and solves the linear system
            f_calclulation = self._solve_timestep(lhs_matrix, rhs_vector)

        sol = np.append(f_calclulation, self._f_values[-1])  # Append the last point

        return sol
