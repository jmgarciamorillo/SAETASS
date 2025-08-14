import numpy as np

from AdvectionSolver import AdvectionSolver
from DiffusionSolver import DiffusionSolver
from LossSolver import LossSolver

# Map operator names to their solver classes
SUBSOLVER_MAP = {
    "advection": AdvectionSolver,
    "diffusion": DiffusionSolver,
    "loss": LossSolver,
}


class Solver:
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        problem_type: str,
        Q_values: np.ndarray = None,
        advection_params: dict = None,
        diffusion_params: dict = None,
        loss_params: dict = None,
        substeps: dict = None,
        **kwargs,
    ):
        """
        Initializes the main Solver with operator splitting.
        problem_type: str, e.g. 'advection-diffusion', 'loss', etc.
        substeps: dict, e.g. {'advection': 1, 'diffusion': 1, 'loss': 1}
        """
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.f_values = np.copy(f_values)
        self.problem_type = problem_type.lower()
        self.Q_values = Q_values if Q_values is not None else np.zeros_like(x_grid)
        self.params = {
            "advection": advection_params or {},
            "diffusion": diffusion_params or {},
            "loss": loss_params or {},
        }
        self.substeps = substeps or {}

        # Determine which operators are present
        self.operators = [op for op in SUBSOLVER_MAP if op in self.problem_type]
        self.n_os = len(self.operators)

        if self.n_os == 0:
            raise ValueError("No valid operators specified in problem_type.")

        # Split Q_values equally among operators
        self.Q_split = self.Q_values / self.n_os

        # Substeps for each operator (default 1)
        self.substeps_per_op = {op: self.substeps.get(op, 1) for op in self.operators}

        # Prepare subsolvers using the mapping
        self.subsolvers = []
        self._initialize_subsolvers(**kwargs)

    def _refined_t_grid(self, t_grid, n_sub):
        """Return a refined t_grid for n_sub substeps per global step."""
        num_timesteps = len(t_grid) - 1
        t_grid_refined = []
        for i in range(num_timesteps):
            t_start = t_grid[i]
            t_end = t_grid[i + 1]
            t_grid_refined.extend(np.linspace(t_start, t_end, n_sub + 1)[:-1])
        t_grid_refined.append(t_grid[-1])
        return np.array(t_grid_refined)

    def _initialize_subsolvers(self, **kwargs):
        """Initialize subsolvers with appropriate t_grids and parameters."""
        for op in self.operators:
            solver_class = SUBSOLVER_MAP[op]
            n_sub = self.substeps_per_op[op]
            if n_sub > 1:
                t_grid_refined = self._refined_t_grid(self.t_grid, n_sub)
            else:
                t_grid_refined = self.t_grid

            self.subsolvers.append(
                solver_class(
                    self.x_grid,
                    t_grid_refined,
                    self.f_values,
                    Q_values=self.Q_split,
                    **self.params[op],
                    **kwargs,
                )
            )

    def _advance(self, f_start, n_steps):
        """
        Advance the solution by n_steps from f_start.
        Returns the new state.
        """
        f_current = np.copy(f_start)
        for _ in range(n_steps):
            for i, op in enumerate(self.operators):
                subsolver = self.subsolvers[i]
                n_sub = self.substeps_per_op[op]
                subsolver._f_values = np.copy(f_current)
                f_current = subsolver.run_simulation(n_sub)
        return f_current

    def run(self):
        """
        Runs the operator splitting simulation.
        Returns the solution at the final time step.
        """
        num_timesteps = len(self.t_grid) - 1
        f_current = self._advance(self.f_values, num_timesteps)
        return f_current

    def step(self, n_steps=1):
        """
        Advance the solution by n_steps from the current state.
        Updates self.f_values in-place.
        Returns the new state.
        """
        f_current = self._advance(self.f_values, n_steps)
        self.f_values = np.copy(f_current)
        return f_current
