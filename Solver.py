import numpy as np
import inspect

from AdvectionSolver import AdvectionSolver
from DiffusionSolver import DiffusionSolver
from LossSolver import LossSolver
from AdvectionFVSolverv2 import AdvectionFVSolver

import logging

logger = logging.getLogger(__name__)

# Map operator names to their solver classes
SUBSOLVER_MAP = {
    "advection": AdvectionSolver,
    "diffusion": DiffusionSolver,
    "loss": LossSolver,
    "advectionFV": AdvectionFVSolver,
    "source": None,  # Placeholder for future SourceSolver
}


class Solver:
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        problem_type: str,
        operator_params: dict = None,
        substeps: dict = None,
        **kwargs,
    ):
        """
        Initializes the main Solver with operator splitting.
        problem_type: str, e.g. 'advection-diffusion', 'loss', etc.
        substeps: dict, e.g. {'advection': 1, 'diffusion': 1, 'loss': 1}
        operator_params: dict, e.g. {
            'advection': {...}, 'diffusion': {...}, 'loss': {...}, 'advectionFV': {...}
        }
        """
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.f_values = np.copy(f_values)
        self.problem_type = problem_type.lower()

        # Store operator parameters (nested dict)
        self.operator_params = operator_params or {}
        self.substeps = substeps or {}

        # Determine which operators are present and print for debugging
        logger.debug(f"Problem type: {self.problem_type}")
        logger.debug(f"Available operators: {list(SUBSOLVER_MAP.keys())}")

        # Parse operators from problem_type, handling exact operator names split by '-'
        self.operator_list = []
        ops_in_type = [op.strip().lower() for op in self.problem_type.split("-")]
        for op in SUBSOLVER_MAP:
            if op.lower() in ops_in_type:
                self.operator_list.append(op)
        logger.debug(f"Using operators: {self.operator_list}")
        self.n_os = len(self.operator_list)

        if self.n_os == 0:
            raise ValueError("No valid operators specified in problem_type.")

        # Substeps for each operator (default 1)
        self.substeps_per_op = {
            op: self.substeps.get(op, 1) for op in self.operator_list
        }

        # Prepare subsolvers using the mapping
        self.operator_subsolvers = []
        self._initialize_subsolvers(**kwargs)

        self.global_step = 0
        self.total_steps = len(self.t_grid) - 1

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
        for op in self.operator_list:
            solver_class = SUBSOLVER_MAP[op]

            # Refine t_grid according to substeps
            n_sub = self.substeps_per_op[op]
            if n_sub > 1:
                t_grid_refined = self._refined_t_grid(self.t_grid, n_sub)
            else:
                t_grid_refined = self.t_grid

            op_params = self.operator_params.get(op, {})

            self.operator_subsolvers.append(
                solver_class(
                    self.x_grid,
                    t_grid_refined,
                    self.f_values,
                    op_params,
                    **kwargs,
                )
            )

            logger.debug(f"Initialized '{op}' solver with params: {op_params}")

    def _advance(self, f_start, n_steps):
        """Advance the solution by n_steps using operator splitting."""

        f_current = np.copy(f_start)

        for step_idx in range(n_steps):
            self.global_step += 1
            logger.debug(
                f"Global step {self.global_step}/{self.total_steps} | max(f)={np.max(f_current):.4g} min(f)={np.min(f_current):.4g}"
            )
            for i, op in enumerate(self.operator_list):
                subsolver = self.operator_subsolvers[i]
                n_sub = self.substeps_per_op[op]
                logger.debug(f"Operation '{op}' with {n_sub} substeps")
                subsolver.f_values = np.copy(f_current)
                f_current = subsolver.advance(n_sub)
        logger.debug(
            f"Advance finished | max(f)={np.max(f_current):.4g} min(f)={np.min(f_current):.4g}"
        )
        return f_current

    def run(self):
        num_timesteps = len(self.t_grid) - 1
        f_current = self._advance(self.f_values, num_timesteps)
        return f_current

    def step(self, n_steps=1):
        f_current = self._advance(self.f_values, n_steps)
        self.f_values = np.copy(f_current)
        return f_current
