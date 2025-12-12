import numpy as np
import inspect

from AdvectionSolver import AdvectionSolver
from DiffusionSolver import DiffusionSolver
from LossSolver import LossSolver

# from AdvectionFVSolver import AdvectionFVSolver
from DiffusionFVSolver import DiffusionFVSolver
from tmpAdvectionFVSolver import AdvectionFVSolver
from LossFVSolver import LossFVSolver
from SourceSolver import SourceSolver
from State import State
from Grid import Grid
from SplittingScheme import StrangSplitting, LieSplitting, create_splitting_scheme
import logging

logger = logging.getLogger(__name__)

# Map operator names to their solver classes
SUBSOLVER_MAP = {
    "advection": AdvectionSolver,
    "diffusion": DiffusionSolver,
    "loss": LossSolver,
    "advectionFV": AdvectionFVSolver,
    "diffusionFV": DiffusionFVSolver,
    "lossFV": LossFVSolver,
    "source": SourceSolver,
}


class Solver:
    def __init__(
        self,
        grid: Grid,
        state: State,
        problem_type: str,
        operator_params: dict = None,
        substeps: dict = None,
        splitting_scheme=None,
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
        self.grid = grid
        self.state = state
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

        self.global_step = 0
        self.total_steps = self.grid.num_timesteps

        # Initialize splitting scheme
        if splitting_scheme is None:
            splitting_scheme = "strang"
            logger.info("No splitting scheme specified; defaulting to 'strang'.")
        self.splitting_scheme = create_splitting_scheme(splitting_scheme)

        # Prepare subsolvers using the mapping
        self.operator_subsolvers = []
        self._initialize_subsolvers(**kwargs)

    def _refined_t_grid(self, n_sub):
        """Return a refined t_grid for n_sub substeps per global step."""
        t_grid = self.grid.t_grid
        num_timesteps = self.total_steps
        t_grid_refined = []
        for i in range(num_timesteps):
            t_start = t_grid[i]
            t_end = t_grid[i + 1]
            t_grid_refined.extend(np.linspace(t_start, t_end, n_sub + 1)[:-1])
        t_grid_refined.append(t_grid[-1])
        return np.array(t_grid_refined)

    def _initialize_subsolvers(self, **kwargs):
        """Initialize subsolvers with appropriate t_grids and parameters."""

        refined_t_grids = self.splitting_scheme.initialize_t_grid(
            self.operator_list, self.substeps_per_op, self.grid.t_grid
        )

        logger.info(
            f"Lengths of refined t_grids: {[len(refined_t_grids[op]) for op in self.operator_list]}"
        )

        for i, op in enumerate(self.operator_list):

            solver_class = SUBSOLVER_MAP[op]
            t_grid_refined = refined_t_grids[op]
            op_params = self.operator_params.get(op, {})

            self.operator_subsolvers.append(
                solver_class(
                    self.grid,
                    t_grid_refined,
                    op_params,
                    **kwargs,
                )
            )

            logger.debug(f"Initialized '{op}' solver with params: {op_params}")

    def _advance(self, n_steps):
        """Advance the solution by n_steps using operator splitting."""

        for _ in range(n_steps):
            self.global_step += 1
            if self.global_step == 505:
                logger.debug("Reached global step 505.")
            logger.info(
                f"Global step {self.global_step}/{self.total_steps} | max(f)={np.max(self.state.f):.4g} min(f)={np.min(self.state.f):.4g}"
            )
            self.splitting_scheme.apply(
                self.operator_list,
                self.operator_subsolvers,
                self.substeps_per_op,
                self.state,
            )
        logger.debug(
            f"Advance finished | max(f)={np.max(self.state.f):.4g} min(f)={np.min(self.state.f):.4g}"
        )

    def run(self):
        num_timesteps = len(self.t_grid) - 1
        self._advance(num_timesteps)
        return self.state

    def step(self, n_steps=1):
        self._advance(n_steps)
        return self.state
