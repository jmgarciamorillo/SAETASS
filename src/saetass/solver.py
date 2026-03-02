"""
The Solver class is the main interface for running simulations.
It manages the overall time loop and coordinates the advancement of the solution according to the specified problem type and parameters.
Hence, it serves as the central orchestrator of the simulation workflow, while delegating the actual numerical updates to specialized subsolvers for each operator (advection, diffusion, loss, source).

In the initialization phase, the Solver class takes in the grid, initial state, problem type, operator parameters, substep counts and splitting scheme.
It parses the problem type to determine which operators are involved and initializes the corresponding subsolvers with their own refined time grids based on the specified splitting scheme.

Any object of the Solver class exposes two main methods: one for advancing the solution by a single time step and another for running the entire simulation.
After each advancement, the Solver updates the State object associated with the solution.

--------------
"""

import numpy as np

from .solvers.diffusion_solver import DiffusionSolver
from .solvers.advection_solver import AdvectionSolver
from .solvers.loss_solver import LossSolver
from .solvers.source_solver import SourceSolver
from .state import State
from .grid import Grid
from .splitting import StrangSplitting, LieSplitting, create_splitting_scheme
import logging

logger = logging.getLogger(__name__)

# Map operator names to their solver classes
SUBSOLVER_MAP = {
    "advection": AdvectionSolver,
    "diffusion": DiffusionSolver,
    "loss": LossSolver,
    "source": SourceSolver,
}


class Solver:
    """
    Main class to manage the simulation workflow. It initializes subsolvers based on the specified problem type and coordinates the time advancement of the solution.

    Parameters
    ----------
    grid : Grid
        Grid object containing spatial and momentum grids and time grid.
    state : State
        State object containing the initial distribution f.
    problem_type : str
        String specifying the type of problem and which operators to include (e.g., "advection-diffusion-loss-source").
    operator_params : dict, optional
        Dictionary mapping operator names to their specific parameters (e.g., {"advection": {...}, "diffusion": {...}}).
        These parameters will be passed to the corresponding subsolvers during initialization and should be structured accordingly.
        For further details on expected parameters for each operator, refer to the documentation of the respective subsolver classes.
    substeps : dict, optional
        Dictionary specifying the number of substeps for each operator (e.g., {"advection": 2, "diffusion": 1}).
        Default is no subrefinement, this is, 1 substep per operator.
    splitting_scheme : str, optional
        String specifying the operator splitting scheme to use (e.g., "strang", "lie"). Default is "strang".
    """

    def __init__(
        self,
        grid: Grid,
        state: State,
        problem_type: str,
        operator_params: dict = None,
        substeps: dict = None,
        splitting_scheme: str = None,
        **kwargs,
    ):
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
            if self.global_step == 371:
                logger.debug("Reached global step 371.")
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

    def run(self) -> State:
        """Advances the solution to the final time, updating and returning the final :class:`State` object.

        Returns
        -------
        State
            The final state of the solution after advancing to the final time.
        """
        num_timesteps = len(self.t_grid) - 1
        self._advance(num_timesteps)
        return self.state

    def step(self, n_steps=1):
        """Advances the solution by n_steps global time steps, updating and returning the current :class:`State` object.

        Parameters
        ----------
        n_steps : int, optional
            The number of global time steps to advance. Default is 1.
        """
        self._advance(n_steps)
        return self.state
