"""
The solver module contains the main interface for running simulations.

It defines the public API that all physical operators must expose through the abstract class :py:class:`~saetass.solver.SubSolver`.

It also defines the :py:class:`~saetass.solver.Solver` class, which manages the overall time loop and coordinates the advancement of the solution according to the specified problem type and parameters.
Hence, it serves as the central orchestrator of the simulation workflow, while delegating the actual numerical updates to the specialized operators (:py:class:`~saetass.solvers.advection_solver.AdvectionSolver`, :py:class:`~saetass.solvers.diffusion_solver.DiffusionSolver`, :py:class:`~saetass.solvers.loss_solver.LossSolver`, :py:class:`~saetass.solvers.source_solver.SourceSolver`...).

--------------
"""

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import numpy as np

from .cli.banner import print_banner
from .cli.progress import create_progress_bar
from .grid import Grid
from .splitting import create_splitting_scheme
from .state import State


class SubSolver(ABC):
    """
    Abstract base class defining the minimal interface every operator must expose.

    Implementations must properly initialize from a domain :py:class:`~saetass.grid.Grid`, a specific time grid,
    and a parameter dictionary, and must provide an :py:meth:`~saetass.solver.SubSolver.advance` method to update the
    global :py:class:`~saetass.state.State`.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        The :py:class:`~saetass.grid.Grid` object containing spatial and/or momentum nodes.
    t_grid : np.ndarray
        The refined time grid for this specific operator's integration steps.
    params : Dict[str, Any]
        Dictionary containing physical and numerical parameters specific to this :py:class:`~saetass.solver.SubSolver`.
    """

    @abstractmethod
    def __init__(
        self, grid: Grid, t_grid: np.ndarray, params: dict[str, Any], **kwargs
    ):
        pass

    @abstractmethod
    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the provided :py:class:`~saetass.state.State` by a specified number of internal steps.

        Parameters
        ----------
        n_steps : int
            The number of internal substeps to advance based on :py:attr:`~saetass.solver.SubSolver.t_grid`.
        state : :py:class:`~saetass.state.State`
            The global tracking :py:class:`~saetass.state.State` to be updated.
        """
        pass


from .solvers.advection_solver import AdvectionSolver  # noqa: E402
from .solvers.diffusion_solver import DiffusionSolver  # noqa: E402
from .solvers.loss_solver import LossSolver  # noqa: E402
from .solvers.source_solver import SourceSolver  # noqa: E402

logger = logging.getLogger(__name__)


class OperatorType(StrEnum):
    """Auxiliary class for correct operator type handling.

    .. note::
       Currently, the supported operator types are: "advection", "diffusion", "loss" and "source".


    Parameters
    ----------
    operator_type : str
        String identifier for the operator type (e.g., "advection", "diffusion"). This
        will raise a ``ValueError`` if an unsupported operator type is provided.
    """

    ADVECTION = ("advection", AdvectionSolver)
    DIFFUSION = ("diffusion", DiffusionSolver)
    LOSS = ("loss", LossSolver)
    SOURCE = ("source", SourceSolver)

    def __new__(cls, operator_type: str, solver_class: type):
        obj = str.__new__(cls, operator_type)
        obj._value_ = operator_type
        obj.solver_class = solver_class
        return obj


class Solver:
    """
    Main class to manage the simulation workflow. It initializes operators based on the specified problem type and coordinates the time advancement of the solution.

    In the initialization phase, the :py:class:`~saetass.solver.Solver` class takes in the associated :py:class:`~saetass.grid.Grid`, initial :py:class:`~saetass.state.State`, problem type, operator parameters, substep counts and splitting scheme.
    It parses the problem type to determine which operators are involved and initializes them with their own refined time grids based on the specified splitting type.

    Any object of the :py:class:`~saetass.solver.Solver` class exposes two main methods: :py:meth:`~saetass.solver.Solver.step` for advancing the solution by some amount of time steps and :py:meth:`~saetass.solver.Solver.run` for running the entire simulation.
    After each advancement, :py:class:`~saetass.solver.Solver` updates the :py:class:`~saetass.state.State` object associated with the solution.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` object containing spatial and momentum grids and time grid.
    state : :py:class:`~saetass.state.State`
        :py:class:`~saetass.state.State` object containing the initial distribution.
    problem_type : str
        String specifying the type of problem and which operators to include. Valid operators are defined in
        :py:class:`~saetass.solver.OperatorType` (e.g., "advection-diffusion-loss-source"). The order of the operators will affect the behaviour of :py:class:`~saetass.splitting.SplittingScheme`.
    operator_params : dict, optional
        Dictionary mapping operator names to their specific parameters (e.g., ``{"advection": {...}, "diffusion": {...}}``).
        These parameters will be passed to the corresponding subsolvers during initialization and should be structured accordingly.
        For further details on expected parameters for each operator, refer to the documentation of the respective subsolver classes.
    substeps : dict, optional
        Dictionary specifying the number of substeps for each operator (e.g., ``{"advection": 2, "diffusion": 1}``).
        Default is no subrefinement, this is, 1 substep per operator.
    splitting_scheme : str or :py:class:`~saetass.splitting.SplittingSchemeType`, optional
        String or :py:class:`~saetass.splitting.SplittingSchemeType` specifying the :py:class:`~saetass.splitting.SplittingScheme` to use. Valid schemes are defined in
        :py:class:`~saetass.splitting.SplittingSchemeType` (e.g., "strang", "lie"). Default is "strang".
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
        # Print the banner on initialization
        print_banner()

        self.grid = grid
        self.state = state
        self.problem_type = problem_type.lower()

        # Store operator parameters (nested dict)
        self.operator_params = operator_params or {}
        self.substeps = substeps or {}

        # Determine which operators are present and print for debugging
        logger.debug(f"Problem type: {self.problem_type}")
        logger.debug(f"Available operators: {[op.value for op in OperatorType]}")

        # Parse operators from problem_type, handling exact operator names split by '-'
        self.operator_list = []
        ops_in_type = [op.strip().lower() for op in self.problem_type.split("-")]

        for op in ops_in_type:
            # This implicitly validates standard operator strings (raises ValueError if invalid)
            operator_type = OperatorType(op)
            self.operator_list.append(operator_type)

        logger.debug(f"Using operators: {[op.value for op in self.operator_list]}")
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
            solver_class = op.solver_class
            t_grid_refined = refined_t_grids[op]
            op_params = self.operator_params.get(op.value, {})

            self.operator_subsolvers.append(
                solver_class(
                    self.grid,
                    t_grid_refined,
                    op_params,
                    **kwargs,
                )
            )

            logger.debug(f"Initialized '{op.value}' solver with params: {op_params}")

    def _advance(self, n_steps):
        """Advance the solution by n_steps using operator splitting."""
        manage_progress = False
        if getattr(self, "_progress", None) is None:
            self._progress = create_progress_bar()
            self._progress.start()
            self._task_id = self._progress.add_task(
                "[bold cyan]Solving...",
                total=self.total_steps,
                completed=self.global_step,
                metrics="max=0.0 min=0.0",
            )
            manage_progress = True

        for _ in range(n_steps):
            self.global_step += 1

            f_max = np.max(self.state.f)
            f_min = np.min(self.state.f)

            self._progress.update(
                self._task_id,
                completed=self.global_step,
                metrics=f"max={f_max:.4g} min={f_min:.4g}",
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

        if manage_progress or self.global_step >= self.total_steps:
            if self._progress is not None:
                self._progress.stop()
                self._progress = None

    def run(self) -> State:
        """Advances the solution to the final time, updating and returning the final :py:class:`~saetass.state.State` object.

        Returns
        -------
        State
            The final state of the solution after advancing to the final time.
        """
        num_timesteps = self.grid.num_timesteps
        self._advance(num_timesteps)
        return self.state

    def step(self, n_steps=1) -> State:
        """Advances the solution by n_steps global time steps, updating and returning the current :py:class:`~saetass.state.State` object.

        Parameters
        ----------
        n_steps : int, optional
            The number of global time steps to advance. Default is 1.
        """
        self._advance(n_steps)
        return self.state
