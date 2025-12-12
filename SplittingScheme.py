from abc import ABC, abstractmethod
from State import State
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SplittingScheme(ABC):
    # def __init__(self):
    #    self.iterations = 0

    @abstractmethod
    def initialize_t_grid(
        self, operator_list: list, substeps_per_op: dict, t_grid: np.ndarray
    ):
        """
        Define how to initialize the time grid based on the operator list,
        their corresponding substeps, and the global time grid.
        """
        pass

    @abstractmethod
    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ):
        """
        Define how to apply operator splitting given a list of operators,
        their corresponding subsolvers, the current state, and the timestep dt.
        """
        pass


class StrangSplitting(SplittingScheme):
    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ):
        """
        For Strang splitting, the time grid is refined by the substeps and the refinment
        for symmetric operators is double (all of them execpt for the last one).

        output: refined_t_grids: dict
        """
        refined_t_grids = {}

        for i, op in enumerate(operator_list):
            n_sub = substeps_per_op[op]
            if i < len(operator_list) - 1:
                n_sub = n_sub * 2  # Double substeps for all but last operator

            refined_t_grids[op] = _refine_t_grid(t_grid, n_sub)

        return refined_t_grids

    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ):
        """
        Apply the Strang splitting scheme.
        """
        # self.iterations += 1
        # if self.iterations % 100 == 0:
        #     logger.info(f"Strang splitting: completed {self.iterations} iterations")

        # First half-step for the first operators
        if np.min(state.f) < 0:
            logger.warning(
                "Strang splitting: Negative values detected in state.f before applying operators."
            )
        for op, subsolver in zip(operator_list[:-1], operator_subsolvers[:-1]):
            logger.debug(f"Strang splitting: first half-step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Full step for the last operator
        logger.debug(f"Strang splitting: full step for operator '{operator_list[-1]}'")
        n_sub = substeps_per_op[operator_list[-1]]
        operator_subsolvers[-1].advance(n_sub, state)

        # Second half-step for the first operators
        for op, subsolver in zip(
            reversed(operator_list[:-1]), reversed(operator_subsolvers[:-1])
        ):
            logger.debug(f"Strang splitting: second half-step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)


class LieSplitting(SplittingScheme):
    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ):
        """
        For Lie splitting, the time grid is refined by the substeps only.
        """
        refined_t_grids = {}

        for op in operator_list:
            n_sub = substeps_per_op[op]
            refined_t_grids[op] = _refine_t_grid(t_grid, n_sub)

        return refined_t_grids

    def apply(
        self,
        operator_list: list,
        operator_subsolvers: list,
        substeps_per_op: dict,
        state: State,
    ):
        """
        Apply the Lie splitting scheme.
        """
        # self.iterations += 1
        # if self.iterations % 100 == 0:
        #     logger.info(f"Lie splitting: completed {self.iterations} iterations")
        for op, subsolver in zip(operator_list, operator_subsolvers):
            logger.debug(f"Lie splitting: full step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)


def create_splitting_scheme(scheme_name: str) -> SplittingScheme:
    """
    Factory function to create a SplittingScheme instance based on the scheme name.
    """
    scheme_name = scheme_name.lower()
    if scheme_name == "strang":
        return StrangSplitting()
    elif scheme_name == "lie":
        return LieSplitting()
    else:
        raise ValueError(f"Unknown splitting scheme: {scheme_name}")


def _refine_t_grid(t_grid: np.ndarray, n_sub: int) -> np.ndarray:
    if n_sub > 1:
        t_grid_refined = []
        for j in range(len(t_grid) - 1):
            t_start = t_grid[j]
            t_end = t_grid[j + 1]
            t_grid_refined.extend(np.linspace(t_start, t_end, n_sub + 1)[:-1])
        t_grid_refined.append(t_grid[-1])
        return np.array(t_grid_refined)
    else:
        return t_grid
