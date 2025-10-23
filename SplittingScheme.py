from abc import ABC, abstractmethod
from State import State
import logging

logger = logging.getLogger(__name__)


class SplittingScheme(ABC):
    @abstractmethod
    def apply(self, operator_list: list, operator_subsolvers: list, state: State):
        """
        Define how to apply operator splitting given a list of operators,
        their corresponding subsolvers, the current state, and the timestep dt.
        """
        pass


class StrangSplitting(SplittingScheme):
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
        # First half-step for the first operators
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
