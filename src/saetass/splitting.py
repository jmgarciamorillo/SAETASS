from abc import ABC, abstractmethod
from enum import StrEnum
from .state import State
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SplittingScheme(ABC):
    """Base class for operator splitting schemes.

    Subclasses must call ``super().__init__()`` and invoke
    ``self._store_t_grid(t_grid)`` inside their ``initialize_t_grid``
    implementation.  At the end of every ``apply`` call they must invoke
    ``self._advance_global_time(state)`` so that ``state.t`` always reflects
    the canonical grid value (no floating-point accumulation drift).
    """

    def __init__(self):
        self.t_grid: np.ndarray = None
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _store_t_grid(self, t_grid: np.ndarray) -> None:
        """Store the global time grid and reset the internal step counter.

        Must be called by every concrete ``initialize_t_grid`` implementation.
        """
        self.t_grid = np.asarray(t_grid, dtype=float)
        self._global_step = 0

    def _advance_global_time(self, state: State) -> None:
        """Increment the internal step counter and set ``state.t`` to the
        exact canonical grid value.

        This replaces any external ``state.advance_time(dt)`` call in the
        ``Solver`` loop.  Using :meth:`State.set_time` (assignment rather than
        accumulation) guarantees that ``state.t`` never drifts away from the
        true grid value due to repeated floating-point additions.
        """
        self._global_step += 1
        state.set_time(self.t_grid[self._global_step])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize_t_grid(
        self, operator_list: list, substeps_per_op: dict, t_grid: np.ndarray
    ):
        """Build per-operator refined time grids and store the global grid.

        Implementations must call ``self._store_t_grid(t_grid)`` so that the
        base class can track global time.
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
        """Apply one global time step of the splitting scheme.

        Implementations must call ``self._advance_global_time(state)`` as
        their final action so that ``state.t`` is updated to the canonical
        end-of-step grid value.
        """
        pass


class StrangSplitting(SplittingScheme):
    def __init__(self):
        super().__init__()

    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ):
        """For Strang splitting, the time grid is refined by the substeps and the refinment
        for symmetric operators is double (all of them except for the last one).

        output: refined_t_grids: dict
        """
        self._store_t_grid(t_grid)
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
        """Apply the Strang splitting scheme."""

        if np.min(state.f) < 0:
            logger.warning(
                "Strang splitting: Negative values detected in state.f before applying operators."
            )

        # First half-step for all but the last operator
        for op, subsolver in zip(operator_list[:-1], operator_subsolvers[:-1]):
            logger.debug(f"Strang splitting: first half-step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Full step for the last operator
        logger.debug(f"Strang splitting: full step for operator '{operator_list[-1]}'")
        n_sub = substeps_per_op[operator_list[-1]]
        operator_subsolvers[-1].advance(n_sub, state)

        # Second half-step for all but the last operator (reversed)
        for op, subsolver in zip(
            reversed(operator_list[:-1]), reversed(operator_subsolvers[:-1])
        ):
            logger.debug(f"Strang splitting: second half-step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Advance state.t to the exact end-of-step grid value
        self._advance_global_time(state)


class LieSplitting(SplittingScheme):
    def __init__(self):
        super().__init__()

    def initialize_t_grid(
        self,
        operator_list: list,
        substeps_per_op: dict,
        t_grid: np.ndarray,
    ):
        """For Lie splitting, the time grid is refined by the substeps only."""
        self._store_t_grid(t_grid)
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
        """Apply the Lie splitting scheme."""
        for op, subsolver in zip(operator_list, operator_subsolvers):
            logger.debug(f"Lie splitting: full step for operator '{op}'")
            n_sub = substeps_per_op[op]
            subsolver.advance(n_sub, state)

        # Advance state.t to the exact end-of-step grid value
        self._advance_global_time(state)


class SplittingSchemeType(StrEnum):
    """Auxiliary class for correct splitting scheme handling.

    Parameters
    ----------
    scheme_type : str
        String identifier for the splitting scheme (e.g., "strang", "lie"). This
        will raise a ``ValueError`` if an unsupported splitting scheme is provided.
    """

    STRANG = ("strang", StrangSplitting)
    LIE = ("lie", LieSplitting)

    def __new__(cls, scheme_type: str, scheme_class: type):
        obj = str.__new__(cls, scheme_type)
        obj._value_ = scheme_type
        obj.scheme_class = scheme_class
        return obj


def create_splitting_scheme(scheme_name: str | SplittingSchemeType) -> SplittingScheme:
    """Factory function to create a :py:class:`~saetass.splitting.SplittingScheme` instance based on the scheme name.

    Parameters
    ----------
    scheme_name : str | SplittingSchemeType
        Name of the scheme or a :py:class:`~saetass.splitting.SplittingSchemeType` enum instance.
        Valid string names are mapped by the enum (e.g., "strang", "lie").
    """
    if isinstance(scheme_name, str):
        scheme_name = scheme_name.lower().strip()

    scheme_type = SplittingSchemeType(scheme_name)
    return scheme_type.scheme_class()


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
