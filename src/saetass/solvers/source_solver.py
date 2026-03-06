import numpy as np
import logging
from ..state import State
from ..grid import Grid
from ..state import State
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SourceSolver:
    """
    Source term operator for FVM operator splitting.

    This operator simply updates the conserved variable f according to:
        df/dt = Q(r, p, t)

    The source function Q can depend on space (r), momentum (p), and time (t),
    and may be provided as a callable or as a fixed array matching the grid shape.

    Attributes:
        x_grid: np.ndarray (spatial or momentum grid, as appropriate)
        t_grid: np.ndarray (optional time grid)
        state: State (the solution to update)
        operator_params: dict (contains at least the 'source' function or array)
    """

    def __init__(self, grid: Grid, t_grid: np.ndarray, params: dict, **kwargs):
        self.grid = grid
        self.t_grid = t_grid
        self.x_grid = grid.r_centers if grid.r_centers is not None else grid.p_centers
        self.params = params or {}

        source_input = self.params.get("source", None)

        if source_input is None:
            raise ValueError("A source function or array must be provided.")

        if callable(source_input):
            self.is_source_dynamic = True

            def _get_source_dynamic(t):
                if self.grid.p_centers is not None and self.grid.r_centers is not None:
                    return np.asarray(
                        source_input(self.grid.r_centers, self.grid.p_centers, t),
                        dtype=float,
                    )
                elif self.grid.r_centers is not None:
                    return np.asarray(
                        source_input(self.grid.r_centers, None, t), dtype=float
                    )
                else:
                    return np.asarray(
                        source_input(None, self.grid.p_centers, t), dtype=float
                    )

            self._get_source = _get_source_dynamic
        else:
            self.is_source_dynamic = False
            self.source_static = np.asarray(source_input, dtype=float)
            self._get_source = lambda t: self.source_static

    def advance(self, n_steps: int, state: State) -> np.ndarray:
        """
        Advance the state by n_steps * dt using simple explicit Euler integration.
        """
        total_dt = float(n_steps) * np.diff(self.t_grid)[0]

        # Process source term efficiently
        if self.is_source_dynamic:
            S = self._get_source(state.t)
        else:
            S = self.source_static

        # Ensure shape compatibility
        if S.shape != state.get_f().shape:
            raise ValueError(
                f"Source shape {S.shape} does not match state shape {state.f.shape}"
            )

        # Explicit update
        f_new = state.f + total_dt * S
        state.update_f(f_new)

        logger.debug(f"Advanced source operator by {n_steps} steps (dt={total_dt})")

        return state.f
