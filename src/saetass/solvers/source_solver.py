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

        # Allow user to pass a callable source or a fixed array
        self.source_func = self.params.get("source", None)

        if self.source_func is None:
            raise ValueError("A source function or array must be provided.")

    def _compute_source(self, t: float) -> np.ndarray:
        """
        Evaluate the source term at time t.
        The function must return an array of shape matching state.f.
        """
        if callable(self.source_func):
            # Handle 1D or 2D case transparently
            if self.grid.p_centers is not None and self.grid.r_centers is not None:
                return self.source_func(self.grid.r_centers, self.grid.p_centers, t)
            elif self.grid.r_centers is not None:
                return self.source_func(self.grid.r_centers, None, t)
            else:
                return self.source_func(None, self.grid.p_centers, t)
        else:
            # Fixed source array
            return np.asarray(self.source_func, dtype=float)

    def advance(self, n_steps: int, state: State) -> np.ndarray:
        """
        Advance the state by n_steps * dt using simple explicit Euler integration.
        """
        total_dt = float(n_steps) * np.diff(self.t_grid)[0]

        # Compute source term at current time
        S = self._compute_source(state.t)

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
