import numpy as np
import logging
from ..state import State
from ..grid import Grid
from ..state import State
from ..solver import SubSolver
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SourceSolver(SubSolver):
    """
    Explicit Euler operator for a source term, inheriting from :py:class:`~saetass.solver.SubSolver`.

    Advances the distribution function according to:

    .. math::

        \\frac{\\partial f}{\\partial t} = Q(t, r, p),

    using a single first-order explicit Euler step over the total requested
    time ``n_steps * dt``.

    Because the source operator does not involve any spatial derivatives, no CFL condition applies and the entire ``n_steps * dt`` interval is consumed in one evaluation of :math:`Q`.
    The source function :math:`Q` may be time-dependent or a fixed array.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` providing ``r_centers`` and/or ``p_centers`` depending on the problem dimension.
    t_grid : ndarray
        Subproblem time grid.
        In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration.  Accepted keys are:

        source : ndarray or callable
            Source term, :math:`Q`.
            If callable, the signature must be ``source(r_centers, p_centers, t) -> ndarray``, where either ``r_centers`` or ``p_centers`` may be ``None`` for 1D problems.
            If an array, its shape must match the :py:class:`~saetass.state.State` distribution function.
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

    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the :py:class:`~saetass.state.State` by ``n_steps`` in :py:attr:`~saetass.solvers.source_solver.SourceSolver.t_grid`.

        Applies a single explicit Euler step with total time ``total_dt = n_steps * dt``.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        state : :py:class:`~saetass.state.State`
            Current simulation state. The distribution function is updated in-place at the end of the call.
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
