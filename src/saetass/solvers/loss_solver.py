import logging
from typing import Any

import numpy as np

from ..grid import Grid
from .hyperbolic_solver import HyperbolicSolver

logger = logging.getLogger(__name__)


class LossSolver(HyperbolicSolver):
    """
    Finite volume solver for energy losses in momentum space, inheriting from :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver`.

    Solves the momentum-loss equation in conservative form,

    .. math::

        \\frac{\\partial f}{\\partial t} + \\frac{\\partial}{\\partial p}\\bigl(\\dot{p}(t,p)\\,f\\bigr) = 0,

    where :math:`\\dot{p} = dp/dt \\leq 0` is the (signed) momentum loss rate. The solver uses the conservative variable :math:`U = p f` and, also, :math:`V(t,y) = \\frac{\\dot{p}}{p\\ln(10)}` and :math:`y = \\log_{10}(p)`.
    The finite volume update is delegated to the base class across the momentum (p) axis.

    Parameters
    ----------
    grid : :py:class:`~saetass.grid.Grid`
        :py:class:`~saetass.grid.Grid` containing at least ``p_centers`` and ``p_faces``; optionally ``r_centers`` and ``r_faces`` for 2D problems.
    t_grid : ndarray
        Subproblem time grid. In the standard SAETASS workflow this is already subrefined during :py:class:`~saetass.solver.Solver` initialization.
    params : dict
        Solver configuration.  Accepted keys are:

        P_dot : ndarray or callable
            Momentum loss rate, :math:`\\dot{p}`, at cell centers. A callable must have signature ``P_dot(t) -> ndarray``.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL number for the adaptive sub-step calculation.
        inflow_value_f : float
            Value of the primitive distribution function at the high-momentum boundary, used as an inflow condition when :math:`\\dot{p} > 0` (i.e. momentum gain).
        order : ``{1, 2}``
            Order of the numerical scheme.
        adiabatic_losses : bool
            If ``True``, include adiabatic losses. The key ``v_centers_physical`` must also be supplied.
        v_centers_physical : ndarray, optional
            Physical advection velocity at cell centres; required when ``adiabatic_losses`` is ``True``.
    """

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict[str, Any],
        **kwargs,
    ) -> None:
        """Initialize the loss solver."""
        # Convert momentum loss parameters to general hyperbolic solver format
        loss_params = params.copy()

        # Set momentum axis as the main axis for losses
        loss_params["axis"] = 0

        # Add adiabatic losses
        if loss_params["adiabatic_losses"]:
            try:
                v_centers_physical = loss_params.pop("v_centers_physical")
            except KeyError:
                raise ValueError(
                    "If adiabatic_losses is True, v_centers_physical must be provided."
                )
            self.P_dot_adiabatic = self._adiabatic_losses(grid, v_centers_physical)

        # Rename loss-specific parameters to match the base class
        if "P_dot" in loss_params:
            P_dot_input = loss_params.pop("P_dot")
            if callable(P_dot_input):

                def dynamic_V_centers(t):
                    P = P_dot_input(t)
                    return self._generalized_velocity(P, grid)

                loss_params["V_centers"] = dynamic_V_centers
            else:
                loss_params["V_centers"] = self._generalized_velocity(P_dot_input, grid)

        if "inflow_value_f" in loss_params:
            loss_params["inflow_value_U"] = self._generalized_variable(
                loss_params.pop("inflow_value_f"), grid
            )[-1]

        # Initialize the base class
        super().__init__(grid, t_grid, loss_params, **kwargs)

    def _generalized_variable(self, f: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the primitive distribution function to the conservative variable.
        """
        p_centers = grid._p_centers_phys
        return p_centers * f

    def _inverse_generalized_variable(self, U: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Map the conservative variable back to the primitive distribution function.
        """
        p = np.asarray(grid._p_centers_phys).flatten()

        # 1D CASE
        if U.ndim == 1:
            if U.shape[0] != p.shape[0]:
                raise ValueError(f"Shape mismatch: U {U.shape}, p {p.shape}")
            mask = p > 0
            f = np.zeros_like(U)
            f[mask] = U[mask] / p[mask]
            if mask.sum() < len(p):
                raise ValueError(
                    "p contains non-positive values, cannot divide by zero."
                )
            return f

        # 2D CASE
        elif U.ndim == 2:
            # U: (n_r, n_p), p: (n_p,)
            if U.shape[1] != p.shape[0]:
                raise ValueError(
                    f"Expected U.shape[1] == p.shape[0], got U {U.shape}, p {p.shape}"
                )

            mask = p > 0.0
            f = np.zeros_like(U)
            # Only divide columns where p > 0
            f[:, mask] = U[:, mask] / p[mask]
            if mask.sum() < len(p):
                raise ValueError(
                    "p contains non-positive values, cannot divide by zero."
                )
            return f

        else:
            raise ValueError(
                "inverse_generalized_variable only supports 1D or 2D arrays."
            )

    def _generalized_velocity(self, P_dot: np.ndarray, grid: Grid) -> np.ndarray:
        """
        Convert the physical momentum loss rate to the generalized velocity used by the base-class finite-volume update.
        """
        p_centers = grid._p_centers_phys
        ln10 = np.log(10)
        denom = p_centers * ln10
        if P_dot.ndim == 2:
            len_x = grid.shape[1]
            denom = np.tile(denom[:, None], (1, len_x))  # Expand for 2D grids

        mask = denom > 0.0
        gen_vel = np.zeros_like(P_dot)
        gen_vel[mask] = P_dot[mask] / denom[mask]

        # For p=0, set generalized velocity to zero to avoid singularity
        if not np.all(mask) and np.any(mask):
            gen_vel[~mask] = 0.0

        return gen_vel

    def _adiabatic_losses(
        self, grid: Grid, v_centers_physical: np.ndarray
    ) -> np.ndarray:
        """
        Compute the adiabatic momentum loss rate due to spherical expansion.

        The adiabatic loss term arises from the divergence of the advection velocity field and is given by

        .. math::

            \\dot{p}_{\\text{ad}} = -\\frac{p}{3}\\,\\nabla \\cdot \\mathbf{v},

        evaluated cell-by-cell on the spatial grid via a finite-difference approximation of the radial flux divergence.
        """
        r_faces = grid.r_faces
        A_face = 4.0 * np.pi * r_faces
        V = (4.0 / 3.0) * np.pi * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)

        # TEMPORARY SOLUTION
        N = len(grid.r_centers)
        v_faces = np.zeros((len(grid._p_centers_phys), N + 1))
        v_faces[:, 1:N] = 0.5 * (v_centers_physical[:, :-1] - v_centers_physical[:, 1:])
        v_faces[:, 0] = v_centers_physical[:, 0]
        v_faces[:, -1] = v_centers_physical[:, -1]

        Phi = A_face * v_faces
        div = (Phi[:, 1:] - Phi[:, :-1]) / V

        return (-grid._p_centers_phys * div.T).T / 3.0 * 0
