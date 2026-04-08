"""
The :py:class:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver` class implements a finite volume method for solving hyperbolic PDEs of the general form

.. math::

    \\frac{\\partial U}{\\partial t} + \\frac{\\partial}{\\partial y}\\big(V(t,y)\\, U\\big) = 0,

where :math:`V(t,y)` is a generalized velocity that can depend on time and on the variable :math:`y`.
It supports both first-order upwind and second-order MUSCL-Hancock schemes with various slope limiters (minmod, van Leer, MC) to ensure stability and non-oscillatory behavior.
The solver can handle both 1D and 2D problems by treating the secondary axis as independent slices.
It automatically computes stable time steps based on the CFL condition and updates the associated :py:class:`~saetass.state.State` object accordingly.

This class serves as a flexible base for specific hyperbolic problems, such as advection or loss terms, by defining the appropriate generalized variable transformations and velocity fields.
Thus, :py:class:`~saetass.solvers.advection_solver.AdvectionSolver` and :py:class:`~saetass.solvers.loss_solver.LossSolver` inherit from this base class and implement the problem-specific logic while reusing the core finite volume update mechanism.
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numba import njit, prange

from ..grid import Grid
from ..solver import SubSolver
from ..state import State

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def _minmod_multi_arr(A, B, C):
    a = np.asarray(A)
    b = np.asarray(B)
    c = np.asarray(C)
    out = np.empty_like(a)
    n = a.size
    for i in prange(n):
        ai, bi, ci = a.flat[i], b.flat[i], c.flat[i]
        if ai * bi > 0.0 and ai * ci > 0.0:
            s = 1.0 if ai > 0 else -1.0
            out.flat[i] = s * min(abs(ai), abs(bi), abs(ci))
        else:
            out.flat[i] = 0.0
    return out


def _minmod_multi(a, b, c):

    if np.isscalar(a) and np.isscalar(b) and np.isscalar(c):
        if a * b > 0.0 and a * c > 0.0:
            s = 1.0 if a > 0 else -1.0
            return s * min(abs(a), abs(b), abs(c))
        else:
            return 0.0

    return _minmod_multi_arr(a, b, c)


class HyperbolicSolver(SubSolver, ABC):
    """
    Base class for hyperbolic PDE subsolvers using finite volume methods.

    It exposes a public API for advancing the solution in time and relies on subclasses to define problem-specific transformations and velocity fields.

    Parameters
    ----------
    grid : :class:`~saetass.grid.Grid`
        A grid object containing both spatial and momentum grids.
    t_grid : ndarray
        Time grid for integration.
        In the standard SAETASS workflow, this is typically subrefined during :class:`~saetass.solver.Solver` initialization.
    params : dict
        A dictionary of solver configuration parameters:

        V_centers : ndarray
            Generalized velocities at cell centers. Shape must match
            grid dimensions.
        limiter : ``{'minmod', 'vanleer', 'mc'}``
            Slope limiter used for second-order schemes.
        cfl : float
            CFL (Courant-Friedrichs-Lewy) number for stable time
            step calculation.
        inflow_value_U : float
            Value of the conservative variable $U$ at the outer boundary
            for inflow conditions.
        order : ``{1, 2}``
            Order of the numerical scheme:
            * 1: First-order upwind.
            * 2: Second-order MUSCL-Hancock.
        axis : ``{0, 1}``
            Integer indicating the main axis of the problem:
            * 0: Momentum axis.
            * 1: Spatial axis.
    """

    def __init__(
        self,
        grid: Grid,
        t_grid: np.ndarray,
        params: dict,
        **kwargs,
    ) -> None:

        self._unpack_params(params)
        self._unpack_grid(grid)
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.params = params or {}

        # Determine if V_centers is static or dynamic
        V_input = params.get("V_centers")
        if callable(V_input):
            self.is_V_dynamic = True

            # Helper to fetch and orient V_centers properly
            def _get_V_centers_dynamic(t):
                V = np.asarray(V_input(t), dtype=float)
                if self.axis == 0:
                    V = V.T
                return V

            self._get_V_centers = _get_V_centers_dynamic

            # Provide an initial value just for shape/metrics purposes
            self.V_centers = self._get_V_centers(self.t_grid[0])

        else:
            self.is_V_dynamic = False
            self.V_centers = np.asarray(V_input, dtype=float)
            if self.axis == 0:
                self.V_centers = self.V_centers.T

            # Caching strategy for static case
            self.V_centers_static = self.V_centers
            self.V_faces_static = self._face_generalized_velocity_interpolated(
                self.V_centers
            )
            self._get_V_centers = lambda t: self.V_centers_static

    @property
    def _main_centers(self) -> np.ndarray:
        """Return the main axis centers based on axis."""
        if self.axis == 0:
            return self.p_centers
        elif self.axis == 1:
            return self.r_centers
        else:
            raise ValueError("Invalid axis for main axis.")

    @property
    def _main_faces(self) -> np.ndarray:
        """Return the main axis faces based on axis."""
        if self.axis == 0:
            return self.p_faces
        elif self.axis == 1:
            return self.r_faces
        else:
            raise ValueError("Invalid axis for main axis.")

    @property
    def _other_axis(self) -> int:
        """Return the non-active axis index (0 or 1)."""
        return 1 - self.axis

    def advance(self, n_steps: int, state: State) -> None:
        r"""
        Advance the :py:class:`~saetass.state.State` by ``n_steps`` in :py:attr:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver.t_grid`.

        The inner loop may subdivide each requested step further to satisfy the CFL stability condition, so the number of actual flux evaluations can exceed ``n_steps``.
        The total elapsed time is always exactly ``n_steps * dt``, where ``dt`` is the time step size inferred from :py:attr:`~saetass.solvers.hyperbolic_solver.HyperbolicSolver.t_grid`.

        .. note:
            A positivity floor :math:`U(t,y)\geq 0` is enforced after the update to suppress negatives that may arise at steep gradient fronts when using the MUSCL-Hancock scheme.

        Parameters
        ----------
        n_steps : int
            Number of time steps to advance.
        state : :py:class:`~saetass.state.State`
            Current simulation state. The distribution function is updated in-place at the end of the call.
        """
        f = np.asarray(state.get_f(), dtype=float)
        if self.axis == 0:
            f = f.T

        U = self._generalized_variable(f, self.grid)

        dx = self.dx if self.M is None else self.dx[None, :]

        dt_requested = np.diff(self.t_grid)[0]
        total_time = n_steps * dt_requested
        t_local = state.t

        while total_time > 1e-40:
            V_centers, V_faces = self._get_velocities(t_local)
            dt_step = min(total_time, float(self._compute_dt(V_faces=V_faces)))

            if self.order == 1:
                F = self._compute_first_order_fluxes(U, V_centers, V_faces)
            elif self.order == 2:
                F = self._compute_second_order_fluxes(
                    U, self.V_centers, V_faces, dt_step
                )
                # Use numba for performance test
                # F = _numba_second_order_fluxes_core(
                #     U,
                #     self.V_centers,
                #     V_faces,
                #     self._main_centers,
                #     self._main_faces,
                #     dt_step,
                #     0,
                # )
            else:
                raise NotImplementedError("order must be 1 or 2")

            U = U - (dt_step / dx) * (F[..., 1:] - F[..., :-1])
            total_time -= dt_step
            t_local += dt_step

        f_new = self._inverse_generalized_variable(U, self.grid)

        # Positivity floor: MUSCL-Hancock is TVD but not strictly positive-definite;
        # clip machine-precision negatives that arise at steep gradient fronts.
        f_new = np.maximum(f_new, 0.0)
        state.update_f(f_new.T if self.axis == 0 else f_new)

        logger.debug(f"max(|f|) after step: {np.max(np.abs(f_new)):.4g}")

    # ---------------- internal helpers ----------------
    def _unpack_params(self, params: dict = None) -> None:
        """Unpack and strictly validate parameters."""
        # expected keys
        # expected_keys = {
        #     "V_centers",
        #     "limiter",
        #     "cfl",
        #     "inflow_value_U",
        #     "order",
        #     "axis",
        # }
        # provided_keys = set(params.keys())

        # check missing / extra
        # missing = expected_keys - provided_keys
        # extra = provided_keys - expected_keys
        # if missing:
        #    raise ValueError(f"Missing required params: {missing}")
        # if extra:
        #    raise ValueError(f"Unexpected params provided: {extra}")

        # --- axis ---
        try:
            self.axis: int = int(params["axis"])
        except Exception:
            raise ValueError("axis must be an integer")
        if not (0 <= self.axis < 2):
            raise ValueError(
                "axis must be 0 (for momentum problems) or 1 (for spatial problems)"
            )

        # Note: V_centers is handled locally in __init__ now due to static/callable checks

        # --- limiter ---
        limiter = params["limiter"]
        if limiter not in {"minmod", "vanleer", "mc"}:
            raise ValueError("limiter must be 'minmod', 'vanleer' or 'mc'")
        self.limiter: Literal["minmod", "vanleer", "mc"] = limiter

        # --- cfl ---
        try:
            self.cfl: float = float(params["cfl"])
        except Exception:
            raise ValueError("cfl must be a float")

        # --- inflow_value_U ---
        self.inflow_value_U = params["inflow_value_U"]
        # convert to 1D plain row array
        self.inflow_value_U = np.asarray(self.inflow_value_U, dtype=float).flatten()

        # --- order ---
        order = int(params["order"])
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        self.order: Literal[1, 2] = order

    def _unpack_grid(self, grid: Grid):
        """Unpack and validate grid object."""
        self.grid = grid

        def _load_axis(name: str) -> bool:
            """Loads an axis ('r' or 'p') from the grid and returns True if it exists."""
            centers = getattr(grid, f"{name}_centers", None)
            faces = getattr(grid, f"{name}_faces", None)

            if centers is not None:
                centers = np.asarray(centers, dtype=float)
                faces = np.asarray(faces, dtype=float)
                setattr(self, f"{name}_centers", centers)
                setattr(self, f"{name}_faces", faces)
                logger.info(f"Using {name}-grid with {len(centers)} points")
                return True
            else:
                setattr(self, f"{name}_centers", None)
                setattr(self, f"{name}_faces", None)
                return False

        def _init_axis_metrics(name: str):
            """Initialize mesh metrics for the main axis."""
            centers = getattr(self, f"{name}_centers")
            faces = getattr(self, f"{name}_faces")
            self.N = len(centers)
            self.dx = getattr(grid, f"d{name}", None)  # for compatibility (dr or dp)
            self.dx_c = centers[1:] - centers[:-1]  # Δ between centers
            self.dx_R = faces[1:] - centers  # distance center to right face
            self.dx_L = centers - faces[:-1]  # distance center to left face

        # --- main logic ---
        axes = {0: "p", 1: "r"}
        main_axis = axes.get(self.axis)

        if main_axis is None:
            raise ValueError(
                f"Invalid axis {self.axis}: expected 0 (momentum) or 1 (spatial)"
            )

        if not _load_axis(main_axis):
            raise ValueError(
                f"Grid must include {main_axis}_centers for {main_axis}-space problems"
            )

        _init_axis_metrics(main_axis)

        other_axis = "p" if main_axis == "r" else "r"
        if _load_axis(other_axis):
            logger.info(f"Found {other_axis}-grid")
            self.M = len(getattr(self, f"{other_axis}_centers"))
        else:
            logger.info(f"No {other_axis}-grid found, operating in 1D mode")
            self.M = None

    def _face_generalized_velocity_interpolated(
        self, V_centers: np.ndarray
    ) -> np.ndarray:
        """
        Linear interpolation of generalized velocities from cell centers to internal face positions x_{i+1/2}.
        Uses distance-weighted linear interpolation (works for non-uniform grids).

        Returns
        -------
        V_face : np.ndarray
            Interpolated generalized velocities at faces, length N-1.
        """
        if self.N <= 1:
            return np.zeros(0)

        V = V_centers

        # Distances from face to neighboring centers
        dist_left = self.dx_L
        dist_right = self.dx_R

        V_left = V[..., :-1]
        V_right = V[..., 1:]

        dist_left_b = (
            np.tile(dist_left[1:], (V.shape[0], 1)) if V.ndim == 2 else dist_left[1:]
        )
        dist_right_b = (
            np.tile(dist_right[:-1], (V.shape[0], 1))
            if V.ndim == 2
            else dist_right[:-1]
        )

        # Compute interpolation denominator safely
        denom = dist_left_b + dist_right_b
        denom = np.where(denom == 0.0, 1.0, denom)

        # Linear interpolation: closer center gets higher weight
        V_face = (V_left * dist_right_b + V_right * dist_left_b) / denom
        return V_face

    def _compute_dt(self, V_faces: np.ndarray = None) -> float:
        """
        Compute a stable time step based on the CFL condition:

            dt = cfl * min(dx) / max(|V|)

        Works for both 1D and 2D problems depending on `self.axis`.
        """
        # Interpolated velocities at faces along the active axis
        if V_faces is None:
            V_faces = self._face_generalized_velocity_interpolated()

        # Take absolute max over all entries (works for scalar, 1D, or 2D)
        if V_faces.size:
            Vmax = float(np.max(np.abs(V_faces)))
        else:
            Vmax = 0.0

        # Include outermost face proxy using last cell center along the active axis
        # Take max across slices if multidimensional
        if self.V_centers.ndim == 1:
            last_center_vel = abs(self.V_centers[-1])
        else:
            # Take the last index along the active axis, all slices on the other axis
            last_center_vel = np.max(
                np.abs(np.take(self.V_centers, indices=-1, axis=self.axis))
            )

        Vmax = max(Vmax, float(last_center_vel))

        # If velocity is zero everywhere → infinite dt
        if Vmax <= 0.0:
            logger.debug("Max velocity is zero, returning infinite dt_cfl")
            return np.inf

        dx_min = np.min(self.dx)

        return float(self.cfl * dx_min / Vmax)

    def _get_velocities(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the current generalized velocities at centers and faces."""
        if self.is_V_dynamic:
            self.V_centers = self._get_V_centers(t)
            V_faces = self._face_generalized_velocity_interpolated(self.V_centers)
            return self.V_centers, V_faces
        return self.V_centers_static, self.V_faces_static

    def _compute_first_order_fluxes(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
    ) -> np.ndarray:
        """
        Compute first-order upwind fluxes at faces given conservative variable U and face generalized velocities for 2D case.
        Returns flux matrix with length N+1 on axis and M on other_axis.
        """
        U_left = U[..., :-1]
        U_right = U[..., 1:]
        flux_int = np.where(V_faces >= 0.0, V_faces * U_left, V_faces * U_right)

        F = np.empty((*U.shape[:-1], U.shape[-1] + 1), dtype=float)
        F[..., 0] = 0.0
        F[..., 1:-1] = flux_int
        # Outflow/inflow at right boundary: upwind using last cell velocity
        V_last = np.take(V_centers, -1, axis=-1)  # shape (...,) — works 1D and 2D
        U_last = np.take(U, -1, axis=-1)
        F[..., -1] = U_last * np.where(V_last >= 0.0, V_last, self.inflow_value_U)

        return F

    def _compute_second_order_fluxes(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute second-order MUSCL-Hancock fluxes at faces given conservative generalized variable U and face generalized velocities for 2D case.
        Returns flux matrix with length N+1 on axis and M on other_axis.
        """
        if self.axis == 0:
            # For debugging breakpoints
            logger.debug("Computing 2D second-order fluxes along r-axis")
        # 1) compute slopes for W in non-uniform grid (minmod / vanleer)
        slopes_U = self._compute_slopes(U)  # shape (N,M) or (M,N)
        # 2) reconstruct left/right states at internal faces (t^n)
        UL, UR = self._recontruct_face_states(U, slopes_U)
        F_centers = V_centers * U  # flux at cell centers (for predictor step)
        slopes_F = self._compute_slopes(F_centers)
        # 3) predictor to t^{n+1/2}
        # ULh, URh = self._predictor_states(UL, UR, V_centers, slopes_U, dt)
        ULh, URh = self._predictor_states_v2(UL, UR, slopes_F, dt)
        # 4) compute V at internal faces and fluxes by upwind using sign of V_face
        flux_int = np.where(
            V_faces >= 0.0, V_faces * ULh[..., 1:-1], V_faces * URh[..., 1:-1]
        )

        F = np.empty((*U.shape[:-1], U.shape[-1] + 1), dtype=float)
        F[..., 1:-1] = flux_int
        F[..., 0], F[..., -1] = self._boundary_fluxes(
            U, ULh, URh, V_centers, slopes_U, dt
        )
        # F[..., 0], F[..., -1] = self._boundary_fluxes_v2(ULh, URh, V_faces)

        return F

    def _compute_slopes(self, U: np.ndarray) -> np.ndarray:
        """
        Compute limited slopes on non-uniform grid for 2D case.
        We'll compute left derivative, right derivative, and central derivative
        with correct Δr factors, then apply the chosen limiter.
        """
        N = self.N
        slopes = np.zeros_like(U)
        if N <= 2:
            return slopes

        centers = self._main_centers

        # left derivative at i (using centers i-1 and i)
        dL = (U[..., 1:-1] - U[..., :-2]) / (centers[1:-1] - centers[:-2])  # length N-2
        # right derivative at i (using centers i and i+1)
        dR = (U[..., 2:] - U[..., 1:-1]) / (centers[2:] - centers[1:-1])  # length N-2
        # central derivative across two cells
        dC = (U[..., 2:] - U[..., :-2]) / (centers[2:] - centers[:-2])  # length N-2

        if self.limiter == "minmod":
            slopes_interior = _minmod_multi(dL, dR, dC)
        elif self.limiter == "vanleer":  # generalized
            prod = dL * dR
            summ = dL + dR
            slopes_interior = np.where(
                prod > 0.0, (2.0 * prod) / np.where(summ != 0, summ, 1.0), 0.0
            )
        elif self.limiter == "mc":
            slopes_interior = _minmod_multi(2.0 * dL, 2.0 * dR, 0.5 * dC)
        else:  # should not happen due to earlier check, but just in case
            raise NotImplementedError("limiter must be 'minmod', 'vanleer' or 'mc'")
        # Assign slopes to correct slices
        slopes[..., 1:-1] = slopes_interior

        # TEMPORARY SOLUTION
        # Left boundary i=0 : forward/backward differences built with same center spacing
        d_fwd = (U[..., 1] - U[..., 0]) / (centers[1] - centers[0])  # forward 1-step
        d_fwd2 = (U[..., 2] - U[..., 0]) / (
            centers[2] - centers[0]
        )  # two-cell centred forward

        if self.limiter == "minmod":
            slopes[..., 0] = _minmod_multi(d_fwd, d_fwd2, d_fwd)
        elif self.limiter == "vanleer":
            prod = d_fwd * d_fwd2
            summ = d_fwd + d_fwd2
            slopes[..., 0] = np.where(
                prod > 0.0, (2.0 * prod) / (summ if summ != 0 else 1.0), 0.0
            )
        elif self.limiter == "mc":
            slopes[..., 0] = _minmod_multi(2.0 * d_fwd, 2.0 * d_fwd2, 0.5 * d_fwd2)
        else:
            slopes[..., 0] = d_fwd  # fallback (shouldn't happen)

        # Right boundary i=N-1 : backward / two-cell centred backward
        d_bwd = (U[..., -1] - U[..., -2]) / (
            centers[-1] - centers[-2]
        )  # backward 1-step
        d_bwd2 = (U[..., -1] - U[..., -3]) / (
            centers[-1] - centers[-3]
        )  # two-cell centred backward

        if self.limiter == "minmod":
            slopes[..., -1] = _minmod_multi(d_bwd, d_bwd2, d_bwd)
        elif self.limiter == "vanleer":
            prod = d_bwd * d_bwd2
            summ = d_bwd + d_bwd2
            slopes[..., -1] = np.where(
                prod > 0.0, (2.0 * prod) / (summ if summ != 0 else 1.0), 0.0
            )
        elif self.limiter == "mc":
            slopes[..., -1] = _minmod_multi(2.0 * d_bwd, 2.0 * d_bwd2, 0.5 * d_bwd2)
        else:
            slopes[..., -1] = d_bwd  # fallback

        return slopes

    def _recontruct_face_states(
        self, U: np.ndarray, slopes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left/right states at internal faces from cell-centered values and slopes for 2D case.
        Returns UL, UR arrays of shape (N+1,M) or (M,N+1).
        UL[i] = left state at face i (from right cell)
        UR[i] = right state at face i (from left cell)
        """
        UL = np.zeros(
            (*U.shape[:-1], U.shape[-1] + 1)
        )  # left state of face i (value from right cell)
        UR = np.zeros(
            (*U.shape[:-1], U.shape[-1] + 1)
        )  # right state of face i (value from left cell)
        UR[..., :-1] = U - slopes * self.dx_L
        UL[..., 1:] = U + slopes * self.dx_R

        UR[..., -1] = U[..., -1]
        UL[..., 0] = U[..., 0]

        return UL, UR

    def _predictor_states(
        self,
        UL: np.ndarray,
        UR: np.ndarray,
        V_centers: np.ndarray,
        slopes: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform MUSCL-Hancock predictor step to advance interface states to t^{n+1/2} for 2D case.
        Each interface state is predicted using the donor cell's slope.
        Returns ULh, URh arrays of shape (N+1,M) or (M,N+1).
        """
        ULh = UL.copy()
        URh = UR.copy()
        if self.N > 1:
            # UL* (state coming from cell i) uses v_i and slope_i
            ULh[..., 1:] -= 0.5 * dt * V_centers * slopes
            # UR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            URh[..., :-1] -= 0.5 * dt * V_centers * slopes

        return ULh, URh

    def _predictor_states_v2(
        self,
        UL,
        UR,
        slopes_F,
        dt,
    ):
        ULh = UL.copy()
        URh = UR.copy()

        if self.N > 1:
            ULh[..., 1:] -= 0.5 * dt * slopes_F
            URh[..., :-1] -= 0.5 * dt * slopes_F

        return ULh, URh

    def _boundary_fluxes(
        self,
        U: np.ndarray,
        ULh: np.ndarray,
        URh: np.ndarray,
        V_centers: np.ndarray,
        slopes: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute fluxes at the domain boundaries (faces 0 and N) for 2D case.
        Left boundary (r=0): assume symmetry => flux=0.
        Right boundary (outer face): use last cell's left state and slope for prediction.
        """
        size = U.shape[0] if U.ndim > 1 else 1
        flux_L = np.zeros(size, dtype=float)
        if self._main_faces[0] <= 0.0 and self.axis == 1:
            # Left face (r=0): symmetry
            flux_L[...] = 0.0
        elif self.axis == 0:
            # Left face (momentum inflow): use first cell right-state, or inflow
            slp = (V_centers[..., 1] - V_centers[..., 0]) / (
                self._main_centers[1] - self._main_centers[0]
            )
            V_in = V_centers[..., 0] - slp * (
                self._main_centers[0] - self._main_faces[0]
            )
            # U_right_inner = U[..., 0]
            flux_L = V_in * np.where(
                V_in >= 0.0,
                0.0,
                URh[..., 0],
            )
        else:
            raise ValueError("Invalid left face position for the given axis.")

        # Right boundary (outer face)
        slp = (V_centers[..., -1] - V_centers[..., -2]) / (
            self._main_centers[-1] - self._main_centers[-2]
        )
        V_out = V_centers[..., -1] + slp * (
            self._main_faces[-1] - self._main_centers[-1]
        )
        # left-state
        flux_R = V_out * np.where(
            V_out >= 0.0,
            ULh[..., -1],
            self.inflow_value_U,
        )

        return flux_L, flux_R

    def _boundary_fluxes_v2(self, ULh, URh, V_faces):

        flux_L = np.where(V_faces[..., 0] >= 0.0, 0.0, V_faces[..., 0] * URh[..., 0])
        flux_R = np.where(
            V_faces[..., -1] >= 0.0,
            V_faces[..., -1] * ULh[..., -1],
            self.inflow_value_U * V_faces[..., -1],
        )
        return flux_L, flux_R

    @abstractmethod
    def _generalized_variable(self, f: np.ndarray, centers: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _inverse_generalized_variable(
        self, U: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        pass


@njit(parallel=True, fastmath=True)
def _numba_second_order_fluxes_core(U, V_centers, V_faces, centers, faces, dt, limiter):
    """
    Numba-accelerated 2D MUSCL-Hancock flux computation core.
    This performs slope computation, reconstruction, predictor, and flux calculation in one pass.
    """
    M, N = U.shape  # M: other axis, N: main axis
    F = np.zeros((M, N + 1))
    dx_L = faces[1:-1] - centers[:-1]
    dx_R = centers[1:] - faces[1:-1]

    for j in prange(M):
        # Extract 1D row (active axis)
        Uj = U[j, :]
        Vcj = V_centers[j, :]
        Vfj = V_faces[j, :]

        slopes = np.zeros(N)

        if N > 2:
            dL = (Uj[1:-1] - Uj[:-2]) / (centers[1:-1] - centers[:-2])
            dR = (Uj[2:] - Uj[1:-1]) / (centers[2:] - centers[1:-1])
            dC = (Uj[2:] - Uj[:-2]) / (centers[2:] - centers[:-2])

            # Apply limiter
            if limiter == 0:  # minmod
                for i in range(1, N - 1):
                    a, b, c = dL[i - 1], dR[i - 1], dC[i - 1]
                    if a * b > 0.0 and a * c > 0.0:
                        s = 1.0 if a > 0 else -1.0
                        slopes[i] = s * min(abs(a), abs(b), abs(c))
                    else:
                        slopes[i] = 0.0
            elif limiter == 1:  # vanleer
                for i in range(1, N - 1):
                    a, b = dL[i - 1], dR[i - 1]
                    prod = a * b
                    slopes[i] = (2.0 * prod / (a + b)) if prod > 0.0 else 0.0
            elif limiter == 2:  # MC
                for i in range(1, N - 1):
                    a, b, c = 2 * dL[i - 1], 2 * dR[i - 1], 0.5 * dC[i - 1]
                    if a * b > 0.0 and a * c > 0.0:
                        s = 1.0 if a > 0 else -1.0
                        slopes[i] = s * min(abs(a), abs(b), abs(c))
                    else:
                        slopes[i] = 0.0

        # Reconstruction at faces
        UL = np.zeros(N + 1)
        UR = np.zeros(N + 1)
        for i in range(1, N):
            UR[i] = Uj[i - 1] + slopes[i - 1] * dx_L[i - 1]
            UL[i] = Uj[i] - slopes[i] * dx_R[i - 1]

        # Predictor step
        ULh = UL.copy()
        URh = UR.copy()
        for i in range(1, N):
            ULh[i] -= 0.5 * dt * Vcj[i] * slopes[i]
            URh[i] -= 0.5 * dt * Vcj[i - 1] * slopes[i - 1]

        # Internal fluxes
        for i in range(1, N):
            F[j, i] = Vfj[i - 1] * (URh[i] if Vfj[i - 1] >= 0.0 else ULh[i])

        # Boundary fluxes (simplified, can adapt later)
        # outflow inner face
        Vin = Vcj[0]
        U_right_inner = Uj[0]
        F[j, 0] = Vin * (U_right_inner if Vin <= 0.0 else 0.0)
        Vout = Vcj[-1]
        F[j, -1] = Vout * (Uj[-1] if Vout >= 0.0 else 0.0)

    return F
