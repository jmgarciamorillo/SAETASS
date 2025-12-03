import numpy as np
from typing import Literal
from State import State, SliceState
from Grid import Grid
from abc import ABC, abstractmethod
from numba import njit, prange

import logging

logger = logging.getLogger(__name__)


@njit(parallel=True, fastmath=True)
def minmod_multi(a, b, c):
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


class HyperbolicFVSolver(ABC):

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

    def _slice_array(self, arr: np.ndarray, idx: int) -> np.ndarray:
        """Return a slice of arr at fixed index along the non-active axis."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            return arr

        other = self._other_axis
        return np.take(arr, indices=idx, axis=other)

    def _face_generalized_velocity_interpolated(self) -> np.ndarray:
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

        V = np.asarray(self.V_centers, dtype=float)

        # Distances from face to neighboring centers
        dist_left = self.dx_L
        dist_right = self.dx_R

        if V.ndim == 1:
            V_left = V[:-1]
            V_right = V[1:]
        else:
            V_left = V[:, :-1]
            V_right = V[:, 1:]

        dist_left_b = np.tile(dist_left, (V.shape[0], 1)) if V.ndim == 2 else dist_left
        dist_right_b = (
            np.tile(dist_right, (V.shape[0], 1)) if V.ndim == 2 else dist_right
        )

        # Compute interpolation denominator safely
        denom = dist_left_b + dist_right_b
        denom = np.where(denom == 0.0, 1.0, denom)

        # Linear interpolation: closer center gets higher weight
        V_face = (V_left * dist_right_b + V_right * dist_left_b) / denom
        return V_face

    def compute_dt(self, V_faces: np.ndarray = None) -> float:
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

    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the state by n_steps * dt.

        Automatically adapts to whether the main grid is spatial (r) or momentum (p).

        If the problem is 1D (no secondary axis), processes it directly.
        If 2D (main × secondary), processes each slice independently along the secondary axis.
        """

        # Determine axis
        main_axis = "r" if self.axis == 1 else "p"
        other_axis = "p" if main_axis == "r" else "r"
        other_centers = getattr(self, f"{other_axis}_centers", None)

        # Detect 1D or 2D case
        is_1d = other_centers is None or state.ndim == 1

        if is_1d:
            # Case 1D: single field dependent on main axis
            logger.debug(f"Advancing 1D state along {main_axis}")
            V_faces_1d = self._face_generalized_velocity_interpolated()
            V_centers_1d = self._slice_array(self.V_centers, 0)
            self._advance_slice(
                n_steps, state, V_centers_1d, V_faces_1d, self.inflow_value_U
            )
        else:
            # Case 2D: process slices along secondary axis
            # n_other = len(other_centers)
            # state_n_other = getattr(state, f"n_{other_axis}", None)

            # if state_n_other != n_other:
            #     raise ValueError(
            #         f"State shape mismatch: expected {n_other} {other_axis}-slices, "
            #         f"got {state_n_other}"
            #     )

            # V_faces_all = self._face_generalized_velocity_interpolated()

            # for i in range(n_other):
            # Create a view/slice of the state at fixed other_axis axis
            # slice_state = SliceState(state, i, axis=self.axis)
            # V_faces_1d = self._slice_array(V_faces_all, i)
            # V_centers_1d = self._slice_array(self.V_centers, i)
            # inflow_value_U = self.inflow_value_U[i]
            # logger.debug(f"Advancing slice {i+1}/{n_other} along {main_axis}")
            # self._advance_slice(
            #    n_steps, slice_state, V_centers_1d, V_faces_1d, inflow_value_U
            # )
            self._advance_slice_2d(n_steps, state)

        return

    def _advance_slice_2d(self, n_steps: int, state: State) -> None:
        """
        Advance a 2D state by n_steps * dt along the main axis.
        This method is currently not used; the main advance method handles 2D via slices.
        """
        f = np.asarray(state.get_f(), dtype=float)
        if self.axis == 0:
            f = f.T
        N = self.N
        M = self.M  # size along other axis

        U = self._generalized_variable(f, self.grid)

        V_centers = self.V_centers
        V_faces = self._face_generalized_velocity_interpolated()
        dx = np.tile(self.dx, (M, 1))

        dt_requested = float(n_steps) * np.diff(self.t_grid)[0]
        total_time = n_steps * dt_requested
        dt_cfl = float(self.compute_dt(V_faces=V_faces))
        dt_step = min(dt_requested, dt_cfl)

        while total_time - dt_step > 0.0:
            if self.order == 1:
                F = self._compute_first_order_fluxes_2d(
                    U,
                    V_centers,
                    V_faces,
                )
            elif self.order == 2:
                F = self._compute_second_order_fluxes_2d(
                    U, V_centers, V_faces, total_time
                )
                # Use numba for performance test
                # F = _numba_second_order_fluxes_2d_core(
                #     U,
                #     V_centers,
                #     V_faces,
                #     self._main_centers,
                #     self._main_faces,
                #     dt_step,
                #     0,
                # )
            else:
                raise NotImplementedError("order must be 1 or 2")

            U = U - (dt_step / dx) * (F[:, 1:] - F[:, :-1])
            total_time -= dt_step

        if self.order == 1:
            F = self._compute_first_order_fluxes_2d(U, V_centers, V_faces)
        elif self.order == 2:
            F = self._compute_second_order_fluxes_2d(U, V_centers, V_faces, total_time)
            # Use numba for performance test
            # F = _numba_second_order_fluxes_2d_core(
            #     U, V_centers, V_faces, self._main_centers, self._main_faces, dt_step, 0
            # )
        else:
            raise NotImplementedError("order must be 1 or 2")

        U_new = U - (total_time / dx) * (F[:, 1:] - F[:, :-1])
        f_new = self._inverse_generalized_variable(U_new, self.grid)
        # TODO?: self._postprocess_solution(f_new,centers)
        # f_new[0] = f_new[1]
        f_new = f_new.T if self.axis == 0 else f_new
        state.update_f(f_new)

        logger.debug(f"max(|f|) after step: {np.max(np.abs(f_new)):.4g}")
        return

    def _advance_slice(
        self,
        n_steps: int,
        state: State,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
        inflow_value_U: float,
    ) -> None:
        """
        Advance a single slice (1D). Works with a generalized conservative variable W
        inside the solver. All axis-dependent behaviors are delegated to hooks.
        """
        f = np.asarray(state.get_f().copy(), dtype=float)
        N = self.N

        if f.shape != (N,):
            raise ValueError(f"U shape mismatch: expected ({N},), got {f.shape}")

        U = self._generalized_variable(f, self.grid)  # Convert to generalized variable

        dt_requested = float(n_steps) * np.diff(self.t_grid)[0]
        total_time = n_steps * dt_requested
        dt_cfl = float(self.compute_dt())
        dt_step = min(dt_requested, dt_cfl)

        while total_time - dt_step > 0.0:
            if self.order == 1:
                F = self._compute_first_order_fluxes(
                    U, V_centers, V_faces, inflow_value_U
                )
            elif self.order == 2:
                F = self._compute_second_order_fluxes(
                    U, V_centers, V_faces, inflow_value_U, dt_step
                )
            else:
                raise NotImplementedError("order must be 1 or 2")

            U = U - (dt_step / self.dx) * (F[1:] - F[:-1])
            total_time -= dt_step

        if self.order == 1:
            F = self._compute_first_order_fluxes(U, V_centers, V_faces, inflow_value_U)
        elif self.order == 2:
            F = self._compute_second_order_fluxes(
                U, V_centers, V_faces, inflow_value_U, total_time
            )
        else:
            raise NotImplementedError("order must be 1 or 2")

        U_new = U - (total_time / self.dx) * (F[1:] - F[:-1])
        f_new = self._inverse_generalized_variable(U_new, self.grid)
        # TODO?: self._postprocess_solution(f_new,centers)
        # f_new[0] = f_new[1]
        state.update_f(f_new)

        logger.debug(f"max(|f|) after step: {np.max(np.abs(f_new)):.4g}")
        return

    # ---------------- internal helpers ----------------
    def _unpack_params(self, params: dict = None) -> None:
        """Unpack and strictly validate parameters."""

        # expected keys
        expected_keys = {
            "V_centers",
            "limiter",
            "cfl",
            "inflow_value_U",
            "order",
            "axis",
        }
        provided_keys = set(params.keys())

        # check missing / extra
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
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

        # --- V_centers ---
        self.V_centers = np.asarray(params["V_centers"], dtype=float)
        if self.axis == 0:
            self.V_centers = self.V_centers.T

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
            self.dx_L = faces[1:-1] - centers[:-1]  # distance center to left face
            self.dx_R = centers[1:] - faces[1:-1]  # distance center to right face

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

    def _compute_slopes(self, U: np.ndarray) -> np.ndarray:
        """
        Compute limited slopes on non-uniform grid.
        We'll compute left derivative, right derivative, and central derivative
        with correct Δr factors, then apply the chosen limiter.
        """
        N = self.N
        slopes = np.zeros_like(U)
        if N <= 2:
            return slopes

        centers = self._main_centers

        # left derivative at i (using centers i-1 and i)
        # we compute for i = 1..N-2 (interior centers)
        dL = (U[1:-1] - U[:-2]) / (centers[1:-1] - centers[:-2])  # length N-2
        # right derivative at i (using centers i and i+1)
        dR = (U[2:] - U[1:-1]) / (centers[2:] - centers[1:-1])  # length N-2
        # central derivative across two cells
        dC = (U[2:] - U[:-2]) / (centers[2:] - centers[:-2])  # length N-2

        if self.limiter == "minmod":
            slopes_interior = minmod_multi(dL, dR, dC)
        elif self.limiter == "vanleer":  # generalized
            prod = dL * dR
            slopes_interior = np.where(prod > 0.0, (2.0 * prod) / (dL + dR), 0.0)
        elif self.limiter == "mc":
            slopes_interior = minmod_multi(
                2.0 * dL, 2.0 * dR, 0.5 * dC
            )  # TODO: check factor 0.5
        else:  # should not happen due to earlier check, but just in case
            raise NotImplementedError("limiter must be 'minmod', 'vanleer' or 'mc'")

        slopes[1:-1] = slopes_interior
        slopes[0] = 0.0
        slopes[-1] = 0.0
        return slopes

    def _recontruct_face_states(
        self, U: np.ndarray, slopes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left/right states at internal faces from cell-centered values and slopes.
        Returns UL, UR arrays of length N+1 (faces).
        UL[i] = left state at face i (from right cell)
        UR[i] = right state at face i (from left cell)
        """
        UL = np.zeros(self.N + 1)  # left state of face i (value from right cell)
        UR = np.zeros(self.N + 1)  # right state of face i (value from left cell)
        UR[1:-1] = U[:-1] + slopes[:-1] * self.dx_L  # UR at face i uses left cell (i-1)
        UL[1:-1] = U[1:] - slopes[1:] * self.dx_R  # UL at face i uses right cell (i)
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
        Perform MUSCL-Hancock predictor step to advance interface states to t^{n+1/2}.
        Each interface state is predicted using the donor cell's slope.
        Returns ULh, URh arrays of length N+1 (faces).
        """
        # donor - cell - based
        ULh = UL.copy()
        URh = UR.copy()
        if self.N > 1:
            # UL* (state coming from cell i) uses v_i and slope_i
            ULh[1:-1] -= 0.5 * dt * V_centers[1:] * slopes[1:]
            # UR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            URh[1:-1] -= 0.5 * dt * V_centers[:-1] * slopes[:-1]
        return ULh, URh

    def _boundary_fluxes(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        slopes: np.ndarray,
        inflow_value_U: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Compute fluxes at the domain boundaries (faces 0 and N).
        Left boundary (r=0): assume symmetry => flux=0.
        Right boundary (outer face): use last cell's left state and slope for prediction.
        """

        if self._main_faces[0] <= 0.0 and self.axis == 1:
            # Left face (r=0): symmetry
            flux_L = 0.0
        elif self.axis == 0:
            # Left face (momentum inflow): use first cell right-state, or inflow
            V_in = V_centers[0]
            U_right_inner = U[0]
            flux_L = V_in * (inflow_value_U if V_in >= 0.0 else U_right_inner)
        else:
            raise ValueError("Invalid left face position for the given axis.")

        # Right boundary (outer face):
        # left-state at outer face (from last cell)
        centers = self._main_centers
        faces = self._main_faces
        dx_to_right = faces[-1] - centers[-1]
        U_left_outer = U[-1] + slopes[-1] * dx_to_right
        U_left_outer_star = U_left_outer - 0.5 * dt * V_centers[-1] * slopes[-1]
        V_out = V_centers[-1]  # proxy for face velocity
        flux_R = V_out * (U_left_outer_star if V_out >= 0.0 else inflow_value_U)

        return flux_L, flux_R

    def _compute_first_order_fluxes(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
        inflow_value_U: float,
    ) -> np.ndarray:
        """
        Compute first-order upwind fluxes at faces given conservative variable U and face generalized velocities.
        Returns flux array of length N+1.
        """
        flux_int = np.where(V_faces >= 0.0, V_faces * U[:-1], V_faces * U[1:])
        F = np.empty(self.N + 1, dtype=float)
        if self._main_faces[0] <= 0.0 and self.axis == 1:
            # Left face (r=0): symmetry
            F[0] = 0.0
        elif self._main_faces[0] > 0.0:
            # Left face (momentum inflow): use first cell right-state, or inflow
            V_in = V_centers[0]
            U_right_inner = U[0]
            F[0] = V_in * (inflow_value_U if V_in >= 0.0 else U_right_inner)
        else:
            raise ValueError("Invalid left face position for the given axis.")
        F[1:-1] = flux_int
        # Outer face (use last cell left-state, or inflow)
        V_out = V_centers[-1]
        U_left_outer = U[-1]
        F[-1] = V_out * (U_left_outer if V_out >= 0.0 else inflow_value_U)
        return F

    def _compute_second_order_fluxes(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
        inflow_value_U: float,
        dt: float,
    ) -> np.ndarray:
        """
        Compute second-order MUSCL-Hancock fluxes at faces given conservative generalized variable U and face generalized velocities.
        Returns flux array of length N+1.
        """
        # 1) compute slopes for W in non-uniform grid (minmod / vanleer)
        slopes = self._compute_slopes(U)  # length N (slopes at centers)
        # 2) reconstruct left/right states at internal faces (t^n)
        UL, UR = self._recontruct_face_states(U, slopes)
        # 3) predictor to t^{n+1/2}
        ULh, URh = self._predictor_states(UL, UR, V_centers, slopes, dt)
        # 4) compute V at internal faces and fluxes by upwind using sign of V_face
        flux_int = np.where(V_faces >= 0.0, V_faces * URh[1:-1], V_faces * ULh[1:-1])
        F = np.empty(self.N + 1)
        F[1:-1] = flux_int
        # Note: if V_face>0 donor is left cell => URh (value from left), else ULh (from right)

        # 5) boundary fluxes:
        F[0], F[-1] = self._boundary_fluxes(U, V_centers, slopes, inflow_value_U, dt)
        return F

    def _compute_first_order_fluxes_2d(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        V_faces: np.ndarray,
    ) -> np.ndarray:
        """
        Compute first-order upwind fluxes at faces given conservative variable U and face generalized velocities for 2D case.
        Returns flux matrix with length N+1 on axis and M on other_axis.
        """
        U_left = np.take(U, indices=range(self.N - 1), axis=self.axis)
        U_right = np.take(U, indices=range(1, self.N), axis=self.axis)
        flux_int = np.where(V_faces >= 0.0, V_faces * U_left, V_faces * U_right)

        if self.axis == 1:
            shape = (U.shape[0], self.N)
            F = np.empty(shape, dtype=float)
            F[:, 0] = 0.0
            F[:, 1:-1] = flux_int
            F[:, -1] = V_centers[:, -1] * (
                U[:, -1] if V_centers[:, -1] >= 0.0 else self.inflow_value_U
            )
            return F
        elif self.axis == 0:
            shape = (self.N, U.shape[1])
            F = np.empty(shape, dtype=float)
            F[0, :] = V_centers[0, :] * (
                self.inflow_value_U if V_centers[0, :] >= 0.0 else U[0, :]
            )  # assuming zero inflow for 2D
            F[1:-1, :] = flux_int
            F[-1, :] = V_centers[-1, :] * (
                U[:, -1] if V_centers[-1, :] >= 0.0 else self.inflow_value_U
            )  # assuming zero inflow for 2D
            return F
        else:
            raise ValueError("Invalid axis for main axis.")

    def _compute_second_order_fluxes_2d(
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
        slopes = self._compute_slopes_2d(U)  # shape (N,M) or (M,N)
        # 2) reconstruct left/right states at internal faces (t^n)
        UL, UR = self._recontruct_face_states_2d(U, slopes)
        # 3) predictor to t^{n+1/2}
        ULh, URh = self._predictor_states_2d(UL, UR, V_centers, slopes, dt)
        # 4) compute V at internal faces and fluxes by upwind using sign of V_face
        flux_int = np.where(
            V_faces >= 0.0, V_faces * URh[:, 1:-1], V_faces * ULh[:, 1:-1]
        )

        # TEST
        # seleccionar solo caras internas (1..N-1)
        # Vf_internal = V_faces  # shape (M, N-1)
        # ULh_internal = ULh[:, 1 : self.N]  # estado izquierdo en esas caras
        # URh_internal = URh[:, 1 : self.N]  # estado derecho en esas caras

        # upwind clásico: v>=0 -> toma estado izquierdo; v<0 -> toma estado derecho
        # flux_int = np.where(
        #     Vf_internal >= 0.0,
        #     Vf_internal * URh_internal,
        #     Vf_internal * ULh_internal,
        # )

        shape = (U.shape[0], self.N + 1)
        F = np.empty(shape, dtype=float)
        F[:, 1:-1] = flux_int
        F[:, 0], F[:, -1] = self._boundary_fluxes_2d(U, V_centers, slopes, dt)
        return F

    def _compute_slopes_2d(self, U: np.ndarray) -> np.ndarray:
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
        dL = (U[:, 1:-1] - U[:, :-2]) / (centers[1:-1] - centers[:-2])  # length N-2
        # right derivative at i (using centers i and i+1)
        dR = (U[:, 2:] - U[:, 1:-1]) / (centers[2:] - centers[1:-1])  # length N-2
        # central derivative across two cells
        dC = (U[:, 2:] - U[:, :-2]) / (centers[2:] - centers[:-2])  # length N-2

        if self.limiter == "minmod":
            slopes_interior = minmod_multi(dL, dR, dC)
        elif self.limiter == "vanleer":  # generalized
            prod = dL * dR
            slopes_interior = np.where(prod > 0.0, (2.0 * prod) / (dL + dR), 0.0)
        elif self.limiter == "mc":
            slopes_interior = minmod_multi(2.0 * dL, 2.0 * dR, 0.5 * dC)
        else:  # should not happen due to earlier check, but just in case
            raise NotImplementedError("limiter must be 'minmod', 'vanleer' or 'mc'")
        # Assign slopes to correct slices
        slopes[:, 1:-1] = slopes_interior

        # TEMPORARY SOLUTION
        # Left boundary i=0 : forward/backward differences built with same center spacing
        d_fwd = (U[:, 1] - U[:, 0]) / (centers[1] - centers[0])  # forward 1-step
        d_fwd2 = (U[:, 2] - U[:, 0]) / (
            centers[2] - centers[0]
        )  # two-cell centred forward

        if self.limiter == "minmod":
            slopes[:, 0] = minmod_multi(d_fwd, d_fwd2, d_fwd)
        elif self.limiter == "vanleer":
            prod = d_fwd * d_fwd2
            slopes[:, 0] = np.where(prod > 0.0, (2.0 * prod) / (d_fwd + d_fwd2), 0.0)
        elif self.limiter == "mc":
            slopes[:, 0] = minmod_multi(2.0 * d_fwd, 2.0 * d_fwd2, 0.5 * d_fwd2)
        else:
            slopes[:, 0] = d_fwd  # fallback (shouldn't happen)

        # Right boundary i=N-1 : backward / two-cell centred backward
        d_bwd = (U[:, -1] - U[:, -2]) / (centers[-1] - centers[-2])  # backward 1-step
        d_bwd2 = (U[:, -1] - U[:, -3]) / (
            centers[-1] - centers[-3]
        )  # two-cell centred backward

        if self.limiter == "minmod":
            slopes[:, -1] = minmod_multi(d_bwd, d_bwd2, d_bwd)
        elif self.limiter == "vanleer":
            prod = d_bwd * d_bwd2
            slopes[:, -1] = np.where(prod > 0.0, (2.0 * prod) / (d_bwd + d_bwd2), 0.0)
        elif self.limiter == "mc":
            slopes[:, -1] = minmod_multi(2.0 * d_bwd, 2.0 * d_bwd2, 0.5 * d_bwd2)
        else:
            slopes[:, -1] = d_bwd  # fallback

        # TESTING: zero slopes at boundaries
        # slopes[:, 0] = 0.0
        # slopes[:, -1] = 0.0

        return slopes

    def _recontruct_face_states_2d(
        self, U: np.ndarray, slopes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left/right states at internal faces from cell-centered values and slopes for 2D case.
        Returns UL, UR arrays of shape (N+1,M) or (M,N+1).
        UL[i] = left state at face i (from right cell)
        UR[i] = right state at face i (from left cell)
        """
        UL = np.zeros(
            (U.shape[0], self.N + 1)
        )  # left state of face i (value from right cell)
        UR = np.zeros(
            (U.shape[0], self.N + 1)
        )  # right state of face i (value from left cell)
        UR[:, 1:-1] = U[:, 0 : self.N - 1] + slopes[:, 0 : self.N - 1] * self.dx_L
        UL[:, 1:-1] = U[:, 1 : self.N] - slopes[:, 1 : self.N] * self.dx_R

        # TEMPORARY FIX FOR BOUNDARIES
        UR[:, 0] = U[:, 0] - slopes[:, 0] * self.dx_L[0]
        UR[:, -1] = U[:, -1]
        UL[:, 0] = U[:, 0]
        UL[:, -1] = U[:, -1] + slopes[:, -1] * self.dx_R[-1]

        # TESTING: direct assignment at boundaries
        # UR[:, 0] = U[:, 0]
        # UL[:, -1] = U[:, -1]
        return UL, UR

    def _predictor_states_2d(
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
            ULh[:, 1:-1] -= 0.5 * dt * V_centers[:, 1:] * slopes[:, 1:]
            # UR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            URh[:, 1:-1] -= 0.5 * dt * V_centers[:, :-1] * slopes[:, :-1]

        # TEMPORARY FIX FOR BOUNDARIES
        # ULh[:, 0] -= 0.5 * dt * V_centers[:, 0] * slopes[:, 0]
        # URh[:, -1] -= 0.5 * dt * V_centers[:, -1] * slopes[:, -1]

        return ULh, URh

    def _boundary_fluxes_2d(
        self,
        U: np.ndarray,
        V_centers: np.ndarray,
        slopes: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute fluxes at the domain boundaries (faces 0 and N) for 2D case.
        Left boundary (r=0): assume symmetry => flux=0.
        Right boundary (outer face): use last cell's left state and slope for prediction.
        """

        flux_L = np.zeros(U.shape[0], dtype=float)
        if self._main_faces[0] <= 0.0 and self.axis == 1:
            # Left face (r=0): symmetry
            flux_L[:] = 0.0
        elif self.axis == 0:
            # Left face (momentum inflow): use first cell right-state, or inflow
            slp = (V_centers[:, 1] - V_centers[:, 0]) / (
                self._main_centers[1] - self._main_centers[0]
            )
            V_in = V_centers[:, 0] - slp * (self._main_centers[0] - self._main_faces[0])
            U_right_inner = U[:, 0] - slopes[:, 0] * (
                self._main_centers[0] - self._main_faces[0]
            )
            # V_in = V_centers[:, 0]  # proxy for face velocity
            # U_right_inner = U[:, 0]
            flux_L = V_in * np.where(
                V_in >= 0.0,
                0.0,
                U_right_inner,
            )
        else:
            raise ValueError("Invalid left face position for the given axis.")

        # Right boundary (outer face):
        dx_to_right = self._main_faces[-1] - self._main_centers[-1]
        U_left_outer = U[:, -1] + slopes[:, -1] * dx_to_right

        U_left_outer_star = U_left_outer - 0.5 * dt * V_centers[:, -1] * slopes[:, -1]
        V_out = V_centers[:, -1]  # proxy for face velocity
        flux_R = V_out * np.where(
            V_out >= 0.0,
            U_left_outer_star,
            self.inflow_value_U,
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
def _numba_second_order_fluxes_2d_core(
    U, V_centers, V_faces, centers, faces, dt, limiter
):
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
