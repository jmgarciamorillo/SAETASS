import numpy as np
from typing import Literal
from State import State, SliceState
from Grid import Grid
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


def minmod_multi(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized minmod for three arrays (same shape)."""
    # returns sign * min(|a|,|b|,|c|) when all same sign, else 0
    s = np.sign(a)
    same = (a * b > 0.0) & (a * c > 0.0)
    out = np.zeros_like(a)
    out[same] = s[same] * np.minimum(
        np.minimum(np.abs(a[same]), np.abs(b[same])), np.abs(c[same])
    )
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
        """Return the main axis centers based on index."""
        if self.index == 0:
            return self.r_centers
        elif self.index == 1:
            return self.p_centers
        else:
            raise ValueError("Invalid index for main axis.")

    @property
    def _main_faces(self) -> np.ndarray:
        """Return the main axis faces based on index."""
        if self.index == 0:
            return self.r_faces
        elif self.index == 1:
            return self.p_faces
        else:
            raise ValueError("Invalid index for main axis.")

    # ---------------- public API ----------------
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

        # Generalized velocities at left/right centers
        V_left = self.V_centers[:-1]
        V_right = self.V_centers[1:]

        # Distances from face to neighboring centers
        dist_left = self.dx_L
        dist_right = self.dx_R
        denom = dist_left + dist_right

        # Prevent division by zero in degenerate cells
        denom = np.where(denom == 0.0, 1.0, denom)

        # Linear interpolation: closer center gets higher weight
        V_face = (V_left * dist_right + V_right * dist_left) / denom
        return V_face

    def compute_dt(self) -> float:
        """Compute stable dt from max(|V_face|) and min dx."""
        V_faces = self._face_generalized_velocity_interpolated()
        Vmax = float(np.max(np.abs(V_faces))) if V_faces.size else 0.0
        # Include outer face using last center as proxy
        Vmax = max(Vmax, abs(self.V_centers[-1]))
        if Vmax <= 0.0:
            return np.inf
        return float(self.cfl * np.min(self.dx) / Vmax)

    def advance(self, n_steps: int, state: State) -> None:
        """
        Advance the state by n_steps * dt.

        Automatically adapts to whether the main grid is spatial (r) or momentum (p).

        If the problem is 1D (no secondary axis), processes it directly.
        If 2D (main × secondary), processes each slice independently along the secondary axis.
        """

        # Determine axis
        main_axis = "r" if self.index == 0 else "p"
        other_axis = "p" if main_axis == "r" else "r"
        other_centers = getattr(self, f"{other_axis}_centers", None)

        # Detect 1D or 2D case
        is_1d = other_centers is None or state.ndim == 1

        if is_1d:
            # Case 1D: single field dependent on main axis
            logger.debug(f"Advancing 1D state along {main_axis}")
            self._advance_slice(n_steps, state)
        else:
            # Case 2D: process slices along secondary axis
            n_other = len(other_centers)
            state_n_other = getattr(state, f"n_{other_axis}", None)

            if state_n_other != n_other:
                raise ValueError(
                    f"State shape mismatch: expected {n_other} {other_axis}-slices, "
                    f"got {state_n_other}"
                )

            for i in range(n_other):
                # Create a view/slice of the state at fixed other_axis index
                slice_state = SliceState(state, i)
                logger.debug(f"Advancing slice {i+1}/{n_other} along {main_axis}")
                self._advance_slice(n_steps, slice_state)

        return

    def _advance_slice(self, n_steps: int, state: State) -> None:
        """
        Advance a single slice (1D). Works with a generalized conservative variable W
        inside the solver. All axis-dependent behaviors are delegated to hooks.
        """
        f = np.asarray(state.get_f().copy(), dtype=float)
        centers = (
            self._main_centers
        )  # this returns e.g. self.r_centers or self.p_centers
        N = self.N

        if f.shape != (N,):
            raise ValueError(f"U shape mismatch: expected ({N},), got {f.shape}")

        U = self._generalized_variable(f, centers)  # Convert to generalized variable

        dt_requested = float(n_steps) * np.diff(self.t_grid)[0]
        total_time = n_steps * dt_requested
        dt_cfl = float(self.compute_dt())
        dt_step = min(dt_requested, dt_cfl)

        V_faces = self._face_generalized_velocity_interpolated()

        while total_time - dt_step > 0.0:
            if self.order == 1:
                F = self._compute_first_order_fluxes(U, V_faces)
            elif self.order == 2:
                F = self._compute_second_order_fluxes(U, V_faces, dt_step)
            else:
                raise NotImplementedError("order must be 1 or 2")

            U = U - (dt_step / self.dx) * (F[1:] - F[:-1])
            total_time -= dt_step

        if self.order == 1:
            F = self._compute_first_order_fluxes(U, V_faces)
        elif self.order == 2:
            F = self._compute_second_order_fluxes(U, V_faces, total_time)
        else:
            raise NotImplementedError("order must be 1 or 2")

        U_new = U - (total_time / self.dx) * (F[1:] - F[:-1])
        f_new = self._inverse_generalized_variable(U_new, centers)
        # TODO?: self._postprocess_solution(f_new,centers)
        f_new[0] = f_new[1]
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
            "index",
        }
        provided_keys = set(params.keys())

        # check missing / extra
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        if missing:
            raise ValueError(f"Missing required params: {missing}")
        if extra:
            raise ValueError(f"Unexpected params provided: {extra}")

        # --- V_centers ---
        self.V_centers = np.asarray(params["V_centers"], dtype=float)

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
        try:
            self.inflow_value_U: float = float(params["inflow_value_U"])
        except Exception:
            raise ValueError("inflow_value_U must be a float")

        # --- order ---
        order = int(params["order"])
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        self.order: Literal[1, 2] = order

        # --- index ---
        try:
            self.index: int = int(params["index"])
        except Exception:
            raise ValueError("index must be an integer")
        if not (0 <= self.index < 2):
            raise ValueError(
                "index must be 0 (for spatial problems) or 1 (for momentum problems)"
            )

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
        axes = {0: "r", 1: "p"}
        main_axis = axes.get(self.index)

        if main_axis is None:
            raise ValueError(
                f"Invalid index {self.index}: expected 0 (spatial) or 1 (momentum)"
            )

        if not _load_axis(main_axis):
            raise ValueError(
                f"Grid must include {main_axis}_centers for {main_axis}-space problems"
            )

        _init_axis_metrics(main_axis)

        other_axis = "p" if main_axis == "r" else "r"
        if _load_axis(other_axis):
            logger.info(f"Found {other_axis}-grid")
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
        self, UL: np.ndarray, UR: np.ndarray, slopes: np.ndarray, dt: float
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
            ULh[1:-1] -= 0.5 * dt * self.V_centers[1:] * slopes[1:]
            # UR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            URh[1:-1] -= 0.5 * dt * self.V_centers[:-1] * slopes[:-1]
        return ULh, URh

    def _boundary_fluxes(
        self, U: np.ndarray, slopes: np.ndarray, dt: float
    ) -> tuple[float, float]:
        """
        Compute fluxes at the domain boundaries (faces 0 and N).
        Left boundary (r=0): assume symmetry => flux=0.
        Right boundary (outer face): use last cell's left state and slope for prediction.
        """

        # Left boundary (r=0): symmetry
        flux_L = 0.0

        # Right boundary (outer face):
        # left-state at outer face (from last cell)
        centers = self._main_centers
        faces = self._main_faces
        dx_to_right = faces[-1] - centers[-1]
        U_left_outer = U[-1] + slopes[-1] * dx_to_right
        U_left_outer_star = U_left_outer - 0.5 * dt * self.V_centers[-1] * slopes[-1]
        V_out = self.V_centers[-1]  # proxy for face velocity
        flux_R = V_out * (U_left_outer_star if V_out >= 0.0 else self.inflow_value_U)

        return flux_L, flux_R

    def _compute_first_order_fluxes(
        self, U: np.ndarray, V_faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute first-order upwind fluxes at faces given conservative variable U and face generalized velocities.
        Returns flux array of length N+1.
        """
        flux_int = np.where(V_faces >= 0.0, V_faces * U[:-1], V_faces * U[1:])
        F = np.empty(self.N + 1, dtype=float)
        F[0] = 0.0
        F[1:-1] = flux_int
        # Outer face (use last cell left-state, or inflow)
        V_out = self.V_centers[-1]
        U_left_outer = U[-1]
        F[-1] = V_out * (U_left_outer if V_out >= 0.0 else self.inflow_value_U)
        return F

    def _compute_second_order_fluxes(
        self, U: np.ndarray, V_faces: np.ndarray, dt: float
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
        ULh, URh = self._predictor_states(UL, UR, slopes, dt)
        # 4) compute V at internal faces and fluxes by upwind using sign of V_face
        flux_int = np.where(V_faces >= 0.0, V_faces * URh[1:-1], V_faces * ULh[1:-1])
        F = np.empty(self.N + 1)
        F[1:-1] = flux_int
        # Note: if V_face>0 donor is left cell => URh (value from left), else ULh (from right)

        # 5) boundary fluxes:
        F[0], F[-1] = self._boundary_fluxes(U, slopes, dt)
        return F

    @abstractmethod
    def _generalized_variable(self, f: np.ndarray, centers: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _inverse_generalized_variable(
        self, U: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        pass
