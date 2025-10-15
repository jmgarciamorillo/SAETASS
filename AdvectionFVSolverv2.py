import numpy as np
from typing import Literal
from State import State

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


class AdvectionFVSolver:
    """
    FV solver for spherical advection using W = r^2 u (reduced variable).

    Key fixes vs earlier attempts:
      - v evaluated at faces by linear interpolation (weights by distances).
      - slope limiter (minmod) written for non-uniform grid; optional vanLeer kept.
      - MUSCL–Hancock predictor: each interface state predicted using donor cell's slope.
      - CFL computed from face velocities and actual cell widths Δr.
    """

    def __init__(
        self,
        r_centers: np.ndarray,
        t_grid: np.ndarray,
        params: dict,
        **kwargs,
    ) -> None:

        self.t_grid = np.asarray(t_grid, dtype=float)

        self.r_centers = np.asarray(r_centers, dtype=float)
        if self.r_centers.ndim != 1 or self.r_centers.size < 2:
            raise ValueError("r_centers must be 1-D array length >= 2")
        if abs(self.r_centers[0]) > 1e-14:
            raise ValueError("r_centers[0] must be 0.0")
        if np.any(np.diff(self.r_centers) <= 0.0):
            raise ValueError("r_centers must be strictly increasing")

        self.N = self.r_centers.size
        self.r_faces = np.empty(self.N + 1, dtype=float)
        self.r_faces[1:-1] = 0.5 * (self.r_centers[1:] + self.r_centers[:-1])
        self.r_faces[0] = self.r_centers[0] - 0.5 * (
            self.r_centers[1] - self.r_centers[0]
        )
        self.r_faces[-1] = self.r_centers[-1] + 0.5 * (
            self.r_centers[-1] - self.r_centers[-2]
        )
        self.dr = self.r_faces[1:] - self.r_faces[:-1]  # Δr_i
        self.dx_c = self.r_centers[1:] - self.r_centers[:-1]  # center-to-center (N-1)
        # distances center->face for non-uniform reconstruction
        self.dx_L = self.r_faces[1:-1] - self.r_centers[:-1]  # for WR (from left cell)
        self.dx_R = self.r_centers[1:] - self.r_faces[1:-1]  # for WL (from right cell)

        self.params = params or {}
        self._unpack_params()

    # ---------------- public API ----------------
    def set_velocity(self, v_centers: np.ndarray) -> None:
        v_centers = np.asarray(v_centers, dtype=float)
        if v_centers.shape != (self.N,):
            raise ValueError("v_centers must have shape (N,)")
        self.v_centers = v_centers

    def _face_velocity_interpolated(self) -> np.ndarray:
        """
        Linear interpolation of velocities from cell centers to internal face positions r_{i+1/2}.
        Uses distance-weighted linear interpolation (works for non-uniform grids).
        Returns
        -------
        v_face : np.ndarray
            Interpolated velocities at faces, length N-1.
        """
        if self.N <= 1:
            return np.zeros(0)

        # velocities at left/right centers
        v_left = self.v_centers[:-1]
        v_right = self.v_centers[1:]

        # distances from face to neighboring centers
        dist_left = self.dx_L
        dist_right = self.dx_R
        denom = dist_left + dist_right

        # prevent division by zero in degenerate cells
        denom = np.where(denom == 0.0, 1.0, denom)

        # linear interpolation: closer center gets higher weight
        v_face = (v_left * dist_right + v_right * dist_left) / denom
        return v_face

    def compute_dt(self) -> float:
        """Compute stable dt from max(|v_face|) and min Δr."""
        v_faces = self._face_velocity_interpolated()
        vmax = float(np.max(np.abs(v_faces))) if v_faces.size else 0.0
        # include outer face using last center as proxy
        vmax = max(vmax, abs(self.v_centers[-1]))
        if vmax <= 0.0:
            return np.inf
        return float(self.cfl * np.min(self.dr) / vmax)

    def advance(self, n_steps: int, state: State) -> np.ndarray:
        """
        Advance U by n_steps * dt. Works on W = r^2 * U internally.
        """
        U = np.asarray(state.f.copy(), dtype=float)
        if U.shape != (self.N,):
            raise ValueError("U shape mismatch")
        W = (self.r_centers**2) * U  # conservative variable

        dt = float(n_steps) * np.diff(self.t_grid)[0]
        total_time = n_steps * dt
        dt_cfl = self.compute_dt()
        dt_step = min(dt, dt_cfl)
        v_faces = self._face_velocity_interpolated()

        if self.order == 1:
            while total_time - dt_step > 0.0:
                # internal fluxes length N-1: if v_face>=0 use left cell W[i-1], else right cell W[i]
                flux_int = np.where(v_faces >= 0.0, v_faces * W[:-1], v_faces * W[1:])
                F = np.empty(self.N + 1)
                F[0] = 0.0
                F[1:-1] = flux_int
                # outer face (use last cell left-state, or inflow)
                v_out = self.v_centers[-1]
                W_left_outer = W[-1]
                F[-1] = v_out * (W_left_outer if v_out >= 0.0 else self.inflow_value_W)
                W_new = W - (dt_step / self.dr) * (F[1:] - F[:-1])
                total_time = total_time - dt_step
            # internal fluxes length N-1: if v_face>=0 use left cell W[i-1], else right cell W[i]
            flux_int = np.where(v_faces >= 0.0, v_faces * W[:-1], v_faces * W[1:])
            F = np.empty(self.N + 1)
            F[0] = 0.0
            F[1:-1] = flux_int
            # outer face (use last cell left-state, or inflow)
            v_out = self.v_centers[-1]
            W_left_outer = W[-1]
            F[-1] = v_out * (W_left_outer if v_out >= 0.0 else self.inflow_value_W)
            W_new = W - (total_time / self.dr) * (F[1:] - F[:-1])
            U_new = W_new / (self.r_centers**2)
            U_new[0] = U_new[1]  # avoid issues at r=0

            state.update_f(U_new)
            return

        # ---------------- second order MUSCL-Hancock ----------------

        while total_time - dt_step > 0.0:
            # 1) compute slopes for W in non-uniform grid (minmod / vanleer)
            slopes = self._compute_slopes(W)  # length N (slopes at centers)

            # 2) reconstruct left/right states at internal faces (t^n)
            WL, WR = self._recontruct_face_states(W, slopes)

            # 3) predictor to t^{n+1/2}
            WLh, WRh = self._predictor_states(WL, WR, slopes, dt_step)

            # 4) compute v at internal faces and fluxes by upwind using sign of v_face
            flux_int = np.where(
                v_faces >= 0.0, v_faces * WRh[1:-1], v_faces * WLh[1:-1]
            )
            F = np.empty(self.N + 1)
            F[1:-1] = flux_int
            # Note: if v_face>0 donor is left cell => WRh (value from left), else WLh (from right)

            # 5) boundary fluxes:
            F[0], F[-1] = self._boundary_fluxes(W, slopes, dt_step)

            # 6) update
            W_new = W - (dt_step / self.dr) * (F[1:] - F[:-1])
            W = W_new
            total_time = total_time - dt_step

        # final step with remaining time
        # 1) compute slopes for W in non-uniform grid (minmod / vanleer)
        slopes = self._compute_slopes(W)  # length N (slopes at centers)

        # 2) reconstruct left/right states at internal faces (t^n)
        WL, WR = self._recontruct_face_states(W, slopes)

        # 3) predictor to t^{n+1/2}
        WLh, WRh = self._predictor_states(WL, WR, slopes, dt_step)

        # 4) compute v at internal faces and fluxes by upwind using sign of v_face
        flux_int = np.where(v_faces >= 0.0, v_faces * WRh[1:-1], v_faces * WLh[1:-1])
        F = np.empty(self.N + 1)
        F[1:-1] = flux_int
        # Note: if v_face>0 donor is left cell => WRh (value from left), else WLh (from right)

        # 5) boundary fluxes:
        F[0], F[-1] = self._boundary_fluxes(W, slopes, dt_step)

        # 6) update
        W_new = W - (total_time / self.dr) * (F[1:] - F[:-1])
        U_new = W_new / (self.r_centers**2)
        U_new[0] = U_new[1]  # avoid issues at r=0

        logger.debug(f"max |U| after step: {np.max(np.abs(U_new)):.4g}")

        state.update_f(U_new)
        return

    # ---------------- internal helpers ----------------
    def _unpack_params(self):
        """Unpack and strictly validate parameters."""

        # expected keys
        expected_keys = {"v_centers", "limiter", "cfl", "inflow_value_W", "order"}
        provided_keys = set(self.params.keys())

        # check missing / extra
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        if missing:
            raise ValueError(f"Missing required params: {missing}")
        if extra:
            raise ValueError(f"Unexpected params provided: {extra}")

        # --- v_centers ---
        self.v_centers = np.asarray(self.params["v_centers"], dtype=float)
        if self.v_centers.shape != (self.N,):
            raise ValueError(f"v_centers must have shape ({self.N},)")

        # --- limiter ---
        limiter = self.params["limiter"]
        if limiter not in {"minmod", "vanleer", "mc"}:
            raise ValueError("limiter must be 'minmod', 'vanleer' or 'mc'")
        self.limiter: Literal["minmod", "vanleer", "mc"] = limiter

        # --- cfl ---
        try:
            self.cfl: float = float(self.params["cfl"])
        except Exception:
            raise ValueError("cfl must be a float")

        # --- inflow_value_W ---
        try:
            self.inflow_value_W: float = float(self.params["inflow_value_W"])
        except Exception:
            raise ValueError("inflow_value_W must be a float")

        # --- order ---
        order = int(self.params["order"])
        if order not in {1, 2}:
            raise ValueError("order must be 1 or 2")
        self.order: Literal[1, 2] = order

    def _compute_slopes(self, Q: np.ndarray) -> np.ndarray:
        """
        Compute limited slopes on non-uniform grid.
        We'll compute left derivative, right derivative, and central derivative
        with correct Δr factors, then apply the chosen limiter.
        """
        N = self.N
        slopes = np.zeros_like(Q)
        if N <= 2:
            return slopes

        # left derivative at i (using centers i-1 and i)
        # we compute for i = 1..N-2 (interior centers)
        dL = (Q[1:-1] - Q[:-2]) / (
            self.r_centers[1:-1] - self.r_centers[:-2]
        )  # length N-2
        # right derivative at i (using centers i and i+1)
        dR = (Q[2:] - Q[1:-1]) / (
            self.r_centers[2:] - self.r_centers[1:-1]
        )  # length N-2
        # central derivative across two cells
        dC = (Q[2:] - Q[:-2]) / (self.r_centers[2:] - self.r_centers[:-2])  # length N-2

        if self.limiter == "minmod":
            slopes_interior = minmod_multi(dL, dR, dC)
        elif self.limiter == "vanleer":  # generalized
            prod = dL * dR
            slopes_interior = np.where(prod > 0.0, (2.0 * prod) / (dL + dR), 0.0)
        elif self.limiter == "mc":
            slopes_interior = minmod_multi(
                2.0 * dL, 2.0 * dR, 0.5 * dC
            )  # TODO: check factor 0.5
        else:  # should not happen due to earlier check, but just in case satate it is not implemented
            raise NotImplementedError("limiter must be 'minmod', 'vanleer' or 'mc'")

        slopes[1:-1] = slopes_interior
        slopes[0] = 0.0
        slopes[-1] = 0.0
        return slopes

    def _recontruct_face_states(
        self, W: np.ndarray, slopes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct left/right states at internal faces from cell-centered values and slopes.
        Returns WL, WR arrays of length N+1 (faces).
        WL[i] = left state at face i (from right cell)
        WR[i] = right state at face i (from left cell)
        """
        WL = np.zeros(self.N + 1)  # left state of face i (value from right cell)
        WR = np.zeros(self.N + 1)  # right state of face i (value from left cell)
        WR[1:-1] = W[:-1] + slopes[:-1] * self.dx_L  # WR at face i uses left cell (i-1)
        WL[1:-1] = W[1:] - slopes[1:] * self.dx_R  # WL at face i uses right cell (i)
        return WL, WR

    def _predictor_states(
        self, WL: np.ndarray, WR: np.ndarray, slopes: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform MUSCL-Hancock predictor step to advance interface states to t^{n+1/2}.
        Each interface state is predicted using the donor cell's slope.
        Returns WLh, WRh arrays of length N+1 (faces).
        """
        # donor - cell - based
        WLh = WL.copy()
        WRh = WR.copy()
        if self.N > 1:
            # WL* (state coming from cell i) uses v_i and slope_i
            WLh[1:-1] -= 0.5 * dt * self.v_centers[1:] * slopes[1:]
            # WR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            WRh[1:-1] -= 0.5 * dt * self.v_centers[:-1] * slopes[:-1]
        return WLh, WRh

    def _boundary_fluxes(
        self, W: np.ndarray, slopes: np.ndarray, dt: float
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
        dx_to_right = self.r_faces[-1] - self.r_centers[-1]
        W_left_outer = W[-1] + slopes[-1] * dx_to_right
        W_left_outer_star = W_left_outer - 0.5 * dt * self.v_centers[-1] * slopes[-1]
        v_out = self.v_centers[-1]  # proxy for face velocity
        flux_R = v_out * (W_left_outer_star if v_out >= 0.0 else self.inflow_value_W)

        return flux_L, flux_R


# ---------------- Example / test ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    R = 1.0
    N = 1500
    r_centers = np.linspace(0.0, R, N)

    # velocity piecewise
    v_centers = np.where(r_centers < 0.5, 1.0, 0.05)

    # --- Initial condition u(r) and plotting
    U = np.where((r_centers > 0.4) & (r_centers < 0.5), 1.0, 0.0)
    t_grid = np.linspace(0.0, 0.2, 100)

    solver_params = {
        "v_centers": v_centers,
        "limiter": "minmod",  # robust for discontinuities
        "cfl": 0.8,
        "inflow_value_W": 0.0,
        "order": 2,
    }

    solver = AdvectionFVSolver(
        r_centers=r_centers,
        f_values=U,
        params=solver_params,
        t_grid=t_grid,
    )

    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(r_centers, U, label="U(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("U")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("t = 0.0000")
    plt.show()

    # also show velocity
    fig2, ax2 = plt.subplots()
    ax2.plot(r_centers, U, label="Initial U(r)")
    ax2.plot(r_centers, v_centers, label="v(r)")
    ax2.set_xlabel("r")
    ax2.set_ylabel("Value")
    ax2.set_title("Initial Profiles")
    ax2.legend()
    plt.show()

    # --- Analytical solution for spherical advection (general v(r)) - REPLACED ---
    def analytical_solution(r, t, v_func, r0=0.45, width=0.1, u0_func=None):
        """
        Compute analytical (characteristic) solution for an initial profile u0 advected
        by a time-independent radial velocity field v(r) > 0.

        This routine finds, for each target radius r_tar, the initial radius r_init such
        that the travel time along characteristics from r_init to r_tar equals t:

            t = ∫_{r_init}^{r_tar} 1 / v(s) ds

        Then conservation of W = r^2 u gives

            u(r_tar, t) = (r_init^2 / r_tar^2) * u0(r_init)

        Notes and assumptions:
        - v_func must be a callable v_func(r_array) -> array_like (vectorized) or
          accept scalar r and return scalar. The implementation vectorizes safety.
        - This handles spatially varying and discontinuous (piecewise) v(r) by
          numerically integrating 1/v and inverting the map s(r)=∫_0^r 1/v.
        - If t is larger than the travel time from r=0 to r_tar, we set r_init=0.
        - If r_tar == 0 we return the value at small radius (use neighbor) to avoid
          division by zero; callers may enforce u(0)=u(1).
        - u0_func(r) provides the initial primitive u at radius r; if None, a top-hat
          centered at r0 with width is assumed (used in the earlier example).
        """
        r = np.asarray(r, dtype=float)
        if np.any(r < 0):
            raise ValueError("r must be non-negative")

        # prepare initial profile function
        if u0_func is None:

            def u0_func_local(rq):
                rq = np.asarray(rq)
                return np.where((rq > 0.0) & (np.abs(rq - r0) < 0.45 * width), 1.0, 0.0)

            u0_fn = u0_func_local
        else:
            u0_fn = u0_func

        # build an evaluation grid for inversion: use a fine grid up to max(r)
        rmax = float(np.max(r))
        if rmax <= 0.0:
            # trivial case: all at zero
            return np.zeros_like(r)

        ngrid = max(2001, int(10 * r.size))  # sufficiently fine
        r_eval = np.linspace(0.0, rmax, ngrid)

        # evaluate v on the grid (vectorize callable if needed)
        v_eval = np.asarray(v_func(r_eval)) if callable(v_func) else np.asarray(v_func)
        v_eval = np.asarray(v_eval, dtype=float)

        # safety: where v_eval <= 0 (stalled or inward flow) we avoid division by zero.
        # For robust behavior, treat non-positive velocities as very small positive values
        # when computing travel times outward; but warn the user.
        if np.any(v_eval <= 0.0):
            # we choose to treat non-positive v as tiny positive to allow forward integration;
            # this avoids infinities but the physical meaning is limited. Prefer a warning.
            import warnings

            warnings.warn(
                "analytical_solution: v(r) has non-positive values; treating them as small positive for travel-time integration."
            )
            v_eval = np.where(v_eval <= 0.0, 1e-12, v_eval)

        # cumulative integral s(r) = ∫_0^r 1/v(s) ds using trapezoidal rule
        invv = 1.0 / v_eval
        # cumulative trapezoid
        s = np.empty_like(r_eval)
        s[0] = 0.0
        # s[i] = s[i-1] + 0.5*(invv[i-1]+invv[i])*(r_eval[i]-r_eval[i-1])
        diffs = np.diff(r_eval)
        s[1:] = np.cumsum(0.5 * (invv[:-1] + invv[1:]) * diffs)

        # now for each target r_tar compute s_tar and target s_init = s_tar - t
        s_tar = np.interp(r, r_eval, s)
        s_init_target = s_tar - float(t)

        # if s_init_target <= 0 => r_init at or below 0, set r_init = 0
        # otherwise invert s -> r_init by linear interpolation of s(r_eval)
        # s is non-decreasing since invv >= 0 (we forced positivity)
        r_init = np.empty_like(r)
        # for values <=0 set to 0
        mask_neg = s_init_target <= 0.0
        r_init[mask_neg] = 0.0
        mask_pos = ~mask_neg
        if np.any(mask_pos):
            # s is monotonically increasing; invert with interp
            r_init[mask_pos] = np.interp(s_init_target[mask_pos], s, r_eval)

        # compute u0 at r_init
        u0_at_rinit = u0_fn(r_init)

        # build solution u(r,t) = (r_init^2 / r^2) * u0(r_init), with care at r==0
        u = np.zeros_like(r)
        nonzero_mask = r > 0.0
        # where r>0 and r_init>=0
        valid = nonzero_mask & (r_init >= 0.0)
        u[valid] = (r_init[valid] ** 2) / (r[valid] ** 2) * u0_at_rinit[valid]

        # handle r==0: set to neighbor value to avoid division by zero
        if r.size >= 2:
            u[r == 0.0] = u[np.where(r > 0.0)[0][0]]
        else:
            u[r == 0.0] = u0_at_rinit[r == 0.0]

        return u

    # v_func
    def v_func(r):
        return np.where(r < 0.5, 1.0, 0.05)

    # --- Time loop

    # Plot analytical solution at t=0 for reference
    U_analytical = analytical_solution(r_centers, 0.0, v_func)
    (analytical_line,) = ax.plot(r_centers, U_analytical, "r--", label="Analytical")
    ax.legend()

    t = 0.0
    t_end = 0.2
    i = 0
    while t < t_end:
        i += 1
        dt = min(solver.compute_dt(), t_end - t)

        U = solver.advance(1)
        t += dt
        line.set_ydata(U)
        solver.f_values = U
        # Update analytical solution
        U_analytical = analytical_solution(r_centers, t_grid[i], v_func)
        analytical_line.set_ydata(U_analytical)
        ax.set_title(f"t = {t:.4f}")
        plt.pause(0.1)

    plt.ioff()
    plt.show()
