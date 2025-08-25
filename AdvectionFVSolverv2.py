import numpy as np
from typing import Literal


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
        r_faces: np.ndarray,  # length N+1, r_faces[0]=0
        v_centers: np.ndarray,  # length N (vel at centers)
        limiter: Literal["minmod", "vanleer"] = "minmod",
        cfl: float = 0.8,
        inflow_value_W: float = 0.0,
        order: Literal[1, 2] = 2,
    ) -> None:
        self.r_faces = np.asarray(r_faces, dtype=float)
        if self.r_faces.ndim != 1 or self.r_faces.size < 2:
            raise ValueError("r_faces must be 1-D array length >= 2")
        if abs(self.r_faces[0]) > 1e-14:
            raise ValueError("r_faces[0] must be 0.0")
        if np.any(np.diff(self.r_faces) <= 0.0):
            raise ValueError("r_faces must be strictly increasing")

        self.N = self.r_faces.size - 1
        self.r_centers = 0.5 * (self.r_faces[1:] + self.r_faces[:-1])
        self.dr = self.r_faces[1:] - self.r_faces[:-1]  # Δr_i
        self.dx_c = self.r_centers[1:] - self.r_centers[:-1]  # center-to-center (N-1)
        # distances center->face for non-uniform reconstruction
        self.dx_L = self.r_faces[1:-1] - self.r_centers[:-1]  # for WR (from left cell)
        self.dx_R = self.r_centers[1:] - self.r_faces[1:-1]  # for WL (from right cell)

        self.v_centers = np.asarray(v_centers, dtype=float)
        if self.v_centers.shape != (self.N,):
            raise ValueError("v_centers must have shape (N,)")

        if limiter not in {"minmod", "vanleer"}:
            raise ValueError("limiter must be 'minmod' or 'vanleer'")
        self.limiter = limiter
        self.cfl = float(cfl)
        self.inflow_value_W = float(inflow_value_W)
        self.order = order

    # ---------------- public API ----------------
    def set_velocity(self, v_centers: np.ndarray) -> None:
        v_centers = np.asarray(v_centers, dtype=float)
        if v_centers.shape != (self.N,):
            raise ValueError("v_centers must have shape (N,)")
        self.v_centers = v_centers

    def _face_velocity_interpolated(self) -> np.ndarray:
        """
        Linear interpolation of v to internal face positions r_{i+1/2}.
        We use a linear weight by distances (more accurate for non-uniform grids).
        Returns array length N-1.
        """
        if self.N <= 1:
            return np.zeros(0)
        # For face i (between centers i-1 and i): weight by distance to centers
        # w_left = dist(center i -> face) / (dist centers)
        # Using our precomputed dx_L/dx_R arrays:
        # dx_total = dx_L + dx_R = r_center[i] - r_center[i-1]
        left_weights = self.dx_R / (
            self.dx_L + self.dx_R
        )  # actually weight for left center when evaluating face position
        # But safer to compute explicit linear interpolation by coordinate:
        r_face = 0.5 * (self.r_faces[1:-1] + self.r_faces[1:-1])  # just r_faces[1:-1]
        # simpler: use linear interpolation by distances:
        v_left = self.v_centers[:-1]
        v_right = self.v_centers[1:]
        # weight left by distance from face to right center:
        # dist_left = r_face - r_center_left = dx_L
        dist_left = self.dx_L
        dist_right = self.dx_R
        denom = dist_left + dist_right
        # avoid division by zero (degenerate)
        denom = np.where(denom == 0.0, 1.0, denom)
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

    def advance(self, U: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance U by dt. Works on W = r^2 * U internally.
        """
        U = np.asarray(U, dtype=float)
        if U.shape != (self.N,):
            raise ValueError("U shape mismatch")
        W = (self.r_centers**2) * U  # conservative variable

        if self.order == 1:
            # Godunov upwind (first order)
            v_faces = self._face_velocity_interpolated()
            # internal fluxes length N-1: if v_face>=0 use left cell W[i-1], else right cell W[i]
            flux_int = np.where(v_faces >= 0.0, v_faces * W[:-1], v_faces * W[1:])
            F = np.empty(self.N + 1)
            F[0] = 0.0
            F[1:-1] = flux_int
            # outer face (use last cell left-state, or inflow)
            v_out = self.v_centers[-1]
            W_left_outer = W[-1]
            F[-1] = v_out * (W_left_outer if v_out >= 0.0 else self.inflow_value_W)
            W_new = W - (dt / self.dr) * (F[1:] - F[:-1])
            U_new = W_new / (self.r_centers**2)
            return U_new

        # ---------------- second order MUSCL-Hancock ----------------
        # 1) compute slopes for W in non-uniform grid (minmod / vanleer)
        slopes = self._compute_slopes(W)  # length N (slopes at centers)

        # 2) reconstruct left/right states at internal faces (t^n)
        WL = np.zeros(self.N + 1)  # left state of face i (value from right cell)
        WR = np.zeros(self.N + 1)  # right state of face i (value from left cell)
        WR[1:-1] = W[:-1] + slopes[:-1] * self.dx_L  # WR at face i uses left cell (i-1)
        WL[1:-1] = W[1:] - slopes[1:] * self.dx_R  # WL at face i uses right cell (i)

        # 3) predictor to t^{n+1/2}: donor-cell-based
        WLh = WL.copy()
        WRh = WR.copy()
        if self.N > 1:
            # WL* (state coming from cell i) uses v_i and slope_i
            WLh[1:-1] -= 0.5 * dt * self.v_centers[1:] * slopes[1:]
            # WR* (state coming from cell i-1) uses v_{i-1} and slope_{i-1}
            WRh[1:-1] -= 0.5 * dt * self.v_centers[:-1] * slopes[:-1]

        # 4) compute v at internal faces and fluxes by upwind using sign of v_face
        v_faces = self._face_velocity_interpolated()  # length N-1
        flux_int = np.where(v_faces >= 0.0, v_faces * WRh[1:-1], v_faces * WLh[1:-1])
        # Note: if v_face>0 donor is left cell => WRh (value from left), else WLh (from right)

        # 5) boundary fluxes:
        F = np.empty(self.N + 1)
        F[0] = 0.0
        F[1:-1] = flux_int
        # rightmost face:
        # left-state at outer face (from last cell)
        dx_to_right = self.r_faces[-1] - self.r_centers[-1]
        W_left_outer = W[-1] + slopes[-1] * dx_to_right
        W_left_outer_star = W_left_outer - 0.5 * dt * self.v_centers[-1] * slopes[-1]
        v_out = self.v_centers[-1]  # proxy for face velocity
        F[-1] = v_out * (W_left_outer_star if v_out >= 0.0 else self.inflow_value_W)

        # 6) update
        W_new = W - (dt / self.dr) * (F[1:] - F[:-1])
        U_new = W_new / (self.r_centers**2)
        return U_new

    # ---------------- internal helpers ----------------
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
        else:  # vanLeer (generalized)
            prod = dL * dR
            slopes_interior = np.where(prod > 0.0, (2.0 * prod) / (dL + dR), 0.0)

        slopes[1:-1] = slopes_interior
        slopes[0] = 0.0
        slopes[-1] = 0.0
        return slopes


# ---------------- Example / test ----------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    R = 1.0
    N = 1500
    r_faces = np.linspace(0.0, R, N + 1)
    r_centers = 0.5 * (r_faces[1:] + r_faces[:-1])

    # velocity piecewise
    v_centers = np.where(r_centers < 0.5, 1.0, 0.5)

    solver = AdvectionFVSolver(
        r_faces=r_faces,
        v_centers=v_centers,
        limiter="minmod",  # robust for discontinuities
        cfl=0.8,
        inflow_value_W=0.0,
        order=2,
    )

    # --- Initial condition u(r) and plotting
    U = np.where((r_centers > 0.2) & (r_centers < 0.3), 1.0, 0.0)

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

    # --- Analytical solution for spherical advection (f_0/r^2)
    def analytical_solution(r, t, v_func, r0=0.25, width=0.1):
        """
        Analytical solution for an initial top-hat (unit amplitude) centered at r0
        advected with velocity v0 = v_func(r0). For constant v0 the exact solution
        in spherical symmetry (conserved W = r^2 u) is

            u(r,t) = (r_init / r)^2 * u0(r_init)   with   r_init = r - v0 * t

        where u0 is the initial profile (here a top-hat of amplitude 1 around r0).
        """
        r = np.asarray(r)
        # evaluate characteristic speed for the initial parcel at r0
        v0 = v_func(r0) if callable(v_func) else float(v_func)

        # initial position corresponding to current r
        r_init = r - v0 * t

        # initial top-hat profile: u0(r_init) = 1.0 if within width around r0, else 0
        init_mask = (r_init > 0.0) & (np.abs(r_init - r0) < 0.5 * width)

        u = np.zeros_like(r, dtype=float)
        # apply conservation of W = r^2 u  => u = (r_init^2 / r^2) * u0(r_init)
        valid = init_mask & (r > 0.0)
        u[valid] = (r_init[valid] / r[valid]) ** 2 * 1.0

        return u

    v_func = lambda r: 1.0 if r < 0.5 else 0.5

    # --- Time loop

    # Plot analytical solution at t=0 for reference
    U_analytical = analytical_solution(r_centers, 0.0, v_func)
    (analytical_line,) = ax.plot(r_centers, U_analytical, "r--", label="Analytical")
    ax.legend()

    t = 0.0
    t_end = 0.2
    while t < t_end:
        dt = min(solver.compute_dt(), t_end - t)
        U = solver.advance(U, dt)
        t += dt
        line.set_ydata(U)
        # Update analytical solution
        U_analytical = analytical_solution(r_centers, t, v_func)
        analytical_line.set_ydata(U_analytical)
        ax.set_title(f"t = {t:.4f}")
        plt.pause(0.1)

    plt.ioff()
    plt.show()
