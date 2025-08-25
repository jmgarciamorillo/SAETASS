import numpy as np
from typing import Optional, Literal


class AdvectionFVSolver:
    r"""
    Finite-Volume solver for spherically symmetric advection with known velocity:

        \partial_t u + (1/r^2) \partial_r [ r^2 v(r,t) u ] = 0

    Discretization:
      - Radial 1D FV on spherical shells.
      - MUSCL–Hancock reconstruction (piecewise-linear) with slope limiter (MC or van Leer).
      - Upwind face flux using velocity at faces (either averaged or upwinded from centers).
      - Geometry handled via shell areas A=4\pi r^2 and volumes V=4/3\pi (r_{i+1/2}^3 - r_{i-1/2}^3).

    Inputs (cell-centered):
      - r_centers: array of cell-center radii (monotone increasing, length N)
      - v_centers: array of advection velocities at cell centers (same length as r_centers).

    Notes:
      * If your velocity is time-dependent, call `set_velocity(...)` before each `advance(...)`.
      * At r=0, the first face area is 0 so the left boundary flux is identically zero (spherical symmetry).
      * The MUSCL–Hancock predictor here uses a local half-step based on center flux gradients; the
        conservative update itself is fully geometric.
    """

    def __init__(
        self,
        r_centers: np.ndarray,
        v_centers: np.ndarray,
        limiter: Literal["mc", "vanleer"] = "mc",
        cfl: float = 0.8,
        face_velocity: Literal["avg", "upwind"] = "avg",
    ) -> None:
        # Store cell centers and velocities
        self.r_centers = np.asarray(r_centers, dtype=float)
        self.v_centers = np.asarray(v_centers, dtype=float)
        if self.r_centers.ndim != 1:
            raise ValueError("r_centers must be 1-D")
        if self.v_centers.shape != self.r_centers.shape:
            raise ValueError("v_centers must have same shape as r_centers")
        if np.any(np.diff(self.r_centers) <= 0):
            raise ValueError("r_centers must be strictly increasing")

        self.ncells = (
            self.r_centers.size
        )  # TODO: this can be drawn from num_points in a more general way

        # TODO: all the geometry stuff can be moved to a separate class and differetiate between uniform and non-uniform grids
        # Build face locations from centers. Set inner face exactly at r=0 for spherical symmetry.
        self.r_faces = np.empty(self.ncells + 1, dtype=float)
        self.r_faces[1:-1] = 0.5 * (self.r_centers[1:] + self.r_centers[:-1])
        self.r_faces[0] = 0.0
        # Extrapolate outer face using last cell width
        dr_last = self.r_centers[-1] - self.r_centers[-2]
        self.r_faces[-1] = self.r_centers[-1] + 0.5 * dr_last

        # Geometry factors (areas at faces, volumes for cells)
        self.areas = 4.0 * np.pi * self.r_faces**2  # length N+1
        self.volumes = (
            (4.0 / 3.0) * np.pi * (self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3)
        )  # length N

        # Precompute distances for non-uniform grids
        self.dx_c = np.diff(self.r_centers)  # length N-1 (center-to-center)
        self.dx_L = (
            self.r_faces[1:-1] - self.r_centers[:-1]
        )  # distance center(i) -> right face(i+1/2)
        self.dx_R = (
            self.r_centers[1:] - self.r_faces[1:-1]
        )  # distance center(i+1) -> left face(i+1/2)

        # Options
        self.limiter = limiter.lower()
        if self.limiter not in {"mc", "vanleer"}:
            raise ValueError("limiter must be 'mc' or 'vanleer'")
        self.cfl = float(cfl)
        self.face_velocity = face_velocity
        if self.face_velocity not in {"avg", "upwind"}:
            raise ValueError("face_velocity must be 'avg' or 'upwind'")

    # ------------------------------- public API -------------------------------
    def set_velocity(self, v_centers: np.ndarray) -> None:
        """Update the array of cell-centered velocities (e.g., if time-dependent)."""
        v_centers = np.asarray(v_centers, dtype=float)
        if v_centers.shape != self.r_centers.shape:
            raise ValueError("new v_centers must match r_centers shape")
        self.v_centers = v_centers

    def compute_dt(self) -> float:
        """CFL-limited time step based on max |v| and smallest center spacing."""
        vmax = float(np.max(np.abs(self.v_centers)))
        if vmax == 0.0:
            return np.inf
        dr_min = float(np.min(self.dx_c))
        return self.cfl * dr_min / vmax

    def advance(self, U: np.ndarray, dt: float) -> np.ndarray:
        """
        Advance cell-averaged solution U by time dt using MUSCL–Hancock + upwind.

        Parameters
        ----------
        U : np.ndarray
            Cell-averaged solution values at centers, shape (N,).
        dt : float
            Time step.

        Returns
        -------
        U_new : np.ndarray
            Updated solution after one step.
        """
        U = np.asarray(U, dtype=float)
        if U.shape != (self.ncells,):
            raise ValueError("U must have shape (N,)")

        # 1) Reconstruction at t^n: piecewise-linear with slope limiting
        slopes = self._compute_slopes(U)
        uL, uR = self._interface_states_from_slopes(
            U, slopes
        )  # states at internal faces [1..N-1]

        # 2) Predictor (Hancock half-step) using center flux gradients
        uLh, uRh = self._predict_half_step(U, uL, uR, dt)

        # 3) Face velocities and upwind fluxes at t^{n+1/2}
        v_faces = self._face_velocities()
        flux_faces = self._upwind_face_fluxes(
            uLh, uRh, v_faces
        )  # physical flux (v*u) at internal faces

        # 4) Geometric fluxes through spherical faces (multiply by area)
        F = self.areas[1:-1] * flux_faces  # length N-1

        # 5) Conservative update with geometric divergence
        U_new = U.copy()
        # Left boundary (r=0): area=0 => flux_in = 0
        # Internal cells
        U_new[0] += -(dt / self.volumes[0]) * (F[0] - 0.0)
        if self.ncells > 2:
            div = F[1:] - F[:-1]
            U_new[1:-1] += -(dt / self.volumes[1:-1]) * div
        # Right boundary: by default, zero-gradient (do-nothing outflow)
        U_new[-1] += -(dt / self.volumes[-1]) * (0.0 - F[-1])

        return U_new

    # ----------------------------- internal helpers ----------------------------
    def _compute_slopes(self, U: np.ndarray) -> np.ndarray:
        """Compute limited slopes at cell centers for non-uniform grid."""
        slopes = np.zeros_like(U)
        # One-sided near boundaries: keep slope = 0 for robustness
        if self.ncells <= 2:
            return slopes

        # Left and right differences as derivatives
        dL = (U[1:-1] - U[:-2]) / self.dx_c[:-1]
        dR = (U[2:] - U[1:-1]) / self.dx_c[1:]

        if self.limiter == "mc":
            # MC limiter: minmod of {dL, dR, 0.5(dL+dR)} with the usual 2x bounds
            sgn = np.sign(dL)
            same_sign = (dL * dR) > 0.0
            avg = 0.5 * (np.abs(dL) + np.abs(dR))
            limited = np.where(
                same_sign,
                np.minimum(np.minimum(2.0 * np.abs(dL), 2.0 * np.abs(dR)), avg),
                0.0,
            )
            slopes[1:-1] = sgn * limited
        elif self.limiter == "vanleer":
            prod = dL * dR
            slopes[1:-1] = np.where(prod > 0.0, (2.0 * prod) / (dL + dR), 0.0)
        else:
            raise NotImplementedError(f"Limiter '{self.limiter}' is not implemented.")

        # Boundaries: keep zero or copy nearest interior slope (optional)
        slopes[0] = 0.0
        slopes[-1] = 0.0
        return slopes

    def _interface_states_from_slopes(
        self, U: np.ndarray, slopes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct left/right states at internal faces i=1..N-1 from limited slopes.
        For face i (between cells i-1 and i):
            uR[i] = U[i-1] + slope[i-1] * (r_face[i]   - r_center[i-1])
            uL[i] = U[i]   - slope[i]   * (r_center[i] - r_face[i])
        Returns arrays uL,uR of length N+1; only entries 1..N-1 are valid.
        """
        uL = np.zeros(self.ncells + 1)
        uR = np.zeros(self.ncells + 1)
        # distances center -> face on each side
        dxL = self.dx_L  # length N-1 (center i-1 to face i)
        dxR = self.dx_R  # length N-1 (center i   to face i)
        # faces 1..N-1
        uR[1:-1] = U[:-1] + slopes[:-1] * dxL
        uL[1:-1] = U[1:] - slopes[1:] * dxR
        return uL, uR

    def _predict_half_step(
        self, U: np.ndarray, uL: np.ndarray, uR: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        MUSCL–Hancock half-step predictor: update interface states using local
        center flux gradients. For linear advection, f(U)=v*U, we use F_c = v_c * U.
        The predictor is applied to internal faces (1..N-1).
        """
        F_c = self.v_centers * U  # center flux values
        uLh = uL.copy()
        uRh = uR.copy()
        if self.ncells <= 1:
            return uLh, uRh

        # For face i between cells i-1 and i, estimate gradient with neighbor centers
        # Use the center-to-center spacing on the left (dx_c[i-1])
        corr = 0.5 * dt * (F_c[1:] - F_c[:-1]) / self.dx_c  # length N-1
        uLh[
            1:-1
        ] -= corr  # left state from cell i gets correction using (i - (i-1)) gradient
        uRh[
            1:-1
        ] -= corr  # right state from cell i-1 gets the same magnitude (symmetric)
        return uLh, uRh

    def _face_velocities(self) -> np.ndarray:
        """Interpolate or upwind cell-centered velocities to internal faces (length N-1)."""
        if self.face_velocity == "avg":
            return 0.5 * (self.v_centers[:-1] + self.v_centers[1:])
        # upwind: take donor-side center velocity for the sign based on average sign
        v_avg = 0.5 * (self.v_centers[:-1] + self.v_centers[1:])
        v_left = self.v_centers[:-1]
        v_right = self.v_centers[1:]
        return np.where(v_avg >= 0.0, v_left, v_right)

    def _upwind_face_fluxes(
        self, uLh: np.ndarray, uRh: np.ndarray, v_faces: np.ndarray
    ) -> np.ndarray:
        """
        Scalar upwind flux for linear advection at internal faces (length N-1):
            flux = v_face * u_upwind
        """
        # interface states at faces 1..N-1 live at indices [1:-1]
        uL_int = uLh[1:-1]
        uR_int = uRh[1:-1]
        return np.where(v_faces >= 0.0, v_faces * uL_int, v_faces * uR_int)


# -----------------------------------------------------------------------------
# Example usage (remove or adapt in your project):
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test setup
    N = 200
    r_c = np.linspace(0, 1.0, N)  # centers for uniform faces with r_face[0]=0
    # Define a velocity profile at centers (here: outward, piecewise)
    v_c = np.where(r_c < 0.5, 1.0, 0.5)

    solver = AdvectionFVSolver(r_c, v_c, limiter="mc", cfl=0.8, face_velocity="upwind")

    # Initial square pulse
    U = np.where((r_c > 0.2) & (r_c < 0.3), 1.0, 0.0)

    import matplotlib.pyplot as plt

    t = 0.0
    t_end = 0.2

    plt.ion()
    fig, ax = plt.subplots()
    (line,) = ax.plot(r_c, U, label="U(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("U")
    ax.set_title(f"t = {t:.4f}")
    plt.show()
    # Plot initial profiles of velocity and density
    fig2, ax2 = plt.subplots()
    ax2.plot(r_c, U, label="Initial Density U(r)")
    ax2.plot(r_c, v_c, label="Velocity v(r)")
    ax2.set_xlabel("r")
    ax2.set_ylabel("Value")
    ax2.set_title("Initial Profiles")
    ax2.legend()
    plt.show()
    while t < t_end:
        dt = min(solver.compute_dt(), t_end - t)
        U = solver.advance(U, dt * 0.1)
        t += dt * 0.1

        line.set_ydata(U)
        ax.set_title(f"t = {t:.4f}")
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    # At this point, U contains the advected profile.
    # (Plotting omitted intentionally.)
