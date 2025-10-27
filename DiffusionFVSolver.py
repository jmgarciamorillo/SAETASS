import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from State import State, SliceState
from Grid import Grid
import logging

logger = logging.getLogger(__name__)


class DiffusionFVSolver:
    def __init__(self, grid: Grid, t_grid: np.ndarray, params: dict, **kwargs) -> None:
        self._unpack_grid(grid)
        self.t_grid = np.asarray(t_grid, dtype=float)
        if self.r_centers.ndim != 1 or self.r_centers.size < 2:
            raise ValueError("r_centers must be 1D with at least 2 points.")
        if abs(self.r_centers[0]) > 1e-14:
            raise ValueError("first r_center must be 0.")
        self.N = len(self.r_centers)
        self.h = np.asarray(self.grid.dr, dtype=float)
        self.V = (self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3) / 3.0
        self.d_centers = self.r_centers[1:] - self.r_centers[:-1]
        self.params = params or {}
        self._unpack_params()

    def _unpack_grid(self, grid):
        self.grid = grid
        if grid.r_centers is None:
            raise ValueError("Grid must include r_centers")
        self.r_centers = np.asarray(grid.r_centers, dtype=float)
        self.r_faces = np.asarray(grid.r_faces, dtype=float)
        if grid.p_centers is not None:
            self.p_centers = np.asarray(grid.p_centers, dtype=float)
            self.p_faces = np.asarray(grid.p_faces, dtype=float)
            logger.info(f"Found momentum grid with {len(self.p_centers)} points")
        else:
            self.p_centers = None
            self.p_faces = None
            logger.info("No momentum grid found, operating in 1D mode")

    def _unpack_params(self):
        expected = {"D_values", "f_end"}
        if not expected.issubset(set(self.params.keys())):
            raise ValueError(f"Missing params: {expected - set(self.params.keys())}")
        self.D_values = np.asarray(self.params["D_values"], dtype=float)
        if self.D_values.shape != self.grid.shape:
            raise ValueError(
                f"D_values shape mismatch: expected {self.grid.shape}, got {self.D_values.shape}"
            )
        self.f_end = float(self.params["f_end"])

    # ------------------- Public advance -------------------
    def advance(self, n_steps: int, state: State) -> None:
        dt = float(np.diff(self.t_grid)[0]) * n_steps

        # If no momentum grid or state is 1D, use scalar path
        if (
            self.p_centers is None
            or getattr(state, "f", None) is None
            or state.get_f().ndim == 1
        ):
            self._advance_slice(n_steps, state)  # existing single-slice code
            return

        # Try to extract the entire 2D f matrix from state.
        # Best if State implements get_f() returning shape (n_p, N).
        try:
            f_all = state.get_f()  # expected (n_p, N)
        except Exception:
            # Fallback: attempt to use state.f attribute
            f_all = np.asarray(state.f, dtype=float)
            if f_all.ndim != 2:
                raise ValueError(
                    "Cannot extract 2D f array from state; implement get_f() for efficiency."
                )

        # Validate shapes
        n_p = len(self.p_centers)
        if f_all.shape != (n_p, self.N):
            raise ValueError(
                f"State f shape {f_all.shape} does not match expected {(n_p, self.N)}"
            )

        # Vectorized batched solve for all slices
        f_new_all = self._advance_all_slices_batched(dt, f_all)

        # Update state: ideally state.update_f(f_new_all)
        if hasattr(state, "update_f"):
            state.update_f(f_new_all)
        else:
            # fallback: update slice by slice
            for i in range(n_p):
                slice_state = SliceState(state, i)
                slice_state.update_f(f_new_all[i, :])

    # ------------------- Core batched advance -------------------
    def _advance_all_slices_batched(self, dt: float, f_all: np.ndarray) -> np.ndarray:
        """
        Vectorized Crank-Nicolson for all momentum slices at once.

        f_all: shape (n_p, N)
        D_values: shape (n_p, N)  (or (N,) for 1D)
        Returns f_new_all shape (n_p, N)
        """
        n_p = f_all.shape[0]

        # D per slice
        D = (
            self.D_values if self.p_centers is None else self.D_values
        )  # D shape should be (n_p,N)
        if D.ndim == 1:
            D = np.broadcast_to(D[None, :], (n_p, self.N))
        # Now D shape (n_p, N)

        # 1) Compute D_face for all slices: shape (n_p, N+1)
        # We'll compute interior faces j = 1..N-1 using harmonic mean with nonuniform h
        # Precompute h_left/h_right arrays: shape (N-1,)
        h = np.asarray(self.h, dtype=float)
        h_left = h[:-1]  # length N-1
        h_right = h[1:]  # length N-1
        # For broadcasting: make shapes (1, N-1)
        hL = h_left[None, :]
        hR = h_right[None, :]

        # D_left and D_right per face (for each slice)
        D_left = D[:, :-1]  # shape (n_p, N-1)
        D_right = D[:, 1:]  # shape (n_p, N-1)
        num = hL + hR  # shape (1, N-1)
        den = hL / D_left + hR / D_right  # broadcasts to (n_p, N-1)
        # prevent division by zero
        den = np.where(den == 0.0, np.finfo(float).tiny, den)
        D_face_interior = num / den  # shape (n_p, N-1)

        # assemble D_face: shape (n_p, N+1)
        D_face = np.zeros((n_p, self.N + 1), dtype=float)
        D_face[:, 1:-1] = D_face_interior
        # Boundary faces: use cell-centered D as proxy
        D_face[:, 0] = D[:, 0]
        D_face[:, -1] = D[:, -1]

        # 2) Compute conductances G on faces: G = r_face^2 * D_face / dist_between_centers
        # For interior faces j = 1..N-1 use self.d_centers[j-1], shape (N-1,)
        r_faces = self.r_faces  # shape (N+1,)
        # G shape (n_p, N+1)
        G = np.zeros_like(D_face)
        denom_centers = self.d_centers  # length N-1
        G[:, 1:-1] = (
            (r_faces[1:-1][None, :] ** 2) * D_face[:, 1:-1] / denom_centers[None, :]
        )

        # boundary G
        G[:, 0] = 0.0
        # outmost face: use difference between last two centers
        if self.N >= 2:
            denom_last = self.r_centers[-1] - self.r_centers[-2]
        else:
            denom_last = self.h[0] / 2.0
        G[:, -1] = (r_faces[-1] ** 2) * D_face[:, -1] / denom_last

        # 3) Assemble A and B coefficients (batched)
        # A_diag shape (n_p, N), A_lower (n_p, N) with zero in first col,
        # A_upper (n_p, N) with zero in last col
        V = self.V  # shape (N,)
        V_b = V[None, :]  # shape (1, N) for broadcast

        a_left = G[:, :-1]  # conductance on left face of cell i  → shape (n_p, N)
        a_right = G[:, 1:]  # right face for cell i           → shape (n_p, N)

        A_diag = 2.0 * V_b / dt + (a_left + a_right)
        B_diag = 2.0 * V_b / dt - (a_left + a_right)

        # A_lower at position i-1 corresponds to -a_left[i], so for i>=1 fill A_lower[:, i-1] = -a_left[:, i]
        A_lower = np.zeros_like(A_diag)
        B_lower = np.zeros_like(A_diag)
        A_upper = np.zeros_like(A_diag)
        B_upper = np.zeros_like(A_diag)

        # For interior lower/upper (positions shifted)
        A_lower[:, 1:] = -a_left[:, 1:]
        B_lower[:, 1:] = a_left[:, 1:]
        A_upper[:, :-1] = -a_right[:, :-1]
        B_upper[:, :-1] = a_right[:, :-1]

        # Boundary r=r_end: Dirichlet replacement (last row)
        A_diag[:, -1] = 1.0
        A_lower[:, -1] = 0.0
        A_upper[:, -1] = 0.0
        B_diag[:, -1] = 0.0
        B_lower[:, -1] = 0.0
        B_upper[:, -1] = 0.0

        # 4) Build RHS: rhs = B @ f_current
        # Implement matrix-vector product for tridiagonal B batched
        # B_diag * f + B_lower * f shifted left + B_upper * f shifted right
        f = f_all  # shape (n_p, N)
        rhs = B_diag * f
        rhs[:, 1:] += B_lower[:, 1:] * f[:, :-1]
        rhs[:, :-1] += B_upper[:, :-1] * f[:, 1:]
        # impose Dirichlet at last entry
        rhs[:, -1] = self.f_end

        # 5) Solve batched tridiagonal systems A * f_new = rhs
        f_new = self._thomas_batched(A_lower, A_diag, A_upper, rhs)

        return f_new

    # ------------------- Batched Thomas solver -------------------
    def _thomas_batched(self, a_lower, a_diag, a_upper, rhs):
        """
        Batched Thomas algorithm solving many independent tridiagonal systems in parallel.
        Input shapes:
          a_lower, a_diag, a_upper, rhs: (n_p, N)
        Note: a_lower[:,0] and a_upper[:,-1] should be zeros.
        Returns x shape (n_p, N)
        """

        # Work on copies to avoid modifying inputs
        a = a_diag.astype(float).copy()  # main diagonal (n_p,N)
        b = a_upper.astype(float).copy()  # upper (n_p,N)
        c = a_lower.astype(float).copy()  # lower (n_p,N)
        d = rhs.astype(float).copy()  # RHS (n_p,N)

        n_p, N = a.shape
        # Forward sweep: compute modified coefficients
        # We will store cp = b'/a' in b, and dp = d'/a' in d (reuse arrays)
        # First row (i=0):
        # a0 = a[:,0]; b0 = b[:,0]; d0 = d[:,0]
        # cp0 = b0 / a0
        # dp0 = d0 / a0
        # We must be careful with zeros on diag (shouldn't happen for well-posed A)
        # To avoid division by zero, add eps
        eps = np.finfo(float).eps

        a0 = a[:, 0]
        # Avoid division by zero
        a0_safe = np.where(np.abs(a0) <= 0.0, eps, a0)
        b[:, 0] = b[:, 0] / a0_safe
        d[:, 0] = d[:, 0] / a0_safe

        for i in range(1, N):
            denom = a[:, i] - c[:, i] * b[:, i - 1]
            # protect denom
            denom_safe = np.where(np.abs(denom) <= 0.0, eps, denom)
            if i < N - 1:
                b[:, i] = b[:, i] / denom_safe
            d[:, i] = (d[:, i] - c[:, i] * d[:, i - 1]) / denom_safe

        # Back substitution:
        x = np.zeros_like(d)
        x[:, -1] = d[:, -1]
        for i in range(N - 2, -1, -1):
            x[:, i] = d[:, i] - b[:, i] * x[:, i + 1]

        return x


# =============================================================================
# Example / Test Case
# =============================================================================
if __name__ == "__main__":
    import matplotlib as mpl

    # Parameters matching DiffValidation5.py and untitled.py
    r_0 = 0.0
    r_end = 1.0
    num_points = 200
    f_end_bc = 0.0

    # Non-uniform grid test option
    is_nonuniform_test = True

    # Create grid
    if is_nonuniform_test:
        # Create refined grid around the diffusion coefficient jump at r=0.5
        grid = Grid.create(
            space_grid_type="clustered",
            r_min=r_0,
            r_max=r_end,
            num_r_cells=num_points - 1,
            r_cluster_center=0.5,
            r_cluster_width=0.1,
            r_cluster_strength=0.9,
            t_min=0.0,
            t_max=0.1,
            num_timesteps=50,
        )
    else:
        grid = Grid.uniform(
            r_min=r_0,
            r_max=r_end,
            num_r_cells=num_points - 1,
            t_min=0.0,
            t_max=0.1,
            num_timesteps=50,
        )

    # Get grid data
    r = grid.r_centers
    t_grid = grid.t_grid

    # Initial profile (Gaussian-like)
    f_initial = np.exp(-((r - 0.3) ** 2) / (2 * 0.05**2))

    # Create state
    state = State(f=f_initial)

    # Discontinuous diffusion coefficient
    D_values = np.ones(num_points)
    D_values[r >= 0.5] = 0.1

    # Source term (Q)
    Q = np.zeros(num_points)

    dif_param = {
        "D_values": D_values,
        "Q_values": Q,
        "f_end": f_end_bc,
    }

    # Prepare solver
    solver = DiffusionFVSolver(grid=grid, t_grid=t_grid, params=dif_param)

    # Run simulation
    num_timesteps = len(t_grid) - 1
    f_evolution = [state.f.copy()]

    for n in range(1, num_timesteps + 1):
        solver.advance(1, state)
        f_evolution.append(state.f.copy())

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4.5))

    num_curves = 10
    indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_curves))

    for idx, curve_idx in enumerate(indices):
        ax.plot(
            r,
            (
                f_evolution[curve_idx][0]
                if f_evolution[curve_idx].ndim > 1
                else f_evolution[curve_idx]
            ),
            color=colors[idx],
            linestyle="-",
            label=f"t={t_grid[curve_idx]:.2f}" if idx in [0, num_curves - 1] else None,
        )

    # Also plot the diffusion coefficient profile
    ax2 = ax.twinx()
    ax2.plot(r, D_values, "r--", label="D(r)", alpha=0.5)
    ax2.set_ylabel("Diffusion Coefficient D(r)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax.set_xlabel("$r$ in parsec")
    ax.set_ylabel("Solution $f(t,r)$")
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="k", linestyle="-", label="Numerical (FV Solver)"),
        Line2D([0], [0], color="r", linestyle="--", label="D(r)"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid()
    fig.tight_layout()

    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("$t$")

    plt.xlim(r_0, r_end)
    plt.ylim(0, 1.1)
    plt.grid(False)

    # Update title based on grid type
    if is_nonuniform_test:
        plt.title(
            "Diffusion with Discontinuous Coefficient (Non-uniform Grid FV Solver)"
        )
    else:
        plt.title("Diffusion with Discontinuous Coefficient (Uniform Grid FV Solver)")

    plt.show()

    # Print grid information
    print(f"\nGrid Information:")
    print(f"Total grid points: {len(r)}")
    if is_nonuniform_test:
        print(f"Grid type: Non-uniform (clustered around r=0.5)")
        # Calculate grid spacing statistics
        dr = np.diff(r)
        print(f"Min grid spacing: {dr.min():.6f}")
        print(f"Max grid spacing: {dr.max():.6f}")
        print(f"Mean grid spacing: {dr.mean():.6f}")
    else:
        print(f"Grid type: Uniform")
        print(f"Grid spacing: {(r_end - r_0)/(len(r)-1):.6f}")
