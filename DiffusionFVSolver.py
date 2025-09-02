import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import logging

logger = logging.getLogger(__name__)


class DiffusionFVSolver:
    """
    Finite Volume solver for spherical diffusion equation:
    ∂f/∂t = 1/r^2 * ∂/∂r (r^2 * D(r) * ∂f/∂r) + Q(r)

    This solver uses the Crank-Nicolson method for time discretization, which is
    second-order accurate and unconditionally stable. The diffusion coefficient D(r)
    can be discontinuous, and its value at cell faces is handled using a harmonic mean
    to ensure flux conservation.

    The implementation follows a similar structure to AdvectionFVSolverv2.py.
    """

    def __init__(
        self,
        r_centers: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        params: dict,
        **kwargs,
    ) -> None:
        """
        Initializes the solver with grid, initial conditions, and physical parameters.
        """
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.f_values = np.asarray(f_values, dtype=float)
        self.r_centers = np.asarray(r_centers, dtype=float)

        if self.r_centers.ndim != 1 or self.r_centers.size < 2:
            raise ValueError("r_centers must be a 1D array with at least 2 points.")
        if abs(self.r_centers[0]) > 1e-14:
            raise ValueError("The first point in r_centers must be 0.")

        self.N = self.r_centers.size
        self.r_faces = np.empty(self.N + 1, dtype=float)
        self.r_faces[1:-1] = 0.5 * (self.r_centers[1:] + self.r_centers[:-1])
        self.r_faces[0] = 0.0
        self.r_faces[-1] = self.r_centers[-1] + 0.5 * (
            self.r_centers[-1] - self.r_centers[-2]
        )
        self.dr = self.r_faces[1:] - self.r_faces[:-1]

        self.params = params or {}
        self._unpack_params()

    def _unpack_params(self):
        """Validates and unpacks parameters from the params dictionary."""
        expected_keys = {"D_values", "Q_values", "f_end"}
        provided_keys = set(self.params.keys())

        if not expected_keys.issubset(provided_keys):
            missing = expected_keys - provided_keys
            raise ValueError(f"Missing required parameters: {missing}")

        self.D_values = np.asarray(self.params["D_values"], dtype=float)
        if self.D_values.shape != (self.N,):
            raise ValueError(f"D_values must have shape ({self.N},)")

        self.Q_values = np.asarray(self.params["Q_values"], dtype=float)
        if self.Q_values.shape != (self.N,):
            raise ValueError(f"Q_values must have shape ({self.N},)")

        self.f_end = float(self.params["f_end"])

    def advance(self, n_steps: int) -> np.ndarray:
        """
        Advances the solution by n_steps, using the time step from t_grid.
        """
        dt = np.diff(self.t_grid)[0] * n_steps
        f_current = self.f_values.copy()

        # Harmonic mean for diffusion coefficient at cell faces
        D_faces = np.zeros(self.N)
        # For internal faces i=1 to N-1 (D_half in untitled.py)
        D_plus_1 = self.D_values[1:]
        D_current = self.D_values[:-1]
        denominator = D_plus_1 + D_current
        # Avoid division by zero
        mask = denominator > 0
        D_faces[1:] = np.where(mask, 2 * D_plus_1 * D_current / denominator, 0)

        # Face at r=0 has D_half[0] = 0
        D_faces[0] = 0

        # Setup Crank-Nicolson matrices A and B
        # A * f_new = B * f_old
        alpha = dt / (2 * self.dr**2)

        A_diag = np.ones(self.N)
        A_lower = np.zeros(self.N - 1)
        A_upper = np.zeros(self.N - 1)
        B_diag = np.ones(self.N)
        B_lower = np.zeros(self.N - 1)
        B_upper = np.zeros(self.N - 1)

        for i in range(1, self.N - 1):
            ri = self.r_centers[i]
            if ri == 0:
                continue

            # Coefficients based on flux form
            c_imh = (self.r_faces[i] ** 2) * D_faces[i] / (ri**2)
            c_iph = (self.r_faces[i + 1] ** 2) * D_faces[i + 1] / (ri**2)

            A_lower[i - 1] = -alpha[i] * c_imh
            A_diag[i] = 1 + alpha[i] * (c_imh + c_iph)
            A_upper[i] = -alpha[i] * c_iph

            B_lower[i - 1] = alpha[i] * c_imh
            B_diag[i] = 1 - alpha[i] * (c_imh + c_iph)
            B_upper[i] = alpha[i] * c_iph

        # Boundary Conditions
        # r=0: Neumann (df/dr = 0) -> Symmetry f_0 = f_1
        A_diag[0] = 1.0
        A_upper[0] = -1.0
        B_diag[0] = 0.0  # Does not depend on previous values
        B_upper[0] = 0.0

        # r=r_end: Dirichlet (f = f_end)
        A_diag[-1] = 1.0
        B_diag[-1] = 0.0  # Does not depend on previous values

        A = diags([A_lower, A_diag, A_upper], offsets=[-1, 0, 1], format="csr")
        B = diags([B_lower, B_diag, B_upper], offsets=[-1, 0, 1], format="csr")

        # Calculate RHS and solve
        rhs = B @ f_current + dt * self.Q_values
        rhs[0] = 0  # Neumann BC
        rhs[-1] = self.f_end  # Dirichlet BC

        f_new = spsolve(A, rhs)

        self.f_values = f_new
        return f_new


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

    # Spatial and temporal grids
    r = np.linspace(r_0, r_end, num_points)
    t_steps = 50
    t_grid = np.linspace(0, 0.1, t_steps)

    # Initial profile (Gaussian-like)
    f_initial = np.exp(-((r - 0.3) ** 2) / (2 * 0.05**2))

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
    solver = DiffusionFVSolver(
        r_centers=r, t_grid=t_grid, f_values=f_initial, params=dif_param
    )

    # Run simulation
    num_timesteps = len(t_grid) - 1
    f_evolution = [np.copy(f_initial)]

    for n in range(1, num_timesteps + 1):
        solver.f_values = solver.advance(1)
        f_evolution.append(np.copy(solver.f_values))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4.5))

    num_curves = 10
    indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, num_curves))

    for idx, curve_idx in enumerate(indices):
        ax.plot(
            r,
            f_evolution[curve_idx],
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
    plt.title("Diffusion with Discontinuous Coefficient (Finite Volume Solver)")
    plt.show()
