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
        Now supports non-uniform grids with proper finite volume formulation.
        """
        self.t_grid = np.asarray(t_grid, dtype=float)
        self.f_values = np.asarray(f_values, dtype=float)
        self.r_centers = np.asarray(r_centers, dtype=float)

        if self.r_centers.ndim != 1 or self.r_centers.size < 2:
            raise ValueError("r_centers must be a 1D array with at least 2 points.")
        if abs(self.r_centers[0]) > 1e-14:
            raise ValueError("The first point in r_centers must be 0.")

        self.N = self.r_centers.size

        # Construct face positions for non-uniform grid
        self.r_faces = np.empty(self.N + 1, dtype=float)
        self.r_faces[1:-1] = 0.5 * (self.r_centers[1:] + self.r_centers[:-1])
        self.r_faces[0] = 0.0
        self.r_faces[-1] = self.r_centers[-1] + 0.5 * (
            self.r_centers[-1] - self.r_centers[-2]
        )

        # Cell widths and volumes
        self.h = self.r_faces[1:] - self.r_faces[:-1]  # cell radial widths
        self.V = (
            self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3
        ) / 3.0  # spherical volumes

        # Distances between cell centers (for gradient computation)
        self.d_centers = np.empty(self.N - 1)
        self.d_centers[:] = self.r_centers[1:] - self.r_centers[:-1]

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
        Updated for non-uniform grid with proper finite volume formulation.
        """
        dt = np.diff(self.t_grid)[0] * n_steps
        f_current = self.f_values.copy()

        # Compute D on faces using non-uniform harmonic average
        D_face = np.zeros(self.N + 1)

        # Interior faces (between cells i-1 and i)
        for j in range(1, self.N):
            i_left = j - 1
            i_right = j
            num = self.h[i_left] + self.h[i_right]
            den = (
                self.h[i_left] / self.D_values[i_left]
                + self.h[i_right] / self.D_values[i_right]
            )
            D_face[j] = num / den

        # Boundary faces
        D_face[0] = self.D_values[0]  # r=0 face
        D_face[self.N] = self.D_values[-1]  # r=r_end face

        # Compute conductances G on faces
        G = np.zeros(self.N + 1)

        # Interior faces
        for j in range(1, self.N):
            i_left = j - 1
            denom = (
                self.r_centers[i_left + 1] - self.r_centers[i_left]
            )  # distance between centers
            G[j] = self.r_faces[j] ** 2 * D_face[j] / denom

        # Boundary conductances
        G[0] = 0.0  # symmetry at r=0 -> zero flux
        if self.N >= 2:
            denom = self.r_centers[-1] - self.r_centers[-2]
            G[self.N] = self.r_faces[self.N] ** 2 * D_face[self.N] / denom
        else:
            G[self.N] = self.r_faces[self.N] ** 2 * D_face[self.N] / (self.h[0] / 2.0)

        # Setup Crank-Nicolson matrices A and B
        # A * f_new = B * f_old + RHS
        A_diag = np.zeros(self.N)
        A_lower = np.zeros(self.N - 1)
        A_upper = np.zeros(self.N - 1)
        B_diag = np.zeros(self.N)
        B_lower = np.zeros(self.N - 1)
        B_upper = np.zeros(self.N - 1)

        # Assemble interior rows
        for i in range(self.N):
            a_left = G[i]  # conductance on left face of cell i
            a_right = G[i + 1]  # conductance on right face of cell i

            A_diag[i] = 2.0 * self.V[i] / dt + a_left + a_right
            B_diag[i] = 2.0 * self.V[i] / dt - (a_left + a_right)

            if i > 0:
                A_lower[i - 1] = -a_left
                B_lower[i - 1] = a_left
            if i < self.N - 1:
                A_upper[i] = -a_right
                B_upper[i] = a_right

        # Boundary Conditions
        # r=0: symmetry already enforced because G[0]=0
        # r=r_end: Dirichlet BC - replace last row
        A_diag[-1] = 1.0
        if self.N > 1:
            A_lower[-1] = 0.0
        B_diag[-1] = 0.0

        A = diags([A_lower, A_diag, A_upper], offsets=[-1, 0, 1], format="csr")
        B = diags([B_lower, B_diag, B_upper], offsets=[-1, 0, 1], format="csr")

        # Calculate RHS and solve
        rhs = B @ f_current + 2.0 * self.V * self.Q_values * dt
        rhs[-1] = self.f_end  # Dirichlet BC

        f_new = spsolve(A, rhs)

        # Enforce boundary conditions explicitly for numerical safety
        # f_new[-1] = self.f_end
        # f_new[0] = f_new[1]  # symmetry at r=0

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
    num_points = 10000
    f_end_bc = 0.2

    # Non-uniform grid test option
    is_nonuniform_test = True

    # Spatial and temporal grids
    if is_nonuniform_test:
        # Create refined grid around the diffusion coefficient jump at r=0.5
        r_jump = 0.5
        refinement_width = 0.1  # Width of refinement region around jump
        refinement_factor = 10  # How much finer the grid should be in this region

        # Base grid points
        base_points = num_points // 2

        # Create three regions: before jump, around jump, after jump
        r_before = np.linspace(r_0, r_jump - refinement_width / 2, base_points // 3)
        r_refined = np.linspace(
            r_jump - refinement_width / 2,
            r_jump + refinement_width / 2,
            base_points * refinement_factor // 3,
        )
        r_after = np.linspace(r_jump + refinement_width / 2, r_end, base_points // 3)

        # Combine all regions, removing duplicate points at boundaries
        r = np.concatenate([r_before[:-1], r_refined[:-1], r_after])
        num_points = len(r)
    else:
        r = np.linspace(r_0, r_end, num_points)

    t_steps = 5000
    t_grid = np.linspace(0, 0.1, t_steps)

    # Initial profile (Gaussian-like)
    f_initial = np.exp(-((r - 0.3) ** 2) / (2 * 0.05**2))

    # Discontinuous diffusion coefficient
    D_values = 10000 * np.ones(num_points)
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
    print(f"Total grid points: {num_points}")
    if is_nonuniform_test:
        print(f"Grid type: Non-uniform (refined around r={r_jump})")
        print(f"Refinement width: {refinement_width}")
        print(f"Refinement factor: {refinement_factor}")
        # Calculate grid spacing statistics
        dr = np.diff(r)
        print(f"Min grid spacing: {dr.min():.6f}")
        print(f"Max grid spacing: {dr.max():.6f}")
        print(f"Mean grid spacing: {dr.mean():.6f}")
    else:
        print(f"Grid type: Uniform")
        print(f"Grid spacing: {(r_end - r_0)/(num_points-1):.6f}")
