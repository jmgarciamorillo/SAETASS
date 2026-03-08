import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from ..state import State, SliceState
from ..grid import Grid
from ..solver import SubSolver
import logging
from numba import njit, prange

logger = logging.getLogger(__name__)


class DiffusionSolver(SubSolver):
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
        self._init_buffers()

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

        self.n_p = 1 if self.p_centers is None else len(self.p_centers)

    def _unpack_params(self):
        expected = {"D_values"}
        if not expected.issubset(self.params):
            raise ValueError(f"Missing params: {expected - set(self.params.keys())}")

        D_input = self.params["D_values"]
        self.boundary_condition = self.params.get(
            "boundary_condition", "dirichlet"
        ).lower()

        if callable(D_input):
            self.is_D_dynamic = True

            def _get_d_dynamic(t):
                D = np.asarray(D_input(t), dtype=float)
                if D.ndim == 1:
                    if D.size != self.N:
                        raise ValueError(
                            "1D D_values must have the same size as r_cells"
                        )
                    return D[None, :]
                elif D.ndim == 2:
                    return D
                else:
                    raise ValueError("D_values must be 1D or 2D")

            self._get_D = _get_d_dynamic

            # Initial value for setup
            self.D_values = self._get_D(self.t_grid[0])

        else:
            self.is_D_dynamic = False
            D = np.asarray(D_input, dtype=float)
            if D.shape != self.grid.shape:
                raise ValueError(
                    f"D_values shape {D.shape} does not match grid shape {self.grid.shape}"
                )
            if D.ndim == 1:
                if D.size != self.N:
                    raise ValueError("1D D_values must have size of r_cells")
                self.D_values_static = D[None, :]
            elif D.ndim == 2:
                self.D_values_static = D
            else:
                raise ValueError("D_values must be 1D or 2D")

            self.D_values = self.D_values_static

        f_end_input = self.params.get("f_end", 0.0)
        if callable(f_end_input):
            self.is_f_end_dynamic = True

            def _get_f_end_dynamic(t):
                f_e = f_end_input(t)
                return float(f_e) if np.isscalar(f_e) else np.asarray(f_e, dtype=float)

            self._get_f_end = _get_f_end_dynamic
            self.f_end = self._get_f_end(self.t_grid[0])
        else:
            self.is_f_end_dynamic = False
            self.f_end = float(f_end_input)

    def _init_buffers(self):
        # Buffers for batched Thomas solver
        n_p = self.n_p
        self._D_face = np.zeros((n_p, self.N + 1), dtype=float)
        self._G = np.zeros((n_p, self.N + 1), dtype=float)
        self._A_lower = np.zeros((n_p, self.N), dtype=float)
        self._A_diag = np.zeros((n_p, self.N), dtype=float)
        self._A_upper = np.zeros((n_p, self.N), dtype=float)
        self._B_lower = np.zeros((n_p, self.N), dtype=float)
        self._B_diag = np.zeros((n_p, self.N), dtype=float)
        self._B_upper = np.zeros((n_p, self.N), dtype=float)
        self._rhs = np.zeros((n_p, self.N), dtype=float)
        self._f_new = np.zeros((n_p, self.N), dtype=float)

        # Additional precomputations
        self.V_b = self.V[None, :]
        self.h_left = self.h[:-1].astype(float)
        self.h_right = self.h[1:].astype(float)
        self.hL_b = self.h_left[None, :]
        self.hR_b = self.h_right[None, :]

        # Precompute static conductances if D is not dynamic
        if getattr(self, "is_D_dynamic", False) is False:
            self._update_conductances(self.D_values_static)

    # ------------------- Public advance -------------------
    def advance(self, n_steps: int, state: State) -> None:
        dt = float(np.diff(self.t_grid)[0]) * n_steps

        f_all = state.get_f()

        # If no momentum grid or state is 1D, use scalar path
        if (
            self.p_centers is None
            or getattr(state, "f", None) is None
            or state.get_f().ndim == 1
        ):
            # Promote a 1D f(r) into shape (1, N)
            f_all = state.get_f()

            if f_all.ndim == 1:
                # 1D case → promote
                if f_all.size != self.N:
                    raise ValueError("1D state f must have size N")
                f_all = f_all[None, :]  # shape = (1, N)

        if f_all.shape != (self.n_p, self.N):
            raise ValueError(
                f"State f shape {f_all.shape} does not match expected {(self.n_p, self.N)}"
            )

        # Dynamic update
        if self.is_D_dynamic:
            D_values = self._get_D(state.t)
            self._update_conductances(D_values)

        if getattr(self, "is_f_end_dynamic", False):
            self.f_end = self._get_f_end(state.t)

        # Vectorized batched solve for all slices
        f_new_all = self._advance_all_slices_batched(dt, f_all)

        # Update state: ideally state.update_f(f_new_all)
        if hasattr(state, "update_f"):
            state.update_f(f_new_all)
        else:
            # fallback: update slice by slice
            for i in range(self.n_p):
                slice_state = SliceState(state, i)
                slice_state.update_f(f_new_all[i, :])

    # ------------------- Core batched advance -------------------
    def _update_conductances(self, D_values: np.ndarray) -> None:
        """
        Recompute face conductances when D_values change.
        """
        # 1) Compute D_face for all slices: shape (n_p, N+1)
        D_left = D_values[:, :-1]
        D_right = D_values[:, 1:]
        num = self.hL_b + self.hR_b
        den = self.hL_b / D_left + self.hR_b / D_right
        den = np.where(den == 0.0, np.finfo(float).tiny, den)
        self._D_face[:, 1:-1] = num / den
        self._D_face[:, 0] = D_values[:, 0]
        self._D_face[:, -1] = D_values[:, -1]

        # 2) Compute conductances G on faces
        rf2 = (self.r_faces[1:-1] ** 2)[None, :]
        self._G[:, 1:-1] = (rf2 * self._D_face[:, 1:-1]) / self.d_centers[None, :]
        self._G[:, 0] = 0.0

        if self.boundary_condition == "dirichlet":
            if self.N >= 2:
                denom_last = self.r_centers[-1] - self.r_centers[-2]
            else:
                denom_last = self.h[0] / 2.0
            self._G[:, -1] = (self.r_faces[-1] ** 2) * self._D_face[:, -1] / denom_last
        elif self.boundary_condition == "outflow":
            # Assume asymptotic behaviour f ~ 1/r -> df/dr = -f/r
            # At boundary, flux is roughly proportional to f itself
            # We fold the outflow into G[:, -1] multiplying f_N
            self._G[:, -1] = self.r_faces[-1] * self._D_face[:, -1]
        elif self.boundary_condition == "neumann":
            self._G[:, -1] = 0.0
        else:
            raise ValueError(f"Unknown boundary_condition: {self.boundary_condition}")

    def _build_matrices(self, dt: float) -> None:
        """
        Assemble A and B matrices given the current conductances and timestep dt.
        """
        a_left = self._G[:, :-1]
        a_right = self._G[:, 1:]
        self._A_diag[:] = 2.0 * self.V_b / dt + (a_left + a_right)
        self._B_diag[:] = 2.0 * self.V_b / dt - (a_left + a_right)

        self._A_lower.fill(0.0)
        self._B_lower.fill(0.0)
        self._A_upper.fill(0.0)
        self._B_upper.fill(0.0)

        self._A_lower[:, 1:] = -a_left[:, 1:]
        self._B_lower[:, 1:] = a_left[:, 1:]
        self._A_upper[:, :-1] = -a_right[:, :-1]
        self._B_upper[:, :-1] = a_right[:, :-1]

        if self.boundary_condition == "dirichlet":
            self._A_diag[:, -1] = 1.0
            self._A_lower[:, -1] = 0.0
            self._A_upper[:, -1] = 0.0
            self._B_diag[:, -1] = 0.0
            self._B_lower[:, -1] = 0.0
            self._B_upper[:, -1] = 0.0
        # If boundary_condition is 'neumann' or 'outflow', the finite volume updates perfectly track the conservative fluxes,
        # so keeping the standard diagonal works correctly without overwriting.

    def _advance_all_slices_batched(self, dt: float, f_all: np.ndarray) -> np.ndarray:
        """
        Vectorized Crank-Nicolson for all momentum slices at once.
        f_all: shape (n_p, N)
        Returns f_new_all shape (n_p, N)
        """
        # Matrices depend on dt and G. We rebuild them every step (dt could change).
        self._build_matrices(dt)

        # Build RHS: rhs = B @ f_current
        f = f_all
        self._rhs = self._B_diag * f
        self._rhs[:, 1:] += self._B_lower[:, 1:] * f[:, :-1]
        self._rhs[:, :-1] += self._B_upper[:, :-1] * f[:, 1:]

        if self.boundary_condition == "dirichlet":
            self._rhs[:, -1] = self.f_end

        # Solve batched tridiagonal systems A * f_new = rhs
        f_new = _thomas_batched_numba(
            self._A_lower, self._A_diag, self._A_upper, self._rhs
        )

        return f_new


@njit(parallel=True, fastmath=True)
def _thomas_batched_numba(a_lower, a_diag, a_upper, rhs):
    n_p, N = a_diag.shape
    x = np.empty_like(rhs)
    eps = np.finfo(np.float64).eps

    # Work on local arrays per slice inside parallel loop
    for s in prange(n_p):
        # allocate small 1D temporaries per slice (fast)
        a = a_diag[s, :].copy()
        b = a_upper[s, :].copy()
        c = a_lower[s, :].copy()
        d = rhs[s, :].copy()

        # forward
        a0 = a[0]
        if a0 == 0.0:
            a0 = eps
        b[0] = b[0] / a0
        d[0] = d[0] / a0
        for i in range(1, N):
            denom = a[i] - c[i] * b[i - 1]
            if denom == 0.0:
                denom = eps
            if i < N - 1:
                b[i] = b[i] / denom
            d[i] = (d[i] - c[i] * d[i - 1]) / denom

        # back substitution
        x[s, N - 1] = d[N - 1]
        for i in range(N - 2, -1, -1):
            x[s, i] = d[i] - b[i] * x[s, i + 1]

    return x
