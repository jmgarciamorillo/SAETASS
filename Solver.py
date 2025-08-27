import numpy as np
import inspect

from AdvectionSolver import AdvectionSolver
from DiffusionSolver import DiffusionSolver
from LossSolver import LossSolver
from AdvectionFVSolverv2 import AdvectionFVSolver

# Map operator names to their solver classes
SUBSOLVER_MAP = {
    "advection": AdvectionSolver,
    "diffusion": DiffusionSolver,
    "loss": LossSolver,
    "advectionFV": AdvectionFVSolver,
}


class Solver:
    def __init__(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        f_values: np.ndarray,
        problem_type: str,
        Q_values: np.ndarray = None,
        advection_params: dict = None,
        diffusion_params: dict = None,
        loss_params: dict = None,
        advectionFV_params: dict = None,
        substeps: dict = None,
        **kwargs,
    ):
        """
        Initializes the main Solver with operator splitting.
        problem_type: str, e.g. 'advection-diffusion', 'loss', etc.
        substeps: dict, e.g. {'advection': 1, 'diffusion': 1, 'loss': 1}
        """
        self.x_grid = x_grid
        self.t_grid = t_grid
        self.f_values = np.copy(f_values)
        self.problem_type = problem_type.lower()
        self.Q_values = Q_values if Q_values is not None else np.zeros_like(x_grid)
        self.params = {
            "advection": advection_params or {},
            "diffusion": diffusion_params or {},
            "loss": loss_params or {},
            "advectionFV": advectionFV_params or {},
        }
        self.substeps = substeps or {}

        # Determine which operators are present and print for debugging
        print(f"[Solver DEBUG] Problem type: {self.problem_type}")
        print(f"[Solver DEBUG] Available operators: {list(SUBSOLVER_MAP.keys())}")
        # Parse operators from problem_type, ensuring 'advectionFV' and 'advection' are not both included by substring
        # Parse operators from problem_type, handling exact operator names split by '-'
        self.operators = []
        # Split by '-' and normalize to lowercase for matching
        ops_in_type = [op.strip().lower() for op in self.problem_type.split("-")]
        for op in SUBSOLVER_MAP:
            if op.lower() in ops_in_type:
                self.operators.append(op)
        print(f"[Solver DEBUG] Using operators: {self.operators}")
        self.n_os = len(self.operators)

        if self.n_os == 0:
            raise ValueError("No valid operators specified in problem_type.")

        # Split Q_values equally among operators
        self.Q_split = self.Q_values / self.n_os

        # Substeps for each operator (default 1)
        self.substeps_per_op = {op: self.substeps.get(op, 1) for op in self.operators}

        # Prepare subsolvers using the mapping
        self.subsolvers = []
        self._initialize_subsolvers(**kwargs)

        self.global_step = 0
        self.total_steps = len(self.t_grid) - 1

    def _refined_t_grid(self, t_grid, n_sub):
        """Return a refined t_grid for n_sub substeps per global step."""
        num_timesteps = len(t_grid) - 1
        t_grid_refined = []
        for i in range(num_timesteps):
            t_start = t_grid[i]
            t_end = t_grid[i + 1]
            t_grid_refined.extend(np.linspace(t_start, t_end, n_sub + 1)[:-1])
        t_grid_refined.append(t_grid[-1])
        return np.array(t_grid_refined)

    def _initialize_subsolvers(self, **kwargs):
        """Initialize subsolvers with appropriate t_grids and parameters."""
        for op in self.operators:
            solver_class = SUBSOLVER_MAP[op]
            n_sub = self.substeps_per_op[op]
            if n_sub > 1:
                t_grid_refined = self._refined_t_grid(self.t_grid, n_sub)
            else:
                t_grid_refined = self.t_grid

            # Special-case the AdvectionFVSolver which does NOT follow the SubproblemSolver API
            if op == "advectionFV":
                # prefer explicit advectionFV_params passed to Solver constructor
                fv_params = (
                    kwargs.get("advectionFV_params")
                    or self.params.get("advectionFV", {})
                    or {}
                )
                # Use v_centers directly from provided keys
                if "v_centers" in fv_params:
                    v_centers = np.asarray(fv_params["v_centers"], dtype=float)
                elif "v_field_n" in fv_params:
                    v_centers = np.asarray(fv_params["v_field_n"], dtype=float)
                else:
                    raise ValueError(
                        "advectionFV requires 'v_centers' or 'v_field_n' in advectionFV_params"
                    )

                # Prepare remaining kwargs accepted by AdvectionFVSolver constructor
                # Inspect signature to filter allowed params
                sig = inspect.signature(solver_class.__init__)
                allowed = set(sig.parameters.keys()) - {"self"}
                init_kwargs = {
                    k: v
                    for k, v in fv_params.items()
                    if k in allowed and k != "v_centers"
                }
                # pass r_faces and v_centers as the first two args
                self.subsolvers.append(
                    solver_class(
                        self.x_grid,
                        v_centers,
                        **init_kwargs,
                    )
                )
            else:
                # Regular subsolver construction (assumes SubproblemSolver interface)
                self.subsolvers.append(
                    solver_class(
                        self.x_grid,
                        t_grid_refined,
                        self.f_values,
                        Q_values=self.Q_split,
                        **self.params[op],
                        **kwargs,
                    )
                )

    def _advance(self, f_start, n_steps):
        """
        Advance the solution by n_steps from f_start.
        Returns the new state.
        """
        f_current = np.copy(f_start)
        for step_idx in range(n_steps):
            self.global_step += 1
            print(
                f"[Solver DEBUG] Global step {self.global_step}/{self.total_steps} | max(f)={np.max(f_current):.4g} min(f)={np.min(f_current):.4g}"
            )
            for i, op in enumerate(self.operators):
                subsolver = self.subsolvers[i]
                n_sub = self.substeps_per_op[op]
                print(f"  [Solver DEBUG] Operator '{op}' | substeps: {n_sub}")

                # Special handling for AdvectionFVSolver (not following SubproblemSolver API)
                if op == "advectionFV":
                    # compute global dt for this global step from the top-level t_grid
                    gs = self.global_step
                    if gs <= 0 or gs >= len(self.t_grid):
                        # fallback: use single delta_t estimate
                        dt_global = float(self.t_grid[-1] - self.t_grid[0]) / max(
                            1, self.total_steps
                        )
                    else:
                        dt_global = float(self.t_grid[gs] - self.t_grid[gs - 1])

                    dt_sub = dt_global / max(1, n_sub)

                    # f_current is already centered values (length M-1)
                    U_centers = np.asarray(f_current, dtype=float)
                    if U_centers.ndim != 1 or U_centers.size != self.x_grid.size - 1:
                        raise ValueError(
                            "f_current must be centered values of length M-1 for advectionFV"
                        )

                    # perform n_sub subcycles calling advance(U_centers, dt_sub)
                    for _ in range(n_sub):
                        U_centers = subsolver.advance(
                            U_centers, dt_sub
                        )  # returns centers U (length M-1)

                    f_current = U_centers  # updated centered array after advectionFV
                else:
                    # Regular SubproblemSolver-based operator
                    subsolver._f_values = np.copy(f_current)
                    f_current = subsolver.run_simulation(n_sub)
        print(
            f"[Solver DEBUG] Advance finished | max(f)={np.max(f_current):.4g} min(f)={np.min(f_current):.4g}"
        )
        return f_current

    def run(self):
        """
        Runs the operator splitting simulation.
        Returns the solution at the final time step.
        """
        num_timesteps = len(self.t_grid) - 1
        f_current = self._advance(self.f_values, num_timesteps)
        return f_current

    def step(self, n_steps=1):
        """
        Advance the solution by n_steps from the current state.
        Updates self.f_values in-place.
        Returns the new state.
        """
        f_current = self._advance(self.f_values, n_steps)
        self.f_values = np.copy(f_current)
        return f_current
