import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Apply unified plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import apply_plot_style

apply_plot_style()

from saetass import State, Grid, Solver


def run_advection_simulation(
    r_grid, t_grid, f_initial, solver_params, source_params=None, sample_count=0
):
    """
    Run advection (optionally with source) and collect sampled snapshots by
    advancing the solver in chunks to sampled timestep indices.
    Returns final array, snapshots list and snapshot times.
    """
    grid = Grid(r_centers=r_grid, t_grid=t_grid, p_centers=None)
    state = State(f_initial)

    operator_params = {"advection": solver_params}
    if source_params is not None:
        operator_params["source"] = source_params
        problem_type = "advection-source"
    else:
        problem_type = "advection"

    solver = Solver(
        grid=grid,
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={"advection": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

    snapshots = [np.copy(state.f.flatten())]
    times = [t_grid[0]]

    if sample_count > 0 and num_timesteps > 0:
        sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
        sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))
    else:
        sample_indices = np.array([0, num_timesteps], dtype=int)

    current_step = 0
    for next_step in sample_indices[1:]:
        steps_to_advance = int(next_step - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_step
        snapshots.append(np.copy(solver.state.f.flatten()))
        times.append(t_grid[current_step])

    return solver.state.f.flatten(), snapshots, times


def validation_variable_velocity(
    resolutions,
    r_end=40.0,
    t_final=2.0,
    v_at_1=10.0,
    r_core=0.5,
    source_r_min=0.9,
    source_r_max=1.1,
    source_strength=40.0,
    plot_results=True,
):
    """
    Point-like source advected with a radially-varying velocity v(r) ~ 1/(r+r_core)^2.
    Expectation: downstream profile tends to a plateau (f ~ const) instead of 1/r^2.
    This function runs a resolution sweep, collects snapshots, computes a fitted
    slope in the plume region, and produces the same style plots as the other
    validation scripts (log-log time evolution and convergence plot).
    """
    slopes = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running N={N}")
        r_grid = np.linspace(0.0, r_end, N)
        dr = r_grid[1] - r_grid[0]

        # time grid (many small steps)
        t_grid = np.linspace(0.0, t_final, 10000)

        # initial condition: zero
        f_initial = np.zeros(N)

        # source
        Q_values = np.zeros(N)
        source_mask = (r_grid >= source_r_min) & (r_grid <= source_r_max)
        Q_values[source_mask] = source_strength

        # velocity field v(r) ~ v_at_1 * (1 / (r + r_core)^2) scaled so v(1)=v_at_1
        v_field = v_at_1 * ((1.0 + r_core) ** 2) / (r_grid + r_core) ** 2

        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0,
        }

        f_num, snapshots, snap_times = run_advection_simulation(
            r_grid,
            t_grid,
            f_initial,
            solver_params,
            source_params={"source": Q_values},
            sample_count=5,
        )

        all_results.append(
            {
                "N": N,
                "r_grid": r_grid,
                "f_num": f_num,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        res = np.array(resolutions)

        last_fig = None
        last_N = None
        for rec in all_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            snapshots = rec["snapshots"]
            snap_times = rec["snap_times"]

            fig_log = plt.figure(figsize=(8, 4))
            colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
            mask_pos_all = [(r_grid > 0) & (s > 0) for s in snapshots]
            for idx, (s, t, mask_pos) in enumerate(
                zip(snapshots, snap_times, mask_pos_all)
            ):
                if np.any(mask_pos):
                    plt.loglog(
                        r_grid[mask_pos],
                        s[mask_pos],
                        color=colors[idx],
                        label=f"$t = {t:.2f}$",
                    )

            plt.xlim(left=max(r_grid[1], 1e-6), right=r_end)
            pos_vals = np.concatenate([s[s > 0] for s in snapshots])
            if pos_vals.size:
                y_min_log = pos_vals.min()
                y_max_log = pos_vals.max()
                plt.ylim(y_min_log * 0.8, y_max_log * 1.2)

            plt.xlabel(r"Radial coordinate: $r$")
            plt.ylabel(r"Solution at time $t$: $f(r,t)$")
            plt.legend()
            plt.ylim([1e-2, 1e2])
            plt.xlim([source_r_min, r_end])
            plt.grid(alpha=0.4, which="both")
            plt.show()
            last_log_fig = fig_log
            last_fig = None
            last_N = N

        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)

            if "last_log_fig" in locals():
                last_log_path = os.path.join(
                    out_dir, f"advection_variable_velocity_last_N{last_N}_loglog.pdf"
                )
                try:
                    last_log_fig.savefig(last_log_path, dpi=200, bbox_inches="tight")
                    print(f"Saved last log-log figure to: {last_log_path}")
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096]
    validation_variable_velocity(
        resolutions,
        r_end=10.0,
        t_final=4.0,
        v_at_1=10.0,
        r_core=0.5,
        source_r_min=0.9,
        source_r_max=1.1,
        source_strength=400.0,
        plot_results=True,
    )
