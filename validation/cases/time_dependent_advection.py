import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Apply unified plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import (
    apply_plot_style,
    get_numerical_style,
    get_analytical_style,
    get_quantitative_style,
    add_time_colorbar,
)

apply_plot_style()

from saetass import State, Grid, Solver


def run_simulation(
    r_grid, t_grid, f_initial, solver_params, source_params, sample_count=0
):
    grid = Grid(r_centers=r_grid, t_grid=t_grid, p_centers=None)
    state = State(f_initial)

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advection-source",
        operator_params={"advection": solver_params, "source": source_params},
        substeps={"advection": 1, "source": 1},
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


def compute_relative_L2(numerical, analytical):
    sum_diff_sq = np.sum((numerical - analytical) ** 2)
    sum_theo_sq = np.sum(analytical**2)
    if sum_theo_sq == 0:
        return np.sqrt(sum_diff_sq)
    return np.sqrt(sum_diff_sq / sum_theo_sq)


def validation_time_dependent_advection(
    resolutions, t_steps_list, r_end=10.0, t_final=np.pi, plot_results=True
):
    """
    Validation of 1D radial advection with space-time dependence.
    Eq: df/dt + (1/r^2)*d/dr(r^2 * u_w * f) = Q(r,t)
    u_w(r,t) = r/(1+t)
    f_exact(r,t) = (2+cos(t))*exp(-r)
    Q(r,t) = exp(-r) * [ -sin(t) + ((2+cos(t))/(1+t)) * (3-r) ]
    Sweep resolutions for different time steps, measure relative L2 error and plot convergence.
    """
    all_errors = {}
    layout_results = []

    def f_exact(r, t):
        return (2.0 + np.cos(t)) * np.exp(-r)

    for Nt in t_steps_list:
        print(f"--- Running temporal resolution Nt={Nt} ---")
        errors = []
        for N in resolutions:
            print(f"  Running N={N}")
            r_grid = np.linspace(0.0, r_end, N)
            dr = r_grid[1] - r_grid[0]

            t_grid = np.linspace(0.0, t_final, Nt)
            f_initial = f_exact(r_grid, 0.0)

            def u_w_func(t):
                return r_grid / (1.0 + t)

            def Q_src_func(r, p, t):
                term1 = -np.sin(t)
                term2 = ((2.0 + np.cos(t)) / (1.0 + t)) * (3.0 - r)
                return np.exp(-r) * (term1 + term2)

            solver_params = {
                "v_centers": u_w_func,
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            }

            source_params = {"source": Q_src_func}

            f_num, snapshots, snap_times = run_simulation(
                r_grid, t_grid, f_initial, solver_params, source_params, sample_count=7
            )

            f_ana = f_exact(r_grid, t_final)

            relL2 = compute_relative_L2(f_num, f_ana)
            errors.append(relL2)

            print(f"    dx={dr:.4e}, steps={Nt-1}, relL2={relL2:.4e}")

            # Save layout plots only for the highest time resolution to avoid spamming plots
            if Nt == t_steps_list[-1]:
                layout_results.append(
                    {
                        "N": N,
                        "r_grid": r_grid,
                        "f_initial": f_initial,
                        "f_num": f_num,
                        "f_ana": f_ana,
                        "relL2": relL2,
                        "snapshots": snapshots,
                        "snap_times": snap_times,
                    }
                )

        all_errors[Nt] = errors

    if plot_results:
        res = np.array(resolutions)

        # Spatial Convergence plot for multiple time resolutions
        plt.figure(figsize=(6, 4))

        for idx, Nt in enumerate(t_steps_list):
            errors = np.array(all_errors[Nt])
            style = get_quantitative_style(step_idx=idx, total_steps=len(t_steps_list))
            plt.loglog(res, errors, label=rf"$N_t={Nt}$", **style)

        plt.xlabel("Number of radial cells: $N$")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(alpha=0.3, which="both")
        plt.legend()
        conv_fig = plt.gcf()
        plt.show()

        last_fig = None
        last_N = None

        ymin, ymax = np.inf, -np.inf
        for rec in layout_results:
            ymin = min(ymin, np.min(rec["f_initial"]))
            ymax = max(ymax, np.max(rec["f_initial"]))
            ymin = min(ymin, np.min(rec["f_num"]))
            ymax = max(ymax, np.max(rec["f_num"]))

        padding = 0.05 * (ymax - ymin) if (ymax - ymin) > 0 else 0.1
        ylims = (max(0.0, ymin - padding), ymax + padding)

        for rec in layout_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            f_ana = rec["f_ana"]
            snapshots = rec["snapshots"]
            snap_times = rec["snap_times"]

            fig = plt.figure(figsize=(6, 4))
            for i, (s, t) in enumerate(zip(snapshots, snap_times)):
                is_initial = i == 0
                is_final = i == len(snapshots) - 1
                style = get_numerical_style(
                    is_initial=is_initial,
                    is_final=is_final,
                    step_idx=i,
                    total_steps=len(snapshots),
                )
                label = (
                    "Initial"
                    if is_initial
                    else ("Numerical (final)" if is_final else None)
                )
                plt.plot(r_grid, s, label=label, **style)

            ana_style = get_analytical_style()
            plt.plot(r_grid, f_ana, label="Analytical (final)", **ana_style)

            add_time_colorbar(fig, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1])
            plt.xlim(0, r_end)
            plt.ylim(ylims)
            plt.xlabel(r"Radial coordinate: $r$")
            plt.ylabel(r"Solution: $f(r,t)$")
            plt.legend()
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()
            last_fig = fig
            last_N = N

        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            if last_fig is not None:
                last_path = os.path.join(
                    out_dir, f"time_dependent_advection_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path, dpi=200, bbox_inches="tight")
                print(f"Saved layout to: {last_path}")
            conv_path = os.path.join(
                out_dir, "time_dependent_advection_convergence.pdf"
            )
            conv_fig.savefig(conv_path, dpi=200, bbox_inches="tight")
            print(f"Saved convergence to: {conv_path}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    plot_res = "--no-plot" not in sys.argv
    resolutions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    t_steps = [500, 1000, 2000, 4000, 8000, 16000]
    validation_time_dependent_advection(
        resolutions,
        t_steps_list=t_steps,
        r_end=10.0,
        t_final=np.pi,
        plot_results=plot_res,
    )
