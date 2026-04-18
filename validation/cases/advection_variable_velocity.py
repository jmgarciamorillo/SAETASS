import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Apply unified plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import (
    add_time_colorbar,
    apply_plot_style,
    get_analytical_style,
    get_numerical_style,
    get_quantitative_style,
)

from saetass import Grid, Solver, State

apply_plot_style()


def compute_relative_L2(numerical, analytical, mask=None):
    if mask is None:
        mask = np.ones_like(numerical, dtype=bool)
    num = numerical[mask]
    theo = analytical[mask]

    sum_diff_sq = np.sum((num - theo) ** 2)
    sum_theo_sq = np.sum(theo**2)

    if sum_theo_sq == 0:
        return np.sqrt(sum_diff_sq)
    return np.sqrt(sum_diff_sq / sum_theo_sq)


def get_steady_state_analytical(r_grid, r_c, v_c, r_a, r_b, Q0):
    """
    Calcula la solución analítica exacta en estado estacionario
    para el perfil de velocidad por tramos.
    """
    f_ss = np.zeros_like(r_grid)

    mask_source = (r_grid >= r_a) & (r_grid <= r_b)
    mask_out = r_grid > r_b

    factor = Q0 / (3.0 * v_c * (r_c**2))

    f_ss[mask_source] = factor * (r_grid[mask_source] ** 3 - r_a**3)
    f_ss[mask_out] = factor * (r_b**3 - r_a**3)

    return f_ss


def run_advection_simulation(
    r_grid, t_grid, f_initial, solver_params, source_params=None, sample_count=0
):
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


def validation_piecewise_velocity(
    resolutions,
    r_end=15.0,
    t_final=20.0,
    v_c=5.0,
    r_c=0.2,
    source_r_min=2.0,
    source_r_max=3.0,
    source_strength=40.0,
    plot_results=True,
):
    """
    Validates advection operator with piecewise velocity v(r):
    v = v_c for r < r_c, and v = v_c * (r_c/r)^2 for r >= r_c.
    Source is localized in [source_r_min, source_r_max].
    """
    errors = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running n_r={N}")
        r_grid = np.linspace(0.0, r_end, N)

        t_grid = np.linspace(0.0, t_final, 2000)

        f_initial = np.zeros(N)

        Q_values = np.zeros(N)
        source_mask = (r_grid >= source_r_min) & (r_grid <= source_r_max)
        Q_values[source_mask] = source_strength

        r_safe = np.maximum(r_grid, 1e-12)
        v_field = np.where(r_grid < r_c, v_c, v_c * (r_c / r_safe) ** 2)

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
            sample_count=8,
        )

        f_ana = get_steady_state_analytical(
            r_grid, r_c, v_c, source_r_min, source_r_max, source_strength
        )

        dr = r_grid[1] - r_grid[0]
        # Calculate relL2 error where analytical > 0 and where r < source_r_max + 1
        mask = (f_ana > 0) & (r_grid < source_r_max + 4.5)
        relL2 = compute_relative_L2(f_num, f_ana, mask) if np.any(mask) else 0.0
        errors.append(relL2)
        dxs.append(dr)

        print(f"  dx={dr:.4e}, steps={len(t_grid) - 1}, relL2={relL2:.4e}")

        all_results.append(
            {
                "N": N,
                "r_grid": r_grid,
                "f_num": f_num,
                "f_ana": f_ana,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "figures")
        )
        os.makedirs(out_dir, exist_ok=True)

        dxs = np.array(dxs)
        res = np.array(resolutions)
        errors = np.array(errors)

        # Convergence plot
        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        plt.loglog(res, errors, label=r"Error ($\mathcal{E}_{L_2}$)", **quant_style)
        plt.xlabel("Number of radial cells: $n_r$")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(which="both")
        conv_fig = plt.gcf()
        plt.show()

        last_fig = None
        last_N = None

        for rec in all_results:
            N, r_grid = rec["N"], rec["r_grid"]
            snapshots, snap_times = rec["snapshots"], rec["snap_times"]
            f_ana = rec["f_ana"]

            fig_log = plt.figure(figsize=(6, 4))
            ax = fig_log.add_subplot(111)

            # Plotea los snapshots numéricos
            mask_pos_all = [(r_grid > 0) & (s > 0) for s in snapshots]
            for idx, (s, t, mask_pos) in enumerate(
                zip(snapshots, snap_times, mask_pos_all)
            ):
                if np.any(mask_pos):
                    is_initial = idx == 0
                    is_final = idx == len(snapshots) - 1
                    style = get_numerical_style(
                        is_initial=is_initial,
                        is_final=is_final,
                        step_idx=idx,
                        total_steps=len(snapshots),
                    )
                    label = (
                        "Initial"
                        if is_initial
                        else ("Numerical (final)" if is_final else None)
                    )
                    ax.loglog(r_grid[mask_pos], s[mask_pos], label=label, **style)

            mask_ana = f_ana > 0
            if np.any(mask_ana):
                ana_style = get_analytical_style()
                ax.loglog(
                    r_grid[mask_ana],
                    f_ana[mask_ana],
                    label="Analytical SS",
                    **ana_style,
                )

            add_time_colorbar(fig_log, ax, t_min=snap_times[0], t_max=snap_times[-1])

            ax.set_xlabel(r"Radial coordinate: $r$ (a. u.)")
            ax.set_ylabel(r"Solution: $f(t,r)$ (a. u.)")
            ax.set_xlim([source_r_min, r_end])
            ax.set_ylim([1e-1, 1e3])
            ax.grid(False)
            ax.legend()
            plt.tight_layout()
            plt.show()

            last_fig = fig_log
            last_N = N

        try:
            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir, f"advection_piecewise_velocity_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")
            conv_path_pdf = os.path.join(
                out_dir, "advection_piecewise_velocity_convergence.pdf"
            )
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    validation_piecewise_velocity(
        resolutions,
        r_end=20.0,
        t_final=20.0,
        v_c=50.0,
        r_c=0.4,
        source_r_min=1.5,
        source_r_max=2.5,
        source_strength=50.0,
        plot_results=True,
    )
