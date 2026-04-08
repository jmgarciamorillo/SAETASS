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


def run_advection_simulation(r_grid, t_grid, f_initial, solver_params, sample_count=5):
    """
    Create and run an advection Solver for the provided grids and params.
    Returns the final distribution (numpy array), the snapshots, and snap times.
    """
    grid = Grid(r_centers=r_grid, t_grid=t_grid, p_centers=None)
    state = State(f_initial)

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advection",
        operator_params={"advection": solver_params},
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


def analytical_spherical_advection(r_grid, f_initial_func, v_const, t_final):
    """
    Analytical solution for spherical radial advection (particles advected outward
    at constant velocity v_const). For a scalar field f(r,t) representing a
    density per unit volume (or per radial coordinate with spherical dilution
    accounted by factor r^2), the transported profile obeys:
        f(r,t) = (r_shifted / r)**2 * f_initial(r_shifted)
    where r_shifted = r - v * t and valid for r_shifted > 0 and r > 0.

    We avoid r=0 by returning 0 there.
    """
    r_shifted = r_grid - v_const * t_final
    f_ana = np.zeros_like(r_grid)
    mask = (r_grid > 0) & (r_shifted > 0)
    f_ana[mask] = (r_shifted[mask] / r_grid[mask]) ** 2 * f_initial_func(
        r_shifted[mask]
    )
    # Keep r=0 as zero
    return f_ana


def gaussian_shell(r, r0, sigma):
    return np.exp(-((r - r0) ** 2) / (2 * sigma**2))


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


def validation_sweep(
    resolutions,
    r_end=100.0,
    t_final=10.0,
    v_const=5.0,
    r0=20.0,
    sigma=2.0,
    cfl=0.8,
    plot_results=True,
):
    """
    Run advection validation for the given list of spatial resolutions `resolutions`.
    Returns arrays of N and relative L2 error.
    """
    errors = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running N={N}")
        r_grid = np.linspace(0.0, r_end, N)
        dr = r_grid[1] - r_grid[0]

        t_grid = np.linspace(0.0, t_final, 20000)

        # initial condition: gaussian shell
        f_initial = gaussian_shell(r_grid, r0, sigma)

        # solver params
        v_field = np.full(N, v_const)
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": cfl,
            "inflow_value_U": 0.0,
        }

        f_num, snapshots, snap_times = run_advection_simulation(
            r_grid, t_grid, f_initial, solver_params, sample_count=7
        )

        # analytical solution at t_final
        f_ana = analytical_spherical_advection(
            r_grid, lambda r: gaussian_shell(r, r0, sigma), v_const, t_final
        )

        # mask valid region where analytical is defined (r_shifted>0 and r>0)
        mask = (r_grid > 0) & (r_grid - v_const * t_final > 0)

        relL2 = compute_relative_L2(f_num, f_ana, mask)
        errors.append(relL2)
        dxs.append(dr)

        print(f"  dx={dr:.4e}, steps={len(t_grid) - 1}, relL2={relL2:.4e}")

        # store results for plotting later so all plots use same y-limits
        all_results.append(
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

    # convergence plot
    if plot_results:
        dxs = np.array(dxs)
        res = np.array(resolutions)
        errors = np.array(errors)

        # compute common y-limits across all simulation results
        ymin = np.inf
        ymax = -np.inf
        for rec in all_results:
            ymin = min(ymin, np.min(rec["f_initial"]))
            ymax = max(ymax, np.max(rec["f_initial"]))
            ymin = min(ymin, np.min(rec["f_num"]))
            ymax = max(ymax, np.max(rec["f_num"]))

        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 0.0, 1.0
        padding = 0.05 * (ymax - ymin) if (ymax - ymin) > 0 else 0.1
        ylims = (max(0.0, ymin - padding), ymax + padding)

        plt.figure(figsize=(6, 4))
        quant_style = get_quantitative_style()
        plt.loglog(res, errors, label=r"Error ($\mathcal{E}_{L_2}$)", **quant_style)
        # # Fit a slope line for reference (power law)
        # if len(res) >= 2:
        #     coeffs = np.polyfit(np.log(res), np.log(errors), 1)
        #     slope = coeffs[0]
        #     print(f"Observed convergence slope ~ {slope:.2f}")
        #     xfit = np.array([res.min(), res.max()])
        #     yfit = np.exp(coeffs[1]) * xfit**slope
        #     plt.loglog(xfit, yfit, "--", label=f"slope {slope:.2f}")

        plt.xlabel("Number of radial cells: $N$")
        plt.ylabel(r"Relative error: $\mathcal{E}_{L_2}$")
        plt.grid(alpha=0.3, which="both")
        # capture convergence figure
        conv_fig = plt.gcf()
        plt.show()

        # plot each simulation with consistent y-limits
        last_fig = None
        last_N = None
        for rec in all_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            f_initial = rec["f_initial"]
            f_num = rec["f_num"]
            f_ana = rec["f_ana"]
            relL2 = rec["relL2"]
            snapshots = rec["snapshots"]
            snap_times = rec["snap_times"]

            fig = plt.figure(figsize=(6, 4))

            for idx, (s, t) in enumerate(zip(snapshots, snap_times)):
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
                plt.plot(r_grid, s, label=label, **style)

            ana_style = get_analytical_style()
            plt.plot(r_grid, f_ana, label="Analytical (final)", **ana_style)

            add_time_colorbar(fig, plt.gca(), t_min=snap_times[0], t_max=snap_times[-1])

            plt.xlim(0, r_end)
            plt.ylim(ylims)
            plt.xlabel("Radial coordinate: $r$")
            plt.ylabel("Solution: $f(r)$")
            plt.legend()
            plt.grid(alpha=0.4)
            plt.tight_layout()
            plt.show()
            last_fig = fig
            last_N = N

        # save figures: last simulation and convergence (PDF)
        try:
            out_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
            os.makedirs(out_dir, exist_ok=True)
            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir, f"advection_gaussian_shell_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")
            conv_path_pdf = os.path.join(
                out_dir, "advection_gaussian_shell_convergence.pdf"
            )
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    validation_sweep(
        resolutions,
        r_end=100.0,
        t_final=10.0,
        v_const=5.0,
        r0=20.0,
        sigma=2.0,
        cfl=0.8,
        plot_results=True,
    )
