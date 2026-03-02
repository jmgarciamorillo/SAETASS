import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX for matplotlib text rendering when available
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        # increased font sizes for publication-quality figures
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "lines.linewidth": 1.5,
    }
)

from saetass import State, Grid, Solver


def run_diffusion_simulation(r_grid, t_grid, f_initial, solver_params, sample_count=0):
    """
    Create and run a diffusion Solver for the provided grids and params.
    Collect sampled snapshots by advancing the solver in chunks to sampled
    timestep indices (mirrors the approach used in other validation scripts).

    Returns the final distribution (numpy array), a list of snapshots and
    the corresponding snapshot times.
    """
    grid = Grid(r_centers=r_grid, t_grid=t_grid, p_centers=None)
    state = State(f_initial)

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="diffusion",
        operator_params={"diffusion": solver_params},
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )

    num_timesteps = len(t_grid) - 1

    # snapshots: include initial
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


def compute_relative_L2(numerical, analytical, mask=None):
    if mask is None:
        mask = np.ones_like(numerical, dtype=bool)
    diff = numerical[mask] - analytical[mask]
    denom = analytical[mask]
    norm_diff = np.sqrt(np.sum(diff**2))
    norm_ana = np.sqrt(np.sum(denom**2))
    if norm_ana == 0:
        return norm_diff
    return norm_diff / norm_ana


def validation_diffusion_analytic(
    resolutions,
    r_end=1.0,
    t_final=0.1,
    D_const=1.0,
    plot_results=True,
):
    """
    Validation of 1D radial diffusion with a sinc-like initial profile.
    Analytical decay: f(r,t) = f_initial(r) * exp(-pi^2 * D_const * t).
    Sweep resolutions, measure relative L2 error and save figures to ../figures.
    """
    errors = []
    dxs = []
    all_results = []

    for N in resolutions:
        print(f"Running N={N}")
        r_grid = np.linspace(0.0, r_end, N)
        dr = r_grid[1] - r_grid[0]

        t_grid = np.linspace(0.0, t_final, 10000)

        # sinc-like initial condition (as in tests)
        f_initial = (np.pi / 2.0) * np.sinc(r_grid)

        solver_params = {"D_values": np.full(N, D_const), "f_end": 0.0}

        f_num, snapshots, snap_times = run_diffusion_simulation(
            r_grid, t_grid, f_initial, solver_params, sample_count=6
        )

        # analytical solution at t_final
        f_ana = f_initial * np.exp(-(np.pi**2) * D_const * t_final)

        relL2 = compute_relative_L2(f_num, f_ana)
        errors.append(relL2)
        dxs.append(dr)

        print(f"  dx={dr:.4e}, steps={len(t_grid)-1}, relL2={relL2:.4e}")

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

    if plot_results:
        dxs = np.array(dxs)
        res = np.array(resolutions)
        errors = np.array(errors)

        # compute common y-limits across all results
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

        # convergence plot
        plt.figure(figsize=(6, 4))
        plt.loglog(res, errors, "o-", label="Relative $L_2$ error")
        plt.xlabel("Number of radial cells: $N$")
        plt.ylabel("Relative $L_2$ error")
        plt.grid(alpha=0.3, which="both")
        conv_fig = plt.gcf()
        plt.show()

        # per-resolution plots with consistent y-limits
        last_fig = None
        last_N = None
        for rec in all_results:
            N = rec["N"]
            r_grid = rec["r_grid"]
            f_initial = rec["f_initial"]
            f_num = rec["f_num"]
            f_ana = rec["f_ana"]
            snapshots = rec.get("snapshots", [f_num])
            snap_times = rec.get("snap_times", [t_final])

            fig = plt.figure(figsize=(8, 4))
            colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
            # plot snapshots with varying alpha; emphasize initial and final
            for i, (s, t, c) in enumerate(zip(snapshots, snap_times, colors)):
                alpha = 0.9 if (i == 0 or i == len(snapshots) - 1) else 0.5

                # Label every snapshot so all lines appear in the legend
                label = f"$t = {t:.2f}$"
                plt.plot(r_grid, s, color=c, alpha=alpha, label=label)

            # analytical solution (final)
            plt.plot(r_grid, f_ana, "k:", label="Analytical (final)")

            plt.xlim(0, r_end)
            plt.ylim(ylims)
            plt.xlabel(r"Radial coordinate: $r$")
            plt.ylabel(r"Solution at time $t$: $f(r,t)$")
            plt.legend()
            plt.grid(alpha=0.4)
            plt.show()
            last_fig = fig
            last_N = N

        # save figures to ../figures
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir, f"diffusion_analytic_last_N{last_N}.pdf"
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")
            conv_path_pdf = os.path.join(out_dir, "diffusion_analytic_convergence.pdf")
            conv_fig.savefig(conv_path_pdf, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")


if __name__ == "__main__":
    resolutions = [128, 256, 512, 1024, 2048, 4096]
    validation_diffusion_analytic(
        resolutions,
        r_end=1.0,
        t_final=0.3,
        D_const=1.0,
        plot_results=True,
    )
