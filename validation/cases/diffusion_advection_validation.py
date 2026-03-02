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


def advdiff_spherical_analytical(r_grid, f_initial, t, v, D):
    """
    Analytical solution for spherical radial advection-diffusion with constant
    coefficients using the Green's function on the transformed variable u = r f.
    """
    if t <= 0:
        return f_initial.copy()

    dr = r_grid[1] - r_grid[0]

    # transformed variable u = r f
    u0 = r_grid * f_initial

    # broadcasting grids
    r = r_grid[:, None]
    rp = r_grid[None, :]

    # 1D Gaussian Green kernel (with advection shift v*t)
    kernel = np.exp(-((r - rp - v * t) ** 2) / (4.0 * D * t))
    kernel /= np.sqrt(4.0 * np.pi * D * t)

    # discrete convolution (integral) over rp
    u = np.sum(kernel * u0[None, :], axis=1) * dr

    f = np.zeros_like(u)
    # avoid division by zero at r=0
    f[1:] = u[1:] / r_grid[1:]
    f[0] = f[1]
    return f


def run_operator_simulation(
    r_grid, t_grid, f_initial, operator_params, problem_type, sample_count=0
):
    """
    Run a solver with the given operators and collect sampled snapshots.
    Returns final f, snapshots list and snapshot times, and the solver.
    """
    grid = Grid(r_centers=r_grid, t_grid=t_grid, p_centers=None)
    state = State(f_initial)

    solver = Solver(
        grid=grid,
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={k: 1 for k in operator_params.keys()},
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

    return solver.state.f.flatten(), snapshots, times, solver


def fwhm_width(dist, r_grid):
    half_max = np.max(dist) / 2.0
    indices = np.where(dist > half_max)[0]
    if len(indices) < 2:
        return 0.0
    return r_grid[indices[-1]] - r_grid[indices[0]]


def validation_diffusion_advection(
    N=500,
    r_end=100.0,
    t_min=0.5,
    t_max=8.0,
    n_times=8,
    v_const=4.0,
    D_const=0.5,
    r_initial_peak=20.0,
    sigma=3.0,
    sample_count=6,
    t_steps=200,
    plot_results=True,
):
    """
    Validation that runs advection-only, diffusion-only and combined advection+diffusion
    for several final times. Produces temporal evolution plots (sampled snapshots)
    and simple diagnostics (peak position and width) as a function of final time.
    """
    r_grid = np.linspace(0.0, r_end, N)

    # initial gaussian pulse
    f_initial = np.exp(-((r_grid - r_initial_peak) ** 2) / (2 * sigma**2))

    t_finals = np.linspace(t_min, t_max, n_times)

    # diagnostics for combined advection+diffusion only
    peak_positions_both = []
    widths_both = []
    analytic_peaks = []
    analytic_widths = []
    # conservation totals (integral of 4*pi*r^2*f) for each final time
    try:
        total_initial = np.trapz(4.0 * np.pi * r_grid**2 * f_initial, r_grid)
    except Exception:
        total_initial = None
    totals_numerical = []
    totals_analytical = []
    all_results = []

    print(
        f"Running diffusion+advection validation on r-grid N={N}, t in [{t_min},{t_max}]"
    )

    for t_final in t_finals:
        print(f" Running t_final={t_final:.4g}")
        t_grid = np.linspace(0.0, t_final, t_steps)

        # combined advection + diffusion
        op_both = {
            "advection": {
                "v_centers": np.full(N, v_const),
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            },
            "diffusion": {"D_values": np.full(N, D_const), "f_end": 0.0},
        }
        f_both, snaps_both, times_both, _ = run_operator_simulation(
            r_grid,
            t_grid,
            f_initial,
            op_both,
            "advection-diffusion",
            sample_count=sample_count,
        )
        # diagnostics (combined only)
        peak_positions_both.append(r_grid[np.argmax(f_both)])
        widths_both.append(fwhm_width(f_both, r_grid))

        # analytical solution for combined advection+diffusion at final time
        f_ana_final = advdiff_spherical_analytical(
            r_grid, f_initial, t_final, v_const, D_const
        )
        analytic_peaks.append(r_grid[np.argmax(f_ana_final)])
        analytic_widths.append(fwhm_width(f_ana_final, r_grid))

        # conservation totals for this final time
        try:
            totals_numerical.append(np.trapz(4.0 * np.pi * r_grid**2 * f_both, r_grid))
        except Exception:
            totals_numerical.append(None)
        try:
            totals_analytical.append(
                np.trapz(4.0 * np.pi * r_grid**2 * f_ana_final, r_grid)
            )
        except Exception:
            totals_analytical.append(None)

        all_results.append(
            {
                "t_final": t_final,
                "r_grid": r_grid,
                "f_both": f_both,
                "snaps_both": snaps_both,
                "times_both": times_both,
            }
        )

    if plot_results:
        # plot metrics vs time
        t_arr = np.array(t_finals)
        # conservation vs time figure
        try:
            plt.figure(figsize=(6, 4))
            # plot numerical and analytical totals; initial as horizontal line
            nums = np.array([x if x is not None else np.nan for x in totals_numerical])
            anas = np.array([x if x is not None else np.nan for x in totals_analytical])
            plt.plot(t_arr, nums, "b-o", label="Numerical total")
            plt.plot(t_arr, anas, "r--s", label="Analytical total")
            if total_initial is not None:
                plt.hlines(
                    total_initial,
                    t_arr[0],
                    t_arr[-1],
                    colors="k",
                    linestyles=":",
                    label="Initial total",
                )
            plt.xlabel("Final simulation time: $t_{final}$")
            plt.ylabel(r"Total integral: $\\int 4\\pi r^2 f(r) \, dr$")
            plt.legend()
            plt.grid(alpha=0.3)
            cons_fig = plt.gcf()
            plt.show()
        except Exception as e:
            print(f"Warning: could not create conservation figure: {e}")
        plt.figure(figsize=(6, 4))
        plt.plot(t_arr, peak_positions_both, "r-", label="Peak adv+diff (numerical)")
        # overlay analytical peak position
        plt.plot(t_arr, np.array(analytic_peaks), "k:", label="Analytical peak")
        plt.xlabel("Final simulation time: $t_{final}$")
        plt.ylabel("Peak position (r)")
        plt.legend()
        plt.grid(alpha=0.3)
        metrics_fig = plt.gcf()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(t_arr, widths_both, "r-", label="Width adv+diff (numerical)")
        # overlay analytical widths
        plt.plot(t_arr, np.array(analytic_widths), "k:", label="Analytical width")
        plt.xlabel("Final simulation time: $t_{final}$")
        plt.ylabel("FWHM width")
        plt.legend()
        plt.grid(alpha=0.3)
        metrics2_fig = plt.gcf()
        plt.show()

        # temporal evolution plot for the run closest to the largest t_final
        target_rec = all_results[-1]
        r_grid = target_rec["r_grid"]
        snaps = target_rec["snaps_both"]
        times = target_rec["times_both"]

        fig = plt.figure(figsize=(8, 4))
        # Plot only: initial condition, numerical final, analytical final
        plt.plot(r_grid, f_initial, "k--", label="Initial condition")
        plt.plot(r_grid, target_rec["f_both"], "b-", label="Numerical final")
        f_ana_final = advdiff_spherical_analytical(
            r_grid, f_initial, target_rec["t_final"], v_const, D_const
        )
        plt.plot(r_grid, f_ana_final, "r--", label="Analytical final")
        plt.xlabel(r"Radial coordinate: $r$")
        plt.ylabel(r"Solution: $f(r)$")
        plt.title(f"Temporal evolution up to t_final={target_rec['t_final']:.3g}")
        plt.legend()
        plt.grid(alpha=0.4)
        last_fig = plt.gcf()
        plt.show()

        # save figures to ../figures
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)

            metrics_path = os.path.join(out_dir, "diffusion_advection_metrics.pdf")
            metrics_fig.savefig(metrics_path, dpi=200, bbox_inches="tight")
            print(f"Saved metrics figure to: {metrics_path}")

            metrics2_path = os.path.join(out_dir, "diffusion_advection_widths.pdf")
            metrics2_fig.savefig(metrics2_path, dpi=200, bbox_inches="tight")
            print(f"Saved widths figure to: {metrics2_path}")

            if last_fig is not None:
                last_path_pdf = os.path.join(
                    out_dir,
                    f"diffusion_advection_last_t{target_rec['t_final']:.3g}.pdf",
                )
                last_fig.savefig(last_path_pdf, dpi=200, bbox_inches="tight")
                print(f"Saved last simulation figure to: {last_path_pdf}")
        except Exception as e:
            print(f"Warning: could not save figures: {e}")

        # --- Convergence sweep over resolutions for the chosen final time ---
        try:
            Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
            t_target = target_rec["t_final"]
            print(f"Running convergence sweep for t_final={t_target} over Ns={Ns}")

            errors = []
            for Ntest in Ns:
                print(f"  Running N={Ntest}")
                r_grid_test = np.linspace(0.0, r_end, Ntest)
                f_init_test = np.exp(
                    -((r_grid_test - r_initial_peak) ** 2) / (2 * sigma**2)
                )
                t_grid_test = np.linspace(0.0, t_target, t_steps)

                op_adv_test = {
                    "advection": {
                        "v_centers": np.full(Ntest, v_const),
                        "order": 2,
                        "limiter": "minmod",
                        "cfl": 0.8,
                        "inflow_value_U": 0.0,
                    }
                }
                op_diff_test = {
                    "diffusion": {"D_values": np.full(Ntest, D_const), "f_end": 0.0}
                }
                op_both_test = {**op_adv_test, **op_diff_test}

                f_test, _, _, _ = run_operator_simulation(
                    r_grid_test,
                    t_grid_test,
                    f_init_test,
                    op_both_test,
                    "advection-diffusion",
                    sample_count=1,
                )

                f_analytical = advdiff_spherical_analytical(
                    r_grid_test, f_init_test, t_target, v_const, D_const
                )

                # compute relative L2 error
                denom = np.linalg.norm(f_analytical)
                if denom == 0:
                    relL2 = np.linalg.norm(f_test - f_analytical)
                else:
                    relL2 = np.linalg.norm(f_test - f_analytical) / denom
                errors.append(relL2)
                print(f"    relL2={relL2:.3e}")

            # plot convergence
            plt.figure(figsize=(6, 4))
            plt.loglog(Ns, errors, "o-", label="Relative $L_2$ error")
            plt.xlabel("Number of radial cells: $N$")
            plt.ylabel("Relative $L_2$ error vs analytic")
            plt.grid(alpha=0.3, which="both")
            conv_fig = plt.gcf()
            plt.show()

            # save convergence figure
            conv_path = os.path.join(
                out_dir, f"diffusion_advection_convergence_t{t_target:.3g}.pdf"
            )
            conv_fig.savefig(conv_path, dpi=200, bbox_inches="tight")
            print(f"Saved convergence figure to: {conv_path}")

        except Exception as e:
            print(f"Warning: convergence sweep failed: {e}")


if __name__ == "__main__":
    validation_diffusion_advection(
        N=4096,
        r_end=100.0,
        t_min=0.5,
        t_max=8.0,
        n_times=15,
        v_const=4.0,
        D_const=0.5,
        r_initial_peak=20.0,
        sigma=3.0,
        sample_count=6,
        t_steps=1000,
        plot_results=True,
    )
