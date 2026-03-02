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


def run_loss_simulation(p_grid, t_grid, f_initial, operator_params, sample_count=0):
    """
    Run a loss+source simulation on the momentum axis (p_grid) and collect
    sampled snapshots. Returns final f, snapshots list, snapshot times, and solver.
    """
    grid = Grid(r_centers=None, t_grid=t_grid, p_centers=p_grid)
    state = State(f_initial)

    # decide problem type
    if "source" in operator_params:
        problem_type = "loss-source"
    else:
        problem_type = "loss"

    solver = Solver(
        grid=grid,
        state=state,
        problem_type=problem_type,
        operator_params=operator_params,
        substeps={"loss": 1},
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


def analytical_steady_state_loss(p_grid, Q0, b0, alpha, beta, p0, p_end):
    """
    Analytical steady-state for the loss+source model used in LossValidation1.py:
    f(p) = [Q0 * p0 / (1-alpha) / b0] * ( (p_end/p0)^(1-alpha) - (p/p0)^(1-alpha) ) * (p/p0)^(-beta)
    """
    pref = Q0 * p0 / (1.0 - alpha) / b0
    term = (p_end / p0) ** (1.0 - alpha) - (p_grid / p0) ** (1.0 - alpha)
    f = pref * term * (p_grid / p0) ** (-beta)
    return f


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


def validation_loss_source_steady_state(
    N=300,
    p_min=1.0,
    p_max=1000.0,
    p0=1.0,
    t_min=0.1,
    t_max=3.0,
    n_times=10,
    Q0=1.0,
    b0=1.0,
    alpha=4.0,
    beta=2.0,
    t_steps=2000,
    sample_count=6,
    plot_results=True,
):
    """
    Validation of a loss+source steady-state problem.
    Sweeps final times from t_min to t_max (n_times points), runs the solver, and
    compares the numerical final state with the analytical steady state.
    Produces temporal-evolution plots (sampled snapshots) and a residuals-vs-time plot.
    """
    p_grid = np.logspace(np.log10(p_min), np.log10(p_max), N)

    f_initial = np.zeros(N)

    # define source and loss rates following LossValidation1.py
    Q_values = Q0 * (p_grid / p0) ** (-alpha)
    P_dot = -b0 * (p_grid / p0) ** beta  # P_dot is negative for losses

    # pack operator params
    loss_params = {
        "P_dot": P_dot,
        "limiter": "minmod",
        "cfl": 0.2,
        "order": 2,
        "inflow_value_U": 0.0,
        "adiabatic_losses": False,
    }
    operator_params = {"loss": loss_params, "source": {"source": Q_values}}

    # analytical steady state (same for all times)
    ana = analytical_steady_state_loss(p_grid, Q0, b0, alpha, beta, p0, p_max)

    t_finals = np.linspace(t_min, t_max, n_times)

    residuals = []
    times_list = []
    all_results = []

    print(
        f"Running loss+source steady-state validation on p-grid N={N}, p in [{p_min},{p_max}]"
    )

    for t_final in t_finals:
        print(f"Running t_final={t_final:.4g}")
        t_grid = np.linspace(0.0, t_final, t_steps)

        f_num, snapshots, snap_times, solver = run_loss_simulation(
            p_grid, t_grid, f_initial, operator_params, sample_count=sample_count
        )

        relL2 = compute_relative_L2(f_num, ana)
        residuals.append(relL2)
        times_list.append(t_final)

        print(f"  t_final={t_final:.4e}, relL2={relL2:.4e}")

        all_results.append(
            {
                "t_final": t_final,
                "p_grid": p_grid,
                "f_num": f_num,
                "ana": ana,
                "relL2": relL2,
                "snapshots": snapshots,
                "snap_times": snap_times,
            }
        )

    if plot_results:
        times_arr = np.array(times_list)
        residuals = np.array(residuals)

        # residuals vs time
        plt.figure(figsize=(6, 4))
        plt.semilogy(times_arr, residuals, "o-", label="Relative $L_2$ error")
        plt.xlabel(r"Final simulation time: $t_\mathrm{end}$")
        plt.ylabel("Relative $L_2$ error")
        plt.grid(alpha=0.3, which="both")
        conv_fig = plt.gcf()
        plt.show()
        # save residuals/convergence figure
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)
            conv_path = os.path.join(out_dir, "loss_source_residuals.pdf")
            conv_fig.savefig(conv_path, dpi=200, bbox_inches="tight")
            print(f"Saved residuals (convergence) figure to: {conv_path}")
        except Exception as e:
            print(f"Warning: could not save residuals figure: {e}")

        # compute y-limits for p^5 * f plotting (similar to LossValidation1)
        ymin = np.inf
        ymax = -np.inf
        for rec in all_results:
            p = rec["p_grid"]
            fnum = rec["f_num"]
            ana_ = rec["ana"]
            vals = (p**5) * np.maximum(fnum, 1e-300)
            vals_ana = (p**5) * np.maximum(ana_, 1e-300)
            ymin = min(ymin, np.min(vals))
            ymax = max(ymax, np.max(vals_ana))
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 1e-20, 1e20

        for rec in all_results:
            t_final = rec["t_final"]
            p = rec["p_grid"]
            snapshots = rec.get("snapshots", [])
            snap_times = rec.get("snap_times", [])
            ana_ = rec["ana"]

            fig = plt.figure(figsize=(8, 4))
            colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
            for idx, (s, t) in enumerate(zip(snapshots, snap_times)):
                plt.loglog(
                    p,
                    (p**5) * np.maximum(s, 1e-300),
                    color=colors[idx],
                    label=f"$t={t:.3f}$",
                )

            plt.loglog(
                p,
                (p**5) * np.maximum(ana_, 1e-300),
                "k--",
                label="Analytical steady state",
            )

            plt.xlim(p[0], p[-1])
            plt.ylim(1e-2, 1)
            plt.xlabel(r"Momentum coordinate: $p$")
            plt.ylabel(r"Slope-corrected spectrum at $t$: $p^5 f(p,t)$")
            plt.legend()
            plt.grid(alpha=0.4, which="both")
            # store the figure object so we can save exactly the same one later
            try:
                rec["fig"] = fig
            except Exception:
                pass
            plt.show()

        # save only the figure corresponding to t_end ~= 0.4
        try:
            out_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "figures")
            )
            os.makedirs(out_dir, exist_ok=True)

            target_t = 0.4
            # pick recorded run closest to target_t
            rec_target = min(all_results, key=lambda r: abs(r["t_final"] - target_t))

            # save the exact figure that was shown for the chosen run, if available
            fig_target = rec_target.get("fig")
            if fig_target is not None:
                try:
                    target_path_pdf = os.path.join(
                        out_dir, f"loss_source_steady_tend0p4.pdf"
                    )
                    fig_target.savefig(target_path_pdf, dpi=200, bbox_inches="tight")
                    print(f"Saved t_end=0.4 simulation figure to: {target_path_pdf}")
                except Exception as e:
                    print(f"Warning: could not save target figure from stored fig: {e}")
            else:
                # fallback: re-create the plot (older behavior)
                try:
                    fig_target = plt.figure(figsize=(8, 4))
                    p = rec_target["p_grid"]
                    snapshots = rec_target.get("snapshots", [])
                    snap_times = rec_target.get("snap_times", [])
                    ana_ = rec_target["ana"]
                    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
                    for idx, (s, t) in enumerate(zip(snapshots, snap_times)):
                        plt.loglog(
                            p,
                            (p**5) * np.maximum(s, 1e-300),
                            color=colors[idx],
                            label=f"$t={t:.3f}$",
                        )
                    plt.loglog(
                        p,
                        (p**5) * np.maximum(ana_, 1e-300),
                        "k--",
                        label="Analytical steady state",
                    )
                    plt.title(
                        f"Temporal evolution up to t_final={rec_target['t_final']:.3g}"
                    )
                    plt.xlim(p[0], p[-1])
                    try:
                        plt.ylim(ymin, ymax)
                    except Exception:
                        pass
                    plt.xlabel(r"Momentum coordinate: $p$")
                    plt.ylabel(r"Slope-corrected spectrum at $t$: $p^5 f(p,t)$")
                    plt.legend()
                    plt.grid(alpha=0.4, which="both")

                    target_path_pdf = os.path.join(
                        out_dir, f"loss_source_steady_tend0p4.pdf"
                    )
                    fig_target.va(target_path_pdf, dpi=200, bbox_inches="tight")
                    print(f"Saved t_end=0.4 simulation figure to: {target_path_pdf}")
                except Exception as e:
                    print(f"Warning: could not save target figure: {e}")

        except Exception as e:
            print(f"Warning: could not save target figure: {e}")


if __name__ == "__main__":
    validation_loss_source_steady_state(
        N=1024,
        p_min=1.0,
        p_max=1000.0,
        p0=1.0,
        t_min=0.1,
        t_max=0.8,
        n_times=8,
        Q0=1.0,
        b0=1.0,
        alpha=4.0,
        beta=2.0,
        t_steps=10000,
        sample_count=8,
        plot_results=True,
    )
