import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Apply unified plot style
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plot_style import apply_plot_style

apply_plot_style()

from saetass import State, Grid, Solver


def run_advection_simulation(r_grid, t_grid, f_initial, solver_params):
    """
    Create and run an advection Solver for the provided grids and params.
    Returns the final distribution (numpy array) and the solver object.
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
    solver.step(num_timesteps)

    return solver.state.f.flatten(), solver


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
    diff = numerical[mask] - analytical[mask]
    denom = analytical[mask]
    norm_diff = np.sqrt(np.sum(diff**2))
    norm_ana = np.sqrt(np.sum(denom**2))
    if norm_ana == 0:
        return norm_diff
    return norm_diff / norm_ana


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

        f_num, solver = run_advection_simulation(
            r_grid, t_grid, f_initial, solver_params
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

        print(f"  dx={dr:.4e}, steps={len(t_grid)-1}, relL2={relL2:.4e}")

        # store results for plotting later so all plots use same y-limits
        all_results.append(
            {
                "N": N,
                "r_grid": r_grid,
                "f_initial": f_initial,
                "f_num": f_num,
                "f_ana": f_ana,
                "relL2": relL2,
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
        plt.loglog(res, errors, "o-", label="Relative $L_2$ error")
        # # Fit a slope line for reference (power law)
        # if len(res) >= 2:
        #     coeffs = np.polyfit(np.log(res), np.log(errors), 1)
        #     slope = coeffs[0]
        #     print(f"Observed convergence slope ~ {slope:.2f}")
        #     xfit = np.array([res.min(), res.max()])
        #     yfit = np.exp(coeffs[1]) * xfit**slope
        #     plt.loglog(xfit, yfit, "--", label=f"slope {slope:.2f}")

        plt.xlabel("Number of radial cells: $N$")
        plt.ylabel("Relative $L_2$ error")
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

            fig = plt.figure(figsize=(8, 4))
            plt.plot(r_grid, f_initial, "k--", label="Initial")
            plt.plot(r_grid, f_num, label=f"Numerical solution ($N$={N})")
            plt.plot(r_grid, f_ana, "r:", label="Analytical solution")
            plt.xlim(0, r_end)
            plt.ylim(ylims)
            plt.xlabel("Radial coordinate: $r$")
            plt.ylabel("Solution: $f(r)$")
            plt.legend()
            plt.grid(alpha=0.4)
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
