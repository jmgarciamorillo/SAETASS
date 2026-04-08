try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pytest

from saetass import Grid, Solver, State


def run_solver_test(grid_params, operator_params, problem_types, initial_f):
    """
    Helper function to initialize and run the solver for multiple operators.
    """
    grid = Grid(
        r_centers=grid_params.get("r_grid", None),
        t_grid=grid_params["t_grid"],
        p_centers=grid_params.get("p_grid", None),
    )
    state = State(initial_f)

    solver = Solver(
        grid=grid,
        state=state,
        problem_type=problem_types,
        operator_params=operator_params,
        splitting_scheme="strang",
    )

    num_timesteps = len(grid_params["t_grid"]) - 1
    solver.step(num_timesteps)

    return solver.state.f


class TestAdvectionDiffusion:
    """
    Tests for problems combining advection and diffusion operators.
    """

    def test_advection_diffusion_qualitative(self, plot_results):
        """
        Validates the combined effect of advection and diffusion qualitatively.
        The final profile should be advected and diffused compared to the initial state.
        """
        # Parameters
        num_r, r_end, t_final = 500, 100.0, 8.0
        v_const = 4.0  # Advection speed
        D_const = 0.5  # Diffusion coefficient
        r_initial_peak = 20.0
        sigma = 3.0

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 200)

        # Initial condition: Gaussian pulse
        f_initial = np.exp(-((r_grid - r_initial_peak) ** 2) / (2 * sigma**2))

        # --- Run 1: Advection + Diffusion ---
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params_both = {
            "advection": {
                "v_centers": np.full(num_r, v_const),
                "limiter": "minmod",
                "order": 2,
                "cfl": 0.8,
                "inflow_value_U": 1,
            },
            "diffusion": {"D_values": np.full(num_r, D_const), "f_end": 0.0},
        }
        f_final_both = run_solver_test(
            grid_params, op_params_both, "advection-diffusion", f_initial
        ).flatten()

        # --- Run 2: Advection Only ---
        op_params_adv = {"advection": op_params_both["advection"]}
        f_final_adv_only = run_solver_test(
            grid_params, op_params_adv, "advection", f_initial
        ).flatten()

        # --- Run 3: Diffusion Only ---
        op_params_diff = {"diffusion": op_params_both["diffusion"]}
        f_final_diff_only = run_solver_test(
            grid_params, op_params_diff, "diffusion", f_initial
        ).flatten()

        # --- Analysis ---
        def get_width(dist):
            # Use Full Width at Half Maximum (FWHM) as a robust measure of width
            half_max = np.max(dist) / 2.0
            indices = np.where(dist > half_max)[0]
            if len(indices) < 2:
                return 0
            return r_grid[indices[-1]] - r_grid[indices[0]]

        width_initial = get_width(f_initial)
        width_adv_only = get_width(f_final_adv_only)
        width_both = get_width(f_final_both)

        peak_pos_adv_only = r_grid[np.argmax(f_final_adv_only)]
        peak_pos_both = r_grid[np.argmax(f_final_both)]

        # --- Plotting ---
        if plot_results:
            plt.figure(figsize=(12, 8))
            plt.plot(r_grid, f_initial, "k--", label="Initial Profile")
            plt.plot(r_grid, f_final_diff_only, "g:", label="Final (Diffusion Only)")
            plt.plot(r_grid, f_final_adv_only, "b-.", label="Final (Advection Only)")
            plt.plot(
                r_grid, f_final_both, "r-", label="Final (Advection + Diffusion)", lw=2
            )
            plt.title("Combined Advection-Diffusion Test")
            plt.xlabel("Radius (pc)")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # --- Assertions ---
        # 1. The peak of the combined solution should be near the advected position.
        assert np.isclose(peak_pos_both, peak_pos_adv_only, rtol=0.1)

        # 2. The combined solution should be wider than the advection-only solution.
        assert width_both > width_adv_only

        # 3. The peak of the combined solution should be lower than the advection-only peak
        #    due to diffusion.
        assert np.max(f_final_both) < np.max(f_final_adv_only)

        # 4. The combined solution should be wider than the initial profile.
        assert width_both > width_initial


class TestAdvectionSource:
    """
    Tests for problems combining advection and a source term.
    """

    def test_constant_velocity_and_source(self, plot_results):
        """
        Validates that a source injects particles that are then carried away
        by a constant velocity field, creating a plume that goes like 1/r**2.
        """
        # Parameters
        num_r, r_end, t_final = 500, 40.0, 2.0
        v_const = 10.0  # pc/Myr

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 200)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r)

        # Source term: a spike at r=[0.9, 1.1]
        source_r_min, source_r_max = 0.9, 1.1
        Q_values = np.zeros(num_r)
        source_mask = (r_grid >= source_r_min) & (r_grid <= source_r_max)
        Q_values[source_mask] = 40.0

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "advection": {
                "v_centers": np.full(num_r, v_const),
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            },
            "source": {"source": Q_values},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "advection-source", f_initial
        ).flatten()

        # Plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_final, label="Final Profile")
            plt.axvspan(
                source_r_min,
                source_r_max,
                color="red",
                alpha=0.3,
                label="Source Region",
            )
            plt.title("Advection-Source Test (Constant Velocity)")
            plt.xlabel("Radius (pc)")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

            # Additional plot: log-log to see 1/r^2 behavior
            plt.figure(figsize=(10, 6))
            plt.loglog(r_grid, f_final, label="Final Profile")
            plt.axvspan(
                source_r_min,
                source_r_max,
                color="red",
                alpha=0.3,
                label="Source Region",
            )
            plt.title("Advection-Source Test (Constant Velocity) - Log-Log Scale")
            plt.xlabel("Radius (pc)")
            plt.ylabel("f(r)")
            plt.xlim(left=0.1)  # Avoid log(0) issues
            plt.ylim(bottom=1e-6)
            plt.legend()
            plt.grid(True, alpha=0.5, which="both")
            plt.show()

        # --- Assertions ---
        # 1. The distribution should be zero (or very close) upstream of the source.
        upstream_mask = r_grid < source_r_min
        assert np.allclose(f_final[upstream_mask], 0, atol=1e-9)

        # 2. The distribution should be non-zero downstream of the source.
        downstream_mask = r_grid > source_r_max
        assert np.any(f_final[downstream_mask] > 1e-9)

        # 3. Due to spherical geometry and constant velocity, f should decrease ~1/r^2.
        plume_mask = (r_grid > source_r_max) & (f_final > 0)
        plume_indices = np.where(plume_mask)[0]
        if len(plume_indices) > 10:  # Only check if the plume is well-developed
            mid_point = plume_indices[0] + len(plume_indices) // 2
            slope = (
                np.log(f_final[mid_point + 5]) - np.log(f_final[mid_point - 5])
            ) / (np.log(r_grid[mid_point + 5]) - np.log(r_grid[mid_point - 5]))
            expected_slope = -2
            assert np.isclose(slope, expected_slope, rtol=1e-3)

    def test_variable_velocity_and_source(self, plot_results):
        """
        Validates behavior with a source and a spatially varying velocity field (1/r^2).
        Particles should slow down and radial profile should be constant.
        """
        # Parameters
        num_r, r_end, t_final = 800, 6.0, 10.0

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 1000)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r)

        # Velocity field: v ~ 1/r^2, with a cap at small r
        v_field = np.where(r_grid < 0.1, 1 / 0.1**2, 1 / r_grid**2)
        v_field[0] = 0  # Ensure velocity is zero at the origin

        # Source term: a spike at r=[0.9, 1.1]
        source_r_min, source_r_max = 0.9, 1.1
        Q_values = np.zeros(num_r)
        source_mask = (r_grid >= source_r_min) & (r_grid <= source_r_max)
        Q_values[source_mask] = 40.0

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "advection": {
                "v_centers": v_field,
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            },
            "source": {"source": Q_values},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "advection-source", f_initial
        ).flatten()

        # Plotting
        if plot_results:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(r_grid, f_final, label="Final Profile", color="C0")
            ax1.axvspan(
                source_r_min,
                source_r_max,
                color="red",
                alpha=0.3,
                label="Source Region",
            )
            ax1.set_xlabel("Radius (pc)")
            ax1.set_ylabel("f(r)", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(r_grid, v_field, "g--", label="Velocity Field")
            ax2.set_ylabel("Velocity (pc/Myr)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2.legend(loc="upper right")

            plt.title("Advection-Source Test (Variable Velocity)")
            fig.tight_layout()
            plt.show()

        # --- Assertions ---
        # 1. The distribution should be zero upstream of the source.
        upstream_mask = r_grid < source_r_min
        assert np.allclose(f_final[upstream_mask], 0, atol=1e-9)

        # 2. The distribution should be non-zero downstream of the source.
        downstream_mask = r_grid > source_r_max
        assert np.any(f_final[downstream_mask] > 1e-9)

        # 3. Due to v decreasing, f should be more or less constant (spherical geometry).
        # We check that the average slope in the plume region is close to zero.
        plume_mask = (r_grid > source_r_max) & (f_final > 0)
        plume_indices = np.where(plume_mask)[0]
        if len(plume_indices) > 10:  # Only check if the plume is well-developed
            mid_point = plume_indices[0] + len(plume_indices) // 2
            avg_slope = (f_final[mid_point + 5] - f_final[mid_point - 5]) / (
                r_grid[mid_point + 5] - r_grid[mid_point - 5]
            )
            assert np.isclose(avg_slope, 0, atol=1e-2)


class TestDiffusionSource:
    """
    Tests for problems combining diffusion and a source term.
    """

    def test_diffusion_source_steady_state(self, plot_results):
        """
        Validates that the solver reaches the correct analytical steady-state
        for a problem with spatially-dependent D(r) and Q(r).
        """
        # Parameters
        num_r, r_end, t_final = 1000, 1.0, 20
        D_0 = 1.0
        Q_0 = 4.0
        eps = 0.01

        # Grids
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 100000)

        # Initial condition: zero everywhere
        f_initial = np.zeros(num_r)

        # Spatially-dependent diffusion and source
        D_values = D_0 * (r_grid + eps) ** 2
        Q_values = Q_0 * r_grid

        # SubSolver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        op_params = {
            "diffusion": {"D_values": D_values, "f_end": 0.0},
            "source": {"source": Q_values},
        }

        # Run simulation
        f_final = run_solver_test(
            grid_params, op_params, "diffusion-source", f_initial
        ).flatten()

        # Analytical steady-state solution from DiffValidation4.py
        C1 = 1 - 2 * eps * np.log(eps + 1) - eps**2 / (eps + 1)
        analytical_steady_state = (Q_0 / (4 * D_0)) * (
            C1 - (r_grid - 2 * eps * np.log(eps + r_grid) - eps**2 / (eps + r_grid))
        )
        # Ensure boundary condition is met
        analytical_steady_state[-1] = 0.0

        # Plotting
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_final, label="Numerical Final State", lw=2)
            plt.plot(
                r_grid,
                analytical_steady_state,
                "r--",
                label="Analytical Steady State",
                lw=2,
            )
            plt.title("Diffusion-Source Steady State Test")
            plt.xlabel("Radius r")
            plt.ylabel("f(r)")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # --- Assertions ---
        # Check that the final numerical solution is close to the analytical steady state.
        # A tolerance is needed as it's an approximation to a steady state.
        assert np.allclose(f_final, analytical_steady_state, atol=1e-3)


if __name__ == "__main__":
    print("Running tests with plotting enabled...")
    pytest.main([__file__, "--plot"])
