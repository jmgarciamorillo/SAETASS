import pytest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the parent directory to the sys.path to allow imports from the project root
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from State import State
from Grid import Grid
from Solver import Solver


# Helper function to set up and run an advection problem
def run_advection_test(grid_params, solver_params, initial_f):
    """
    Helper function to initialize and run the solver for an advection problem.
    Handles both 1D and 2D cases.
    """
    # Create grid and initial state
    grid = Grid(
        r_centers=grid_params["r_grid"],
        t_grid=grid_params["t_grid"],
        p_centers=grid_params.get("p_grid", np.array([1.0])),
    )
    state = State(initial_f)

    # Create solver
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advectionFV",
        operator_params={"advectionFV": solver_params},
        substeps={"advectionFV": 1},
        splitting_scheme="strang",
    )

    # Run simulation
    num_timesteps = len(grid_params["t_grid"]) - 1
    solver.step(num_timesteps)

    return solver.state.f


class Test1DRadialAdvection:
    """
    Tests for pure 1D radial advection problems.
    """

    def test_simple_translation(self, plot_results):
        """
        Validates that a Gaussian pulse advects correctly against the analytical
        solution, including spherical dilution.
        """
        num_r, r_end, t_final = 2000, 100.0, 10.0
        v_const = 5.0
        r_initial_peak = 20.0
        sigma = 2.0
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 1000)

        # Initial condition: Gaussian pulse
        f_initial = np.exp(-((r_grid - r_initial_peak) ** 2) / (2 * sigma**2))

        # Advection parameters
        v_field = np.full(num_r, v_const)
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0,
        }

        f_final_numerical = run_advection_test(
            grid_params, solver_params, f_initial
        ).flatten()

        # Analytical solution at t_final for spherical advection
        r_shifted = r_grid - v_const * t_final
        f_analytical = np.zeros_like(r_grid)
        # Mask for valid regions (r > 0 and r_shifted > 0)
        mask = (r_grid > 0) & (r_shifted > 0)
        f_analytical[mask] = ((r_shifted[mask] / r_grid[mask]) ** 2) * np.exp(
            -((r_shifted[mask] - r_initial_peak) ** 2) / (2 * sigma**2)
        )

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_initial, "k--", label="Initial Profile")
            plt.plot(r_grid, f_final_numerical, label="Final (Numerical)", lw=2)
            plt.plot(
                r_grid,
                f_analytical,
                "r:",
                label="Final (Analytical)",
                lw=3,
                alpha=0.8,
            )
            plt.title("1D Advection: Numerical vs Analytical Solution")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        assert np.allclose(f_final_numerical, f_analytical, atol=1e-3)

    def test_spherical_dilution(self, plot_results):
        """
        Tests that the peak amplitude decreases as ~1/r^2 due to spherical expansion.
        """
        num_r, r_end, t_final = 4000, 100.0, 10.0
        v_const = 5.0
        r_initial = 20.0
        sigma = 2.0
        r_grid = np.linspace(0.0, r_end, num_r)
        t_grid = np.linspace(0, t_final, 2000)

        f_initial = np.exp(-((r_grid - r_initial) ** 2) / (2 * sigma**2))
        peak_initial = np.max(f_initial)

        v_field = np.full(num_r, v_const)
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0,
        }

        f_final = run_advection_test(grid_params, solver_params, f_initial).flatten()

        peak_final_numerical = np.max(f_final)
        r_final_analytical = r_initial + v_const * t_final

        # Analytical peak height decrease due to spherical dilution (use analytical formula)
        peak_final_analytical = peak_initial * (r_initial / r_final_analytical) ** 2

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(
                r_grid, f_initial, "k--", label=f"Initial (Peak={peak_initial:.3f})"
            )
            plt.plot(r_grid, f_final, label=f"Final (Peak={peak_final_numerical:.3f})")
            plt.axhline(
                peak_final_analytical,
                color="r",
                ls="--",
                label=f"Analytical Peak Height ({peak_final_analytical:.3f})",
            )
            plt.title("1D Advection: Spherical Dilution Test")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # Check that the final peak height is close to the analytical prediction
        # Use a larger tolerance due to numerical diffusion affecting the peak
        assert np.isclose(peak_final_numerical, peak_final_analytical, rtol=1e-2)


class Test2DEnergyRadiusAdvection:
    """
    Tests for 2D advection problems (energy and radius).
    """

    def test_energy_independent_advection(self, plot_results):
        """
        Verifies that if velocity is the same for all energies, all energy slices
        advect identically.
        """
        num_r, num_E, r_end, t_final = 1000, 10, 50.0, 5.0
        v_const = 4.0
        r_initial = 10.0
        r_grid = np.linspace(0.0, r_end, num_r)
        p_grid = np.logspace(0, 2, num_E)  # Dummy momentum grid
        t_grid = np.linspace(0, t_final, 1000)

        # Same velocity field for all energies
        v_field = np.full((num_E, num_r), v_const)

        # Initial condition: Gaussian in radius, same for all energies
        f_initial_1d = np.exp(-((r_grid - r_initial) ** 2) / (2 * 1.0**2))
        f_initial_2d = np.tile(f_initial_1d, (num_E, 1))

        grid_params = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params = {
            "v_centers": v_field,
            "order": 2,
            "limiter": "minmod",
            "cfl": 0.8,
            "inflow_value_U": 0.0,
        }

        f_final_2d = run_advection_test(grid_params, solver_params, f_initial_2d)

        # Get the final profiles for the lowest and highest energies
        f_final_low_E = f_final_2d[0, :]
        f_final_high_E = f_final_2d[-1, :]

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_initial_1d, "k--", label="Initial Profile")
            plt.plot(r_grid, f_final_low_E, label="Final (Low E)", lw=3, alpha=0.8)
            plt.plot(r_grid, f_final_high_E, "r:", label="Final (High E)", lw=3)
            plt.title("2D Energy-Independent Advection")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        # The profiles for all energies should be identical
        assert np.allclose(f_final_low_E, f_final_high_E, atol=1e-7)


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It calls pytest and passes the --plot flag to enable plotting.
    print("Running tests with plotting enabled...")
    pytest.main([__file__, "--plot"])
