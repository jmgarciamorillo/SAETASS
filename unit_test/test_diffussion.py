import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
import math
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


# Helper function to set up and run a diffusion problem
def run_diffusion_test(grid_params, solver_params, initial_f):
    """
    Helper function to initialize and run the solver for a diffusion problem.
    Handles both 1D and 2D cases.
    """
    # Create grid and initial state
    grid = Grid(
        r_centers=grid_params["r_grid"],
        t_grid=grid_params["t_grid"],
        p_centers=grid_params.get("p_grid", None),
    )
    state = State(initial_f)

    # Create solver
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="diffusionFV",
        operator_params={"diffusionFV": solver_params},
        substeps={"diffusionFV": 1},
        splitting_scheme="strang",
    )

    # Run simulation
    num_timesteps = len(grid_params["t_grid"]) - 1
    solver.step(num_timesteps)

    return solver.state.f


class Test1DRadialDiffusion:
    """
    Tests for pure 1D radial diffusion problems.
    A dummy energy dimension (num_E=1) is used to fit the solver's 2D structure.
    """

    @pytest.mark.parametrize("num_points", [200, 400, 800])
    def test_analytical_sinc(self, num_points, plot_results):
        """
        Validates against an analytical solution with a sinc-like initial profile.
        """
        r_0, r_end, t_final, D_const = 0.0, 1.0, 0.1, 1.0
        r_grid = np.linspace(r_0, r_end, num_points)
        t_grid = np.linspace(0, t_final, 100)
        f_initial = (np.pi / 2) * np.sinc(r_grid)

        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": np.full(num_points, D_const),
            "f_end": 0.0,
        }

        f_final_numerical = run_diffusion_test(
            grid_params, solver_params, f_initial
        ).flatten()
        f_final_analytical = f_initial * np.exp(-(np.pi**2) * D_const * t_final)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_final_numerical, label="Numerical", lw=2)
            plt.plot(r_grid, f_final_analytical, "r--", label="Analytical", lw=2)
            plt.title(f"1D Sinc Test - {num_points} points")
            plt.legend()
            plt.grid(True, alpha=0.5)
            plt.show()

        assert np.allclose(f_final_numerical, f_final_analytical, atol=1e-4)

    def test_particle_conservation(self, plot_results):
        """
        Tests that the total number of particles is conserved in a closed system (where no enough time has passed for significant loss).
        """
        num_points, r_end, t_final, D_const = 800, 500.0, 1000.0, 0.1
        r_grid = np.linspace(0.0, r_end, num_points)
        t_grid = np.linspace(0, t_final, 100)
        dr = r_grid[1] - r_grid[0]
        f_initial = np.exp(-((r_grid - 50.0) ** 2) / (2 * 5**2))

        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": np.full(num_points, D_const),
            "f_end": 0.0,
        }

        integrand_initial = 4 * np.pi * r_grid**2 * f_initial
        n_particles_initial = np.sum(integrand_initial * dr)

        f_final_numerical = run_diffusion_test(
            grid_params, solver_params, f_initial
        ).flatten()

        integrand_final = 4 * np.pi * r_grid**2 * f_final_numerical
        n_particles_final = np.sum(integrand_final * dr)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_initial, label=f"Initial, N={n_particles_initial:.4f}")
            plt.plot(
                r_grid, f_final_numerical, label=f"Final, N={n_particles_final:.4f}"
            )
            plt.title("1D Particle Conservation Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        assert np.isclose(n_particles_initial, n_particles_final, rtol=1e-3)

    def test_discontinuous_diffusion_coefficient(self, plot_results):
        """
        Tests behavior with a discontinuous diffusion coefficient in 1D.
        A jump downwards in D should cause an accumulation of particles.
        """
        # Parameters
        num_points = 200
        r_end = 1.0
        t_final = 0.1

        # Grids
        r_grid = np.linspace(0.0, r_end, num_points)
        t_grid = np.linspace(0, t_final, 100)

        # Initial condition (Gaussian centered before the discontinuity)
        f_initial = np.exp(-((r_grid - 0.3) ** 2) / (2 * 0.05**2))

        # Discontinuous diffusion coefficient
        D_values = np.ones(num_points)
        discontinuity_idx = int(num_points / 2)
        D_values[discontinuity_idx:] = 0.1  # D drops by a factor of 10

        # Solver parameters
        grid_params = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params = {
            "D_values": D_values,
            "f_end": 0.0,
        }

        # Run simulation
        f_final = run_diffusion_test(grid_params, solver_params, f_initial).flatten()

        # Plotting for visual validation
        if plot_results:
            fig, ax1 = plt.subplots(figsize=(12, 7))
            # Plot distribution
            ax1.plot(r_grid, f_initial, "k--", label="Initial Profile", alpha=0.5)
            ax1.plot(r_grid, f_final, label="Final Profile", lw=2)
            ax1.axvline(
                r_grid[discontinuity_idx],
                color="r",
                linestyle="--",
                label="Discontinuity in D",
            )
            ax1.set_xlabel("Radius r")
            ax1.set_ylabel("f(r)", color="C0")
            ax1.tick_params(axis="y", labelcolor="C0")
            ax1.legend(loc="upper left")
            ax1.grid(True, alpha=0.3)

            # Plot diffusion coefficient on a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(r_grid, D_values, "g-", label="Diffusion Coeff. D(r)", alpha=0.6)
            ax2.set_ylabel("D(r)", color="g")
            ax2.tick_params(axis="y", labelcolor="g")
            ax2.legend(loc="upper right")
            plt.title("1D Diffusion with Discontinuous Coefficient")
            fig.tight_layout()
            plt.show()

        # Check for accumulation: f should be higher just before the boundary
        f_before = f_final[discontinuity_idx - 1]
        f_after = f_final[discontinuity_idx]
        assert f_before > f_after

        # Check for gradient change: gradient should be steeper in the low-D region
        grad_high_D = np.mean(
            np.abs(np.diff(f_final[discontinuity_idx - 5 : discontinuity_idx - 1]))
        )
        grad_low_D = np.mean(
            np.abs(np.diff(f_final[discontinuity_idx : discontinuity_idx + 4]))
        )

        # Since D_high > D_low, we expect |grad_high| < |grad_low| for flux continuity
        assert grad_high_D < grad_low_D


class Test2DEnergyRadiusDiffusion:
    """
    Tests for 2D problems involving both energy and radius dimensions.
    """

    def test_energy_dependent_diffusion(self, plot_results):
        """
        Verifies that diffusion is faster for energies with a higher diffusion coefficient.
        """
        num_r, num_E = 200, 10
        r_grid = np.linspace(0.0, 10.0, num_r)
        p_grid = np.logspace(0, 2, num_E)  # Dummy momentum grid
        t_grid = np.linspace(0, 0.1, 100)

        # D(E) = D_0 * (p / p_0), i.e., diffusion is faster for higher "momentum"
        D_values = (p_grid / p_grid[0])[:, np.newaxis] * np.ones((num_E, num_r))

        # Initial condition: Gaussian in radius, same for all energies
        f_initial = np.exp(-((r_grid - 5.0) ** 2) / (2 * 0.5**2))
        f_initial_2d = np.tile(f_initial, (num_E, 1))

        grid_params = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params = {"D_values": D_values, "f_end": 0.0}

        f_final_2d = run_diffusion_test(grid_params, solver_params, f_initial_2d)

        # Calculate the standard deviation (width) of the profile for low and high energy
        def get_width(dist, r_coords):
            mean = np.sum(dist * r_coords) / np.sum(dist)
            variance = np.sum(dist * (r_coords - mean) ** 2) / np.sum(dist)
            return np.sqrt(variance)

        width_low_E = get_width(f_final_2d[0, :], r_grid)
        width_high_E = get_width(f_final_2d[-1, :], r_grid)

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_final_2d[0, :], label=f"Low E (width={width_low_E:.2f})")
            plt.plot(
                r_grid, f_final_2d[-1, :], label=f"High E (width={width_high_E:.2f})"
            )
            plt.plot(r_grid, f_initial, "k--", label="Initial Profile")
            plt.title("2D Energy-Dependent Diffusion Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        # The profile for higher energy (and higher D) must be wider
        assert width_high_E > width_low_E

    def test_2d_decoupling_vs_1d(self, plot_results):
        """
        Verifies that a 2D run with D constant in energy matches a 1D run.
        """
        # 1. Run a standard 1D simulation
        num_r, D_const, t_final = 200, 1.0, 0.1
        r_grid = np.linspace(0.0, 1.0, num_r)
        t_grid = np.linspace(0, t_final, 200)
        f_initial_1d = (np.pi / 2) * np.sinc(r_grid)
        grid_params_1d = {"r_grid": r_grid, "t_grid": t_grid}
        solver_params_1d = {
            "D_values": np.full(num_r, D_const),
            "f_end": 0.0,
        }
        f_final_1d = run_diffusion_test(
            grid_params_1d, solver_params_1d, f_initial_1d
        ).flatten()

        # 2. Run a 2D simulation with the same parameters
        num_E = 5
        p_grid = np.linspace(1, 5, num_E)
        f_initial_2d = np.tile(f_initial_1d, (num_E, 1))
        grid_params_2d = {"r_grid": r_grid, "t_grid": t_grid, "p_grid": p_grid}
        solver_params_2d = {
            "D_values": np.full((num_E, num_r), D_const),
            "f_end": 0.0,
        }
        f_final_2d = run_diffusion_test(grid_params_2d, solver_params_2d, f_initial_2d)

        # 3. Compare the result of the 1D run with one slice of the 2D run
        f_slice_from_2d = f_final_2d[num_E // 2, :]

        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.plot(r_grid, f_final_1d, "b-", label="1D Run Result", lw=4, alpha=0.7)
            plt.plot(
                r_grid,
                f_slice_from_2d,
                "r--",
                label="Slice from 2D Run",
                lw=2,
            )
            plt.title("2D Decoupling vs 1D Test")
            plt.legend()
            plt.grid(True)
            plt.show()

        # The results should be identical
        assert np.allclose(f_final_1d, f_slice_from_2d, atol=1e-7)


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It calls pytest and passes the --plot flag to enable plotting.
    print("Running tests with plotting enabled...")
    pytest.main([__file__, "--plot"])
