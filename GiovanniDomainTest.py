import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
import astropy.units as u
import astropy.constants as const
from matplotlib.gridspec import GridSpec
from State import State
from Grid import Grid
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("GiovanniDomainTest")

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Solver import Solver
from giovanni_profiles import create_giovanni_setup, get_theoretical_profile


def obtain_parameters(
    r_end_factor, E_k=10 * u.GeV, base_points=2000, num_timesteps=8000
):
    """
    Run a simulation with a specific domain size, adjusting grid points to maintain density.

    Parameters:
    -----------
    r_end_factor : float
        Factor by which to multiply R_b to determine r_end
    E_k : astropy Quantity
        Particle kinetic energy
    base_points : int
        Base number of spatial grid points (for reference domain)
    num_timesteps : int
        Number of time steps

    Returns:
    --------
    dict
        Results dictionary containing grid, solver, and parameters
    """
    # First create a reference model to get R_TS and R_b
    reference_model = create_giovanni_setup(
        r_0=0.0 * u.pc,
        r_end=300.0 * u.pc,  # Temporary large value
        num_points=100,  # Coarse grid for quick reference
        L_wind=1e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=const.m_p / u.cm**3,
        t_b=1 * u.Myr,
        eta_B=0.1,
        eta_inj=0.1,
        E_k=E_k,
        diffusion_model="kolmogorov",
    )

    # Get reference values
    R_TS = reference_model["R_TS"]
    R_b = reference_model["R_b"]

    # Calculate r_end based on factor
    r_end = r_end_factor * R_b

    # Adjust number of points to maintain point density
    # Using a reference of base_points for a domain of R_b
    num_points = int(base_points * r_end_factor)

    print(
        f"Domain: r_end = {r_end.to('pc'):.1f} ({r_end_factor:.1f}×R_b), Points: {num_points}"
    )

    # Create the actual simulation model
    setup = create_giovanni_setup(
        r_0=0.0 * u.pc,
        r_end=r_end,
        num_points=num_points,
        L_wind=1e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=const.m_p / u.cm**3,
        t_b=1 * u.Myr,
        eta_B=0.1,
        eta_inj=0.1,
        E_k=E_k,
        diffusion_model="kolmogorov",
    )

    # Extract parameters
    r = setup["r"]
    t_b = setup["t_b"]
    R_TS = setup["R_TS"]
    R_b = setup["R_b"]
    v_field = setup["v_field"]
    D_values = setup["D_values"]
    Q = setup["Q"]
    masks = setup["masks"]

    # Create temporal grid (slightly longer than bubble age)
    t_end = 1.2 * t_b.to("Myr").value
    t_grid = np.linspace(0, t_end, num_timesteps)

    # Initial profile: zero everywhere
    f_values = np.zeros(len(r))

    # Prepare solver parameters
    advectionFV_params = {
        "v_centers": v_field,
        "order": 2,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_W": 0.0,
    }

    diffusion_params = {
        "D_values": D_values.to("pc**2/Myr").value,
        "Q_values": Q,
        "f_end": 0.0,  # No galactic background
    }

    source_params = {"Q_values": Q}

    op_params = {
        "advectionFV": advectionFV_params,
        "diffusionFV": diffusion_params,
        "source": source_params,
    }

    # Create grid and solver
    grid = Grid(r_centers=r, t_grid=t_grid, p_centers=None)

    solver = Solver(
        grid=grid,
        state=State(f_values),
        problem_type="advectionFV-diffusionFV",
        operator_params=op_params,
        substeps={"advectionFV": 1, "diffusionFV": 1},
    )

    # Store simulation configuration
    results = {
        "setup": setup,
        "grid": grid,
        "solver": solver,
        "r": r,
        "t_grid": t_grid,
        "R_TS": R_TS,
        "R_b": R_b,
        "masks": masks,
        "r_end_factor": r_end_factor,
    }

    return results


def save_results(results_list, filename="domain_test_results.pkl"):
    """
    Save simulation results to a file.

    Parameters:
    -----------
    results_list : list
        List of simulation results
    filename : str
        File to save results to
    """
    # Extract essential data from results to save
    # We can't pickle the full solver, so we extract just what we need
    save_data = []

    for results in results_list:

        # Create a dict with just the necessary data
        save_item = {
            "r": results["r"],
            "R_TS": results["R_TS"],
            "R_b": results["R_b"],
            "r_end_factor": results["r_end_factor"],
            "f_final": results["f_final"],
            "masks": results["masks"],
            "v_w": results["v_w"],
            "D_values": results["D_values"],
        }
        save_data.append(save_item)

    # Save to file
    with open(filename, "wb") as f:
        pickle.dump(save_data, f)

    logger.info(f"Saved results to {filename}")


def load_results(filename="domain_test_results.pkl"):
    """
    Load simulation results from file.

    Parameters:
    -----------
    filename : str
        File to load results from

    Returns:
    --------
    list
        List of simulation result dictionaries
    """
    if not os.path.exists(filename):
        logger.info(f"Results file {filename} not found")
        return None

    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None


def recalculate_single_case(
    r_end_factor, num_timesteps=8000, base_points=1000, E_k=10 * u.GeV
):
    """
    Recalculate a single case and update the saved data.

    Parameters:
    -----------
    r_end_factor : float
        Factor by which to multiply R_b to determine r_end
    num_timesteps : int, optional
        Number of time steps, if None will be estimated based on domain size
    base_points : int, optional
        Base number of spatial grid points
    E_k : astropy Quantity, optional
        Particle kinetic energy

    Returns:
    --------
    dict
        Updated simulation results
    """

    logger.info(
        f"Recalculating for domain factor {r_end_factor:.1f} with {num_timesteps} time steps"
    )

    # Create and run the simulation
    results = obtain_parameters(
        r_end_factor=r_end_factor,
        E_k=E_k,
        num_timesteps=num_timesteps,
        base_points=base_points,
    )

    # Run the simulation
    solver = results["solver"]
    solver.step(solver.grid.t_grid.size - 1)  # Run to final time
    logger.info(f"Completed simulation for domain factor {r_end_factor:.1f}")

    # Load existing results
    results_file = "domain_test_results.pkl"
    all_results = load_results(results_file)

    if all_results is None:
        # No existing results, create a new list
        all_results = [results]
    else:
        # Update existing results
        found = False
        for i, res in enumerate(all_results):
            if abs(res["r_end_factor"] - r_end_factor) < 1e-6:
                # Found existing result with this factor, replace it
                logger.info(f"Replacing existing result for factor {r_end_factor:.1f}")

                # Extract the final state from solver
                f_final = results["solver"].state.f[0].copy()

                # Create a dict with just the necessary data
                save_item = {
                    "r": results["r"],
                    "R_TS": results["R_TS"],
                    "R_b": results["R_b"],
                    "r_end_factor": results["r_end_factor"],
                    "f_final": f_final,
                    "masks": results["masks"],
                    "v_w": results["setup"]["v_w"],
                    "D_values": results["setup"]["D_values"],
                }

                all_results[i] = save_item
                found = True
                break

        if not found:
            # No existing result with this factor, add a new one
            logger.info(f"Adding new result for factor {r_end_factor:.1f}")
            all_results.append(results)

    # Save the updated results
    save_results(all_results, results_file)

    return results


def boundary_test(use_saved=True, save=True, recalculate_factor=None):
    """
    Test the effect of domain size on the solution accuracy.

    Parameters:
    -----------
    use_saved : bool
        Whether to try loading saved results first
    save : bool
        Whether to save results after computation
    recalculate_factor : float, optional
        If provided, recalculate only the case with this r_end_factor
    """
    results_file = "domain_test_results.pkl"
    all_results = None

    # Handle single case recalculation
    if recalculate_factor is not None:
        # Recalculate just one case and return
        logger.info(f"Recalculating only the case with factor {recalculate_factor}")
        recalculate_single_case(
            recalculate_factor, num_timesteps=40000, base_points=1000
        )
        # Reload all results including the updated one
        all_results = load_results(results_file)
        if not all_results:
            logger.error("Failed to load results after recalculation")
            return
    else:
        # Standard flow - try to load saved results if requested
        if use_saved:
            all_results = load_results(results_file)
            if all_results:
                logger.info(f"Loaded saved results from {results_file}")

        # Compute if no saved results or loading failed
        if all_results is None:
            # Set up simulation parameters
            E_k = 10 * u.GeV  # Fixed energy

            # Define domain sizes to test (multiples of R_b)
            r_end_factors = [1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
            time_steps = [40000, 40000, 40000, 40000, 40000, 40000, 40000, 60000]

            # Keep all simulation results
            all_results = []

            # Run simulations for all domain sizes
            for i, factor in enumerate(r_end_factors):
                logger.info(f"Starting simulation for domain factor {factor:.1f}")
                results = obtain_parameters(
                    r_end_factor=factor,
                    E_k=E_k,
                    num_timesteps=time_steps[i],
                    base_points=1000,
                )

                # Run the simulation
                solver = results["solver"]
                solver.step(solver.grid.t_grid.size - 1)  # Run to final time
                logger.info(f"Completed simulation for domain factor {factor:.1f}")

                # Store results
                all_results.append(results)

            # Save results if requested
            if save:
                save_results(all_results, results_file)

    # Create visualization
    fig1 = plt.figure(figsize=(18, 12))
    gs1 = GridSpec(2, 4, figure=fig1)

    # Determine the smallest domain (for consistent x-axis limits)
    min_factor = min(result["r_end_factor"] for result in all_results)
    min_result = next(r for r in all_results if r["r_end_factor"] == min_factor)
    min_domain = min_result["r"][-1] * 1.05  # Add 5% margin

    # Plot all solutions
    for i, results in enumerate(all_results):
        # Get subplot (arrange in 2x4 grid)
        row = i // 4
        col = i % 4
        ax = fig1.add_subplot(gs1[row, col])

        # Extract data
        r = results["r"]
        R_TS = results["R_TS"]
        R_b = results["R_b"]
        r_end_factor = results["r_end_factor"]

        # Get solver state (if from saved data or full results)
        if "f_final" in results:
            f_final = results["f_final"]
        else:
            f_final = results["solver"].state.f[0]

        # Find index closest to TS for normalization
        r_bubble = r >= R_TS.to("pc").value
        ts_idx = np.where(r_bubble)[0][0] + 5  # Slightly past TS

        # Normalize by TS value
        ts_level = f_final[ts_idx] if f_final[ts_idx] > 0 else 1.0
        f_normalized = f_final / ts_level

        # Calculate theoretical profile
        if "masks" in results and "v_w" in results and "D_values" in results:
            # From saved data
            f_theoretical = get_theoretical_profile(
                r,
                results["masks"],
                results["v_w"],
                results["D_values"],
                R_TS,
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )
        else:
            # From full results
            f_theoretical = get_theoretical_profile(
                r,
                results["masks"],
                results["setup"]["v_w"],
                results["setup"]["D_values"],
                R_TS,
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )

        # Plot numerical and theoretical solutions
        ax.semilogy(r, f_normalized, "b-", lw=1.5, label="Numerical")
        ax.semilogy(r, f_theoretical, "r--", lw=1.5, label="Theoretical")

        # Add vertical lines for key radii
        ax.axvline(
            R_TS.to("pc").value, color="gray", linestyle=":", linewidth=1, label="R_TS"
        )
        ax.axvline(
            R_b.to("pc").value, color="gray", linestyle="-.", linewidth=1, label="R_b"
        )

        # Mark domain boundary
        ax.axvline(
            r[-1],
            color="black",
            linestyle="-",
            alpha=0.3,
            linewidth=1,
            label="Domain boundary",
        )

        # Set plot limits and labels
        # Use fixed x-axis limits for all plots based on the smallest domain
        ax.set_xlim(0, min_domain)
        ax.set_ylim(1e-4, 2)
        ax.set_title(f"Domain = {r_end_factor:.1f}×R_b")

        if row == 1:
            ax.set_xlabel("Radius (pc)")
        if col == 0:
            ax.set_ylabel("CR Density f/f_TS")

        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc="upper right", fontsize=9)

        # Add grid
        ax.grid(True, alpha=0.3)

    # Add overall title to first figure
    title = f"Effect of Domain Size on Cosmic Ray Transport Solution\nKolmogorov Diffusion, E = 10 GeV (Fixed x-axis view)"
    if recalculate_factor:
        title += f"\nRecalculated factor: {recalculate_factor:.1f}"
    fig1.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save first figure
    plt.savefig("giovanni_domain_test_solutions.png", dpi=150, bbox_inches="tight")

    # Create a second figure to show error convergence
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    # Calculate errors and plot convergence
    r_end_factors = []
    errors = []
    rel_errors = []

    for results in all_results:
        r = results["r"]
        R_b = results["R_b"]
        r_end_factor = results["r_end_factor"]
        r_end_factors.append(r_end_factor)

        # Get the normalized solution
        if "f_final" in results:
            f_final = results["f_final"]
        else:
            f_final = results["solver"].state.f[0]

        r_bubble = (r < R_b.to("pc").value) & (r >= results["R_TS"].to("pc").value)
        ts_idx = np.where(r_bubble)[0][0] + 5
        ts_level = f_final[ts_idx] if f_final[ts_idx] > 0 else 1.0
        f_normalized = f_final / ts_level

        # Get theoretical solution
        if "masks" in results and "v_w" in results and "D_values" in results:
            # From saved data
            f_theoretical = get_theoretical_profile(
                r,
                results["masks"],
                results["v_w"],
                results["D_values"],
                results["R_TS"],
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )
        else:
            # From full results
            f_theoretical = get_theoretical_profile(
                r,
                results["masks"],
                results["setup"]["v_w"],
                results["setup"]["D_values"],
                results["R_TS"],
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )

        # Find the index right after R_b
        rb_idx = np.abs(r - R_b.to("pc").value).argmin()
        rb_plus_idx = min(rb_idx, len(r) - 1)

        # Calculate error at that point
        error = abs(f_normalized[rb_plus_idx] - f_theoretical[rb_plus_idx])
        rel_error = error / f_theoretical[rb_plus_idx]

        errors.append(error)
        rel_errors.append(rel_error)

    # Sort by r_end_factor for better plotting
    sort_idx = np.argsort(r_end_factors)
    r_end_factors = [r_end_factors[i] for i in sort_idx]
    errors = [errors[i] for i in sort_idx]
    rel_errors = [rel_errors[i] for i in sort_idx]

    # Plot absolute error
    ax2.semilogy(r_end_factors, errors, "bo-", lw=2, ms=8, label="Absolute Error")

    # Plot relative error on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.semilogy(
        r_end_factors, rel_errors, "ro-", lw=2, ms=8, label="Relative Error"
    )

    # Set labels and title
    ax2.set_xlabel("Domain Size (×R_b)")
    ax2.set_ylabel("Absolute Error at R_b+", color="blue")
    ax2_twin.set_ylabel("Relative Error at R_b+", color="red")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2_twin.tick_params(axis="y", labelcolor="red")

    # Add legend with both lines
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    title = f"Solution Error vs. Domain Size\nKolmogorov Diffusion, E = 10 GeV"
    if recalculate_factor:
        title += f"\nRecalculated factor: {recalculate_factor:.1f}"
    ax2.set_title(title)
    ax2.grid(True, alpha=0.3)

    # Save second figure
    plt.tight_layout()
    plt.savefig("giovanni_domain_error_convergence.png", dpi=150, bbox_inches="tight")

    print("Analysis completed. Figures saved.")
    plt.show()


if __name__ == "__main__":
    # Ejemplo de uso:
    # Para recalcular un caso específico con factor 2.0:
    # boundary_test(recalculate_factor=2.0)

    # Para usar los datos guardados sin recalcular nada:
    # boundary_test(use_saved=True, save=True)

    # Para recalcular todo desde cero:
    # boundary_test(use_saved=False, save=True)

    boundary_test(save=True, use_saved=True, recalculate_factor=8.0)
