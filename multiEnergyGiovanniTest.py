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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Solver import Solver
from giovanni_profiles import create_giovanni_setup, get_theoretical_profile


def run_simulation(diffusion_model, E_k, num_points=2000, num_timesteps=8000):
    """
    Run a simulation for a specific diffusion model and particle energy.

    Parameters:
    -----------
    diffusion_model : str
        Diffusion model ('kolmogorov', 'kraichnan', or 'bohm')
    E_k : astropy Quantity
        Particle kinetic energy
    num_points : int
        Number of spatial grid points
    num_timesteps : int
        Number of time steps

    Returns:
    --------
    dict
        Results dictionary containing grid, solver, and parameters
    """
    # Create Giovanni model setup
    setup = create_giovanni_setup(
        r_0=0.0 * u.pc,
        r_end=300.0 * u.pc,
        num_points=num_points,
        L_wind=1e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=const.m_p / u.cm**3,
        t_b=1 * u.Myr,
        eta_B=0.1,
        eta_inj=0.1,
        E_k=E_k,
        diffusion_model=diffusion_model,
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
    }

    return results


def simulate_and_visualize(save_data=True):
    """
    Main function to run simulations and create visualization.

    Parameters:
    -----------
    save_data : bool, optional
        Whether to save the simulation data for future replotting
    """
    # Define diffusion models with their properties
    diffusion_models = {
        "kolmogorov": {
            "name": "Kolmogorov",
            "color": "red",
            "row": 0,
            "extra_num_timesteps": 0,
        },
        "kraichnan": {
            "name": "Kraichnan",
            "color": "green",
            "row": 1,
            "extra_num_timesteps": 4000,
        },
        "bohm": {
            "name": "Bohm",
            "color": "blue",
            "row": 2,
            "extra_num_timesteps": 10000,
        },
    }

    # Define particle energies with their properties
    energies = [1, 100, 1000, 100000]  # GeV

    # Create figure for the results
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    # Sample points for plotting (we don't want to store every timestep)
    sample_count = 10  # Number of curves to plot for time evolution

    # Dense grid for theoretical profile
    r_theo = np.linspace(0, 300, 5000)

    # Dictionary to store all simulation data
    if save_data:
        all_data = {
            "energies": energies,
            "diffusion_models": list(diffusion_models.keys()),
            "simulations": {},
        }

    # Loop through diffusion models and energies
    for diff_model_name, diff_props in diffusion_models.items():
        for col_idx, energy in enumerate(energies):
            print(f"Processing {diff_model_name} with E = {energy} GeV")

            # Get subplot
            ax = fig.add_subplot(gs[diff_props["row"], col_idx])

            # Run simulation
            E_k = energy * u.GeV
            results = run_simulation(
                diffusion_model=diff_model_name,
                E_k=E_k,
                num_points=2000,
                num_timesteps=int(
                    8000 + 6000 * np.log10(energy) + diff_props["extra_num_timesteps"]
                ),
            )

            # Extract key parameters
            solver = results["solver"]
            setup = results["setup"]
            r = results["r"]
            t_grid = results["t_grid"]
            R_TS = results["R_TS"]
            R_b = results["R_b"]
            masks = results["masks"]

            # Sample timesteps to visualize
            num_timesteps = len(t_grid) - 1
            sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
            sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))

            # Store results for each sampled timestep
            stored_curves = []
            stored_times = []

            # Initial state
            f_current = solver.state.f.copy()
            f_slice = f_current[0]  # Extract first slice (2D -> 1D)

            # Find index closest to TS for normalization
            r_bubble = r >= R_TS.to("pc").value
            ts_idx = np.where(r_bubble)[0][0] + 5  # Slightly past TS

            # Save initial curve
            stored_curves.append(f_slice)
            stored_times.append(t_grid[0])

            # Run simulation and store intermediate results
            current_step = 0
            for next_step in sample_indices[1:]:
                steps_to_advance = int(next_step - current_step)
                if steps_to_advance > 0:
                    solver.step(steps_to_advance)
                    current_step = next_step

                # Store state
                f_current = solver.state.f.copy()
                f_slice = f_current[0]
                stored_curves.append(f_slice)
                stored_times.append(t_grid[current_step])

            # Normalize all curves by TS value in final state (if non-zero)
            f_final = stored_curves[-1]
            ts_level = f_final[ts_idx] if f_final[ts_idx] > 0 else 1.0
            normalized_curves = [curve / ts_level for curve in stored_curves]

            # Calculate theoretical steady-state profile
            f_theoretical = get_theoretical_profile(
                r,
                masks,
                setup["v_w"],
                setup["D_values"],
                R_TS,
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )

            # Save simulation data
            if save_data:
                # Create a unique key for this simulation
                sim_key = f"{diff_model_name}_{energy}"

                # Save only the numerical data (not objects like the solver)
                sim_data = {
                    "r": r,
                    "t_grid": t_grid,
                    "R_TS": R_TS.to("pc").value,
                    "R_b": R_b.to("pc").value,
                    "stored_times": stored_times,
                    "stored_curves": stored_curves,
                    "normalized_curves": normalized_curves,
                    "ts_level": ts_level,
                    "theoretical_profile": f_theoretical,
                    "D_values": setup["D_values"].to("pc**2/Myr").value,
                    "v_field": setup["v_field"],
                    "Q": setup["Q"],
                    "energy": energy,
                    "diffusion_model": diff_model_name,
                }

                all_data["simulations"][sim_key] = sim_data

            # Plot time evolution with varying transparency
            alphas = np.linspace(0.1, 0.8, len(normalized_curves))
            for i, (curve, t_val, alpha) in enumerate(
                zip(normalized_curves, stored_times, alphas)
            ):
                if i == 0 or i == len(normalized_curves) - 1:
                    label = (
                        f"t = {t_val:.1f} Myr"
                        if i == len(normalized_curves) - 1
                        else "t = 0"
                    )
                    ls = "-" if i == len(normalized_curves) - 1 else "--"
                    ax.semilogy(
                        r,
                        curve,
                        color=diff_props["color"],
                        alpha=0.9,
                        linestyle=ls,
                        label=label,
                    )
                else:
                    ax.semilogy(
                        r,
                        curve,
                        color=diff_props["color"],
                        alpha=alpha,
                        linestyle="-",
                        linewidth=0.8,
                    )

            # Calculate theoretical steady-state profile
            # Get full theoretical profile
            f_theoretical = get_theoretical_profile(
                r,
                masks,
                setup["v_w"],
                setup["D_values"],
                R_TS,
                R_b,
                f_gal=0.0,
                f_TS=1.0,
            )

            # Plot theoretical profile
            ax.semilogy(
                r,
                f_theoretical,
                "k--",
                lw=1.5,
                label=(
                    "Steady state" if diff_props["row"] == 0 and col_idx == 0 else None
                ),
            )

            # Add vertical lines for key radii
            ax.axvline(
                R_TS.to("pc").value,
                color="gray",
                linestyle=":",
                linewidth=1,
                label="R_TS" if diff_props["row"] == 0 and col_idx == 0 else None,
            )
            ax.axvline(
                R_b.to("pc").value,
                color="gray",
                linestyle="-.",
                linewidth=1,
                label="R_b" if diff_props["row"] == 0 and col_idx == 0 else None,
            )

            # Set plot limits and labels
            ax.set_xlim(0, 250)
            ax.set_ylim(1e-4, 2)

            # Add labels only to the left and bottom subplots
            if col_idx == 0:
                ax.set_ylabel(f"{diff_props['name']}\nCR Density $f/f_{{TS}}$")
            if diff_props["row"] == 2:
                ax.set_xlabel("Radius (pc)")

            # Add energy to top subplots
            if diff_props["row"] == 0:
                ax.set_title(f"E = {energy} GeV")

            # Add legend only to the first subplot
            if diff_props["row"] == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=9)

            # Add grid
            ax.grid(True, alpha=0.3)

    # Save all simulation data
    if save_data:
        data_filename = "multiEnergy_simulation_data2.npz"

        # Convert to a format suitable for numpy saving
        np_data = {
            "energies": np.array(energies),
            "diffusion_models": np.array(list(diffusion_models.keys())),
        }

        # Add each simulation's data
        for sim_key, sim_data in all_data["simulations"].items():
            for data_key, data_value in sim_data.items():
                np_data[f"{sim_key}_{data_key}"] = np.array(data_value)

        # Save the data
        np.savez_compressed(data_filename, **np_data)
        print(f"Saved simulation data to {data_filename}")

    # Add overall title
    plt.suptitle(
        "Cosmic Ray Transport in Wind Bubble: Time Evolution vs. Steady State\n"
        "Different Diffusion Models and Particle Energies",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)

    # Save the figure
    plt.savefig("multiEnergy_diffusion_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to 'multiEnergy_diffusion_comparison.png'")

    plt.show()


def replot_from_saved_data(data_file="multiEnergy_simulation_data.npz"):
    """
    Replot the figure using previously saved simulation data.

    Parameters:
    -----------
    data_file : str
        Path to the saved simulation data file
    """
    # Load saved data
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)

    # Extract basic parameters
    energies = data["energies"]
    diffusion_models_list = data["diffusion_models"]

    # Define diffusion models with their properties
    diffusion_models = {
        "kolmogorov": {"name": "Kolmogorov", "color": "red", "row": 0},
        "kraichnan": {"name": "Kraichnan", "color": "green", "row": 1},
        "bohm": {"name": "Bohm", "color": "blue", "row": 2},
    }

    # Create figure for the results
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 4, figure=fig)

    # Loop through diffusion models and energies
    for diff_model_name, diff_props in diffusion_models.items():
        for col_idx, energy in enumerate(energies):
            print(f"Plotting {diff_model_name} with E = {energy} GeV")

            # Get subplot
            ax = fig.add_subplot(gs[diff_props["row"], col_idx])

            # Create key for this simulation
            sim_key = f"{diff_model_name}_{energy}"

            # Extract data for this simulation
            r = data[f"{sim_key}_r"]
            R_TS = data[f"{sim_key}_R_TS"]
            R_b = data[f"{sim_key}_R_b"]
            normalized_curves = []
            stored_times = data[f"{sim_key}_stored_times"]

            # Reconstruct normalized curves
            for i in range(len(stored_times)):
                curve_key = f"{sim_key}_normalized_curves"
                if curve_key in data:
                    normalized_curves = data[curve_key]
                else:
                    # Reconstruct from stored curves and ts_level
                    stored_curves = data[f"{sim_key}_stored_curves"]
                    ts_level = data[f"{sim_key}_ts_level"]
                    normalized_curves = [curve / ts_level for curve in stored_curves]

            # Theoretical profile
            f_theoretical = data[f"{sim_key}_theoretical_profile"]

            # Plot time evolution with varying transparency
            alphas = np.linspace(0.1, 0.8, len(normalized_curves))
            for i, (curve, t_val, alpha) in enumerate(
                zip(normalized_curves, stored_times, alphas)
            ):
                if i == 0 or i == len(normalized_curves) - 1:
                    label = (
                        f"t = {t_val:.1f} Myr"
                        if i == len(normalized_curves) - 1
                        else "t = 0"
                    )
                    ls = "-" if i == len(normalized_curves) - 1 else "--"
                    ax.semilogy(
                        r,
                        curve,
                        color=diff_props["color"],
                        alpha=0.9,
                        linestyle=ls,
                        label=label,
                    )
                else:
                    ax.semilogy(
                        r,
                        curve,
                        color=diff_props["color"],
                        alpha=alpha,
                        linestyle="-",
                        linewidth=0.8,
                    )

            # Plot theoretical profile
            ax.semilogy(
                r,
                f_theoretical,
                "k--",
                lw=1.5,
                label=(
                    "Steady state" if diff_props["row"] == 0 and col_idx == 0 else None
                ),
            )

            # Add vertical lines for key radii
            ax.axvline(
                R_TS,
                color="gray",
                linestyle=":",
                linewidth=1,
                label="R_TS" if diff_props["row"] == 0 and col_idx == 0 else None,
            )
            ax.axvline(
                R_b,
                color="gray",
                linestyle="-.",
                linewidth=1,
                label="R_b" if diff_props["row"] == 0 and col_idx == 0 else None,
            )

            # Set plot limits and labels
            ax.set_xlim(0, 150)
            ax.set_ylim(1e-4, 2)

            # Add labels only to the left and bottom subplots
            if col_idx == 0:
                ax.set_ylabel(f"{diff_props['name']}\nCR Density $f/f_{{TS}}$")
            if diff_props["row"] == 2:
                ax.set_xlabel("Radius (pc)")

            # Add energy to top subplots
            if diff_props["row"] == 0:
                ax.set_title(f"E = {energy} GeV")

            # Add legend only to the first subplot
            if diff_props["row"] == 0 and col_idx == 0:
                ax.legend(loc="upper right", fontsize=9)

            # Add grid
            ax.grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle(
        "Cosmic Ray Transport in Wind Bubble: Time Evolution vs. Steady State\n"
        "Different Diffusion Models and Particle Energies (Replotted from saved data)",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.3)

    # Save the figure
    plt.savefig("multiEnergy_diffusion_replot.png", dpi=150, bbox_inches="tight")
    print("Saved replotted visualization to 'multiEnergy_diffusion_replot.png'")

    plt.show()


if __name__ == "__main__":
    # Run simulations and save data
    simulate_and_visualize(save_data=True)

    # Or replot from previously saved data
    # replot_from_saved_data()
