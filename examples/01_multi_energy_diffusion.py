"""
Example 01: Multi-Energy Diffusion Comparison
=============================================

This example demonstrates how to set up and run a cosmic ray transport simulation
using SAETASS within a stellar wind bubble. It compares the spatial distribution
of particles at different energy levels (1, 100, 1000, 100000 GeV) for three different
diffusion models: Kolmogorov, Kraichnan, and Bohm.

The script visualizes the time evolution of the cosmic ray spatial distribution,
comparing it against theoretical steady-state profiles.
"""

import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import astropy.units as u
import astropy.constants as const
from matplotlib.gridspec import GridSpec

# 0. Import SAETASS modules
from saetass import State, Grid, Solver
from saetass.utils.bubble_profiles import BubbleProfileCalculator

# 0.1 Import end

# ---------------------------------------------------------
# Configuration Setup
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "legend.fontsize": 14,
        "legend.title_fontsize": 16,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    }
)


def get_truncated_cmap(cmap, min_val=0.25, max_val=1.0):
    """Truncate a colormap to avoid its lightest colors."""
    colors = cmap(np.linspace(min_val, max_val, 256))
    return mcolors.LinearSegmentedColormap.from_list(f"trunc_{cmap.name}", colors)


# Define diffusion models with their properties
diffusion_models = {
    "kolmogorov": {
        "name": "Kolmogorov",
        "cmap": get_truncated_cmap(plt.cm.Reds),
        "row": 0,
        "extra_num_timesteps": 0,
    },
    "kraichnan": {
        "name": "Kraichnan",
        "cmap": get_truncated_cmap(plt.cm.Greens),
        "row": 1,
        "extra_num_timesteps": 4000,
    },
    "bohm": {
        "name": "Bohm",
        "cmap": get_truncated_cmap(plt.cm.Blues),
        "row": 2,
        "extra_num_timesteps": 100000,
    },
}

# Define particle energies
energies = [1, 100, 1000, 100000]  # GeV

# ---------------------------------------------------------
# Helper Plotting Functions
# ---------------------------------------------------------


def setup_figure():
    """Initializes the main Matplotlib figure and GridSpec layout."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 0.02], figure=fig)
    return fig, gs


def plot_simulation_step(
    ax,
    r,
    R_TS,
    R_b,
    energy,
    diff_props,
    col_idx,
    stored_curves,
    stored_times,
    f_theoretical,
):
    """Renders the diffusion curves for a single (energy, diffusion_model) parameter grid point."""

    # Negative vmin to avoid pure white at t=0
    cmap = diff_props["cmap"]
    norm = plt.Normalize(vmin=0.0, vmax=1.2)

    # Plot time evolution (skipping t=0)
    for i, (curve, t_val) in enumerate(zip(stored_curves[1:], stored_times[1:])):
        color = cmap(norm(t_val))

        is_last = i == len(stored_curves[1:]) - 1
        label = f"$t = {t_val:.1f}$ Myr" if is_last else None
        ls = "-" if is_last else "--"
        lw = 2.0 if is_last else 1.0

        ax.semilogy(
            r,
            curve,
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=label if label else "_nolegend_",
        )

    # Plot theoretical profile
    ax.semilogy(
        r,
        f_theoretical,
        "k--",
        lw=1.5,
        label=("Steady state" if col_idx == 0 else None),
    )

    # Add vertical lines for key radii
    ax.axvline(
        R_TS,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=r"$R_\mathrm{TS}$" if col_idx == 0 else None,
    )
    ax.axvline(
        R_b,
        color="gray",
        linestyle="-.",
        linewidth=1.5,
        label=r"$R_\mathrm{B}$" if col_idx == 0 else None,
    )

    # Set plot limits and labels
    ax.set_xlim(0, 250)
    ax.set_ylim(1e-4, 2)

    # Add labels only to the left and bottom subplots
    if col_idx == 0:
        ax.set_ylabel(r"Norm. dist.: $f(r,t)/f_\mathrm{TS}$")
    else:
        ax.set_yticklabels([])  # Remove yticks from internal subplots

    if diff_props["row"] == 2:
        ax.set_xlabel(r"Radial coordinate: $r$ (pc)")

    # Add energy to top subplots
    if diff_props["row"] == 0:
        if energy >= 1000:
            ax.set_title(f"$E = {math.floor(energy/1000):.0f}$ TeV")
        else:
            ax.set_title(f"$E = {energy}$ GeV")

    # Add legend only to the first subplot of each row
    if col_idx == 0:
        ax.legend(loc="upper right", title=diff_props["name"])

    # Add grid
    ax.grid(True, alpha=0.3)


def finalize_and_save_figure(fig, gs, diffusion_models):
    """Adds the layout colorbars and exports the finalized figure."""
    for diff_model_name, diff_props in diffusion_models.items():
        cbar_ax = fig.add_subplot(gs[diff_props["row"], 4])
        sm = plt.cm.ScalarMappable(
            cmap=diff_props["cmap"], norm=plt.Normalize(vmin=0.0, vmax=1.2)
        )
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(r"Time: $t$ (Myr)", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multi_energy_diffusion_comparison.pdf")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to '{output_path}'")


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------

if __name__ == "__main__":

    fig, gs = setup_figure()
    sample_count = 12

    for diff_model_name, diff_props in diffusion_models.items():
        for col_idx, energy in enumerate(energies):
            print(f"Processing {diff_model_name} with E = {energy} GeV")
            ax = fig.add_subplot(gs[diff_props["row"], col_idx])

            # 1. Generate Physical Setup Parameters
            # Calculate number of timesteps
            E_k = energy * u.GeV
            num_timesteps = (
                int(6000 + 6000 * np.log10(energy) + diff_props["extra_num_timesteps"])
                if diff_model_name != "bohm"
                else 120000
            )  # Special care needs to be taken in num_timesteps choice due to stability contraints of this specific system

            # We use the new BubbleProfileCalculator from saetass.utils.bubble_profiles
            t_b = 1 * u.Myr
            calculator = BubbleProfileCalculator(
                r_grid=np.linspace(0.0, 300.0, 800) * u.pc,
                model="Morlino21",
                L_wind=1e38 * u.erg / u.s,
                M_dot=1e-4 * const.M_sun / u.yr,
                rho_0=const.m_p / u.cm**3,
                t_b=t_b,
            )

            setup = calculator.get_all_profiles(
                E_k=E_k, eta_B=0.1, eta_inj=0.1, diffusion_model=diff_model_name
            )

            r = setup["r_grid"].to("pc").value
            R_TS = setup["R_TS"]
            R_b = setup["R_b"]
            t_end = 1.2 * t_b  # 1 Myr from t_b

            # 2. Create Solver arguments
            t_grid = np.linspace(0, t_end, num_timesteps)
            f_values = np.zeros(len(r))  # Initially no distribution

            # We create an instance of Grid class
            grid = Grid(r_centers=r, t_grid=t_grid, p_centers=None)

            # We generate the parameters for the subsolvers
            op_params = {
                "advection": {
                    "v_centers": setup["v_field"],
                    "order": 2,
                    "limiter": "minmod",
                    "cfl": 0.8,
                    "inflow_value_U": 0.0,
                },
                "diffusion": {
                    "D_values": setup["D_values"].to("pc**2/Myr").value,
                    "f_end": 0.0,
                },
                "source": {"source": setup["Q"]},
            }

            # 3. Instantiate Solver
            # We then instantiate Solver class
            solver = Solver(
                grid=grid,
                state=State(f_values),
                problem_type="advection-diffusion-source",
                operator_params=op_params,
                substeps={"advection": 1, "diffusion": 1, "source": 1},
            )

            # 4. Calculate the time sampling checkpoints to plot curves over time
            num_timesteps = len(t_grid) - 1
            sample_indices = np.unique(
                np.append(
                    np.linspace(0, num_timesteps, sample_count, dtype=int),
                    [0, num_timesteps],
                )
            )

            stored_curves = [solver.state.f.copy()[0]]
            stored_times = [t_grid[0]]

            # 5. Simulation loop
            current_step = 0
            for next_step in sample_indices[1:]:
                steps_to_advance = int(next_step - current_step)
                if steps_to_advance > 0:
                    solver.step(steps_to_advance)  # SIMULATION CORE
                    current_step = next_step

                stored_curves.append(solver.state.f.copy()[0])
                stored_times.append(t_grid[current_step])

            # 6. Result Normalization
            ts_idx = np.where(r >= R_TS.to("pc").value)[0][0] + 5
            ts_level = (
                stored_curves[-1][ts_idx] if stored_curves[-1][ts_idx] > 0 else 1.0
            )
            normalized_curves = [curve / ts_level for curve in stored_curves]

            # 7. Generate Baseline Theoretical Configuration
            f_theoretical = calculator.compute_analytical_CR_profile(
                D_values=setup["D_values"],
                f_gal=0.0,
                f_TS=1.0,
            )

            # 8. Render local plot data to the axis GridSpec
            plot_simulation_step(
                ax,
                r,
                R_TS.to("pc").value,
                R_b.to("pc").value,
                energy,
                diff_props,
                col_idx,
                normalized_curves,
                stored_times,
                f_theoretical,
            )

    # 9. Render figure
    finalize_and_save_figure(fig, gs, diffusion_models)
