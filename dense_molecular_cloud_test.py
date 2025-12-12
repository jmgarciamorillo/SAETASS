"""
Test script for diffusion through a dense molecular cloud.

This script simulates cosmic ray diffusion through a multi-zone environment:
- Core (0-12 pc): Low density (0.1 cm⁻³)
- Ultra-dense cloud (12-13 pc): Very high density (10⁵ cm⁻³)
- Big low-density cloud (13-50 pc): Medium density (10² cm⁻³)
- ISM (>50 pc): Standard density (1 cm⁻³)

Initial condition: Delta function at 12 pc (termination shock position)
Energy range: 100-300 TeV
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import logging
from matplotlib.gridspec import GridSpec

from State import State
from Grid import Grid
from Solver import Solver

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DenseCloudTest")


def create_density_profile(r_grid):
    """
    Create multi-zone density profile.

    Zones:
    - Core (0-12 pc): 0.1 cm⁻³
    - Ultra-dense cloud (12-13 pc): 10⁵ cm⁻³
    - Big low-density cloud (13-50 pc): 10² cm⁻³
    - ISM (>50 pc): 1 cm⁻³

    Parameters:
    -----------
    r_grid : array-like
        Radial coordinates (pc)

    Returns:
    --------
    np.array
        Density profile (cm⁻³)
    """
    n_gas = np.zeros_like(r_grid)

    # Core: 0-12 pc
    mask_core = r_grid < 12.0
    n_gas[mask_core] = 0.1

    # Ultra-dense cloud: 12-13 pc
    mask_ultra_dense = (r_grid >= 12.0) & (r_grid < 13.0)
    n_gas[mask_ultra_dense] = 1e5

    # Big low-density cloud: 13-50 pc
    mask_big_cloud = (r_grid >= 13.0) & (r_grid < 50.0)
    n_gas[mask_big_cloud] = 1e2

    # ISM: >50 pc
    mask_ISM = r_grid >= 50.0
    n_gas[mask_ISM] = 1.0

    return n_gas


def create_diffusion_profile(r_grid, E_grid):
    """
    Create diffusion coefficient profile with energy dependence.

    D(E) = D_0 × (E / 1 GeV)^(1/3)

    Different D_0 values for each zone:
    - Core: D_0 = 1e28 cm²/s (moderately fast diffusion)
    - Ultra-dense cloud: D_0 = 1e25 cm²/s (very slow diffusion due to high density)
    - Big low-density cloud: D_0 = 1e27 cm²/s (intermediate diffusion)
    - ISM: D_0 = 3e28 cm²/s (standard ISM diffusion)

    Parameters:
    -----------
    r_grid : array-like
        Radial coordinates (pc)
    E_grid : astropy Quantity
        Energy grid

    Returns:
    --------
    np.array
        Diffusion coefficient (pc²/Myr), shape (num_E, num_r)
    """
    num_E = len(E_grid)
    num_r = len(r_grid)

    D_values = np.zeros((num_E, num_r))

    # Energy dependence: D ∝ E^(1/3)
    E_GeV = E_grid.to("GeV").value
    energy_factor = (E_GeV / 1.0) ** (1 / 3)

    # Core: 0-12 pc
    mask_core = r_grid < 12.0
    D_0_core = 5e33 * u.cm**2 / u.s
    D_core = (D_0_core * energy_factor).to("pc**2/Myr").value
    for j in range(num_r):
        if mask_core[j]:
            D_values[:, j] = D_core

    # Ultra-dense cloud: 12-13 pc (VERY SLOW diffusion)
    mask_ultra_dense = (r_grid >= 12.0) & (r_grid < 13.0)
    D_0_ultra = 1e22 * u.cm**2 / u.s  # 1000x slower than ISM
    D_ultra = (D_0_ultra * energy_factor).to("pc**2/Myr").value
    for j in range(num_r):
        if mask_ultra_dense[j]:
            D_values[:, j] = D_ultra

    # Big low-density cloud: 13-50 pc
    mask_big_cloud = (r_grid >= 13.0) & (r_grid < 50.0)
    D_0_big = 1e27 * u.cm**2 / u.s
    D_big = (D_0_big * energy_factor).to("pc**2/Myr").value
    for j in range(num_r):
        if mask_big_cloud[j]:
            D_values[:, j] = D_big

    # ISM: >50 pc (standard diffusion)
    mask_ISM = r_grid >= 50.0
    D_0_ISM = 3e28 * u.cm**2 / u.s
    D_ISM = (D_0_ISM * energy_factor).to("pc**2/Myr").value
    for j in range(num_r):
        if mask_ISM[j]:
            D_values[:, j] = D_ISM

    return D_values


def create_initial_condition(r_grid, E_grid):
    """
    Create delta function initial condition at r = 12 pc.

    Parameters:
    -----------
    r_grid : array-like
        Radial coordinates (pc)
    E_grid : astropy Quantity
        Energy grid

    Returns:
    --------
    np.array
        Initial distribution function, shape (num_E, num_r)
    """
    num_E = len(E_grid)
    num_r = len(r_grid)

    f_init = np.zeros((num_E, num_r))

    # Find index closest to 12 pc
    r_shock = 12.0
    shock_idx = np.abs(r_grid - r_shock).argmin()

    # Create delta function (normalized by grid spacing)
    dr = np.diff(r_grid).mean()
    f_init[:, shock_idx] = 1.0 / dr  # Normalized delta function

    logger.info(f"Initial condition: Delta function at r = {r_grid[shock_idx]:.2f} pc")

    return f_init


def run_dense_cloud_test():
    """Run dense molecular cloud diffusion test with real-time plotting."""

    # Physical and grid parameters
    num_r = 500
    num_E = 200
    r_0 = 0.0 * u.pc
    r_end = 500.0 * u.pc
    E_min = 100.0 * u.TeV
    E_max = 300.0 * u.TeV

    # Log-spaced energy grid
    E_grid = (
        np.logspace(
            np.log10(E_min.to("GeV").value), np.log10(E_max.to("GeV").value), num_E
        )
        * u.GeV
    )

    # Momentum grid
    p_grid = np.sqrt((E_grid**2 + 2 * E_grid * (const.m_p * const.c**2))) / const.c
    p_grid_numeric = p_grid.to("g*pc/Myr").value

    # Spatial grid
    r_grid = np.linspace(r_0.to("pc").value, r_end.to("pc").value, num_r)

    # Create profiles
    logger.info("Creating density and diffusion profiles...")
    n_gas = create_density_profile(r_grid)
    D_values_2d = create_diffusion_profile(r_grid, E_grid)

    # Log profile information
    logger.info("\nZone definitions:")
    logger.info(f"  Core (0-12 pc): n = 0.1 cm⁻³")
    logger.info(f"  Ultra-dense cloud (12-13 pc): n = 10⁵ cm⁻³")
    logger.info(f"  Big low-density cloud (13-50 pc): n = 10² cm⁻³")
    logger.info(f"  ISM (>50 pc): n = 1 cm⁻³")
    logger.info(f"\nDiffusion coefficients at 100 TeV:")
    logger.info(f"  Core: {D_values_2d[0, 5]:.3e} pc²/Myr")
    logger.info(
        f"  Ultra-dense: {D_values_2d[0, np.abs(r_grid - 12.5).argmin()]:.3e} pc²/Myr"
    )
    logger.info(
        f"  Big cloud: {D_values_2d[0, np.abs(r_grid - 30).argmin()]:.3e} pc²/Myr"
    )
    logger.info(f"  ISM: {D_values_2d[0, -1]:.3e} pc²/Myr")

    # Create initial condition (delta function at 12 pc)
    f_init = create_initial_condition(r_grid, E_grid)
    state = State(f_init)

    # Time grid
    t_max = 0.02  # Myr
    num_timesteps = 20000
    t_grid = np.linspace(0, t_max, num_timesteps)

    # Operator parameters (ONLY diffusion)
    diffusionFV_params = {
        "D_values": D_values_2d,
        "Q_values": None,
        "f_end": 0.0,
    }

    op_params = {
        "diffusionFV": diffusionFV_params,
    }

    # Create grid and solver
    grid = Grid(
        r_centers=r_grid,
        t_grid=t_grid,
        p_centers=p_grid_numeric,
    )

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="diffusionFV",
        operator_params=op_params,
        substeps={"diffusionFV": 1},
        splitting_scheme="strang",
    )

    # Setup plotting
    energies_to_plot = [100, 150, 200, 250, 300]  # TeV
    energy_indices = [
        np.abs(E_grid.to("TeV").value - e).argmin() for e in energies_to_plot
    ]

    # Ultra-dense cloud mask (12-13 pc)
    ultra_dense_mask = (r_grid >= 12.0) & (r_grid < 13.0)

    # Create interactive figure
    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Spatial profiles (top row and middle left/center)
    axes_spatial = [
        fig.add_subplot(gs[0, 0]),  # 100 TeV
        fig.add_subplot(gs[0, 1]),  # 150 TeV
        fig.add_subplot(gs[0, 2]),  # 200 TeV
        fig.add_subplot(gs[1, 0]),  # 250 TeV
        fig.add_subplot(gs[1, 1]),  # 300 TeV
    ]

    # Environment profiles (middle right)
    ax_env = fig.add_subplot(gs[1, 2])

    # Ultra-dense cloud spectrum (bottom left)
    ax_spectrum = fig.add_subplot(gs[2, 0])

    # Ultra-dense cloud time evolution (bottom center)
    ax_time_evolution = fig.add_subplot(gs[2, 1])

    # Diffusion profile (bottom right)
    ax_diffusion = fig.add_subplot(gs[2, 2])

    # Initialize spatial profile lines
    lines = []
    for i, (ax, e_idx, E_val) in enumerate(
        zip(axes_spatial, energy_indices, energies_to_plot)
    ):
        (line,) = ax.semilogy(
            [], [], label=f"{E_val} TeV", alpha=0.7, linewidth=2, color=f"C{i}"
        )
        lines.append(line)

        # Configure axes
        ax.set_title(f"E = {E_val} TeV", fontsize=12, fontweight="bold")
        ax.axvline(
            12, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="Core edge"
        )
        ax.axvline(
            13,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Cloud edge",
        )
        ax.axvline(
            50, color="g", linestyle="--", linewidth=1.5, alpha=0.7, label="ISM edge"
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-8, 1e2)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Radius (pc)", fontsize=10)
        ax.set_ylabel("f(r, E) [arb. units]", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Plot environment profiles (density)
    ax_env.plot(r_grid, n_gas, "b-", linewidth=2, label="Gas density")
    ax_env.axvline(12, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax_env.axvline(13, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax_env.axvline(50, color="g", linestyle="--", linewidth=1, alpha=0.5)
    ax_env.set_xlabel("Radius (pc)", fontsize=10)
    ax_env.set_ylabel("Density (cm⁻³)", fontsize=10, color="b")
    ax_env.set_yscale("log")
    ax_env.set_xlim(0, 100)
    ax_env.set_ylim(1e-2, 1e6)
    ax_env.tick_params(axis="y", labelcolor="b")
    ax_env.grid(True, alpha=0.3)
    ax_env.set_title("Environment Profiles", fontsize=12, fontweight="bold")
    ax_env.legend(loc="upper left", fontsize=8)

    # Plot diffusion profile at 100 TeV
    ax_diff = ax_diffusion
    D_100TeV = D_values_2d[0, :]
    ax_diff.semilogy(r_grid, D_100TeV, "purple", linewidth=2, label="D at 100 TeV")
    ax_diff.axvline(12, color="r", linestyle="--", linewidth=1, alpha=0.5)
    ax_diff.axvline(13, color="orange", linestyle="--", linewidth=1, alpha=0.5)
    ax_diff.axvline(50, color="g", linestyle="--", linewidth=1, alpha=0.5)
    ax_diff.set_xlabel("Radius (pc)", fontsize=10)
    ax_diff.set_ylabel("D (pc²/Myr)", fontsize=10, color="purple")
    ax_diff.set_xlim(0, 100)
    ax_diff.tick_params(axis="y", labelcolor="purple")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.set_title("Diffusion Coefficient (100 TeV)", fontsize=12, fontweight="bold")
    ax_diff.legend(loc="best", fontsize=8)

    # Initialize ultra-dense cloud spectrum line
    (spectrum_line,) = ax_spectrum.loglog(
        [], [], label="Integrated in cloud", alpha=0.7, linewidth=2, color="C0"
    )
    ax_spectrum.set_title(
        "Spectrum in Ultra-Dense Cloud (12-13 pc)", fontsize=12, fontweight="bold"
    )
    ax_spectrum.set_xlabel("Energy (TeV)", fontsize=10)
    ax_spectrum.set_ylabel("Integrated f(E) [arb. units]", fontsize=10)
    ax_spectrum.set_xlim(E_min.to("TeV").value, E_max.to("TeV").value)
    ax_spectrum.set_ylim(1e-6, 1e2)
    ax_spectrum.grid(True, alpha=0.3)
    ax_spectrum.legend(loc="upper right", fontsize=8)

    # Initialize time evolution plot
    (time_line,) = ax_time_evolution.plot(
        [], [], "b-", linewidth=2, label="Total in cloud"
    )
    ax_time_evolution.set_title(
        "Time Evolution in Ultra-Dense Cloud", fontsize=12, fontweight="bold"
    )
    ax_time_evolution.set_xlabel("Time (Myr)", fontsize=10)
    ax_time_evolution.set_ylabel("Integrated f [arb. units]", fontsize=10)
    ax_time_evolution.set_xlim(0, t_max)
    ax_time_evolution.set_ylim(0, 1)
    ax_time_evolution.grid(True, alpha=0.3)
    ax_time_evolution.legend(loc="upper right", fontsize=8)

    # Time text
    time_text = fig.text(
        0.5, 0.935, "", ha="center", va="top", fontsize=14, weight="bold"
    )

    plt.suptitle(
        "Cosmic Ray Diffusion Through Dense Molecular Cloud\n"
        "Pure Diffusion - Initial Delta Function at 12 pc",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )

    plt.show(block=False)
    plt.pause(0.1)

    # Precompute volume weights for integration
    dr = np.diff(r_grid)
    dr = np.append(dr, dr[-1])
    volume_weights = 4 * np.pi * r_grid**2 * dr

    # Storage for time evolution
    time_array = []
    cloud_integral_array = []

    # Simulation loop with real-time updates
    update_interval = 50
    total_frames = num_timesteps // update_interval

    logger.info("Starting simulation with real-time plotting...")

    for frame in range(total_frames):
        # Advance solver
        solver.step(update_interval)

        current_step = min((frame + 1) * update_interval, num_timesteps - 1)
        current_time = t_grid[current_step]

        # Get current state
        f_current = solver.state.f.copy()

        # Update spatial profile lines
        for i, (line, e_idx) in enumerate(zip(lines, energy_indices)):
            line.set_data(r_grid, f_current[e_idx, :])

        # Calculate integrated spectrum in ultra-dense cloud
        cloud_integral = np.sum(
            f_current[:, ultra_dense_mask] * volume_weights[ultra_dense_mask], axis=1
        )

        spectrum_line.set_data(E_grid.to("TeV").value, cloud_integral)

        # Store time evolution data
        total_in_cloud = np.sum(cloud_integral)
        time_array.append(current_time)
        cloud_integral_array.append(total_in_cloud)

        # Update time evolution plot
        time_line.set_data(time_array, cloud_integral_array)

        # Auto-scale y-axis for time evolution
        if len(cloud_integral_array) > 1:
            max_val = max(cloud_integral_array)
            ax_time_evolution.set_ylim(0, max_val * 1.1)

        # Update time text
        time_text.set_text(f"Time: {current_time:.4f} Myr")

        # Redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

        # Log progress
        if frame % 10 == 0:
            logger.info(
                f"Frame {frame}/{total_frames}, Time: {current_time:.4f} Myr, Particles in cloud: {total_in_cloud:.3e}"
            )

    logger.info("Simulation complete!")

    # Turn off interactive mode and save final plot
    plt.ioff()
    fig.savefig("dense_cloud_diffusion_final.png", dpi=150, bbox_inches="tight")
    logger.info("Saved final state to dense_cloud_diffusion_final.png")
    plt.show()


if __name__ == "__main__":
    run_dense_cloud_test()
