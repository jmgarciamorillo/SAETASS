"""
Test script for EnergyLossCalculator class.

This script tests the energy loss calculations for both protons and electrons,
computing relevant loss mechanisms and plotting their characteristic timescales.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import logging
from giovanni_profiles import create_giovanni_setup

from energy_losses import EnergyLossCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LossesTest")


def create_test_environment():
    """
    Create a test environment with energy grids, spatial grid, and density profile
    using Giovanni setup.

    Returns:
        dict: Dictionary containing test parameters
    """
    # Energy grids
    num_E = 200
    E_min_proton = 0.001 * u.GeV
    E_max_proton = 1e7 * u.GeV
    E_grid_proton = (
        np.logspace(np.log10(E_min_proton.value), np.log10(E_max_proton.value), num_E)
        * u.GeV
    )

    E_min_electron = 0.001 * u.GeV
    E_max_electron = 1e7 * u.GeV
    E_grid_electron = (
        np.logspace(
            np.log10(E_min_electron.value), np.log10(E_max_electron.value), num_E
        )
        * u.GeV
    )

    # Bubble parameters
    num_r = 500
    r_0 = 0.0 * u.pc
    r_end = 300.0 * u.pc
    L_wind = 1e38 * u.erg / u.s
    M_dot = 1e-4 * const.M_sun / u.yr
    rho_0 = const.m_p / u.cm**3
    t_b = 1 * u.Myr
    eta_B = 0.1
    eta_inj = 0.1

    # Create Giovanni setup for reference energy
    E_ref = 100 * u.GeV
    setup = create_giovanni_setup(
        r_0=r_0,
        r_end=r_end,
        num_points=num_r,
        L_wind=L_wind,
        M_dot=M_dot,
        rho_0=rho_0,
        t_b=t_b,
        eta_B=eta_B,
        eta_inj=eta_inj,
        E_k=E_ref,
        diffusion_model="kolmogorov",
    )

    # Extract spatial grid and gas density from Giovanni setup
    r_grid = setup["r"] * u.pc
    n_gas = setup["n_profile_weaver"]

    # Extract other profiles
    B_field = setup["delta_B"]
    T_gas = setup["T_profile_weaver"]

    # ===========================================================
    # Radiation field for inverse Compton: stellar photon field
    # ===========================================================

    M = 100.0 * const.M_sun
    R_sun = const.R_sun
    L_sun = const.L_sun

    R_star = 0.85 * (M / (const.M_sun)) ** 0.67 * R_sun

    def stellar_luminosity(M):
        # Constants from your capture
        Lb1 = 3191 * L_sun
        Lb2 = 368874 * L_sun
        Mb1 = 7.0 * const.M_sun
        Mb2 = 36.089 * const.M_sun
        a1, a2, a3 = 3.97, 2.86, 1.34
        D1, D2 = 0.01, 0.15
        K = 0.817

        if 2.4 <= M / const.M_sun < 12:
            term = 0.5 + 0.5 * (M / Mb1)
            return Lb1 * (M / Mb1) ** a1 * term ** (-(a1 + a2) * D1)

        if M / const.M_sun >= 12:
            term = 0.5 + 0.5 * (M / Mb2)
            return K * Lb2 * (M / Mb2) ** a2 * term ** (-(a2 + a3) * D2)

        raise ValueError("Mass below MLR range.")

    L_star = stellar_luminosity(M)

    T_eff = (L_star / (4 * np.pi * R_star**2 * const.sigma_sb)) ** 0.25

    num_eps = 800
    eps_min = 1e-2 * u.eV
    eps_max = 1e4 * u.eV

    eps_grid = (
        np.exp(np.linspace(np.log(eps_min.value), np.log(eps_max.value), num_eps))
        * eps_min.unit
    )

    nu_grid = eps_grid / const.h

    # Planck spectrum B_nu(T)
    def B_nu(nu, T):
        """Planck function [SI]: W m^-2 Hz^-1 sr^-1"""
        h = const.h
        c = const.c
        kB = const.k_B
        nu = nu
        x = (h * nu / (kB * T)).decompose()
        return (2 * h * nu**3 / c**2) / (np.exp(x) - 1)

    f_r = 1 - np.sqrt(1 - (R_star / r_grid) ** 2)
    # u_nu = 2π/c * B_nu(T) * f(r)
    u_nu = (2 * np.pi / const.c) * B_nu(nu_grid[:, None], T_eff) * f_r[None, :]
    # convert to energy density per energy
    u_rad = u_nu

    # u_rad now has shape (num_eps, num_r)

    logger.info("=" * 60)
    logger.info("TEST ENVIRONMENT CREATED USING GIOVANNI SETUP")
    logger.info("=" * 60)
    logger.info(f"Spatial range: {r_grid.min():.1f} - {r_grid.max():.1f}")
    logger.info(f"Number of radial points: {len(r_grid)}")
    logger.info(f"Termination shock radius: {setup['R_TS']:.2f}")
    logger.info(f"Bubble radius: {setup['R_b']:.2f}")
    logger.info(f"Wind velocity: {setup['v_w'].to('km/s'):.2f}")
    logger.info(f"Gas density range: {n_gas.min():.3e} - {n_gas.max():.3e}")
    logger.info(f"Magnetic field range: {B_field.min():.3e} - {B_field.max():.3e}")
    logger.info(f"Temperature range: {T_gas.min():.1f} - {T_gas.max():.1f}")
    logger.info("=" * 60 + "\n")

    return {
        "E_grid_proton": E_grid_proton,
        "E_grid_electron": E_grid_electron,
        "r_grid": r_grid,
        "n_gas": n_gas,
        "B_field": B_field,
        "T_gas": T_gas,
        "u_rad": u_rad,
        "eps_grid": eps_grid,
        "setup": setup,
    }


def get_region_indices(r_grid, masks, shell_mask):
    """
    Get representative radial indices for each physical region.

    Parameters:
        r_grid: Radial grid array
        masks: Dictionary of boolean masks from Giovanni setup

    Returns:
        dict: Dictionary with region names and their representative indices
    """
    regions = {}

    # Wind region: middle of wind zone
    wind_mask = masks["r_wind"]
    if np.any(wind_mask):
        wind_indices = np.where(wind_mask)[0]
        regions["Wind"] = wind_indices[len(wind_indices) // 2]

    # Bubble region: middle of bubble (between TS and bubble edge)
    bubble_mask = masks["r_bubble"]
    if np.any(bubble_mask):
        bubble_indices = np.where(bubble_mask)[0]
        regions["Bubble"] = bubble_indices[len(bubble_indices) // 2]

    # Shell region: middle of shell
    if np.any(shell_mask):
        shell_indices = np.where(shell_mask)[0]
        regions["Shell"] = shell_indices[len(shell_indices) // 2]

    # ISM region: well into ISM
    ISM_mask = masks["r_ISM"]
    if np.any(ISM_mask):
        ISM_indices = np.where(ISM_mask)[0]
        # Take a point ~20% into the ISM region
        idx_offset = max(1, len(ISM_indices) // 5)
        regions["ISM"] = ISM_indices[idx_offset]

    return regions


def test_proton_losses():
    """
    Test energy loss calculations for protons.

    Computes:
    - Ionization losses
    - Pion production losses
    - Coulomb losses

    Plots timescales at multiple radii (Wind, Bubble, Shell, ISM).
    """
    logger.info("=" * 60)
    logger.info("TESTING PROTON LOSSES")
    logger.info("=" * 60)

    # Setup environment
    env = create_test_environment()

    # Create calculator for protons
    calc = EnergyLossCalculator(
        E_grid=env["E_grid_proton"],
        r_grid=env["r_grid"],
        n_gas=env["n_gas"],
        particle_mass=const.m_p,
    )

    logger.info(
        f"\nEnergy range: {env['E_grid_proton'].min():.3e} - {env['E_grid_proton'].max():.3e}"
    )

    # Compute individual mechanisms
    logger.info("\nComputing loss mechanisms...")

    E_dot_ion = calc.compute_ionization_losses(species="hadronic")
    logger.info(f"✓ Ionization losses: {E_dot_ion.min():.3e} - {E_dot_ion.max():.3e}")

    E_dot_pion = calc.compute_pion_production_losses()
    logger.info(
        f"✓ Pion production losses: {E_dot_pion.min():.3e} - {E_dot_pion.max():.3e}"
    )

    E_dot_coulomb = calc.compute_coulomb_losses(T_gas=env["T_gas"], species="hadronic")
    logger.info(
        f"✓ Coulomb losses: {E_dot_coulomb.min():.3e} - {E_dot_coulomb.max():.3e}"
    )

    # Compute total losses
    E_dot_total = calc.compute_total_losses()
    logger.info(f"✓ Total losses: {E_dot_total.min():.3e} - {E_dot_total.max():.3e}")

    # Get momentum loss rate for solver
    P_dot = calc.get_momentum_loss_rate()
    logger.info(
        f"\nMomentum loss rate (for solver): {P_dot.min():.3e} - {P_dot.max():.3e}"
    )

    # Get region indices
    setup = env["setup"]
    regions = get_region_indices(
        env["r_grid"].value, setup["masks"], setup["weaver_shell_mask"]
    )

    logger.info("\nPlotting regions:")
    for region_name, idx in regions.items():
        logger.info(f"  {region_name}: r = {env['r_grid'][idx]:.2f}")

    # Create figure
    num_regions = len(regions)
    if num_regions == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flat
    elif num_regions == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, num_regions, figsize=(6 * num_regions, 6))
        if num_regions == 1:
            axes = [axes]

    colors = {
        "ionization": "red",
        "pion": "orange",
        "coulomb": "blue",
        "total": "black",
    }

    for i, (region_name, r_idx) in enumerate(regions.items()):
        ax = axes[i]
        r_val = env["r_grid"][r_idx]

        timescales = calc.get_loss_timescales(r_index=r_idx)

        for mechanism, tau in timescales.items():
            label = mechanism.capitalize()
            color = colors.get(mechanism, "gray")
            lw = 2.5 if mechanism == "total" else 1.5
            ls = "-" if mechanism == "total" else "--"

            ax.loglog(
                env["E_grid_proton"].value,
                tau.to("yr").value,
                label=label,
                color=color,
                linewidth=lw,
                linestyle=ls,
            )

        ax.set_xlabel("Energy (GeV)", fontsize=11)
        ax.set_ylabel("Loss Timescale (yr)", fontsize=11)
        ax.set_title(f"{region_name}: r = {r_val:.1f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Proton Energy Loss Timescales (Giovanni Setup)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("proton_loss_timescales.png", dpi=150)
    logger.info("\n✓ Saved proton timescales plot: proton_loss_timescales.png")
    plt.show()

    return calc


def test_electron_losses():
    """
    Test energy loss calculations for electrons.

    Computes:
    - Ionization losses
    - Synchrotron losses
    - Bremsstrahlung losses
    - Inverse Compton losses
    - Coulomb losses

    Plots timescales at multiple radii (Wind, Bubble, Shell, ISM).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ELECTRON LOSSES")
    logger.info("=" * 60)

    # Setup environment
    env = create_test_environment()

    # Create calculator for electrons
    calc = EnergyLossCalculator(
        E_grid=env["E_grid_electron"],
        r_grid=env["r_grid"],
        n_gas=env["n_gas"],
        particle_mass=const.m_e,
    )

    logger.info(
        f"\nEnergy range: {env['E_grid_electron'].min():.3e} - {env['E_grid_electron'].max():.3e}"
    )

    # Compute individual mechanisms
    logger.info("\nComputing loss mechanisms...")

    E_dot_ion = calc.compute_ionization_losses(species="leptonic")
    logger.info(f"✓ Ionization losses: {E_dot_ion.min():.3e} - {E_dot_ion.max():.3e}")

    E_dot_synch = calc.compute_sychrotron_losses(B_field=env["B_field"])
    logger.info(
        f"✓ Synchrotron losses: {E_dot_synch.min():.3e} - {E_dot_synch.max():.3e}"
    )

    # Assume fully ionised for simplicity
    wind = env["setup"]["masks"]["r_wind"]
    bubble = env["setup"]["masks"]["r_bubble"]
    ionised_mask = wind + bubble
    E_dot_brems = calc.compute_bremsstrahlung_losses(
        n_gas=env["n_gas"], ionised_mask=ionised_mask
    )
    logger.info(
        f"✓ Bremsstrahlung losses: {E_dot_brems.min():.3e} - {E_dot_brems.max():.3e}"
    )

    # E_dot_IC = calc.compute_inverse_compton_losses(
    #     u_rad=env["u_rad"], eps_grid=env["eps_grid"]
    # )
    # logger.info(
    #     f"✓ Inverse Compton losses: {E_dot_IC.min():.3e} - {E_dot_IC.max():.3e}"
    # )

    E_dot_coulomb = calc.compute_coulomb_losses(T_gas=env["T_gas"], species="leptonic")
    logger.info(
        f"✓ Coulomb losses: {E_dot_coulomb.min():.3e} - {E_dot_coulomb.max():.3e}"
    )

    # Compute total losses
    E_dot_total = calc.compute_total_losses()
    logger.info(f"✓ Total losses: {E_dot_total.min():.3e} - {E_dot_total.max():.3e}")

    # Get momentum loss rate for solver
    P_dot = calc.get_momentum_loss_rate()
    logger.info(
        f"\nMomentum loss rate (for solver): {P_dot.min():.3e} - {P_dot.max():.3e}"
    )

    # Get region indices
    setup = env["setup"]
    regions = get_region_indices(
        env["r_grid"].value, setup["masks"], setup["weaver_shell_mask"]
    )

    logger.info("\nPlotting regions:")
    for region_name, idx in regions.items():
        logger.info(f"  {region_name}: r = {env['r_grid'][idx]:.2f}")

    # Create figure
    num_regions = len(regions)
    if num_regions == 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flat
    elif num_regions == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, num_regions, figsize=(6 * num_regions, 6))
        if num_regions == 1:
            axes = [axes]

    colors = {
        "ionization": "red",
        "synchrotron": "yellow",
        "bremsstrahlung": "green",
        "inverse_compton": "purple",
        "coulomb": "blue",
        "total": "black",
    }

    for i, (region_name, r_idx) in enumerate(regions.items()):
        ax = axes[i]
        r_val = env["r_grid"][r_idx]

        timescales = calc.get_loss_timescales(r_index=r_idx)

        for mechanism, tau in timescales.items():
            label = mechanism.capitalize().replace("_", " ")
            color = colors.get(mechanism, "gray")
            lw = 2.5 if mechanism == "total" else 1.5
            ls = "-" if mechanism == "total" else "--"

            ax.loglog(
                env["E_grid_electron"].value,
                tau.to("yr").value,
                label=label,
                color=color,
                linewidth=lw,
                linestyle=ls,
            )

        ax.set_xlabel("Energy (GeV)", fontsize=11)
        ax.set_ylabel("Loss Timescale (yr)", fontsize=11)
        ax.set_title(f"{region_name}: r = {r_val:.1f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Electron Energy Loss Timescales (Giovanni Setup)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("electron_loss_timescales.png", dpi=150)
    logger.info("\n✓ Saved electron timescales plot: electron_loss_timescales.png")
    plt.show()

    return calc


def compare_proton_electron_losses():
    """
    Create a comparison plot of dominant loss mechanisms for protons vs electrons
    in each physical region.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING PROTON VS ELECTRON LOSSES")
    logger.info("=" * 60)

    env = create_test_environment()

    # Create calculators
    calc_p = EnergyLossCalculator(
        E_grid=env["E_grid_proton"],
        r_grid=env["r_grid"],
        n_gas=env["n_gas"],
        particle_mass=const.m_p,
    )

    calc_e = EnergyLossCalculator(
        E_grid=env["E_grid_electron"],
        r_grid=env["r_grid"],
        n_gas=env["n_gas"],
        particle_mass=const.m_e,
    )

    # Compute losses
    calc_p.compute_ionization_losses(species="hadronic")
    calc_p.compute_pion_production_losses()
    calc_p.compute_total_losses()

    calc_e.compute_ionization_losses(species="leptonic")
    calc_e.compute_sychrotron_losses(B_field=env["B_field"])
    wind = env["setup"]["masks"]["r_wind"]
    bubble = env["setup"]["masks"]["r_bubble"]
    ionised_mask = wind + bubble
    calc_e.compute_bremsstrahlung_losses(n_gas=env["n_gas"], ionised_mask=ionised_mask)
    calc_e.compute_total_losses()

    # Get region indices
    setup = env["setup"]
    regions = get_region_indices(
        env["r_grid"].value, setup["masks"], setup["weaver_shell_mask"]
    )

    # Plot comparison for Bubble region (or first available region)
    region_name = "Bubble" if "Bubble" in regions else list(regions.keys())[0]
    r_idx = regions[region_name]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Proton plot
    timescales_p = calc_p.get_loss_timescales(r_index=r_idx)
    for mechanism, tau in timescales_p.items():
        if mechanism != "total":
            ax1.loglog(
                env["E_grid_proton"].value,
                tau.to("yr").value,
                label=mechanism.capitalize(),
                linewidth=1.5,
                linestyle="--",
            )
    ax1.loglog(
        env["E_grid_proton"].value,
        timescales_p["total"].to("yr").value,
        "k-",
        label="Total",
        linewidth=2.5,
    )
    ax1.set_xlabel("Energy (GeV)", fontsize=12)
    ax1.set_ylabel("Loss Timescale (yr)", fontsize=12)
    ax1.set_title("Protons", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Electron plot
    timescales_e = calc_e.get_loss_timescales(r_index=r_idx)
    for mechanism, tau in timescales_e.items():
        if mechanism != "total":
            ax2.loglog(
                env["E_grid_electron"].value,
                tau.to("yr").value,
                label=mechanism.capitalize().replace("_", " "),
                linewidth=1.5,
                linestyle="--",
            )
    ax2.loglog(
        env["E_grid_electron"].value,
        timescales_e["total"].to("yr").value,
        "k-",
        label="Total",
        linewidth=2.5,
    )
    ax2.set_xlabel("Energy (GeV)", fontsize=12)
    ax2.set_ylabel("Loss Timescale (yr)", fontsize=12)
    ax2.set_title("Electrons", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f'{region_name}: r = {env["r_grid"][r_idx].value:.1f} pc',
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("proton_vs_electron_comparison.png", dpi=150)
    logger.info("\n✓ Saved comparison plot: proton_vs_electron_comparison.png")
    plt.show()


def main():
    """Main test function."""
    logger.info("Starting Energy Loss Calculator Tests")
    logger.info("=" * 60 + "\n")

    # Test 1: Proton losses
    calc_proton = test_proton_losses()

    # Test 2: Electron losses
    calc_electron = test_electron_losses()

    # Comparison plot
    compare_proton_electron_losses()

    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
