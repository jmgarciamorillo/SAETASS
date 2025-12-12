"""
Test script to reproduce DRAGON paper loss timescales.

This script calculates energy loss timescales for protons and electrons using
the same assumptions as the DRAGON code, without spatial dependence.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import logging

from energy_losses import EnergyLossCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DRAGON_Timescales")

# ============================================================================
# DRAGON REFERENCE DATA - PROTONS (extracted from plots)
# ============================================================================
E_ion = np.array(
    [
        0.1515134292476053,
        0.6832607387157402,
        3.081213588714002,
        33.59818286283781,
        297.6351441631316,
        3478.1906513141266,
        13894.95494373136,
    ]
)  # GeV

tau_ion = np.array(
    [
        4824812648.3006315,
        10029078578.410118,
        24034344531.18135,
        152795643203.626,
        1190315373821.397,
        10690641917089.768,
        39259705416827.125,
    ]
)  # years

E_coul = np.array(
    [0.1515134292476053, 5.455594781168517, 22.956319242369098, 168.09852774253636]
)  # GeV

tau_coul = np.array(
    [292807770454.33203, 2852550169142.318, 10058241713192.473, 65257133801315.664]
)  # years

E_pion = np.array(
    [
        0.12309079138896024,
        0.5746434968715973,
        2.87505804094488,
        14.635701180190798,
        93.30927435461663,
        855.7183144940002,
        5550.8680300598,
        38589.23467029899,
    ]
)  # GeV

tau_pion = np.array(
    [
        8699020728.391531,
        8019760749.949104,
        6157541024.471211,
        4100742953.4045405,
        2676027315.9512243,
        1856095219.5889122,
        1545806421.859974,
        1340801350.86272,
    ]
)  # years

# ============================================================================
# DRAGON REFERENCE DATA - ELECTRONS (extracted from plots)
# TODO: Add electron reference data from DRAGON paper
# ============================================================================
E_ion_e = np.array(
    [
        # TODO: Add ionization energies for electrons (GeV)
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]
)  # GeV - PLACEHOLDER

tau_ion_e = np.array(
    [
        # TODO: Add ionization timescales for electrons (years)
        1e10,
        1e11,
        1e12,
        1e13,
        1e14,
    ]
)  # years - PLACEHOLDER

E_synch_e = np.array(
    [
        # TODO: Add synchrotron energies for electrons (GeV)
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]
)  # GeV - PLACEHOLDER

tau_synch_e = np.array(
    [
        # TODO: Add synchrotron timescales for electrons (years)
        1e8,
        1e9,
        1e10,
        1e11,
        1e12,
    ]
)  # years - PLACEHOLDER

E_brems_e = np.array(
    [
        # TODO: Add bremsstrahlung energies for electrons (GeV)
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]
)  # GeV - PLACEHOLDER

tau_brems_e = np.array(
    [
        # TODO: Add bremsstrahlung timescales for electrons (years)
        1e9,
        1e10,
        1e11,
        1e12,
        1e13,
    ]
)  # years - PLACEHOLDER

E_IC_e = np.array(
    [
        # TODO: Add inverse Compton energies for electrons (GeV)
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ]
)  # GeV - PLACEHOLDER

tau_IC_e = np.array(
    [
        # TODO: Add inverse Compton timescales for electrons (years)
        1e8,
        1e9,
        1e10,
        1e11,
        1e12,
    ]
)  # years - PLACEHOLDER


def create_dragon_environment():
    """
    Create environment matching DRAGON paper assumptions.

    No spatial dependence - uniform ISM conditions.

    Returns:
        dict: Dictionary containing test parameters
    """
    # Energy grids
    num_E = 300
    E_min_proton = 1e-1 * u.GeV
    E_max_proton = 1e5 * u.GeV
    E_grid_proton = (
        np.logspace(np.log10(E_min_proton.value), np.log10(E_max_proton.value), num_E)
        * u.GeV
    )

    # Electron energy grid
    E_min_electron = 1e-1 * u.GeV
    E_max_electron = 1e5 * u.GeV
    E_grid_electron = (
        np.logspace(
            np.log10(E_min_electron.value), np.log10(E_max_electron.value), num_E
        )
        * u.GeV
    )

    # Uniform ISM conditions (no spatial dependence)
    num_r = 1  # Single point - no spatial variation
    r_grid = np.array([0.0]) * u.pc

    # DRAGON ISM parameters
    n_gas = np.array([0.9]) * u.cm**-3  # Uniform gas density
    T_gas = np.array([1e4]) * u.K  # Gas temperature
    B_field = np.array([4.0]) * u.uG  # Magnetic field

    # TODO: Add radiation field parameters if different from default
    # For now, we'll use a simple CMB-like field in the calculator
    T_CMB = 2.725 * u.K
    num_eps = 120
    eps_min = 1e-9 * u.eV
    eps_max = 1e3 * u.eV
    eps_grid = (
        np.exp(np.linspace(np.log(eps_min.value), np.log(eps_max.value), num_eps))
        * eps_min.unit
    )

    # Planck spectrum for CMB
    u_rad_CMB = (8 * np.pi * (eps_grid / (const.h * const.c)) ** 3) / (
        np.exp(eps_grid / (const.k_B * T_CMB)) - 1
    )

    # Single spatial point
    u_rad = u_rad_CMB[:, np.newaxis]

    logger.info("=" * 60)
    logger.info("DRAGON ENVIRONMENT CREATED")
    logger.info("=" * 60)
    logger.info(f"Energy range (protons): {E_min_proton:.3e} - {E_max_proton:.3e}")
    logger.info(
        f"Energy range (electrons): {E_min_electron:.3e} - {E_max_electron:.3e}"
    )
    logger.info(f"Number of energy bins: {num_E}")
    logger.info(f"Gas density: {n_gas[0]:.1f}")
    logger.info(f"Gas temperature: {T_gas[0]:.1e}")
    logger.info(f"Magnetic field: {B_field[0]:.1f}")
    logger.info(f"CMB temperature: {T_CMB:.3f}")
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
    }


def test_proton_losses_dragon():
    """
    Test proton energy loss calculations matching DRAGON paper.

    Computes:
    - Ionization losses (hadronic)
    - Pion production losses
    - Coulomb losses

    Plots timescales vs energy (no spatial dependence).
    """
    logger.info("=" * 60)
    logger.info("TESTING PROTON LOSSES (DRAGON SETUP)")
    logger.info("=" * 60)

    # Setup environment
    env = create_dragon_environment()

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

    # Get timescales (single spatial point, so r_index=0)
    timescales = calc.get_loss_timescales(r_index=0)

    # Create plot matching DRAGON paper style with comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        "ionization": "red",
        "pion": "orange",
        "coulomb": "blue",
        "total": "black",
    }

    labels = {
        "ionization": "Ionization (our calc)",
        "pion": "Pion production (our calc)",
        "coulomb": "Coulomb (our calc)",
        "total": "Total (our calc)",
    }

    # Plot our calculations
    for mechanism, tau in timescales.items():
        label = labels.get(mechanism, mechanism.capitalize())
        color = colors.get(mechanism, "gray")
        lw = 3.0 if mechanism == "total" else 2.0
        ls = "-"
        alpha = 1.0 if mechanism == "total" else 0.7

        ax.loglog(
            env["E_grid_proton"].value,
            tau.to("yr").value,
            label=label,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    # Plot DRAGON reference data
    ax.loglog(
        E_ion,
        tau_ion,
        "o",
        color="red",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Ionization (DRAGON)",
        zorder=10,
    )

    ax.loglog(
        E_coul,
        tau_coul,
        "s",
        color="blue",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Coulomb (DRAGON)",
        zorder=10,
    )

    ax.loglog(
        E_pion,
        tau_pion,
        "^",
        color="orange",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Pion production (DRAGON)",
        zorder=10,
    )

    ax.set_xlabel("Energy (GeV)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Loss Timescale (yr)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Proton Energy Loss Timescales - Comparison with DRAGON\n"
        + f'(n = {env["n_gas"][0].value:.1f} cm$^{{-3}}$, T = {env["T_gas"][0].value:.1e} K)',
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.tick_params(labelsize=12)

    # Set reasonable axis limits
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1e7, 1e14)

    plt.tight_layout()
    plt.savefig(
        "dragon_proton_loss_timescales_comparison.png", dpi=150, bbox_inches="tight"
    )
    logger.info(
        "\n✓ Saved DRAGON comparison plot: dragon_proton_loss_timescales_comparison.png"
    )
    plt.show()

    # Compute comparison statistics
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH DRAGON REFERENCE DATA (PROTONS)")
    logger.info("=" * 60)

    # Interpolate our results at DRAGON energy points
    from scipy.interpolate import interp1d

    # Ionization comparison
    interp_ion = interp1d(
        env["E_grid_proton"].value,
        timescales["ionization"].to("yr").value,
        kind="linear",
        fill_value="extrapolate",
    )
    tau_ion_calc = interp_ion(E_ion)
    relative_diff_ion = (tau_ion_calc - tau_ion) / tau_ion * 100

    logger.info("\nIonization losses:")
    logger.info(f"  Energy (GeV)  | DRAGON (yr)    | Our calc (yr)  | Rel. diff (%)")
    logger.info("  " + "-" * 70)
    for i in range(len(E_ion)):
        logger.info(
            f"  {E_ion[i]:12.3e} | {tau_ion[i]:12.3e} | {tau_ion_calc[i]:12.3e} | {relative_diff_ion[i]:+8.2f}"
        )
    logger.info(
        f"  Mean relative difference: {np.mean(np.abs(relative_diff_ion)):.2f}%"
    )

    # Coulomb comparison
    interp_coul = interp1d(
        env["E_grid_proton"].value,
        timescales["coulomb"].to("yr").value,
        kind="linear",
        fill_value="extrapolate",
    )
    tau_coul_calc = interp_coul(E_coul)
    relative_diff_coul = (tau_coul_calc - tau_coul) / tau_coul * 100

    logger.info("\nCoulomb losses:")
    logger.info(f"  Energy (GeV)  | DRAGON (yr)    | Our calc (yr)  | Rel. diff (%)")
    logger.info("  " + "-" * 70)
    for i in range(len(E_coul)):
        logger.info(
            f"  {E_coul[i]:12.3e} | {tau_coul[i]:12.3e} | {tau_coul_calc[i]:12.3e} | {relative_diff_coul[i]:+8.2f}"
        )
    logger.info(
        f"  Mean relative difference: {np.mean(np.abs(relative_diff_coul)):.2f}%"
    )

    # Pion production comparison
    interp_pion = interp1d(
        env["E_grid_proton"].value,
        timescales["pion"].to("yr").value,
        kind="linear",
        fill_value="extrapolate",
    )
    tau_pion_calc = interp_pion(E_pion)
    relative_diff_pion = (tau_pion_calc - tau_pion) / tau_pion * 100

    logger.info("\nPion production losses:")
    logger.info(f"  Energy (GeV)  | DRAGON (yr)    | Our calc (yr)  | Rel. diff (%)")
    logger.info("  " + "-" * 70)
    for i in range(len(E_pion)):
        logger.info(
            f"  {E_pion[i]:12.3e} | {tau_pion[i]:12.3e} | {tau_pion_calc[i]:12.3e} | {relative_diff_pion[i]:+8.2f}"
        )
    logger.info(
        f"  Mean relative difference: {np.mean(np.abs(relative_diff_pion)):.2f}%"
    )

    logger.info("=" * 60 + "\n")

    return calc


def test_electron_losses_dragon():
    """
    Test electron energy loss calculations matching DRAGON paper.

    Computes:
    - Ionization losses (leptonic)
    - Synchrotron losses
    - Bremsstrahlung losses
    - Inverse Compton losses
    - Coulomb losses (optional, usually negligible)

    Plots timescales vs energy (no spatial dependence).
    """
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ELECTRON LOSSES (DRAGON SETUP)")
    logger.info("=" * 60)

    # Setup environment
    env = create_dragon_environment()

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

    # Assume fully ionized ISM
    ionised_mask = np.zeros(len(env["r_grid"]), dtype=bool)
    E_dot_brems = calc.compute_bremsstrahlung_losses(
        n_gas=env["n_gas"], ionised_mask=ionised_mask
    )
    logger.info(
        f"✓ Bremsstrahlung losses: {E_dot_brems.min():.3e} - {E_dot_brems.max():.3e}"
    )

    E_dot_IC = calc.compute_inverse_compton_losses(
        u_rad=env["u_rad"], eps_grid=env["eps_grid"]
    )
    logger.info(
        f"✓ Inverse Compton losses: {E_dot_IC.min():.3e} - {E_dot_IC.max():.3e}"
    )

    # Coulomb losses (usually negligible for electrons at high energies)
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

    # Get timescales (single spatial point, so r_index=0)
    timescales = calc.get_loss_timescales(r_index=0)

    # Create plot matching DRAGON paper style with comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        "ionization": "red",
        "synchrotron": "yellow",
        "bremsstrahlung": "green",
        "inverse_compton": "purple",
        "coulomb": "blue",
        "total": "black",
    }

    labels = {
        "ionization": "Ionization (our calc)",
        "synchrotron": "Synchrotron (our calc)",
        "bremsstrahlung": "Bremsstrahlung (our calc)",
        "inverse_compton": "Inverse Compton (our calc)",
        "coulomb": "Coulomb (our calc)",
        "total": "Total (our calc)",
    }

    # Plot our calculations
    for mechanism, tau in timescales.items():
        label = labels.get(mechanism, mechanism.capitalize())
        color = colors.get(mechanism, "gray")
        lw = 3.0 if mechanism == "total" else 2.0
        ls = "-"
        alpha = 1.0 if mechanism == "total" else 0.7

        ax.loglog(
            env["E_grid_electron"].value,
            tau.to("yr").value,
            label=label,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    # Plot DRAGON reference data (if available)
    # TODO: Uncomment and adjust when real data is added
    """
    ax.loglog(
        E_ion_e,
        tau_ion_e,
        "o",
        color="red",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Ionization (DRAGON)",
        zorder=10,
    )

    ax.loglog(
        E_synch_e,
        tau_synch_e,
        "s",
        color="yellow",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Synchrotron (DRAGON)",
        zorder=10,
    )

    ax.loglog(
        E_brems_e,
        tau_brems_e,
        "^",
        color="green",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Bremsstrahlung (DRAGON)",
        zorder=10,
    )

    ax.loglog(
        E_IC_e,
        tau_IC_e,
        "d",
        color="purple",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        label="Inverse Compton (DRAGON)",
        zorder=10,
    )
    """

    ax.set_xlabel("Energy (GeV)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Loss Timescale (yr)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Electron Energy Loss Timescales - Comparison with DRAGON\n"
        + f'(n = {env["n_gas"][0].value:.1f} cm$^{{-3}}$, T = {env["T_gas"][0].value:.1e} K, B = {env["B_field"][0].value:.1f} μG)',
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, which="both", linestyle=":")
    ax.tick_params(labelsize=12)

    # Set reasonable axis limits
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(1e4, 1e12)

    plt.tight_layout()
    plt.savefig(
        "dragon_electron_loss_timescales_comparison.png", dpi=150, bbox_inches="tight"
    )
    logger.info(
        "\n✓ Saved DRAGON electron comparison plot: dragon_electron_loss_timescales_comparison.png"
    )
    plt.show()

    # TODO: Add comparison statistics when DRAGON reference data is available
    logger.info("\n" + "=" * 60)
    logger.info("ELECTRON LOSS CALCULATIONS COMPLETED")
    logger.info("=" * 60)
    logger.info("NOTE: Add DRAGON reference data for detailed comparison")
    logger.info("=" * 60 + "\n")

    return calc


def compare_proton_electron_losses_dragon():
    """
    Create a comparison plot of protons vs electrons loss timescales.

    Shows dominant loss mechanisms for each particle type side by side.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING PROTON VS ELECTRON LOSSES")
    logger.info("=" * 60)

    env = create_dragon_environment()

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

    # Compute proton losses
    logger.info("\nComputing proton losses...")
    calc_p.compute_ionization_losses(species="hadronic")
    calc_p.compute_pion_production_losses()
    calc_p.compute_coulomb_losses(T_gas=env["T_gas"], species="hadronic")
    calc_p.compute_total_losses()

    # Compute electron losses
    logger.info("Computing electron losses...")
    calc_e.compute_ionization_losses(species="leptonic")
    calc_e.compute_sychrotron_losses(B_field=env["B_field"])
    ionised_mask = np.ones(len(env["r_grid"]), dtype=bool)
    calc_e.compute_bremsstrahlung_losses(n_gas=env["n_gas"], ionised_mask=ionised_mask)
    calc_e.compute_inverse_compton_losses(u_rad=env["u_rad"], eps_grid=env["eps_grid"])
    calc_e.compute_total_losses()

    # Get timescales
    timescales_p = calc_p.get_loss_timescales(r_index=0)
    timescales_e = calc_e.get_loss_timescales(r_index=0)

    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Proton plot
    colors_p = {
        "ionization": "red",
        "pion": "orange",
        "coulomb": "blue",
        "total": "black",
    }

    for mechanism, tau in timescales_p.items():
        color = colors_p.get(mechanism, "gray")
        lw = 3.0 if mechanism == "total" else 2.0
        ls = "-" if mechanism == "total" else "--"
        alpha = 1.0 if mechanism == "total" else 0.7

        ax1.loglog(
            env["E_grid_proton"].value,
            tau.to("yr").value,
            label=mechanism.capitalize(),
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    ax1.set_xlabel("Energy (GeV)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Loss Timescale (yr)", fontsize=13, fontweight="bold")
    ax1.set_title("Protons", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3, which="both", linestyle=":")
    ax1.tick_params(labelsize=11)
    ax1.set_xlim(0.1, 1e5)
    ax1.set_ylim(1e5, 1e17)

    # Electron plot
    colors_e = {
        "ionization": "red",
        "synchrotron": "yellow",
        "bremsstrahlung": "green",
        "inverse_compton": "purple",
        "coulomb": "blue",
        "total": "black",
    }

    for mechanism, tau in timescales_e.items():
        label = mechanism.capitalize().replace("_", " ")
        color = colors_e.get(mechanism, "gray")
        lw = 3.0 if mechanism == "total" else 2.0
        ls = "-" if mechanism == "total" else "--"
        alpha = 1.0 if mechanism == "total" else 0.7

        ax2.loglog(
            env["E_grid_electron"].value,
            tau.to("yr").value,
            label=label,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    ax2.set_xlabel("Energy (GeV)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Loss Timescale (yr)", fontsize=13, fontweight="bold")
    ax2.set_title("Electrons", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="best")
    ax2.grid(True, alpha=0.3, which="both", linestyle=":")
    ax2.tick_params(labelsize=11)
    ax2.set_xlim(0.1, 1e5)
    ax2.set_ylim(1e5, 1e17)

    plt.suptitle(
        f"Proton vs Electron Loss Timescales (DRAGON Setup)\n"
        f'n = {env["n_gas"][0].value:.1f} cm$^{{-3}}$, T = {env["T_gas"][0].value:.1e} K, B = {env["B_field"][0].value:.1f} μG',
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        "dragon_proton_vs_electron_comparison.png", dpi=150, bbox_inches="tight"
    )
    logger.info("\n✓ Saved comparison plot: dragon_proton_vs_electron_comparison.png")
    plt.show()


def plot_individual_loss_rates(calc_p, calc_e, env):
    """
    Create diagnostic plots showing dE/dt vs energy for both protons and electrons.

    Parameters:
        calc_p: Proton EnergyLossCalculator instance
        calc_e: Electron EnergyLossCalculator instance
        env: Environment dictionary
    """
    logger.info("Creating diagnostic plots of loss rates (dE/dt)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Proton loss rates
    colors_p = {
        "ionization": "red",
        "pion": "orange",
        "coulomb": "blue",
        "total": "black",
    }

    for mechanism, E_dot in calc_p._E_dot_components.items():
        color = colors_p.get(mechanism, "gray")
        ax1.loglog(
            env["E_grid_proton"].value,
            -E_dot[:, 0].to("GeV/yr").value,
            label=mechanism.capitalize(),
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.7,
        )

    ax1.loglog(
        env["E_grid_proton"].value,
        -calc_p._E_dot_total[:, 0].to("GeV/yr").value,
        label="Total",
        color="black",
        linewidth=3.0,
        linestyle="-",
    )

    ax1.set_xlabel("Energy (GeV)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Energy Loss Rate |dE/dt| (GeV/yr)", fontsize=13, fontweight="bold")
    ax1.set_title("Protons", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which="both", linestyle=":")
    ax1.tick_params(labelsize=11)

    # Electron loss rates
    colors_e = {
        "ionization": "red",
        "synchrotron": "yellow",
        "bremsstrahlung": "green",
        "inverse_compton": "purple",
        "total": "black",
    }

    for mechanism, E_dot in calc_e._E_dot_components.items():
        label = mechanism.capitalize().replace("_", " ")
        color = colors_e.get(mechanism, "gray")
        ax2.loglog(
            env["E_grid_electron"].value,
            -E_dot[:, 0].to("GeV/yr").value,
            label=label,
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.7,
        )

    ax2.loglog(
        env["E_grid_electron"].value,
        -calc_e._E_dot_total[:, 0].to("GeV/yr").value,
        label="Total",
        color="black",
        linewidth=3.0,
        linestyle="-",
    )

    ax2.set_xlabel("Energy (GeV)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Energy Loss Rate |dE/dt| (GeV/yr)", fontsize=13, fontweight="bold")
    ax2.set_title("Electrons", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both", linestyle=":")
    ax2.tick_params(labelsize=11)

    plt.suptitle(
        "Energy Loss Rates (DRAGON Setup)",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("dragon_loss_rates_comparison.png", dpi=150, bbox_inches="tight")
    logger.info("✓ Saved loss rates comparison: dragon_loss_rates_comparison.png")
    plt.show()


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("DRAGON PAPER LOSS TIMESCALES TEST")
    logger.info("Reproducing loss calculations from DRAGON code")
    logger.info("=" * 60 + "\n")

    # Test 1: Proton losses (DRAGON setup) with comparison
    calc_proton = test_proton_losses_dragon()

    # Test 2: Electron losses (DRAGON setup) with comparison
    calc_electron = test_electron_losses_dragon()

    # Test 3: Side-by-side comparison
    compare_proton_electron_losses_dragon()

    # Test 4: Loss rates diagnostic plots
    env = create_dragon_environment()
    plot_individual_loss_rates(calc_proton, calc_electron, env)

    logger.info("\n" + "=" * 60)
    logger.info("DRAGON TESTS COMPLETED!")
    logger.info("=" * 60)
    logger.info("\nGenerated plots:")
    logger.info("  1. dragon_proton_loss_timescales_comparison.png")
    logger.info("  2. dragon_electron_loss_timescales_comparison.png")
    logger.info("  3. dragon_proton_vs_electron_comparison.png")
    logger.info("  4. dragon_loss_rates_comparison.png")
    logger.info("\nTODO for better comparison:")
    logger.info("  - Add electron reference data from DRAGON paper")
    logger.info("  - Update E_ion_e, tau_ion_e arrays (lines 88-91)")
    logger.info("  - Update E_synch_e, tau_synch_e arrays (lines 93-96)")
    logger.info("  - Update E_brems_e, tau_brems_e arrays (lines 98-101)")
    logger.info("  - Update E_IC_e, tau_IC_e arrays (lines 103-106)")
    logger.info(
        "  - Uncomment plotting of DRAGON data in test_electron_losses_dragon()"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
