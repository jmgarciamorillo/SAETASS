import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import logging
from matplotlib.gridspec import GridSpec
import pickle
import os
import sys
import argparse

from State import State
from Grid import Grid
from Solver import Solver
from giovanni_profiles import create_giovanni_setup, get_theoretical_profile

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logger = logging.getLogger("GiovanniLossesTest")


def run_giovanni_losses_test(filename="giovanni_losses_results.pkl"):
    # Physical and grid parameters
    num_r = 2000
    num_E = 400
    r_0 = 0.0 * u.pc
    r_end = 300.0 * u.pc
    E_min = 0.001 * u.GeV
    E_max = 1000000 * u.GeV

    # Log-spaced energy grid
    E_grid = np.logspace(np.log10(E_min.value), np.log10(E_max.value), num_E) * u.GeV

    # Spatial grid
    r_grid = np.linspace(r_0.to("pc").value, r_end.to("pc").value, num_r)

    # Bubble parameters
    L_wind = 1e38 * u.erg / u.s
    M_dot = 1e-4 * const.M_sun / u.yr
    rho_0 = const.m_p / u.cm**3
    t_b = 1 * u.Myr
    eta_B = 0.1
    eta_inj = 0.1

    # Setup Giovanni profiles for a reference energy (middle of the range)
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

    # Extract key parameters from reference
    R_TS = setup["R_TS"]
    R_b = setup["R_b"]
    v_field = setup["v_field"]

    # Precompute energy-dependent quantities and Giovanni theoretical profiles
    # D_values_2d, Q_2d are numeric arrays used by the solver; we also store
    # per-energy theoretical profiles computed with get_theoretical_profile.
    D_values_2d = np.zeros((num_E, num_r))
    Q_2d = np.zeros((num_E, num_r))
    f_theoretical_list = []
    masks_list = []
    v_w_list = []

    for j, E_k in enumerate(E_grid):
        setup_E = create_giovanni_setup(
            r_0=r_0,
            r_end=r_end,
            num_points=num_r,
            L_wind=L_wind,
            M_dot=M_dot,
            rho_0=rho_0,
            t_b=t_b,
            eta_B=eta_B,
            eta_inj=eta_inj,
            E_k=E_k,
            diffusion_model="kolmogorov",
        )

        # store numeric arrays for operators
        D_values_2d[j, :] = setup_E["D_values"].to("pc**2/Myr").value
        Q_2d[j, :] = setup_E["Q"]

        # store auxiliary data for theoretical profile calculation / plotting
        masks_list.append(setup_E["masks"])
        v_w_list.append(setup_E["v_w"])

        # compute theoretical steady-state profile for this energy (f_TS=1.0)
        f_theo = get_theoretical_profile(
            setup_E["r"],
            setup_E["masks"],
            setup_E["v_w"],
            setup_E["D_values"],
            setup_E["R_TS"],
            setup_E["R_b"],
            f_gal=0.0,
            f_TS=1.0,
        )
        f_theoretical_list.append(f_theo)

    # Losses: compute energy loss rates (dE/dt) from detailed processes
    E0 = 1.0 * u.GeV
    n_gas = setup["n_profile_weaver"]  # gas density (astropy quantity)
    n_gas = n_gas.to("cm**-3").value  # in cm^-3
    E_grid_GeV = (E_grid / E0).value  # dimensionless

    E_dot_pion = (
        (-3.85e-16 * np.outer(n_gas, E_grid_GeV**1.28 * (E_grid_GeV + 200) ** -0.2).T)
        * u.GeV
        / u.s
    )  # shape (num_E, num_r)

    IH = 19 * u.eV
    gamma = 1 + E_grid / (const.m_p * const.c**2)
    beta = np.sqrt(1 - 1 / gamma**2)
    q_max = (
        2
        * const.m_e
        * const.c**2
        * beta**2
        * gamma**2
        / (1 + 2 * gamma * const.m_e / const.m_p)
    )
    A = (
        np.log(2 * const.m_e * const.c**2 * beta**2 * gamma**2 * q_max / IH**2)
        - 2 * beta**2
    )
    E_dot_ionisation = (
        -7.64e-18 * np.outer(n_gas, A).T * u.GeV / u.s
    )  # shape (num_E, num_r)

    E_dot = E_dot_pion + E_dot_ionisation  # shape (num_E, num_r)

    # plot timescales for reference
    numplot = num_r // 5
    timescale_ion = (
        E_grid.to("GeV").value / (-E_dot_ionisation[:, numplot]).to("GeV/Myr").value
    )
    timescale_pion = (
        E_grid.to("GeV").value / (-E_dot_pion[:, numplot]).to("GeV/Myr").value
    )
    timescale_total = E_grid.to("GeV").value / (-E_dot[:, numplot]).to("GeV/Myr").value
    plt.figure(figsize=(8, 6))
    plt.loglog(E_grid.to("GeV").value, timescale_ion, label="Ionization")
    plt.loglog(E_grid.to("GeV").value, timescale_pion, label="Pion Production")
    plt.loglog(
        E_grid.to("GeV").value, timescale_total, label="Total Losses", linewidth=2
    )
    plt.xlabel("Energy (GeV)")
    plt.ylabel("Loss Timescale (Myr)")
    plt.title("Loss Timescales at r = {:.1f} pc".format(r_grid[numplot]))
    plt.legend()
    plt.grid(True)
    plt.show()

    # calculate P_dot from E_dot
    v_particle = beta * const.c
    P_dot = (E_dot / v_particle[:, np.newaxis]).to("g*cm/Myr**2")  # keep units
    # convert to numeric array in solver units (g*cm/Myr)
    P_dot_numeric = P_dot.value  # shape (num_E, num_r)

    # Time grid
    t_max = 0.8  # Myr
    num_timesteps = 200000
    t_grid = np.linspace(0, t_max, num_timesteps)

    # Initial condition: zero everywhere (shape expected: (n_p, n_r) or (num_E, num_r))
    f_init = np.zeros((num_E, num_r))
    state = State(f_init)  # State holds array; solver expects this shape

    # Operator parameters
    advectionFV_params = {
        "v_centers": np.tile(v_field, (num_E, 1)),  # shape (num_E, num_r)
        "order": 2,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_U": np.zeros(num_E, dtype=float),
    }
    diffusionFV_params = {
        "D_values": D_values_2d,  # numeric (num_E, num_r)
        "Q_values": None,
        "f_end": 0.0,
    }
    lossFV_params = {
        "P_dot": P_dot_numeric,  # numeric (num_E, num_r)
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_U": np.zeros((num_r, 1), dtype=float),
        "order": 2,
        "adiabatic_losses": True,
        "v_centers_physical": np.tile(v_field, (num_E, 1)),  # shape (num_E, num_r)
    }

    # Calculate momentum from kinetic energy for grid setup for protons (same as before)
    p_grid = (
        (np.sqrt((E_grid**2 + 2 * E_grid * (const.m_p * const.c**2))) / const.c)
        .to("g*cm/Myr")
        .value
    )

    index_1GeV = np.abs(E_grid.to("GeV").value - 1.0).argmin()
    index_1PeV = np.abs(E_grid.to("GeV").value - 1e6).argmin()

    s = 4
    mask_cols = np.any(Q_2d > 0, axis=0)
    spectrum = (p_grid / p_grid[index_1GeV]) ** (-s) * np.exp(
        -p_grid / p_grid[index_1PeV]
    )
    Q_2d[:, mask_cols] = spectrum[:, np.newaxis]

    source_params = {
        "source": Q_2d,
    }

    op_params = {
        "advectionFV": advectionFV_params,
        "diffusionFV": diffusionFV_params,
        "lossFV": lossFV_params,
        "source": source_params,
    }

    # Create grid
    grid = Grid(
        r_centers=r_grid,
        t_grid=t_grid,
        p_centers=p_grid,  # Use momentum as "momentum" axis
    )

    # Create solver
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advectionFV-lossFV-source-diffusionFV",
        operator_params=op_params,
        substeps={"advectionFV": 1, "diffusionFV": 1, "lossFV": 1, "source": 1},
    )

    # Run simulation and store results at selected times
    sample_count = 8
    sample_indices = np.linspace(0, num_timesteps - 1, sample_count, dtype=int)
    stored_curves = []
    stored_times = []

    current_step = 0
    for next_sample in sample_indices:
        steps_to_advance = int(next_sample - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_sample

        f_current = solver.state.f.copy()  # shape (num_E, num_r)
        stored_curves.append(f_current)
        stored_times.append(t_grid[current_step])

    # ----------- SAVE RESULTS -----------
    results_dict = {
        "r_grid": r_grid,
        "E_grid": E_grid.value,
        "stored_curves": stored_curves,
        "stored_times": stored_times,
        "R_TS": R_TS.to("pc").value,
        "R_b": R_b.to("pc").value,
        "f_theoretical_list": [ft.tolist() for ft in f_theoretical_list],
        "params": {
            "num_r": num_r,
            "num_E": num_E,
            "r_0": r_0.value,
            "r_end": r_end.value,
            "E_min": E_min.value,
            "E_max": E_max.value,
            "L_wind": L_wind.value,
            "M_dot": M_dot.value,
            "rho_0": rho_0.value,
            "t_b": t_b.value,
            "eta_B": eta_B,
            "eta_inj": eta_inj,
        },
    }
    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)
    print(f"Resultados guardados en '{filename}'.")

    # Plot results for a few energies, overlay Giovanni steady-state profile per energy
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    energies_to_plot = [0.001, 0.01, 0.1, 1, 10, 1000, 100000, 1000000]  # GeV
    energy_indices = [np.abs(E_grid.value - e).argmin() for e in energies_to_plot]

    for i, (ax, e_idx, E_val) in enumerate(
        zip(axes.flat, energy_indices, energies_to_plot)
    ):
        # get final curve (last stored) and find TS index for normalization
        final_curve = stored_curves[-1]  # shape (num_E, num_r)
        # numerical at this energy
        for curve, tval in zip(stored_curves, stored_times):
            # normalize each curve by TS level found in final state
            # find index just past R_TS
            r_bubble = r_grid >= R_TS.to("pc").value
            ts_idx = np.where(r_bubble)[0][0] + 5
            ts_idx = min(ts_idx, len(r_grid) - 1)
            ts_level = (
                final_curve[e_idx, ts_idx] if final_curve[e_idx, ts_idx] > 0 else 1.0
            )
            normalized_curve = curve[e_idx, :] / ts_level

            ax.semilogy(
                r_grid,
                normalized_curve,
                label=f"t={tval:.2f} Myr" if tval == stored_times[-1] else None,
                alpha=0.7,
            )

        # Giovanni theoretical profile for this energy (already computed)
        f_theoretical = np.array(f_theoretical_list[e_idx])
        # Theoretical profile was computed with f_TS=1.0; plot directly
        ax.semilogy(r_grid, f_theoretical, "k--", lw=1.5, label="Giovanni steady state")

        ax.set_title(f"E = {E_val:.1f} GeV")
        ax.axvline(R_TS.to("pc").value, color="r", linestyle="--", linewidth=1)
        ax.axvline(R_b.to("pc").value, color="g", linestyle="--", linewidth=1)
        ax.set_xlim(0, r_end.to("pc").value)
        ax.set_ylim(1e-6, 1e1)
        ax.grid(True)
        if i % 4 == 0:
            ax.set_ylabel("f(r, E) (normalized)")
        if i >= 4:
            ax.set_xlabel("Radius (pc)")
        if i == 0:
            ax.legend(loc="upper right")

    plt.suptitle(
        "Giovanni Model: Advection-Diffusion-Loss-Source\nTime Evolution for Different Energies (with Giovanni reference)"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("giovanni_losses_test.png", dpi=150)
    plt.show()

    # semilogz plot in 3d
    # Transformaciones a escala logarítmica
    E_log = np.log10(E_grid.to("GeV").value)  # E en log10
    final_curve_log = np.log10(final_curve)  # final_curve en log10
    R, E = np.meshgrid(r_grid, E_log)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        R,
        E,
        final_curve_log,
        cmap="viridis",
        edgecolor="none",
        alpha=0.8,
    )

    ax.set_yticks(np.log10([1e0, 1e2, 1e4, 1e6]))
    ax.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$", r"$10^6$"])

    ax.set_zticks(np.log10([1e-5, 1e-3, 1e-1, 1e1]))
    ax.set_zticklabels([r"$10^{-5}$", r"$10^{-3}$", r"$10^{-1}$", r"$10^{1}$"])
    ax.set_zlim(np.log10(1e-6), np.log10(1e1))

    ax.set_xlabel("Radius (pc)")
    ax.set_ylabel("Energy (GeV)")
    ax.set_zlabel("f(r, E)")
    ax.set_title("Giovanni Model: Final Distribution f(r, E)")
    plt.tight_layout()
    plt.savefig("giovanni_losses_3d.png", dpi=150)
    plt.show()


def load_results_file(filename="giovanni_losses_results.pkl"):
    """Load previously saved results produced by this script."""
    if not os.path.exists(filename):
        logger.error(f"Results file '{filename}' not found.")
        return None
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # restore arrays if they were converted to lists
    if "f_theoretical_list" in data:
        data["f_theoretical_list"] = [np.array(ft) for ft in data["f_theoretical_list"]]
    return data


def plot_from_results(results_dict, out_prefix="giovanni_losses_fromfile"):
    """Plot the same figures using a loaded results dictionary (no simulation)."""
    r_grid = results_dict["r_grid"]
    E_grid = np.array(results_dict["E_grid"])
    stored_curves = results_dict["stored_curves"]
    stored_times = results_dict["stored_times"]
    R_TS = results_dict["R_TS"]
    R_b = results_dict["R_b"]
    f_theoretical_list = results_dict.get("f_theoretical_list", None)

    # Ensure stored_curves are numpy arrays
    stored_curves = [np.asarray(c) for c in stored_curves]

    # Main multi-panel figure (same energies as in simulation)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    energies_to_plot = [1, 3, 10, 30, 100, 300, 600, 1000]  # GeV
    energy_indices = [np.abs(E_grid - e).argmin() for e in energies_to_plot]

    for i, (ax, e_idx, E_val) in enumerate(
        zip(axes.flat, energy_indices, energies_to_plot)
    ):
        final_curve = stored_curves[-1]
        for curve, tval in zip(stored_curves, stored_times):
            # find index just past R_TS for normalization
            r_bubble = r_grid >= R_TS
            ts_idx = np.where(r_bubble)[0][0] + 5
            ts_idx = min(ts_idx, len(r_grid) - 1)
            ts_level = (
                final_curve[e_idx, ts_idx] if final_curve[e_idx, ts_idx] > 0 else 1.0
            )
            normalized_curve = curve[e_idx, :] / ts_level

            ax.semilogy(
                r_grid,
                normalized_curve,
                label=f"t={tval:.2f} Myr" if tval == stored_times[-1] else None,
                alpha=0.7,
            )

        # Giovanni theoretical profile if available
        if f_theoretical_list is not None:
            f_theoretical = np.array(f_theoretical_list[e_idx])
            ax.semilogy(
                r_grid, f_theoretical, "k--", lw=1.5, label="Giovanni steady state"
            )

        ax.set_title(f"E = {E_val:.1f} GeV")
        ax.axvline(R_TS, color="r", linestyle="--", linewidth=1)
        ax.axvline(R_b, color="g", linestyle="--", linewidth=1)
        ax.set_xlim(0, r_grid[-1])
        ax.set_ylim(1e-6, 1e1)
        ax.grid(True)
        if i % 4 == 0:
            ax.set_ylabel("f(r, E) (normalized)")
        if i >= 4:
            ax.set_xlabel("Radius (pc)")
        if i == 0:
            ax.legend(loc="upper right")

    plt.suptitle("Giovanni Model: Results from file")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_file = out_prefix + "_panels.png"
    plt.savefig(fig_file, dpi=150)
    logger.info(f"Saved panel figure to {fig_file}")
    plt.show()

    # Optionally also save a pickle copy (already loaded) or additional diagnostics here
    return


if __name__ == "__main__":
    # Edit the two variables below to choose behaviour (no CLI needed):
    LOAD_ONLY = False  # set to False to run the simulation
    RESULTS_FILE = "giovanni_losses_results_subgevProt.pkl"

    if LOAD_ONLY:
        data = load_results_file(RESULTS_FILE)
        if data is None:
            sys.exit(1)
        plot_from_results(
            data, out_prefix=os.path.splitext(os.path.basename(RESULTS_FILE))[0]
        )
    else:
        run_giovanni_losses_test(filename=RESULTS_FILE)
