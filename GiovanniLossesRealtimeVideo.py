# filepath: /Users/jmorillo/SolverAlpha/SolverAlpha/GiovanniLossesRealtimeVideo.py
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import logging
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, FFMpegWriter

from State import State
from Grid import Grid
from Solver import Solver
from giovanni_profiles import create_giovanni_setup, get_theoretical_profile

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logger = logging.getLogger("GiovanniLossesRealtimeVideo")


def run_giovanni_losses_realtime_video():
    """Run Giovanni losses test with MP4 video generation."""

    # Physical and grid parameters
    num_r = 500
    num_E = 300
    r_0 = 0.0 * u.pc
    r_end = 300.0 * u.pc
    E_min = 0.001 * u.GeV
    E_max = 1000000 * u.GeV

    # Log-spaced energy grid
    E_grid = np.logspace(np.log10(E_min.value), np.log10(E_max.value), num_E) * u.GeV

    # Source term
    p_grid = np.sqrt((E_grid**2 + 2 * E_grid * (const.m_p * const.c**2))) / const.c
    index_1GeV = np.abs(E_grid.to("GeV").value - 1.0).argmin()
    index_100TeV = np.abs(E_grid.to("GeV").value - 1e5).argmin()
    p_grid_numeric = p_grid.to("g*pc/Myr").value

    # Spatial grid
    r_grid = np.linspace(r_0.to("pc").value, r_end.to("pc").value, num_r)

    # Bubble parameters
    L_wind = 1e38 * u.erg / u.s
    M_dot = 1e-4 * const.M_sun / u.yr
    rho_0 = const.m_p / u.cm**3
    t_b = 1 * u.Myr
    eta_B = 0.1
    eta_inj = 0.1

    # Setup Giovanni profiles for reference energy
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

    # Extract key parameters
    R_TS = setup["R_TS"]
    R_b = setup["R_b"]
    v_field = setup["v_field"]

    # Precompute energy-dependent quantities
    D_values_2d = np.zeros((num_E, num_r))
    Q_2d = np.zeros((num_E, num_r))
    s = 4
    mask_cols = setup["Q"] > 0
    if sum(mask_cols) == 0:
        raise ValueError("No source injection columns found in Q_2d.")
    spectrum = (p_grid_numeric / p_grid_numeric[index_1GeV]) ** (-s) * np.exp(
        -p_grid_numeric / p_grid_numeric[index_100TeV]
    )

    f_theoretical_list = []

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

        D_values_2d[j, :] = setup_E["D_values"].to("pc**2/Myr").value
        Q_2d[j, :] = setup_E["Q"]

        # Compute theoretical steady-state profile
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
        f_theoretical_list.append(f_theo * spectrum[j])

    Q_2d[:, mask_cols] = spectrum[:, np.newaxis]
    # Losses computation
    E0 = 1.0 * u.GeV
    n_gas = setup["n_profile_weaver"].to("cm**-3").value
    E_grid_GeV = (E_grid / E0).value

    E_dot_pion = (
        (-3.85e-16 * np.outer(n_gas, E_grid_GeV**1.28 * (E_grid_GeV + 200) ** -0.2).T)
        * u.GeV
        / u.s
    )

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
    E_dot_ionisation = -7.64e-18 * np.outer(n_gas, A).T * u.GeV / u.s

    E_dot = E_dot_pion + E_dot_ionisation

    # Calculate P_dot
    P_dot = (
        E_dot
        * ((E_grid + const.m_p * const.c**2) / (p_grid * const.c**2))[:, np.newaxis]
    ).to("g*pc/Myr**2")
    P_dot_numeric = P_dot.value

    # Time grid
    t_max = 1  # Myr
    num_timesteps = 20000
    t_grid = np.linspace(0, t_max, num_timesteps)

    # Initial condition
    f_init = np.zeros((num_E, num_r))
    state = State(f_init)

    # Tiempos de permanencia dentro de la región de inyección
    delta_r = sum(mask_cols) * np.diff(r_grid).mean()
    tau_adv = delta_r / v_field[mask_cols] * np.ones(num_E)
    tau_diff = delta_r**2 / np.transpose(D_values_2d[:, mask_cols])[0]
    tau_permanence = 1 / (1 / tau_adv + 1 / tau_diff)

    # Operator parameters
    advectionFV_params = {
        "v_centers": np.tile(v_field, (num_E, 1)),
        "order": 2,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_U": np.zeros(num_E, dtype=float),
    }
    diffusionFV_params = {
        "D_values": D_values_2d,
        "Q_values": None,
        "f_end": 0.0,
    }
    lossFV_params = {
        "P_dot": P_dot_numeric * 10,
        "limiter": "minmod",
        "cfl": 0.2,
        "inflow_value_U": np.zeros((num_r, 1), dtype=float),
        "order": 2,
        "adiabatic_losses": True,
        "v_centers_physical": np.tile(v_field, (num_E, 1)),
    }

    source_params = {"source": Q_2d}

    op_params = {
        "advectionFV": advectionFV_params,
        "diffusionFV": diffusionFV_params,
        "lossFV": lossFV_params,
        "source": source_params,
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
        problem_type="advectionFV-lossFV-source-diffusionFV",
        operator_params=op_params,
        substeps={
            "advectionFV": 1,
            "diffusionFV": 1,
            "source": 1,
        },
        splitting_scheme="strang",
    )

    # Setup plotting
    energies_to_plot = [0.001, 0.01, 0.1, 1, 10000, 100000]  # GeV
    energy_indices = [np.abs(E_grid.value - e).argmin() for e in energies_to_plot]

    # Find bubble mask for integration
    bubble_mask = r_grid <= R_b.to("pc").value - 5 * np.diff(r_grid).mean()

    # Create figure for animation
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flat
    lines = []
    theo_lines = []

    # Plot spatial profiles (first 6 subplots)
    for i, (ax, e_idx, E_val) in enumerate(
        zip(axes[:6], energy_indices, energies_to_plot)
    ):
        # Initialize empty line for numerical solution
        (line,) = ax.semilogy([], [], label="Numerical", alpha=0.7, linewidth=2)
        lines.append(line)

        # Plot theoretical profile
        f_theoretical = np.array(f_theoretical_list[e_idx])
        f_theoretical_ts_level = f_theoretical[r_grid >= R_TS.to("pc").value][0]
        f_theoretical /= f_theoretical_ts_level
        (theo_line,) = ax.semilogy(
            r_grid, f_theoretical, "k--", lw=1.5, label="Giovanni steady state"
        )
        theo_lines.append(theo_line)

        # Configure axes
        ax.set_title(f"E = {E_val:.3g} GeV")
        ax.axvline(
            R_TS.to("pc").value, color="r", linestyle="--", linewidth=1, alpha=0.5
        )
        ax.axvline(
            R_b.to("pc").value, color="g", linestyle="--", linewidth=1, alpha=0.5
        )
        ax.set_xlim(0, r_end.to("pc").value)
        ax.set_ylim(1e-6, 1e1)
        ax.grid(True, alpha=0.3)

        if i % 4 == 0:
            ax.set_ylabel("f(r, E) (normalized)")
        if i >= 4:
            ax.set_xlabel("Radius (pc)")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Setup spectrum plot (7th subplot)
    ax_spectrum = axes[6]
    (spectrum_line,) = ax_spectrum.loglog(
        [], [], label="Integrated Spectrum", alpha=0.7, linewidth=2, color="C0"
    )
    (corrected_spectrum_line,) = ax_spectrum.loglog(
        [], [], label="Corrected Spectrum", alpha=0.7, linewidth=2, color="C2"
    )
    (injected_spectrum_line,) = ax_spectrum.loglog(
        [], [], label="Injected Spectrum", alpha=0.7, linewidth=2, color="C3"
    )
    (theoretical_line,) = ax_spectrum.loglog(
        [], [], "k--", lw=1.5, label="Giovanni steady state"
    )

    ax_spectrum.set_title("E² × Integrated Spectrum in Bubble")
    ax_spectrum.set_xlabel("Energy (GeV)")
    ax_spectrum.set_ylabel("E² × Integrated f(E) [arb. units]")
    ax_spectrum.set_xlim(E_min.value, E_max.value)
    ax_spectrum.set_ylim(1e-3, 1e3)
    ax_spectrum.grid(True, alpha=0.3)
    ax_spectrum.legend(loc="upper right", fontsize=8)

    # Setup normalized integrated spectrum plot (8th subplot)
    ax_normalized = axes[7]
    (normalized_spectrum_line,) = ax_normalized.semilogx(
        [], [], label="Normalized Spectrum", alpha=0.7, linewidth=2, color="C1"
    )
    (normalized_theoretical_line,) = ax_normalized.semilogx(
        [], [], "k--", lw=1.5, label="Normalized Giovanni"
    )

    ax_normalized.set_title("Normalized Integrated Spectrum")
    ax_normalized.set_xlabel("Energy (GeV)")
    ax_normalized.set_ylabel("Integrated f(E) / f(E) at TS")
    ax_normalized.set_xlim(E_min.value, E_max.value)
    ax_normalized.set_ylim(0, 1)
    ax_normalized.grid(True, alpha=0.3)
    ax_normalized.legend(loc="upper right", fontsize=8)

    # Time text
    time_text = fig.text(
        0.5, 0.96, "", ha="center", va="top", fontsize=12, weight="bold"
    )

    plt.suptitle(
        "Giovanni Model: Real-Time Evolution\nAdvection-Diffusion-Loss-Source", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Precompute volume weights and dp_dE
    dr = np.diff(r_grid)
    dr = np.append(dr, dr[-1])
    volume_weights = 4 * np.pi * r_grid**2 * dr
    dp_dE = (E_grid + const.m_p * const.c**2) / (p_grid * const.c**2)

    # Theoretical integrated spectrum for comparison
    integrated_theoretical = (
        np.sum(
            np.array(f_theoretical_list)[:, bubble_mask] * volume_weights[bubble_mask],
            axis=1,
        )
        * dp_dE.to("g*pc/(Myr*GeV)").value
        * 4
        * np.pi
        * (p_grid.to("g*pc/Myr").value) ** 2
    )

    ts_idx = np.where(r_grid >= R_TS.to("pc").value)[0][0]
    ts_idx = min(ts_idx, len(r_grid) - 1)

    f_at_TS_theo = (
        np.array(f_theoretical_list)[:, ts_idx]
        * dp_dE.to("g*pc/(Myr*GeV)").value
        * 4
        * np.pi
        * (p_grid.to("g*pc/Myr").value) ** 2
    )
    integrated_theoretical_normalized = integrated_theoretical / f_at_TS_theo
    integrated_theoretical_normalized /= 4 / 3 * np.pi * R_b.to("pc").value ** 3

    injected_spectrum = (
        spectrum
        * dp_dE.to("g*pc/(Myr*GeV)").value
        * 4
        * np.pi
        * (p_grid.to("g*pc/Myr").value) ** 2
    )

    # Animation parameters for 60fps and 30 seconds
    fps = 30
    duration_seconds = 30
    total_frames = fps * duration_seconds  # 1800 frames

    # Calculate update interval to distribute frames across simulation time
    update_interval = max(1, num_timesteps // total_frames)

    logger.info(f"Animation parameters:")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Duration: {duration_seconds} seconds")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Update interval: {update_interval} timesteps per frame")

    def update_frame(frame):
        """Update function for animation."""
        # Advance solver
        solver.step(update_interval)

        current_step = min((frame + 1) * update_interval, num_timesteps - 1)
        current_time = t_grid[current_step]

        # Get current state
        f_current = solver.state.f.copy()

        # Update spatial profile lines
        for i, (line, e_idx) in enumerate(zip(lines, energy_indices)):
            ts_level = f_current[e_idx, ts_idx] if f_current[e_idx, ts_idx] > 0 else 1.0
            normalized_curve = f_current[e_idx, :] / ts_level
            line.set_data(r_grid, normalized_curve)

        # Calculate integrated spectrum
        integrated_spectrum = (
            np.sum(f_current[:, bubble_mask] * volume_weights[bubble_mask], axis=1)
            * dp_dE.to("g*pc/(Myr*GeV)").value
            * 4
            * np.pi
            * (p_grid.to("g*pc/Myr").value) ** 2
        )

        corrected_spectrum = integrated_spectrum / tau_permanence

        f_at_TS = (
            f_current[:, ts_idx]
            * dp_dE.to("g*pc/(Myr*GeV)").value
            * 4
            * np.pi
            * (p_grid.to("g*pc/Myr").value) ** 2
        )
        integrated_spectrum_normalized = integrated_spectrum / f_at_TS
        integrated_spectrum_normalized /= 4 / 3 * np.pi * R_b.to("pc").value ** 3

        # Update spectrum plots
        spectrum_line.set_data(E_grid.value, E_grid.value**2 * integrated_spectrum)
        theoretical_line.set_data(
            E_grid.value, E_grid.value**2 * integrated_theoretical
        )
        corrected_spectrum_line.set_data(
            E_grid.value, E_grid.value**2 * corrected_spectrum
        )
        injected_spectrum_line.set_data(
            E_grid.value, E_grid.value**2 * injected_spectrum * 10000000
        )
        normalized_spectrum_line.set_data(E_grid.value, integrated_spectrum_normalized)
        normalized_theoretical_line.set_data(
            E_grid.value, integrated_theoretical_normalized
        )

        # Auto-scale y-axis for spectrum if needed
        if np.any(integrated_spectrum > 0):
            maxlim = max(
                E_grid[corrected_spectrum > 0].value ** 2
                * corrected_spectrum[corrected_spectrum > 0]
            )
            minlim = min(
                E_grid[integrated_spectrum > 0].value ** 2
                * integrated_spectrum[integrated_spectrum > 0]
            )
            ax_spectrum.set_ylim(minlim * 0.5, maxlim * 1000)

        # Update time text
        time_text.set_text(f"Time: {current_time:.4f} Myr")

        # Log progress every 5% of frames
        if frame % (total_frames // 20) == 0:
            progress = (frame / total_frames) * 100
            logger.info(
                f"Progress: {progress:.1f}% (Frame {frame}/{total_frames}, Time: {current_time:.4f} Myr)"
            )

        return lines + [
            spectrum_line,
            corrected_spectrum_line,
            injected_spectrum_line,
            theoretical_line,
            normalized_spectrum_line,
            normalized_theoretical_line,
            time_text,
        ]

    # Create animation
    logger.info("Creating animation...")
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=total_frames,
        interval=1000 / fps,  # milliseconds between frames
        blit=False,
        repeat=False,
    )

    # Save animation as MP4
    output_filename = "giovanni_losses_simulation_60fps_30s_v2.mp4"
    logger.info(
        f"Saving {duration_seconds}s animation at {fps} fps to {output_filename}..."
    )

    writer = FFMpegWriter(
        fps=fps,
        metadata=dict(artist="SolverAlpha"),
        bitrate=3600,  # Higher bitrate for better quality at 60fps
        codec="h264",
    )
    anim.save(output_filename, writer=writer, dpi=400)

    logger.info(f"✓ Animation saved to {output_filename}")
    logger.info(f"  Duration: {duration_seconds} seconds")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info("Simulation complete!")

    # Save final frame as image
    fig.savefig("giovanni_losses_realtime_video_final.png", dpi=400)
    logger.info("✓ Saved final state to giovanni_losses_realtime_video_final.png")

    plt.close()


if __name__ == "__main__":
    run_giovanni_losses_realtime_video()
