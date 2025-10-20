import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import sys
import os
import logging
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the giovanni_profiles module
from giovanni_profiles import (
    create_giovanni_setup,
    get_theoretical_profile,
    get_cosmic_ray_sea,
    get_larmor_radius,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)


def test1():
    """Test and visualize Giovanni profiles."""

    # Set figure style
    plt.style.use("default")
    plt.rcParams.update({"font.size": 12})

    # 1. Create a Giovanni model setup with default parameters
    print("Creating Giovanni model with default parameters...")

    # Specify particle kinetic energy (1 GeV)
    E_k = 1 * u.GeV

    # Create the model (r_end reduced for clearer visualization)
    r_end = 300.0 * u.pc
    t_b = 1 * u.Myr
    setup = create_giovanni_setup(
        r_0=0.0 * u.pc,
        r_end=r_end,
        num_points=2000,
        L_wind=1e38 * u.erg / u.s,
        M_dot=1e-4 * const.M_sun / u.yr,
        rho_0=const.m_p / u.cm**3,
        t_b=t_b,
        eta_B=0.1,
        eta_inj=0.1,
        E_k=E_k,
        diffusion_model="kolmogorov",
    )

    # Extract key parameters and profiles
    r = setup["r"]
    R_TS = setup["R_TS"]
    R_b = setup["R_b"]
    v_field = setup["v_field"]
    D_values = setup["D_values"]
    delta_B = setup["delta_B"]
    Q = setup["Q"]
    masks = setup["masks"]

    # Print key parameters
    print(f"Termination shock radius: R_TS = {R_TS.to('pc'):.3f}")
    print(f"Bubble radius: R_b = {R_b.to('pc'):.3f}")
    print(f"Wind velocity: v_w = {setup['v_w'].to('km/s'):.2f}")

    # 2. Compute the theoretical CR profile
    print("Computing theoretical CR profile...")

    # Get the theoretical profile (normalized to f_TS=1)
    f_theoretical = get_theoretical_profile(
        r,
        masks,
        setup["v_w"],
        setup["D_values"],
        R_TS,
        R_b,
        f_gal=0.00,  # No galactic background
        f_TS=1.0,  # Normalized to 1.0 at termination shock
    )

    # 3. Create visualization
    print("Creating visualizations...")

    # Create a figure with subplots arranged in a grid
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 2, figure=fig)

    # A. Velocity Profile
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r, v_field, "b-", lw=2)
    ax1.axvline(
        R_TS.to("pc").value,
        color="r",
        linestyle="--",
        lw=1.5,
        label="Termination Shock",
    )
    ax1.axvline(
        R_b.to("pc").value, color="g", linestyle="--", lw=1.5, label="Bubble Boundary"
    )
    ax1.set_xlabel("Radius (pc)")
    ax1.set_ylabel("Velocity (pc/Myr)")
    ax1.set_title("Giovanni Model: Velocity Field")
    ax1.set_xlim(0, min(200, r[-1]))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # B. Diffusion Coefficient
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(r, D_values.to("cm²/s").value, "g-", lw=2)
    ax2.axvline(R_TS.to("pc").value, color="r", linestyle="--", lw=1.5)
    ax2.axvline(R_b.to("pc").value, color="g", linestyle="--", lw=1.5)
    ax2.set_xlabel("Radius (pc)")
    ax2.set_ylabel("Diffusion Coefficient (cm²/s)")
    ax2.set_title("Diffusion Coefficient Profile")
    ax2.set_xlim(0, min(200, r[-1]))
    ax2.grid(True, alpha=0.3)

    # C. Magnetic Field
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.semilogy(r, delta_B.to("μG").value, "r-", lw=2)
    ax3.axvline(R_TS.to("pc").value, color="r", linestyle="--", lw=1.5)
    ax3.axvline(R_b.to("pc").value, color="g", linestyle="--", lw=1.5)
    ax3.set_xlabel("Radius (pc)")
    ax3.set_ylabel("Magnetic Field (μG)")
    ax3.set_title("Magnetic Field Profile")
    ax3.set_xlim(0, min(200, r[-1]))
    ax3.grid(True, alpha=0.3)

    # D. Source Term
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(r, Q, "orange", lw=2)
    ax4.axvline(R_TS.to("pc").value, color="r", linestyle="--", lw=1.5)
    ax4.axvline(R_b.to("pc").value, color="g", linestyle="--", lw=1.5)
    ax4.set_xlabel("Radius (pc)")
    ax4.set_ylabel("Source Intensity (arb. units)")
    ax4.set_title("CR Source Term (Injection at Shock)")
    ax4.set_xlim(0, min(200, r[-1]))
    ax4.grid(True, alpha=0.3)

    # E. Theoretical CR Profile
    ax5 = fig.add_subplot(gs[2, :])
    ax5.semilogy(r, f_theoretical, "b-", lw=2, label="Theoretical Profile")
    ax5.axvline(
        R_TS.to("pc").value,
        color="r",
        linestyle="--",
        lw=1.5,
        label="Termination Shock",
    )
    ax5.axvline(
        R_b.to("pc").value, color="g", linestyle="--", lw=1.5, label="Bubble Boundary"
    )

    # Add markers at key positions
    ax5.plot(R_TS.to("pc").value, 1.0, "ro", ms=8, label="f_TS (normalized)")
    # Find value at bubble boundary
    idx_rb = np.abs(r - R_b.to("pc").value).argmin()
    ax5.plot(
        r[idx_rb],
        f_theoretical[idx_rb],
        "go",
        ms=8,
        label=f"f_Rb ≈ {f_theoretical[idx_rb]:.3f}",
    )

    ax5.set_xlabel("Radius (pc)")
    ax5.set_ylabel("CR Density f(r)/f_TS")
    ax5.set_title("Giovanni Model: Theoretical CR Profile")
    ax5.set_xlim(0, min(200, r[-1]))
    ax5.set_ylim(0.001, 2)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Overall title with key parameters
    plt.suptitle(
        f"Giovanni Wind Bubble Model\n"
        f"R_TS = {R_TS.to('pc'):.1f}, R_b = {R_b.to('pc'):.1f}, "
        f"Age = {t_b.to('Myr'):.1f}, "
        f"E_CR = {E_k.to('GeV'):.1f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.9)

    # Save the figure
    plt.savefig("giovanni_profiles_test.png", dpi=150, bbox_inches="tight")
    print(f"Saved visualization to 'giovanni_profiles_test.png'")

    # Show the figure
    plt.show()

    # 4. Parameter exploration - create a simpler figure showing how R_b changes with age
    print("Exploring parameter dependencies...")

    ages = np.linspace(0.1, 5, 10) * u.Myr
    bubble_radii = []
    shock_radii = []

    for age in ages:
        params = create_giovanni_setup(t_b=age)
        bubble_radii.append(params["R_b"].to("pc").value)
        shock_radii.append(params["R_TS"].to("pc").value)

    plt.figure(figsize=(10, 6))
    plt.plot(ages.to("Myr").value, bubble_radii, "g-", lw=2, label="Bubble Radius")
    plt.plot(
        ages.to("Myr").value, shock_radii, "r-", lw=2, label="Termination Shock Radius"
    )
    plt.xlabel("Bubble Age (Myr)")
    plt.ylabel("Radius (pc)")
    plt.title("Giovanni Model: Evolution with Age")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig("giovanni_age_evolution.png", dpi=150)
    print(f"Saved age evolution to 'giovanni_age_evolution.png'")

    plt.show()


def test2():
    """
    Compare CR profiles for different diffusion models and particle energies.
    Creates a plot with 3 subfigures showing how diffusion models and particle
    momentum affect the CR distribution in a wind bubble.
    """
    # Set figure style
    plt.style.use("default")
    plt.rcParams.update({"font.size": 12})

    # Define particle energies to compare (in GeV)
    energies = [1, 100, 1000, 10000]

    # Define diffusion models with their colors
    diffusion_models = {
        "kolmogorov": {"name": "Kolmogorov", "color": "red"},
        "kraichnan": {"name": "Kraichnan", "color": "green"},
        "bohm": {"name": "Bohm", "color": "blue"},
    }

    # Define line styles for different energies
    line_styles = ["-", "--", "-.", ":"]

    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Common parameters for all models
    r_end = 150.0 * u.pc
    t_b = 1 * u.Myr

    # Loop through diffusion models
    for ax_idx, (diff_model, properties) in enumerate(diffusion_models.items()):
        ax = axs[ax_idx]
        base_color = properties["color"]

        # Set up a reference for the legend (will store first profile from each model)
        first_profile = None
        first_r = None
        first_R_TS = None
        first_R_b = None

        # Loop through energies
        for ener_idx, energy in enumerate(energies):
            # Calculate color intensity (get lighter with increasing energy)
            # Start with base color at 100% intensity, reduce to 30% for highest energy
            intensity = 1.0 - 0.7 * (ener_idx / (len(energies) - 1))

            # Create energy with units
            E_k = energy * u.GeV

            # Create Giovanni setup for this energy and diffusion model
            setup = create_giovanni_setup(
                r_0=0.0 * u.pc,
                r_end=r_end,
                num_points=7000,
                L_wind=1e38 * u.erg / u.s,
                M_dot=1e-4 * const.M_sun / u.yr,
                rho_0=const.m_p / u.cm**3,
                t_b=t_b,
                eta_B=0.1,
                eta_inj=0.1,
                E_k=E_k,
                diffusion_model=diff_model,
            )

            # Extract key parameters
            r = setup["r"]
            R_TS = setup["R_TS"]
            R_b = setup["R_b"]
            masks = setup["masks"]

            # Store first model's parameters for reference lines
            if ener_idx == 0:
                first_r = r
                first_R_TS = R_TS
                first_R_b = R_b

            # Compute theoretical profile
            f_theoretical = get_theoretical_profile(
                r,
                masks,
                setup["v_w"],
                setup["D_values"],
                R_TS,
                R_b,
                f_gal=0.00,
                f_TS=1.0,
            )

            # Plot profile with appropriate color and line style
            (line,) = ax.semilogy(
                r,
                f_theoretical,
                linestyle=line_styles[ener_idx],
                color=base_color,
                alpha=intensity,
                linewidth=2.5,
                label=f"{energy} GeV",
            )

            if ener_idx == 0:
                first_profile = line

        # Add vertical markers for shock and bubble boundary (using first model's values)
        ax.axvline(
            first_R_TS.to("pc").value,
            color="black",
            linestyle="--",
            linewidth=1,
            label="Termination Shock",
        )
        ax.axvline(
            first_R_b.to("pc").value,
            color="black",
            linestyle="-.",
            linewidth=1,
            label="Bubble Boundary",
        )

        # Configure subplot
        ax.set_xlabel("Radius (pc)")
        ax.set_xlim(0, 150)
        ax.set_ylim(1e-3, 2)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{properties['name']} Diffusion")

        # Add legend for the first subplot only
        if ax_idx == 0:
            ax.set_ylabel("CR Density f(r)/f_TS")

    # Add legend below the subplots
    handles = [
        plt.Line2D([0], [0], color="k", linestyle=ls, label=f"{ener} GeV")
        for ener, ls in zip(energies, line_styles)
    ]
    handles.append(
        plt.Line2D([0], [0], color="k", linestyle="--", label="Termination Shock")
    )
    handles.append(
        plt.Line2D([0], [0], color="k", linestyle="-.", label="Bubble Boundary")
    )

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=6)

    # Add overall title
    fig.suptitle(
        "CR Distribution for Different Diffusion Models and Particle Energies\n"
        f"Wind Bubble Age: {t_b.to('Myr'):.1f}",
        fontsize=14,
    )

    plt.tight_layout()

    # Save figure
    plt.savefig("giovanni_diffusion_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved diffusion model comparison to 'giovanni_diffusion_comparison.png'")

    plt.show()


def test3(diffusion_only=True):
    """
    Analyze how physical parameters scale with particle energy.

    This test computes key parameters from Giovanni models across a range of
    particle energies and diffusion models, showing how diffusion coefficients,
    timescales, and length scales depend on energy.

    Args:
        diffusion_only (bool): If True, create only the diffusion coefficient scaling plot.
    """
    # Set figure style
    plt.style.use("default")
    plt.rcParams.update({"font.size": 12})

    # Define energy range (logarithmically spaced)
    E_values = np.logspace(-2, 10, 40)  # 0.1 GeV to 10^6 GeV

    # Diffusion models to compare
    diffusion_models = {
        "kolmogorov": {"name": "Kolmogorov (δ = 1/3)", "color": "red", "marker": "o"},
        "kraichnan": {"name": "Kraichnan (δ = 1/2)", "color": "green", "marker": "s"},
        "bohm": {"name": "Bohm (δ = 1)", "color": "blue", "marker": "^"},
    }

    # Fixed model parameters
    L_wind = 1e38 * u.erg / u.s
    M_dot = 1e-4 * const.M_sun / u.yr
    rho_0 = const.m_p / u.cm**3
    t_b = 1 * u.Myr
    eta_B = 0.1

    # Create reference model for common parameters
    reference_model = create_giovanni_setup(
        L_wind=L_wind, M_dot=M_dot, rho_0=rho_0, t_b=t_b, eta_B=eta_B
    )

    R_TS = reference_model["R_TS"]
    R_b = reference_model["R_b"]
    v_w = reference_model["v_w"]

    print(f"Reference model: R_TS = {R_TS.to('pc'):.2f}, R_b = {R_b.to('pc'):.2f}")
    print(f"Wind velocity: v_w = {v_w.to('km/s'):.2f}")

    # Create figure - either just diffusion plot or full panel
    if diffusion_only:
        fig, ax1 = plt.subplots(figsize=(10, 8))
    else:
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Diffusion coefficient vs momentum
        ax1 = fig.add_subplot(gs[0, 0])

        # 2. Larmor radius vs momentum
        ax2 = fig.add_subplot(gs[0, 1])

        # 3. Escape time vs momentum
        ax3 = fig.add_subplot(gs[1, 0])

        # 4. Diffusion length scale vs momentum
        ax4 = fig.add_subplot(gs[1, 1])

        # 5. CR profile steepness vs momentum
        ax5 = fig.add_subplot(gs[2, 0])

        # 6. Dimensionless parameter (alpha_b) vs momentum
        ax6 = fig.add_subplot(gs[2, 1])

    # Results storage
    results = {
        model: {
            "E_values": E_values,
            "D_bubble": np.zeros(len(E_values)) * u.cm**2 / u.s,
        }
        for model in diffusion_models.keys()
    }

    if not diffusion_only:
        # Add additional result storage for full panel
        for model in diffusion_models.keys():
            results[model].update(
                {
                    "r_L": np.zeros(len(E_values)) * u.pc,
                    "escape_time": np.zeros(len(E_values)) * u.Myr,
                    "diff_length": np.zeros(len(E_values)) * u.pc,
                    "profile_slope": np.zeros(len(E_values)),
                    "alpha_b": np.zeros(len(E_values)),
                }
            )

    # Compute parameters for each energy and diffusion model
    for model_name in diffusion_models.keys():
        print(f"Computing {model_name} model parameters...")

        for i, E in enumerate(E_values):
            # Convert to momentum with units
            E_k = E * u.GeV

            # Create model for this momentum
            setup = create_giovanni_setup(
                r_0=0.0 * u.pc,
                r_end=200.0 * u.pc,
                num_points=2000,  # Low resolution for speed
                L_wind=L_wind,
                M_dot=M_dot,
                rho_0=rho_0,
                t_b=t_b,
                eta_B=eta_B,
                E_k=E_k,
                diffusion_model=model_name,
            )

            # Extract diffusion coefficient inside bubble (mid-point)
            bubble_mask = setup["masks"]["r_bubble"]
            if np.any(bubble_mask):
                D_bubble = setup["D_values"][bubble_mask][np.sum(bubble_mask) // 2]
            else:
                D_bubble = setup["D_values"][0]  # Fallback

            # Store the diffusion coefficient
            results[model_name]["D_bubble"][i] = D_bubble

            # Skip the rest if we're only computing diffusion
            if diffusion_only:
                continue

            # Extract additional parameters for full panel...
            # (all the other calculations for the full panel view)
            r_L = setup["r_L"][0]
            escape_time = R_b**2 / D_bubble
            diff_length = np.sqrt(D_bubble * t_b)
            v_b = setup["v_w"] / 4
            alpha_b = (v_b * R_TS / D_bubble) * (1.0 - R_TS / R_b)

            # Calculate profile slope...
            r = setup["r"]
            masks = setup["masks"]
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
            idx_rb = np.abs(r - R_b.to("pc").value).argmin()
            if idx_rb > 0 and idx_rb < len(r) - 1:
                dr = r[idx_rb + 1] - r[idx_rb - 1]
                dlogf = np.log10(f_theoretical[idx_rb + 1]) - np.log10(
                    f_theoretical[idx_rb - 1]
                )
                slope = dlogf / dr if dr > 0 else 0
            else:
                slope = 0

            # Store results for full panel
            results[model_name]["r_L"][i] = r_L
            results[model_name]["escape_time"][i] = escape_time
            results[model_name]["diff_length"][i] = diff_length
            results[model_name]["profile_slope"][i] = slope
            results[model_name]["alpha_b"][i] = alpha_b

    # Plot diffusion coefficient results
    for model_name, props in diffusion_models.items():
        color = props["color"]
        marker = props["marker"]
        label = props["name"]

        E_ax = results[model_name]["E_values"]
        D = results[model_name]["D_bubble"].to("cm**2/s").value

        # Plot diffusion coefficient vs energy
        if diffusion_only:
            # For the dedicated plot, make larger markers and lines
            ax1.loglog(
                E_ax,
                D,
                marker=marker,
                color=color,
                ms=8,
                lw=2.5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=label,
            )
        else:
            # For the panel plot, keep it more compact
            ax1.loglog(E_ax, D, marker=marker, color=color, ms=5, label=label)

    # Configure diffusion plot
    ax1.set_title("Diffusion Coefficient Inside Bubble vs Particle Energy")
    ax1.set_xlabel("Energy (GeV)")
    ax1.set_ylabel("Diffusion Coefficient (cm²/s)")
    ax1.grid(True, alpha=0.3)

    if diffusion_only:
        # For the dedicated plot, add more details
        ax1.set_xlim(-2, 1e10)
        ax1.set_ylim(1e19, 1e33)

        # Add bubble characteristic size for reference
        # ax1.axhline(
        #     R_b.to("pc").value ** 2 / t_b.to("s").value,
        #     ls=":",
        #     color="gray",
        #     alpha=0.7,
        #     label=f"D_crit = R_b²/t_b",
        # )

        # Add legend in a good position
        ax1.legend(loc="upper left", fontsize=11)

        # Save figure
        plt.savefig("giovanni_diffusion_scaling.png", dpi=150, bbox_inches="tight")
        print(f"Saved diffusion scaling analysis to 'giovanni_diffusion_scaling.png'")

        plt.show()
        return

    # Continue with the full panel plotting if not diffusion_only
    for model_name, props in diffusion_models.items():
        color = props["color"]
        marker = props["marker"]
        label = props["name"]

        E_ax = results[model_name]["E_values"]
        r_L = results[model_name]["r_L"].to("pc").value
        t_esc = results[model_name]["escape_time"].to("Myr").value
        L_diff = results[model_name]["diff_length"].to("pc").value
        slope = results[model_name]["profile_slope"]
        alpha_b = results[model_name]["alpha_b"]

        # 2. Larmor radius vs energy
        ax2.loglog(E_ax, r_L, marker=marker, color=color, ms=5, label=label)

        # 3. Escape time vs energy
        ax3.loglog(E_ax, t_esc, marker=marker, color=color, ms=5, label=label)

        # 4. Diffusion length scale vs energy
        ax4.loglog(E_ax, L_diff, marker=marker, color=color, ms=5, label=label)

        # 5. CR profile slope vs energy
        ax5.semilogx(E_ax, slope, marker=marker, color=color, ms=5, label=label)

        # 6. Dimensionless parameter (alpha_b) vs energy
        ax6.loglog(E_ax, alpha_b, marker=marker, color=color, ms=5, label=label)

    # Larmor radius scaling (proportional to energy)
    E_ref = np.logspace(-1, 6, 3)
    scale_rL = results["kolmogorov"]["r_L"][np.abs(E_ax - 1).argmin()].value
    ax2.plot(E_ref, scale_rL * E_ref, color="gray", ls="-", alpha=0.5, label="∝ E")

    # Configure plots for the comprehensive view
    ax2.set_title("Larmor Radius vs Energy")
    ax2.set_xlabel("Energy (GeV)")
    ax2.set_ylabel("Larmor Radius (pc)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_title("Escape Time vs Energy")
    ax3.set_xlabel("Energy (GeV)")
    ax3.set_ylabel("Escape Time (Myr)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.set_title("Diffusion Length Scale vs Energy")
    ax4.set_xlabel("Energy (GeV)")
    ax4.set_ylabel("√(D·t) (pc)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5.set_title("CR Profile Slope at Bubble Boundary")
    ax5.set_xlabel("Energy (GeV)")
    ax5.set_ylabel("log(f) Slope")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6.set_title("Dimensionless Parameter α_b")
    ax6.set_xlabel("Energy (GeV)")
    ax6.set_ylabel("α_b = (v_b·R_TS/D)·(1-R_TS/R_b)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add horizontal line at alpha_b = 1 (transition point)
    ax6.axhline(1.0, color="black", ls="--", alpha=0.5)
    ax6.text(10, 1.2, "α_b = 1 (Advection = Diffusion)", fontsize=10)

    # Overall title
    plt.suptitle(
        "Energy Dependence of Transport Parameters in Giovanni Model\n"
        f"Wind Bubble: R_TS = {R_TS.to('pc'):.1f}, R_b = {R_b.to('pc'):.1f}, Age = {t_b.to('Myr'):.1f}",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.9)

    # Save figure
    plt.savefig("giovanni_energy_scaling.png", dpi=150, bbox_inches="tight")
    print(f"Saved energy scaling analysis to 'giovanni_energy_scaling.png'")

    plt.show()


if __name__ == "__main__":
    test2()
