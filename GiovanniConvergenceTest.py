import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import astropy.units as u
import astropy.constants as const
import logging

logging.basicConfig(
    level=logging.DEBUG,
    # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    format="[%(levelname)s] %(name)s: %(message)s",
)
# Suppress matplotlib chatter
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.ticker").setLevel(logging.WARNING)

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Solver import Solver


########################
def compute_theoretical_profile_overrefined(
    R_TS,
    R_b,
    v_w,
    D_b_pc2Myr,
    D_out_pc2Myr,
    f_gal,
    f_TS,
    r_0_pc,
    r_end_pc,
    n_refine=20000,
):
    """Compute a high-resolution (grid-independent) theoretical profile.

    Returns
    -------
    r_plot : ndarray
        Concatenated radial coordinates (bubble + ISM) in pc (excluding wind region < R_TS).
    f_plot : ndarray
        Dimensionless theoretical profile f/f_TS on r_plot.
    f_b_boundary : float
        The value f/f_TS at the bubble boundary R_b (used for analytic extension inside ISM for point sampling).
    """
    R_TS_pc = R_TS.to("pc").value
    R_b_pc = R_b.to("pc").value
    r_hi = np.linspace(r_0_pc, r_end_pc, n_refine)
    mask_bubble = (r_hi >= R_TS_pc) & (r_hi <= R_b_pc)
    mask_ISM = r_hi > R_b_pc

    # Flow speed inside shocked bubble
    v_b_pcMyr = (v_w.to("pc/Myr") / 4).value

    # Alpha(r)
    alpha = (v_b_pcMyr * R_TS_pc / D_b_pc2Myr) * (1.0 - R_TS_pc / r_hi[mask_bubble])
    alpha_b = (v_b_pcMyr * R_TS_pc / D_b_pc2Myr) * (1.0 - R_TS_pc / R_b_pc)

    # Beta
    beta = (D_out_pc2Myr * R_b_pc) / (v_b_pcMyr * R_TS_pc**2)

    numerator = (np.exp(alpha) + beta * (np.exp(alpha_b) - np.exp(alpha))) + (
        f_gal / f_TS
    ) * beta * (np.exp(alpha) - 1.0)
    denominator = 1.0 + beta * (np.exp(alpha_b) - 1.0)
    f_b_over_ts = numerator / denominator

    # Outside bubble (ISM)
    f_out_over_ts = f_b_over_ts[-1] * (R_b_pc / r_hi[mask_ISM]) + (f_gal / f_TS) * (
        1.0 - R_TS_pc / r_hi[mask_ISM]
    )

    r_plot = np.concatenate([r_hi[mask_bubble], r_hi[mask_ISM]])
    f_plot = np.concatenate([f_b_over_ts, f_out_over_ts])
    return r_plot, f_plot, f_b_over_ts[-1]


# ======== Definitions =======
Mp = const.m_p
c = const.c
E = lambda Ek: Ek + Mp * c**2
p = lambda Ek: np.sqrt((E(Ek) ** 2 - (Mp * c**2) ** 2) / c**2)
LorenzBeta = lambda Ek: np.sqrt(1 - (Mp * c**2 / E(Ek)) ** 2)

# =========== Cosmic ray sea ========================
Kcr = (
    0.4544
    * 10**-4
    / (45**2 * const.c.value)
    * (u.cm * u.GeV / const.c.to("cm s-1")) ** -3
    / 100
)
f_sea = (
    4
    * np.pi
    / c.to("cm s-1")
    * (1 * u.GeV / c.to("cm s-1")) ** 2
    * Kcr
    * (LorenzBeta(1 * u.GeV) ** -1)
    * (1 * u.GeV / (45 * u.GeV)) ** -4.85
    * (1 + (1 * u.GeV / (336 * u.GeV)) ** 5.54) ** 0.024
)
C = 1.882 * 10**-9 * (u.eV**-1 * u.cm**-2 * u.s**-1 * u.sr**-1)
# Jsea_Voy=C*(E(Eax)/(1*u.MeV))**0.129*(1+E(Eax)/(624.5*u.MeV))**-2.829
Jsea_Voy = (
    C * (1 * u.GeV / (1 * u.MeV)) ** 0.129 * (1 + 1 * u.GeV / (624.5 * u.MeV)) ** -2.829
).to("eV-1 cm-2 s-1 sr-1")
fsea_Voy = Jsea_Voy * (4 * np.pi * u.sr) / c
f_sea_Mix = f_sea * (1 * u.GeV > 90 * u.GeV) + fsea_Voy * (1 * u.GeV < 90 * u.GeV)

f_end = f_sea_Mix.value

f_end = 0


########################

# Parameters
r_0 = 0.0 * u.pc
r_Inj = 1.0 * u.pc  # parsec
r_end = 500.0 * u.pc  # parsec
eta_B = 0.1  # Magnetic field efficiency
L_wind = 1e38 * u.erg / u.s  # erg/s
M_dot = 1e-4 * const.M_sun / u.yr
rho_0 = const.m_p / u.cm**3
t_b = 1 * u.Myr
eta_inj = 0.1
v_w = np.sqrt(2 * L_wind / M_dot)
t_end = 1.2 * u.Myr
p_chosen = (1 * u.GeV / const.c).to("cm*g/s")
# convert p_chosen to velocity using relativistic formula
v_p = (
    p_chosen
    * const.c**2
    / np.sqrt((p_chosen * const.c) ** 2 + (const.m_p * const.c**2) ** 2)
)

R_b = (
    (250 / (308 * math.pi)) ** (1 / 5)
    * L_wind ** (1 / 5)
    * rho_0 ** (-1 / 5)
    * t_b ** (3 / 5)
)

R_TS = (
    np.sqrt((3850 * math.pi) ** (2 / 5) / (28 * math.pi) * M_dot * v_w)
    * L_wind ** (-1 / 5)
    * rho_0 ** (-3 / 10)
    * t_b ** (2 / 5)
)
rho_w = 3 * M_dot / (4 * math.pi * (R_TS**2) * v_w)

# print(f"R_TS: {R_TS.to('pc')}, R_b: {R_b.to('pc')}")

num_points_list = [700, 1000, 2000, 5000, 10000]
t_points_list = [1000, 2000, 4000, 15000, 60000]
error = []

# --- Precompute grid-independent theoretical profile (uses constant D_b & D_out) ---
# Magnetic field in bubble (same formula as inside loop, independent of r discretization)
delta_B_bubble = (
    np.sqrt(11)
    / R_TS.to("cm").value
    * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
) * u.G
r_L_bubble = (
    (const.c.cgs * p_chosen).to("erg").value
    / (const.e.esu.value * delta_B_bubble.to("G").value)
) * u.cm
D_bubble = 1 / 3 * v_p * r_L_bubble ** (1 / 3) * r_Inj ** (2 / 3)
D_out = 3e28 * u.cm**2 / u.s

D_bubble_pc2Myr = D_bubble.to("pc**2/Myr").value
D_out_pc2Myr = D_out.to("pc**2/Myr").value

r_theo_plot, f_theo_plot, f_b_boundary_value = compute_theoretical_profile_overrefined(
    R_TS,
    R_b,
    v_w,
    D_bubble_pc2Myr,
    D_out_pc2Myr,
    0.0,  # f_gal
    1.0,  # f_TS
    r_0.value,
    r_end.value,
    n_refine=20000,
)
R_TS_pc = R_TS.to("pc").value
R_b_pc = R_b.to("pc").value

for num_points in num_points_list:
    # Spatial and temporal grids
    r = np.linspace(r_0.value, r_end.value, num_points)

    r_wind = r < R_TS.to("pc").value
    r_buble = (r >= R_TS.to("pc").value) & (r <= R_b.to("pc").value)
    r_ISM = r > R_b.to("pc").value

    delta_B = np.zeros_like(r) * u.G  # Magnetic field strength in Gauss
    delta_B[r_wind] = (
        1
        / ((r[r_wind] * u.pc).to("cm").value)
        * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
    ) * u.G
    delta_B[0] = (
        1
        / ((r[1] * u.pc).to("cm").value)
        * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
    ) * u.G
    delta_B[r_buble] = (
        np.sqrt(11)
        / R_TS.to("cm").value
        * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
    ) * u.G

    r_L = np.zeros_like(r) * u.cm
    r_L[r_wind] = (
        (const.c.cgs * p_chosen).to("erg").value
        / (const.e.esu.value * delta_B[r_wind].to("G").value)
    ) * u.cm
    r_L[r_buble] = (
        (const.c.cgs * p_chosen).to("erg").value
        / (const.e.esu.value * delta_B[r_buble].to("G").value)
    ) * u.cm
    r_L[r_ISM] = 0 * u.cm

    t_steps = t_points_list[num_points_list.index(num_points)]
    t_grid = np.linspace(0, t_end.value, t_steps)

    # Initial profile: zero everywhere, but the end, where small gausssian until f_end
    f_values = np.zeros(num_points)

    # Velocity profile: inside TS, v(r) = v_w, in bubble, v(r) = v_w/4*(R_TS/r)**2, outside bubble, v(r) = 0
    v_field = np.zeros_like(r)
    v_field[r_wind] = v_w.to("pc/Myr").value
    v_field[r_buble] = (
        v_w.to("pc/Myr").value / 4 * (R_TS.to("pc").value / r[r_buble]) ** 2
    )
    plt.plot(r, v_field, label="Velocity field v(r)")
    # Caracteristic advection time
    print(
        f"Characteristic advection time: {(R_TS*4/(3*v_w)*((R_b/R_TS)**3-1)).to('kyr').value} kyr"
    )

    # Diffusion coefficient:
    D_values = 1 / 3 * v_p * r_L ** (1 / 3) * r_Inj ** (2 / 3)
    # D_values = 1 / 3 * v_p * np.sqrt(r_L * r_Inj)
    # D_values = 1 / 3 * v_p * r_L
    D_values[r_ISM] = 3e28 * u.cm**2 / u.s  # constant diffusion in ISM
    plt.plot(r, D_values.to("pc**2/Myr").value, label="Diffusion Coefficient D(r)")
    # Caracteristic diffusion time
    print(
        f"Characteristic diffusion time: {((R_b - R_TS) ** 2 / (4 * D_values[int(num_points/2)])).to('kyr').value} kyr"
    )

    # Source term: constant injection at termination shock
    Q = np.zeros_like(r)
    Q[(r >= 0.99 * R_TS.to("pc").value) & (r <= 1.01 * R_TS.to("pc").value)] = (
        eta_inj
        * rho_w
        * v_w
        / (4 * math.pi * p_chosen**2)
        * u.pc ** (-1)
        * (u.GeV / const.c) ** (-1)
    )
    Q[Q != 0] = 1000
    # plt.plot(r, Q, label="Source term Q(r)")

    # Prepare solver parameters
    # For advectionFV we pass v_centers via advectionFV_params
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
        "f_end": f_end,
    }
    source_params = {"Q_values": Q}
    op_params = {
        "advectionFV": advectionFV_params,
        "diffusionFV": diffusion_params,
        "source": source_params,
    }
    # Operator splitting: advectionFV then diffusion
    # Important: pass f_values as centers for advectionFV (length M-1)

    solver = Solver(
        x_grid=r,
        t_grid=t_grid,
        f_values=f_values,
        problem_type="advectionFV-diffusionFV",
        operator_params=op_params,
        substeps={"advectionFV": 1, "diffusionFV": 1},
    )
    import matplotlib as mpl
    from datetime import datetime

    # Batch mode: sample 20 curves (including first and last), overlay Giovanni model,
    # and save to ./GiovanniConv with unique filenames including N.
    num_timesteps = len(t_grid) - 1
    sample_count = 20
    sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
    sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))

    stored_curves = []  # normalized curves
    stored_times = []

    current_step = 0
    f_current = solver.f_values
    # Save initial curve
    ts_level = f_current[r_buble][10] if np.any(r_buble) else 1.0
    stored_curves.append(f_current / ts_level)
    stored_times.append(t_grid[0])

    for next_sample in sample_indices[1:]:
        steps_to_advance = int(next_sample - current_step)
        if steps_to_advance > 0:
            f_current = solver.step(steps_to_advance)
            current_step = next_sample
        else:
            f_current = solver.f_values

        ts_level = f_current[r_buble][10] if np.any(r_buble) else 1.0
        stored_curves.append(f_current / ts_level)
        stored_times.append(t_grid[current_step])

    # Plot all stored curves together
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(stored_curves)))
    for idx, (tval, curve) in enumerate(zip(stored_times, stored_curves)):
        label = f"t={tval:.3f} Myr" if idx in (0, len(stored_curves) - 1) else None
        ax.semilogy(r, curve, color=colors[idx], linestyle="-", label=label)

    # Add Giovanni theoretical profile (dimensionless f/f_TS)
    ax.semilogy(
        r_theo_plot,
        f_theo_plot,
        "k--",
        linewidth=2,
        label="Theoretical (20000 pts)",
    )

    ax.set_xlabel("$r$ (pc)")
    ax.set_ylabel("$f(t, r)$ / $f_{TS}$")
    ax.set_xlim(0, r_end.value)
    ax.set_ylim(1e-6, 1e1)
    ax.grid(True)

    # Vertical lines for boundaries
    # (R_TS_pc & R_b_pc precomputed outside loop)
    ax.axvline(
        R_TS_pc, color="r", linestyle="--", linewidth=1.5, label="Termination shock"
    )
    ax.axvline(
        R_b_pc, color="g", linestyle="--", linewidth=1.5, label="Bubble boundary"
    )

    # Colorbar for time
    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.rainbow,
        norm=plt.Normalize(vmin=min(stored_times), vmax=max(stored_times)),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$t$ (Myr)")

    # Title and legend include N
    title = f"Giovanni convergence (N={num_points} points)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    # Markers at the last bubble radius comparing theory vs last numerical curve
    if np.any(r_ISM):
        idx_r_ism_first = np.where(r_ISM)[0][0]
        r_ism_first = r[idx_r_ism_first]
        # Theoretical value at this ISM radius using analytic extension from bubble boundary
        y_theo_ism = f_b_boundary_value * (R_b_pc / r_ism_first)  # f_gal = 0
        y_num_ism = stored_curves[-1][idx_r_ism_first]
        ax.plot(r_ism_first, y_theo_ism, "ko", markersize=7, label="Theo @ first ISM")
        ax.plot(r_ism_first, y_num_ism, "ro", markersize=7, label="Num @ first ISM")
        error.append(np.abs(y_theo_ism - y_num_ism))
    else:
        error.append(np.nan)

    plt.tight_layout()

    # Save without overwriting into GiovanniConv folder
    out_dir = os.path.join(os.path.dirname(__file__), "GiovanniConv")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"GiovanniConv_N{num_points}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot: {out_path}")

    plt.close(fig)


plt.figure()
plt.loglog(num_points_list, error, marker="o")
plt.xlabel("Number of Points")
plt.ylabel("Error")
plt.title("Convergence Study")
# Save error plot
out_dir = os.path.join(os.path.dirname(__file__), "GiovanniConv")
os.makedirs(out_dir, exist_ok=True)
fname = f"GiovanniConv_Error.png"
out_path = os.path.join(out_dir, fname)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved error plot: {out_path}")
plt.close()
