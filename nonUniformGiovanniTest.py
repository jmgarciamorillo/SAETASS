import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import astropy.units as u
import astropy.constants as const
import logging
from State import State
from Grid import Grid

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
def get_theoretical_profile(r, r_bubble, r_ISM, v_w, D_values, R_TS, R_b, f_gal, f_TS):

    # masks
    r_bubble_teo = r <= R_b.to("pc").value
    r_ISM_teo = r > R_b.to("pc").value

    # Ensure correct units for calculations
    v_b = v_w.to("pc/Myr") / 4
    D_b = D_values[r_bubble][int(sum(r_bubble) // 2)].to("pc**2/Myr")
    D_out = D_values[r_ISM][0].to("pc**2/Myr")

    # α(r,p)
    alpha = (v_b * R_TS / D_b) * (1.0 - R_TS / (r[r_bubble_teo] * u.pc))

    # α_b = α(r=R_b,p)
    alpha_b = (v_b * R_TS / D_b) * (1.0 - R_TS / R_b)

    # β(p)
    beta = (D_out * R_b) / (v_b * R_TS**2)

    # f_b(r,p) / f_TS
    numerator = (
        np.exp(alpha) + beta * (np.exp(alpha_b) - np.exp(alpha))
    ) + f_gal / f_TS * beta * (np.exp(alpha) - 1.0)
    denominator = 1.0 + beta * (np.exp(alpha_b) - 1.0)
    f_b_over_ts = numerator / denominator

    f_b_over_ts_RB = (
        (np.exp(alpha_b)) + f_gal / f_TS * beta * (np.exp(alpha_b) - 1.0)
    ) / denominator

    # f_out(r,p) / f_TS
    f_out_over_ts = f_b_over_ts_RB * (R_b / (r[r_ISM_teo] * u.pc)) + f_gal / f_TS * (
        1.0 - R_TS / (r[r_ISM_teo] * u.pc)
    )

    # Concatenate the profiles
    f_b = f_b_over_ts
    f_out = f_out_over_ts

    return np.append(f_b, f_out)


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

f_end = 3.8682e-7


########################

# Parameters
r_0 = 0.0 * u.pc
r_Inj = 1.0 * u.pc  # parsec
r_end = 500.0 * u.pc  # parsec
num_points = 3000
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

# Spatial and temporal grids
r = np.linspace(r_0.value, r_end.value, num_points)
# include exact R_TS and R_b if possible
if R_TS.to("pc").value > r_0.value and R_TS.to("pc").value < r_end.value:
    r = np.append(r, R_TS.to("pc").value)
if R_b.to("pc").value > r_0.value and R_b.to("pc").value < r_end.value:
    r = np.append(r, R_b.to("pc").value)
r = np.unique(np.sort(r))
num_points = len(r)

"""
# Spatial grid: three concatenated linspaces with a dense middle around R_b
R_b_pc = R_b.to("pc").value
r0_pc = r_0.to("pc").value
r_end_pc = r_end.to("pc").value

# Choose a middle-region width around R_b (clipped to reasonable bounds)
# mid_width = min(max(0.02 * r_end_pc, 0.1 * R_b_pc), 0.2 * r_end_pc)
mid_width = 1  # pc
mid_start = max(r0_pc, R_b_pc - mid_width / 2.0)
mid_end = min(r_end_pc, R_b_pc + mid_width / 2.0)

# Allocate points among the three segments (ensure at least 2 points per side, >=3 for middle)
n_mid = max(3, int(num_points * 0.3))
n_side = max(2, (num_points - n_mid) // 2)
n1 = n_side
n3 = num_points - n_mid - n1
if n3 < 2:
    # Fix rounding edge cases
    n3 = 2
    n1 = num_points - n_mid - n3

# Build the three segments, avoid duplicating segment boundaries
r1 = np.linspace(r0_pc, mid_start, n1, endpoint=False)
r2 = np.linspace(mid_start, mid_end, n_mid, endpoint=False)
r3 = np.linspace(mid_end, r_end_pc, n3)

r = np.concatenate([r1, r2, r3])"""


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


t_steps = 8000
t_grid = np.linspace(0, t_end.value, t_steps)

# Initial profile: zero everywhere, but the end, where small gausssian until f_end
f_values = np.zeros(num_points)


# Velocity profile: inside TS, v(r) = v_w, in bubble, v(r) = v_w/4*(R_TS/r)**2, outside bubble, v(r) = 0
v_field = np.zeros_like(r)
v_field[r_wind] = v_w.to("pc/Myr").value
v_field[r_buble] = v_w.to("pc/Myr").value / 4 * (R_TS.to("pc").value / r[r_buble]) ** 2
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

grid = Grid(r_centers=r, t_grid=t_grid, p_centers=None)

solver = Solver(
    grid=grid,
    state=State(f_values),
    problem_type="advectionFV-diffusionFV",
    operator_params=op_params,
    substeps={"advectionFV": 1, "diffusionFV": 1},
)
import matplotlib as mpl

# Toggle: True -> live plotting each step; False -> store 20 curves and plot once at end
plot_in_runtime = False

num_timesteps = len(t_grid) - 1

r_gio = np.linspace(R_TS.to("pc").value, r_end.value, 50000)
if plot_in_runtime:
    # Dense live plotting (runtime)
    num_curves = min(4000, max(2, num_timesteps))
    indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
    indices = np.unique(
        np.append(indices, num_timesteps)
    )  # Ensure last step is included

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.semilogy(r, f_values, color="b")
    ax.set_xlabel("$r$ (pc)")
    ax.set_ylabel("$f(t, r)$")

    ax.set_title(f"Giovanni model test ($t$=0 Myr)")
    ax.set_xlim(0, r_end.value)
    ax.set_ylim(1e-6, 1e1)
    ax.grid(True)
    fig.tight_layout()

    # Draw dashed vertical lines for the injection zone
    R_TS_pc = R_TS.to("pc").value
    R_b_pc = R_b.to("pc").value
    ax.axvline(
        0.99 * R_TS_pc, color="k", linestyle="--", linewidth=1, label="Injection zone"
    )
    ax.axvline(1.01 * R_TS_pc, color="k", linestyle="--", linewidth=1)

    # Additional vertical lines for termination shock and bubble
    ax.axvline(
        R_TS_pc, color="r", linestyle="--", linewidth=1.5, label="Termination shock"
    )
    ax.axvline(
        R_b_pc, color="g", linestyle="--", linewidth=1.5, label="Bubble boundary"
    )

    ax.legend(loc="upper right")

    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.rainbow,
        norm=plt.Normalize(vmin=t_grid[0], vmax=t_grid[-1]),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$t$ (Myr)")

    current_step = 0
    for next_plot_step in indices:
        steps_to_advance = int(next_plot_step - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_plot_step

        f_current = solver.state.f.copy()
        # Normalize by instantaneous TS level (no theoretical curve)
        ts_level = f_current[r_buble][10] if np.any(r_buble) else 1.0
        line.set_ydata(f_current / ts_level)
        ax.set_title(f"Giovanni model test ($t$={t_grid[int(current_step)]:.4f} Myr)")
        plt.pause(0.05)

    plt.ioff()
    plt.show()
else:
    # Batch mode: store 20 curves (including first and last) and plot once at the end
    sample_count = 20
    sample_indices = np.linspace(0, num_timesteps, sample_count, dtype=int)
    sample_indices = np.unique(np.append(sample_indices, [0, num_timesteps]))

    stored_curves = []  # normalized curves
    stored_times = []

    current_step = 0
    f_current = solver.state.f.copy()
    # Save initial curve
    ts_level = f_current[r_buble][10] if np.any(r_buble) else 1.0
    stored_curves.append(f_current / ts_level)
    stored_times.append(t_grid[0])

    for next_sample in sample_indices[1:]:
        steps_to_advance = int(next_sample - current_step)
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = next_sample

        f_current = solver.state.f.copy()
        ts_level = f_current[r_buble][10] if np.any(r_buble) else 1.0
        stored_curves.append(f_current / ts_level)
        stored_times.append(t_grid[current_step])

    # Plot all stored curves together
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(stored_curves)))
    for idx, (tval, curve) in enumerate(zip(stored_times, stored_curves)):
        label = f"t={tval:.3f} Myr" if idx in (0, len(stored_curves) - 1) else None
        ax.semilogy(r, curve, color=colors[idx], linestyle="-")

    # Add Giovanni theoretical profile (dimensionless f/f_TS)
    f_gio_profile = get_theoretical_profile(
        r_gio, r_buble, r_ISM, v_w, D_values, R_TS, R_b, 0, 1.0
    )
    ax.semilogy(
        r_gio,
        f_gio_profile,
        "k--",
        linewidth=2,
        label="Theoretical",
    )

    ax.set_xlabel("$r$ (pc)")
    ax.set_ylabel("$f(t, r)$ / $f_{TS}$")
    ax.set_title("Giovanni model test")
    ax.set_xlim(0, r_end.value)
    ax.set_ylim(1e-6, 1e1)
    ax.grid(True)

    # Vertical lines for boundaries
    R_TS_pc = R_TS.to("pc").value
    R_b_pc = R_b.to("pc").value
    # ax.axvline(
    #    0.99 * R_TS_pc, color="k", linestyle="--", linewidth=1, label="Injection zone"
    # )
    # ax.axvline(1.01 * R_TS_pc, color="k", linestyle="--", linewidth=1)
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

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
