import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import astropy.units as u
import astropy.constants as const
import logging
import matplotlib as mpl

from Solver import Solver

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Parameters from GiovanniTest.py
r_0 = 0.0 * u.pc
r_Inj = 1.0 * u.pc
r_end = 500.0 * u.pc
num_points = 2000
eta_B = 0.1
L_wind = 1e38 * u.erg / u.s
M_dot = 1e-4 * const.M_sun / u.yr
rho_0 = const.m_p / u.cm**3
t_b = 1 * u.Myr
eta_inj = 0.1
v_w = np.sqrt(2 * L_wind / M_dot)
t_end = 1 * u.Myr
p_chosen = (1 * u.GeV / const.c).to("cm*g/s")
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

# Spatial and temporal grids
r = np.linspace(r_0.value, r_end.value, num_points)
t_steps = 10000
t_grid = np.linspace(0, t_end.value, t_steps)

# Regions
r_wind = r < R_TS.to("pc").value
r_buble = (r >= R_TS.to("pc").value) & (r <= R_b.to("pc").value)
r_ISM = r > R_b.to("pc").value

# Magnetic field
delta_B = np.zeros_like(r) * u.G
delta_B[r_wind] = (
    1
    / ((r[r_wind] * u.pc).to("cm").value)
    * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
) * u.G
delta_B[0] = delta_B[1]
delta_B[r_buble] = (
    np.sqrt(11)
    / R_TS.to("cm").value
    * np.sqrt(0.5 * eta_B * M_dot.to("g/s").value * v_w.to("cm/s").value)
) * u.G

# Larmor radius
r_L = np.zeros_like(r) * u.cm
non_zero_B = delta_B > 0 * u.G
r_L[non_zero_B] = (
    (const.c.cgs * p_chosen).to("erg").value
    / (const.e.esu.value * delta_B[non_zero_B].to("G").value)
) * u.cm

# Initial profile
f_end_bc = 0.0001
f_initial = np.zeros(num_points)
f_initial += f_end_bc * np.exp(-((r - r_end.to("pc").value) ** 2) / 2)

# Velocity profile
v_field = np.zeros_like(r)
v_field[r_wind] = v_w.to("pc/Myr").value
v_field[r_buble] = v_w.to("pc/Myr").value / 4 * (R_TS.to("pc").value / r[r_buble]) ** 2

# Diffusion coefficient
D_values = 1 / 3 * v_p * np.sqrt(r_L * r_Inj)
D_values[r_ISM] = 10e2 * 3e28 * u.cm**2 / u.s
D_values_pc_Myr = D_values.to("pc**2/Myr").value

# Source term
Q = np.zeros_like(r)
Q[(r >= 0.99 * R_TS.to("pc").value) & (r <= 1.01 * R_TS.to("pc").value)] = 100

# --- Operator Parameters ---
adv_params = {
    "v_centers": v_field,
    "order": 2,
    "limiter": "minmod",
    "cfl": 0.8,
    "inflow_value_W": 0.0,
}

dif_params_fv = {
    "D_values": 10 * D_values_pc_Myr,
    "Q_values": Q,
    "f_end": f_end_bc,
}

operator_params = {
    "advectionFV": adv_params,
    "diffusionFV": dif_params_fv,
}

# --- Solver Initialization ---
solver = Solver(
    x_grid=r,
    t_grid=t_grid,
    f_values=f_initial,
    problem_type="advectionFV-diffusionFV",
    operator_params=operator_params,
    substeps={"advectionFV": 1, "diffusionFV": 1},
)

# --- Run Simulation ---
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_initial)]

for n in range(num_timesteps):
    if n % 100 == 0:
        logging.info(f"Step {n+1}/{num_timesteps}")
    f_new = solver.step(1)
    f_evolution.append(np.copy(f_new))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))

num_curves = 15
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, num_curves))

for idx, curve_idx in enumerate(indices):
    ax.semilogy(
        r,
        f_evolution[curve_idx],
        color=colors[idx],
        linestyle="-",
        label=f"t={t_grid[curve_idx]:.1f}" if idx in [0, num_curves - 1] else None,
    )

ax.set_xlabel("$r$ (pc)")
ax.set_ylabel("$f(t, r)$")
ax.set_title("Advection-Diffusion with Injection (FV Advection, FV Diffusion)")
ax.set_xlim(0, r_end.value)
ax.set_ylim(1e-6, 1e3)
ax.grid(True)

# Draw physical boundaries
R_TS_pc = R_TS.to("pc").value
R_b_pc = R_b.to("pc").value
ax.axvline(
    0.99 * R_TS_pc, color="k", linestyle="--", linewidth=1, label="Injection zone"
)
ax.axvline(1.01 * R_TS_pc, color="k", linestyle="--", linewidth=1)
ax.axvline(R_TS_pc, color="r", linestyle="--", linewidth=1.5, label="Termination shock")
ax.axvline(R_b_pc, color="g", linestyle="--", linewidth=1.5, label="Bubble boundary")
ax.legend(loc="upper right")

# Colorbar
sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=t_grid[0], vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (Myr)")

plt.tight_layout()
plt.show()
