import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import astropy.units as u
import astropy.constants as const

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Solver import Solver

# Parameters
r_0 = 0.0 * u.pc
r_Inj = 1.0 * u.pc  # parsec
r_end = 100.0 * u.pc  # parsec
num_points = 2000
eta_B = 0.1  # Magnetic field efficiency
L_wind = 1e38 * u.erg / u.s  # erg/s
M_dot = 1e-4 * const.M_sun / u.yr
rho_0 = const.m_p / u.cm**3
t_b = 1 * u.Myr
eta_inj = 0.1
v_w = np.sqrt(2 * L_wind / M_dot)
t_end = 1 * u.Myr
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
r = np.linspace(r_0.value, r_end.value, num_points + 1)
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


t_steps = 4000
t_grid = np.linspace(0, t_end.value, t_steps)

# Initial profile: zero everywhere
f_values = np.zeros(num_points + 1)

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
D_values = 1 / 3 * v_p * np.sqrt(r_L * r_Inj)
D_values[r_ISM] = 3e18 * u.cm**2 / u.s  # constant diffusion in ISM
plt.plot(r, D_values.to("pc**2/Myr").value, label="Diffusion Coefficient D(r)")
# Caracteristic diffusion time
print(
    f"Characteristic diffusion time: {((R_b- R_TS)**2 / 4/D_values[int(num_points/2)]).to("kyr").value} kyr"
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
Q[Q != 0] = 100
# plt.plot(r, Q, label="Source term Q(r)")

# Prepare solver parameters
# For advectionFV we pass v_centers via advectionFV_params
advectionFV_params = {
    "v_centers": v_field * 1000,
    "order": 2,
    "limiter": "minmod",
    "cfl": 0.8,
}

diffusion_params = {"D_values": D_values.to("pc**2/Myr").value * 0.01}

# Operator splitting: advectionFV then diffusion
# Important: pass f_values as centers for advectionFV (length M-1)
solver = Solver(
    x_grid=r,
    t_grid=t_grid,
    f_values=f_values,
    problem_type="advectionFV-diffusion",
    Q_values=Q,
    advectionFV_params=advectionFV_params,  # will be used for advectionFV in Solver
    diffusion_params=diffusion_params,
    substeps={"advectionFV": 1, "diffusion": 1},
)

import matplotlib as mpl

num_timesteps = len(t_grid) - 1
num_curves = 100
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
indices = np.append(indices, num_timesteps)  # Ensure last step is included

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.semilogy(r, f_values, color="b")
ax.set_xlabel("$r$ (pc)")
ax.set_ylabel("$f(t, r)$")

diff_str = f"D = {D_values.to('pc**2/yr').value[0]:.2f} pc$^2$/yr"
adv_str = r"$v(r)$ as defined in code"
inj_str = r"$Q=100$ at $r \approx R_{TS}$"

ax.set_title(
    "Advection-Diffusion with Injection at $R_{TS}$\n"
    f"{diff_str}, {adv_str}\n{inj_str}"
)
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
ax.axvline(R_TS_pc, color="r", linestyle="--", linewidth=1.5, label="Termination shock")
ax.axvline(R_b_pc, color="g", linestyle="--", linewidth=1.5, label="Bubble boundary")

ax.legend(loc="upper right")

sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow,
    norm=plt.Normalize(vmin=t_grid[indices[0]], vmax=t_grid[indices[-1]]),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (yr)")

current_step = 0
for next_plot_step in indices:
    steps_to_advance = int(next_plot_step - current_step)
    if steps_to_advance > 0:
        f_current = solver.step(steps_to_advance)
        current_step = next_plot_step
    else:
        f_current = solver.f_values  # Already at this step

    line.set_ydata(f_current)
    ax.set_title(
        f"Advection-Diffusion, $t$={t_grid[int(current_step)]:.2f} yr\n"
        f"{diff_str}, {adv_str}\n{inj_str}"
    )
    plt.pause(0.05)

plt.ioff()
plt.show()  #
