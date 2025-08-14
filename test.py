import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure parent directory is in sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Solver import Solver

# Parameters
r_0 = 0.0
r_end = 10.0  # parsec
num_points = 4000

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points + 1)
t_steps = 600
t_grid = np.linspace(0, 1, t_steps)  # 5 yr total

# Initial profile: zero everywhere
f_values = np.zeros(num_points + 1)

# Velocity profile: v(r) = 1 / r^2 if r>0.95pc, otherwise v(r)= 1/4 * 1/r^2
v0 = 0.1  # pc/yr
v_field = np.where(r < 0.95, 0.25 * v0 / 0.95**2, v0 / r**2)

# Diffusion coefficient: constant
D_values = np.ones_like(r) * 1  # pc^2/yr

# Source term: constant injection at r ~ 1 pc
Q = np.zeros_like(r)
Q[(r >= 0.95) & (r <= 1.05)] = 100.0

# Prepare solver parameters
advection_params = {"v_field_n": v_field, "v_field_n1": v_field}
diffusion_params = {"D_values": D_values}

# Operator splitting: advection then diffusion
solver = Solver(
    x_grid=r,
    t_grid=t_grid,
    f_values=f_values,
    problem_type="advection-diffusion",
    Q_values=Q,
    advection_params=advection_params,
    diffusion_params=diffusion_params,
    substeps={"advection": 3, "diffusion": 1},
)

import matplotlib as mpl

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
(line,) = ax.plot(r, f_values, color="b")
ax.set_xlabel("$r$ (pc)")
ax.set_ylabel("$f(t, r)$")
ax.set_title("Advection-Diffusion with Constant Injection at $r=1$ pc")
ax.set_xlim(0.1, r_end)
ax.set_ylim(0, 2)
ax.grid(True)
fig.tight_layout()

# Draw dashed vertical lines for the injection zone
ax.axvline(0.95, color="k", linestyle="--", linewidth=1, label="Injection zone")
ax.axvline(1.05, color="k", linestyle="--", linewidth=1)
ax.legend(loc="upper right")

sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (yr)")

plt.show(block=False)

num_timesteps = len(t_grid) - 1
num_curves = 20
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
indices = np.append(indices, num_timesteps)  # Ensure last step is included

current_step = 0
for next_plot_step in indices:
    steps_to_advance = next_plot_step - current_step
    if steps_to_advance > 0:
        f_current = solver.step(steps_to_advance)
        current_step = next_plot_step
    else:
        f_current = solver.f_values  # Already at this step

    line.set_ydata(f_current)
    ax.set_title(f"Advection-Diffusion, $t$={t_grid[current_step]:.2f} yr")
    plt.pause(0.01)

plt.ioff()
plt.show()
