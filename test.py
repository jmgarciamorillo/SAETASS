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
r = np.linspace(r_0, r_end, num_points)
t_steps = 1500
t_grid = np.linspace(0, 10, t_steps)  # yr total

# Initial profile: zero everywhere
f_values = np.zeros(num_points)

# Velocity profile: v(r) = 1 / r^2 if r>0.95pc, otherwise v(r)= 1/4 * 1/r^2
v0 = 10  # pc/yr
v_field = np.where(r < 0.95, 0.25 * v0 / 0.95**2, v0 / r**2)

# Diffusion coefficient: constant
D_values = np.ones_like(r) * 0.0001  # pc^2/yr

# Source term: constant injection at r ~ 1 pc
Q = np.zeros_like(r)
Q[(r >= 0.95) & (r <= 1.05)] = 100.0  # Myr^-1

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
    substeps={"advection": 1, "diffusion": 1},
)

import matplotlib as mpl

num_timesteps = len(t_grid) - 1
num_curves = 20
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
indices = np.append(indices, num_timesteps)  # Ensure last step is included

f_curves = []
current_step = 0
for next_plot_step in indices:
    steps_to_advance = next_plot_step - current_step
    if steps_to_advance > 0:
        f_current = solver.step(steps_to_advance)
        current_step = next_plot_step
    else:
        f_current = solver.f_values  # Already at this step
    f_curves.append(np.copy(f_current))

# Plot all curves with rainbow colormap
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.rainbow(np.linspace(0, 1, len(f_curves)))
for i, (f_curve, idx) in enumerate(zip(f_curves, indices)):
    label = None
    if i == 0:
        label = f"t={t_grid[indices[0]]:.2f} yr"
    elif i == len(f_curves) - 1:
        label = f"t={t_grid[indices[-1]]:.2f} yr"
    ax.loglog(r, f_curve, color=colors[i], label=label)

ax.set_xlabel("$r$ (pc)")
ax.set_ylabel("$f(t, r)$")

# Compose parameter info string
diff_str = f"D = {D_values[0]:.2f} pc$^2$/yr"
adv_str = r"$v(r) = \frac{10}{r^2}$ pc/yr for $r \geq 0.95$; $v(r) = \frac{10}{4 \times 0.95^2}$ for $r < 0.95$"
inj_str = r"$Q=100$ for $0.95 \leq r \leq 1.05$ pc"

ax.set_title(
    "Advection-Diffusion with Constant Injection at $r=1$ pc\n"
    f"{diff_str}, {adv_str}\n{inj_str}"
)
ax.set_xlim(0.1, r_end)
ax.set_ylim(1e-3, 1e1)
ax.grid(True)
fig.tight_layout()

# Draw dashed vertical lines for the injection zone
ax.axvline(0.95, color="k", linestyle="--", linewidth=1, label="Injection zone")
ax.axvline(1.05, color="k", linestyle="--", linewidth=1)

# Colorbar for time
sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow,
    norm=plt.Normalize(vmin=t_grid[indices[0]], vmax=t_grid[indices[-1]]),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (yr)")

# Legend for first and last curve and injection zone
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right")

plt.savefig("advection_diffusion_evolution.png", dpi=300)
plt.show()
