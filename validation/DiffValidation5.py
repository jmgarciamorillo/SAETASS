import numpy as np
import matplotlib.pyplot as plt
import math

import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from DiffusionSolver import DiffusionSolver

# Parameters
r_0 = 0.0
r_end = 1.0
num_points = 200
f_end = 0.0

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points)
t_steps = 50
t_grid = np.linspace(0, 0.1, t_steps)

# Initial profile (Gaussian-like)
f_values = np.exp(-((r - 0.3) ** 2) / (2 * 0.05 ** 2))

# Discontinuous diffusion coefficient
D_values = np.ones(num_points)
D_values[r >= 0.5] = 0.1

# Source term (Q)
Q = np.zeros(num_points)

dif_param = {
    "D_values": D_values,
    "Q_values": Q,
    "f_end": f_end,
}

# Prepare solver
solver = DiffusionSolver(x_grid=r, t_grid=t_grid, f_values=f_values, params=dif_param)

# Run simulation
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_values)]  # Store initial condition for plotting

for n in range(1, num_timesteps + 1):
    solver.f_values = solver.advance(1)
    f_evolution.append(np.copy(solver.f_values))

# Plotting
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(10, 4.5))

num_curves = 10
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
colors = plt.cm.viridis(np.linspace(0, 1, num_curves))

for idx, curve_idx in enumerate(indices):
    ax.plot(
        r,
        f_evolution[curve_idx],
        color=colors[idx],
        linestyle="-",
        label=f"t={t_grid[curve_idx]:.2f}" if idx in [0, num_curves - 1] else None,
    )

# Also plot the diffusion coefficient profile
ax2 = ax.twinx()
ax2.plot(r, D_values, 'r--', label='D(r)', alpha=0.5)
ax2.set_ylabel('Diffusion Coefficient D(r)', color='r')
ax2.tick_params(axis='y', labelcolor='r')


ax.set_xlabel("$r$ in parsec")
ax.set_ylabel("Solution $f(t,r)$")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    Line2D([0], [0], color="r", linestyle="--", label="D(r)"),
]
ax.legend(handles=legend_elements)
ax.grid()
fig.tight_layout()

sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("$t$")

plt.xlim(r_0, r_end)
plt.ylim(0, 1.1)
plt.grid(False)
plt.title("Diffusion with Discontinuous Coefficient")
plt.show()