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
r = np.linspace(r_0, r_end, num_points + 1)
t_steps = 21
t_grid = np.linspace(0, 0.2, t_steps)

# Initial profile (sinc-like)
f_values = np.zeros(num_points + 1)
for i in range(num_points + 1):
    if r[i] == 0:
        f_values[i] = math.pi / 2
    else:
        f_values[i] = 1 / (2 * r[i]) * np.sin(math.pi * r[i])

# Diffusion coefficient
D_values = np.ones(num_points + 1)

# Source term (Q)
Q = np.zeros(num_points + 1)
# Q = np.delete(1 / r**2, len(r) - 1)

# Prepare solver
solver = DiffusionSolver(
    x_grid=r,
    t_grid=t_grid,
    f_values=f_values,
    Q_values=Q,
    D_values=D_values,
)

# Run simulation
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_values)]  # Store initial condition for plotting

for n in range(1, num_timesteps + 1):
    solver._f_values = solver.run_simulation(1)  # Advance one step
    # Store a copy of the current solution (including boundary for plotting)
    f_evolution.append(np.copy(solver._f_values))


# Analytical solution for comparison
f_analytical = []
for n in range(t_steps):
    t = t_grid[n]
    f_ana = np.zeros(num_points + 1)
    for i in range(num_points + 1):
        if r[i] == 0:
            f_ana[i] = math.pi / 2 * np.exp(-math.pi**2 * D_values[i] * t)
        else:
            f_ana[i] = (
                1
                / (2 * r[i])
                * np.sin(math.pi * r[i])
                * np.exp(-math.pi**2 * D_values[i] * t)
            )
    f_analytical.append(f_ana)

# Plotting
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = plt.cm.rainbow(np.linspace(0, 1, t_steps))

for idx in range(t_steps):
    alpha = 0.2 + 0.8 * (idx + 1) / t_steps
    ax.plot(
        r,
        f_analytical[idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="--",
        label="Analytical" if idx == t_steps - 1 else None,
    )
    ax.plot(
        r,
        f_evolution[idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="-",
        label="Numerical" if idx == t_steps - 1 else None,
    )

ax.set_xlabel("$r$")
ax.set_ylabel("Solution $f(t,r)$")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
]
ax.legend(handles=legend_elements)
ax.grid()
fig.tight_layout()

sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$")

plt.xlim(r_0, r_end)
plt.ylim(0, 1.6)
plt.grid(False)
plt.show()
