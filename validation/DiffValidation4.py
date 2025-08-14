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
r_end = 1
num_points = 600
f_end = 0.0

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points + 1)
t_steps = 200
t_grid = np.linspace(0, 3, t_steps)  # yr

# Initial profile: zero everywhere
f_values = np.zeros(num_points + 1)

# Diffusion coefficient
D_0 = 1
eps = 0.01
D_values = D_0 * (r + eps) ** 2

# Print characteristic diffusion time
print("Characteristic diffusion time (yr):", r_end**2 / D_values[0])

# Source term (Q)
Q_0 = 4
Q = Q_0 * r


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
    print(f"Step {n}: max={solver._f_values.max()}, min={solver._f_values.min()}")
    solver._f_values = solver.run_simulation(1)
    f_evolution.append(np.copy(solver._f_values))

# Plotting
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(10, 4.5))  # Wider figure

num_curves = 20
indices = np.linspace(0, num_timesteps, num_curves, dtype=int)
colors = plt.cm.rainbow(np.linspace(0, 1, num_curves))


for idx, curve_idx in enumerate(indices):
    # Make earlier curves wider, later curves thinner
    ax.plot(
        r,
        f_evolution[curve_idx],
        color=colors[idx],
        alpha=1.0,
        linestyle="-",
        label=f"t={t_grid[curve_idx]:.0f}" if idx in [0, num_curves - 1] else None,
    )

# Analytical steady-state: f(r) = Q_0/(4*D_0)*(1 - r)
analytical = (Q_0 / (4 * D_0)) * (
    (1 - 2 * eps * np.log(eps + 1) - eps**2 / (eps + 1))
    - (r - 2 * eps * np.log(eps + r) - eps**2 / (eps + r))
)
ax.plot(r, analytical, "k--")

ax.set_xlabel("$r$")
ax.set_ylabel("Solution $f(t,r)$")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    Line2D(
        [0],
        [0],
        color="k",
        linestyle="--",
        label="Steady state",
    ),
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
plt.ylim(0, 1.1)
plt.grid(False)
plt.show()
