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
r_0 = 0.0  # pc
r_end = 100  # pc
num_points = 6000
f_end = 0.0

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points + 1)
t_steps = 200
t_grid = np.linspace(0, 100, t_steps)  # yr

# Initial profile: zero everywhere
f_values = np.zeros(num_points + 1)

# Diffusion coefficient for particles at E = 1 GeV
D_values = np.ones(num_points + 1) * 3e28  # cm^2/s
# in parsec^2/yr this is
D_values = D_values * (1.05027e-37) * (3600 * 24 * 365)  # convert to pc^2/yr
print(D_values)

# Print characteristic diffusion time
print("Characteristic diffusion time (yr):", r_end**2 / D_values[0])

# Source term (Q): constant injection in a narrow region near r=0 (delta-like)
Q = np.zeros(num_points)
for i in range(num_points):
    if r[i] < 0.05:  # Width of the Dirac delta
        Q[i] = 1000  # Constant injection rate
    else:
        Q[i] = 0  # No injection outside the delta region

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
    ax.loglog(
        r,
        f_evolution[curve_idx],
        color=colors[idx],
        alpha=1.0,
        linestyle="-",
        label=f"t={t_grid[curve_idx]:.0f}" if idx in [0, num_curves - 1] else None,
    )

# Analytical steady-state: f(r) ~ 1/r^2 (normalized for visibility)
analytical = 1 / (r[1:])
ax.loglog(r[1:], analytical, "k--", label="Analytical ($1/r^2$)")

ax.set_xlabel("$r$ (in pc)")
ax.set_ylabel("Solution $f(t,r)$")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    Line2D([0], [0], color="k", linestyle="--", label="Analytical ($\propto 1/r$)"),
]
ax.legend(handles=legend_elements)
ax.grid()
fig.tight_layout()

sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (in yr)")

plt.xlim(r_0, r_end)
plt.ylim(0.001, 100)
plt.grid(False)
plt.show()
