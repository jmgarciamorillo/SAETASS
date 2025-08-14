import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from AdvectionSolver import AdvectionSolver

# Parameters
r_0 = 0.0
r_end = 10.0  # parsec
num_points = 4000

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points + 1)
t_steps = 2001  # At least 200 time steps
t_grid = np.linspace(0, 2.0, t_steps)


# Initial profile (zero everywhere for this test)
def f_0(r):
    return np.zeros_like(r)


f_values = f_0(r)


# Velocity field 1/r^2 and constant inside r=0.1
v_field_n = np.where(r < 0.1, 0, 10 / r**2)  # Avoid division by zero
v_field_n1 = np.where(r < 0.1, 0, 10 / r**2)  # Avoid division by zero

# Source term (Q) a spike at r=1 parsec
Q = np.zeros_like(r)
Q = np.where((r >= 0.9) & (r <= 1.1), 40, 0)  # Small spike in the source term

# Prepare solver
solver = AdvectionSolver(
    x_grid=r,
    t_grid=t_grid,
    f_values=f_values,
    v_field_n=v_field_n,
    v_field_n1=v_field_n1,
    Q_values=Q,
)

# Run simulation
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_values)]  # Store initial condition for plotting

for n in range(1, num_timesteps + 1):
    solver._f_values = solver.run_simulation(1)  # Advance one step
    f_evolution.append(np.copy(solver._f_values))

# Analytical solution for comparison
# f_analytical = []
# for n in range(t_steps):
#     t = t_grid[n]
#     u0 = 1.0
#     r_shifted = r - u0 * t
#     mask = r_shifted > 0
#     f_ana = np.zeros_like(r)
#     f_ana[mask] = ((r[mask] - u0 * t) / r[mask]) ** 2 * f_0(r_shifted[mask])
#     f_analytical.append(f_ana)

# Plotting
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = plt.cm.rainbow(np.linspace(0, 1, 20))

# Plot only 20 evenly spaced curves
plot_indices = np.linspace(0, t_steps - 1, 20, dtype=int)

for idx, plot_idx in enumerate(plot_indices):
    alpha = 0.2 + 0.8 * (idx + 1) / 20
    # ax.plot(
    #     r,
    #     f_analytical[plot_idx],
    #     color=colors[idx],
    #     #alpha=alpha,
    #     linestyle="--",
    #     label="Analytical" if idx == 19 else None,
    # )
    ax.plot(
        r,
        f_evolution[plot_idx],
        color=colors[idx],
        # alpha=alpha,
        linestyle="-",
        label="Numerical" if idx == 19 else None,
    )

ax.set_xlabel("$r$")
ax.set_ylabel("Solution $f(t,r)$")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    # Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
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

plt.grid(False)
plt.xlim(r_0, r_end)
plt.ylim(0, 1.1)
plt.show()
