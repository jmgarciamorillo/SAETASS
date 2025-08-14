import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from LossSolver import LossSolver

# Parameters
p_0 = 1.0
p_end = 1000.0
num_points = 3000

# Spatial and temporal grids
p = np.linspace(p_0, p_end, num_points + 1)
t_steps = 1000
t_grid = np.linspace(0, 2.0, t_steps)

# Initial profile (zero)
f_values = np.zeros(num_points + 1)

# Source term and loss rate
beta = 2
alpha = 4
b0 = 1
Q0 = 1
p0 = 1

Q = Q0 * (p / p0) ** (-alpha)
P_dt = -b0 * (p / p0) ** beta

# Prepare solver
solver = LossSolver(
    x_grid=p,
    t_grid=t_grid,
    f_values=f_values,
    P_dot=P_dt[:-1],
    Q_values=Q,  # match calculation grid size
    is_homogeneous=True,
)

# Run simulation
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_values)]  # Store initial condition for plotting

for n in range(1, num_timesteps + 1):
    solver._f_values = solver.run_simulation(1)  # Advance one step
    f_evolution.append(np.copy(solver._f_values))

# Analytical solution for comparison (steady-state)
f_analytical = []
for n in range(t_steps):
    # Steady-state analytical solution (from momentum_test.py)
    f_ana = (
        Q0
        * p0
        / (1 - alpha)
        / b0
        * ((p_end / p0) ** (1 - alpha) - (p / p0) ** (1 - alpha))
        * (p / p0) ** (-beta)
    )
    f_analytical.append(f_ana)

# Plotting
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = plt.cm.rainbow(np.linspace(0, 1, 20))

# Plot only 20 evenly spaced curves
plot_indices = np.linspace(0, t_steps - 1, 20, dtype=int)

for idx, plot_idx in enumerate(plot_indices):
    alpha = min(0.2 + 0.8 * (idx + 1) / 20, 1.0)
    ax.loglog(
        p,
        p**5 * f_evolution[plot_idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="-",
        label="Numerical" if idx == 19 else None,
    )
    ax.loglog(
        p,
        p**5 * f_analytical[plot_idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="--",
        label="Analytical" if idx == 19 else None,
    )

ax.set_xlabel("$p$")
ax.set_ylabel("$p^5 f(t,p)$")
ax.set_ylim(1e-13, 1e0)
ax.set_xlim(p_0, p_end)
ax.grid(False)
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
    Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
]
ax.legend(handles=legend_elements)
fig.tight_layout()

# Add colorbar for time gradient
sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=t_grid[-1])
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$")

plt.show()
