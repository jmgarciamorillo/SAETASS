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
p_0 = 1.0  # GeV / c
p_max = 100000.0  # GeV / c
p_end = 10000000.0  # GeV / c
num_points = 2000
t_end = 3 * 1000000 * 365 * 24 * 3600  # 3 Myr in seconds

# Spatial and temporal grids
p = np.linspace(p_0, p_end, num_points + 1)
p = np.logspace(np.log10(p_0), np.log10(p_end), num_points + 1)
t_steps = 5000
t_grid = np.linspace(0, t_end, t_steps)

# Initial profile
f_0 = 1
f_values = f_0 * (p / p_0) ** (-4) * np.exp(-p / p_max)

# plot initial profile
plt.figure(figsize=(6, 3.5))
plt.loglog(p, f_values, label="Initial profile")
plt.xlabel("$p$ (GeV/c)")
plt.ylabel("$f(0,p)$")
plt.xlim(p_0, p_end)
plt.show()

# Source term and loss rate
me = 0.511e-3  # GeV/c^2

Q = np.zeros(len(p))
P_dt = -2.53e-18 * (10) ** 2 * (me**2 + p**2)  # GeV / c / s (using 10 microGauss)

# Prepare solver
solver = LossSolver(
    x_grid=p,
    t_grid=t_grid,
    f_values=f_values,
    P_dot=P_dt[:-1],
    Q_values=Q,  # match calculation grid size
)

# Run simulation
num_timesteps = len(t_grid) - 1
f_evolution = [np.copy(f_values)]  # Store initial condition for plotting
n_particles = np.zeros(num_timesteps + 1)
n_particles[0] = np.sum(f_values * p**4)  # Initial number of particles

for n in range(1, num_timesteps + 1):
    solver._f_values = solver.run_simulation(1)  # Advance one step
    f_evolution.append(np.copy(solver._f_values))
    n_particles[n] = np.sum(solver._f_values * p**4)

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
        p**4 * f_evolution[plot_idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="-",
        label="Numerical" if idx == 19 else None,
    )

ax.set_xlabel("$p$ (GeV/c)")
ax.set_ylabel("$p^4 f(t,p)$")
# ax.set_ylim(1e-30, 1e0)
ax.set_xlim(p_0, p_end)
ax.grid(False)
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], color="k", linestyle="-", label="Numerical")]
ax.legend(handles=legend_elements)
fig.tight_layout()

# Add colorbar for time gradient
sm = mpl.cm.ScalarMappable(
    cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=t_grid[-1] / (3.1e13))
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("$t$ (Myrs)")

plt.show()

plt.figure(figsize=(6, 3.5))
plt.plot(t_grid / (3.1e13), n_particles, label="Number of particles")
plt.xlabel("Time (Myrs)")
plt.show()
