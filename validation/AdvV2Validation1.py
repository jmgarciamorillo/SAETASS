import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Use FV solver
from AdvectionFVSolver import AdvectionFVSolver

# Parameters
r_0 = 0.0
r_Inj = 1.0
r_end = 3.0
num_points = 1000

# Spatial and temporal grids
r = np.linspace(r_0, r_end, num_points + 1)  # faces
t_steps = 2001
t_grid = np.linspace(0, 2.0, t_steps)


# Initial profile (Gaussian) at nodes -> centers
def f_0(r):
    return np.exp(-0.5 * ((r - 0.3) / 0.1) ** 2)


f_nodes = f_0(r)
r_centers = 0.5 * (r[:-1] + r[1:])
U0_centers = 0.5 * (f_nodes[:-1] + f_nodes[1:])

# velocity centers (constant)
v_centers = np.ones_like(r_centers)

# Prepare FV solver
solver = AdvectionFVSolver(
    r_faces=r, v_centers=v_centers, limiter="minmod", cfl=0.8, order=2
)

# Run simulation with CFL subcycling
num_timesteps = len(t_grid) - 1
U = U0_centers.copy()
f_evolution = [np.copy(U)]

for n in range(num_timesteps):
    t0 = t_grid[n]
    t1 = t_grid[n + 1]
    dt_global = float(t1 - t0)
    dt_cfl = solver.compute_dt()
    if not np.isfinite(dt_cfl) or dt_cfl <= 0.0:
        n_subcycles = 1
    else:
        n_subcycles = int(np.ceil(dt_global / dt_cfl))
        n_subcycles = max(1, n_subcycles)
    dt_sub = dt_global / n_subcycles
    for s in range(n_subcycles):
        U = solver.advance(U, dt_sub)
    f_evolution.append(np.copy(U))

# Analytical solution (centers) for constant v=1
f_analytical = []
u0 = 1.0
for n in range(t_steps):
    t = t_grid[n]
    r_shifted = r_centers - u0 * t
    mask = r_shifted > 0
    f_ana = np.zeros_like(r_centers)
    f_ana[mask] = ((r_shifted[mask]) / (r_centers[mask])) ** 2 * f_0(r_shifted[mask])
    f_analytical.append(f_ana)

# Plot
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = plt.cm.rainbow(np.linspace(0, 1, 20))
plot_indices = np.linspace(0, t_steps - 1, 20, dtype=int)

for idx, plot_idx in enumerate(plot_indices):
    alpha = 0.2 + 0.8 * (idx + 1) / 20
    ax.plot(
        r_centers,
        f_analytical[plot_idx],
        color=colors[idx],
        alpha=alpha,
        linestyle="--",
    )
    ax.plot(
        r_centers, f_evolution[plot_idx], color=colors[idx], alpha=alpha, linestyle="-"
    )

ax.set_xlabel("$r$")
ax.set_ylabel("Solution $u(t,r)$ (centers)")
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
plt.show()
