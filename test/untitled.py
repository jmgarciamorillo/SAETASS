import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib as mpl

# Parámetros del problema para que coincida con DiffValidation5.py
Nr = 199  # num_points = 200 -> Nr = 199 para que r tenga 200 puntos
r_end = 1.0
dr = r_end / Nr
f_end = 0.3

# Grilla temporal
t_steps = 200
t_grid = np.linspace(0, 2, t_steps)
dt = t_grid[1] - t_grid[0]
Nt = len(t_grid) - 1

# Coeficientes de difusión
D0, D1 = 1.0, 0.1

# Mallado radial
r = np.linspace(0, r_end, Nr + 1)

# Difusividad D(r)
D = np.where(r < 0.5, D0, D1)

# Promedio armónico
D_half = np.zeros(Nr)
for i in range(Nr):
    # Avoid division by zero if both are zero
    if D[i] + D[i + 1] == 0:
        D_half[i] = 0
    else:
        D_half[i] = 2 * D[i] * D[i + 1] / (D[i] + D[i + 1])

# Matrices Crank-Nicolson
alpha = dt / (2 * dr**2)

A_diag = np.zeros(Nr + 1)
A_lower = np.zeros(Nr)
A_upper = np.zeros(Nr)
B_diag = np.zeros(Nr + 1)
B_lower = np.zeros(Nr)
B_upper = np.zeros(Nr)

for i in range(1, Nr):
    ri = r[i]
    if ri == 0:
        continue
    D_imh = D_half[i - 1]
    D_iph = D_half[i]
    c_imh = (ri - dr / 2.0) ** 2 * D_imh / (ri**2)
    c_iph = (ri + dr / 2.0) ** 2 * D_iph / (ri**2)
    A_lower[i - 1] = -alpha * c_imh
    A_diag[i] = 1 + alpha * (c_imh + c_iph)
    A_upper[i] = -alpha * c_iph
    B_lower[i - 1] = alpha * c_imh
    B_diag[i] = 1 - alpha * (c_imh + c_iph)
    B_upper[i] = alpha * c_iph

# Condición en r=0 (Neumann simétrica: df/dr=0)
A_diag[0] = 1.0
B_diag[0] = 1.0

# Condición en r=1 (Dirichlet f=0)
A_diag[-1] = 1.0
B_diag[-1] = 0

A = diags([A_lower, A_diag, A_upper], offsets=[-1, 0, 1], format="csr")
B = diags([B_lower, B_diag, B_upper], offsets=[-1, 0, 1], format="csr")

# Condición inicial
f = np.exp(-((r - 0.3) ** 2) / (2 * 0.05**2))
f_evolution = [f.copy()]

# Evolución temporal
for n in range(Nt):
    rhs = B @ f
    rhs[-1] = f_end  # Forzar condicion de contorno Dirichlet

    f = spsolve(A, rhs)

    # Forzar condiciones de contorno explícitamente después del solver
    f[0] = f[1]  # Forzar condicion de contorno Neumann en r=0
    f[-1] = f_end  # Forzar condicion de contorno Dirichlet en r=1

    f_evolution.append(f.copy())

# Plotting
fig, ax = plt.subplots(figsize=(10, 4.5))

num_curves = 10
indices = np.linspace(0, Nt, num_curves, dtype=int)
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
ax2.plot(r, D, "r--", label="D(r)", alpha=0.5)
ax2.set_ylabel("Diffusion Coefficient D(r)", color="r")
ax2.tick_params(axis="y", labelcolor="r")


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

plt.xlim(0.0, 1.0)
plt.ylim(0, 1.1)
plt.grid(False)
plt.title("Diffusion with Discontinuous Coefficient")
plt.show()
