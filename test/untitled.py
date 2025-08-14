import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

# Enable LaTeX font rendering
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 12})

fig, ax = plt.subplots(figsize=(8, 8))

# Radii for the different regions
r1 = 1  # Stellar wind
r3 = 3  # Contact discontinuity
r4 = 3.2  # Forward shock

# Draw concentric circles
circles = [(r1, "black"), (r3, "black"), (r4, "black")]
for r, color in circles:
    circle = plt.Circle((0, 0), r, fill=False, color=color, linewidth=2)
    ax.add_patch(circle)

# Draw shaded gray area between r3 and r4
annulus = Wedge(
    center=(0, 0), r=r4, theta1=0, theta2=360, width=r4 - r3, color="gray", alpha=0.3
)
ax.add_patch(annulus)

# Add arrows for the wind (region 1)
angles = np.linspace(0, 2 * np.pi, 9)[:-1]
for angle in angles:
    ax.arrow(
        0,
        0,
        0.8 * np.cos(angle),
        0.8 * np.sin(angle),
        head_width=0.1,
        head_length=0.1,
        fc="black",
        ec="black",
    )

# Labels
ax.text(
    0,
    0,
    r"\textbf{(1) Stellar wind}",
    ha="center",
    va="center",
    fontsize=17,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)


ax.text(0, r1 + 0.1, r"\textbf{Termination shock}", ha="center", fontsize=15)

ax.text(
    0,
    1.8,
    r"\textbf{(2) Shocked stellar wind}",
    ha="center",
    fontsize=17,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)

ax.text(
    0,
    r3 - 0.6,
    r"\textbf{(3) Shocked interstellar gas}",
    color="black",
    ha="center",
    va="center",
    fontsize=17,
    bbox=dict(facecolor="gray", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.3),
)

ax.text(
    0,
    r4 + 0.2,
    r"\textbf{(4) Interstellar medium}",
    ha="center",
    fontsize=17,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
)

ax.text(2.5, 1.8, r"\textbf{Forward shock}", ha="center", fontsize=15, rotation=-45)
ax.text(
    -2, -2.5, r"\textbf{Contact discontinuity}", ha="center", fontsize=15, rotation=-50
)

# Final plot formatting
ax.set_xlim(-r4 - 0.5, r4 + 0.5)
ax.set_ylim(-r4 - 0.5, r4 + 0.5)
ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
plt.show()
