import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
import math


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# Parameters 167
r_0 = 0.0  # Start of the domain
r_end = 1.0  # End of the domain
num_points = 200  # Number of points in the spatial grid
f_end = 0.0  # End value for f(r) at r_end

# Create spatial grid
r = np.linspace(r_0, r_end, num_points + 2)

# Define the initial test profile f_values(0, r)
# Example: Sinc function profile
f_values = np.zeros(num_points + 2)
for i in range(num_points + 2):
    if r[i] == 0:
        f_values[i] = math.pi / 2  # Sinc function at r=0
    else:
        f_values[i] = (
            1 / (2 * r[i]) * np.sin(math.pi * r[i])
        )  # Sinc function centered at r=0.5

# Example: initial zero profile but injection at r=0
# f_values = np.zeros(num_points + 1)
Q = np.zeros(num_points + 1)
Q = np.delete(1 / r**2, len(r) - 1)  # Remove the last element to match the grid size;
# Q[0:10] = 1000.0

# Example: Gaussian profile centered at r=0.5 with standard deviation 0.1
# center = 0
# std_dev = 0.1
# f_values = np.exp(-((r - center) ** 2) / (2 * std_dev**2))


f_values = np.delete(
    f_values, len(f_values) - 1
)  # Remove the last element to match the grid size

# Define the diffusion coefficient D(r) as a function of r
# Example: Constant diffusion coefficient
# D_values = 1.0 * np.ones(num_points + 2)
# Example: Linearly increasing diffusion coefficient
D_values = r**2

# Plot the diffusion coefficient profile
r_calc = np.delete(r, len(r) - 1)
plt.plot(r, D_values, label="Diffusion Coefficient (D)")
plt.xlabel("r")
plt.ylabel("D(r)")
plt.title("Diffusion Coefficient Profile")
plt.legend()
plt.grid()
plt.show()

# Plot the initial profile
plt.plot(r, np.append(f_values, 0), label="Initial Profile")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.ylim(-2, 2)
plt.xlabel("r")
plt.ylabel("f_values")
plt.title("Initial Test Profile")
plt.legend()
plt.grid()
plt.show()

# Save the initial profile to a file (optional)
# np.savetxt("initial_profile.txt", np.column_stack((r, f_values)), header="r f_values")


##


def compute_crank_nicolson_coefficients(delta_t, delta_r, r_0, D_values, num_points):
    """
    Compute the coefficients for the Crank-Nicolson scheme at a given spatial index.

    Args:
        index (int): The spatial index (node of the grid).
        delta_t (float): Time step size.
        delta_r (float): Spatial step size.
        r_0 (float): Start of the domain.
        D_values (array): Diffusion coefficient values at each spatial node.

    Returns:
        tuple: Coefficients (a, b, c) for the Crank-Nicolson scheme.
    """
    # Initialize coefficients array q and s with same size of grid
    q = np.zeros(num_points + 1)
    s = np.zeros(num_points + 1)

    # First compute the case r=0 which is different
    q[0] = 3 * D_values[0] * delta_t / (delta_r**2)

    # Compute the rest of the coefficients
    for i in range(1, num_points + 1):
        q[i] = D_values[i] * delta_t / (2 * delta_r**2)
        s[i] = (
            delta_t
            / (4 * delta_r)
            * (
                2 * D_values[i] / (r_0 + i * delta_r)
                + (D_values[i + 1] - D_values[i - 1]) / (2 * delta_r)
            )
        )

    return q, s


def compute_crank_nicolson_matrix(n, s, q):
    """
    Compute the Crank-Nicolson matrix for the given timestep, s, and q coefficients.

    Args:
        timestep (float): Time step size.
        s (array): Array of s coefficients.
        q (array): Array of q coefficients.

    Returns:
        tuple: Two matrices (A, tildeA) for the Crank-Nicolson scheme.
    """
    num_points = len(q)

    Adiag = 1 - 2 * q
    Adiag[0] = 1 - q[0]
    tildeAdiag = 1 + 2 * q
    tildeAdiag[0] = 1 + q[0]

    A = np.diag(Adiag)
    tildeA = np.diag(tildeAdiag)

    for i in range(1, num_points - 1):
        A[i, i - 1] = q[i] - s[i]
        A[i, i + 1] = s[i] + q[i]
        tildeA[i, i - 1] = s[i] - q[i]
        tildeA[i, i + 1] = -s[i] - q[i]

    A[0, 1] = q[0]
    A[-1, -2] = q[-1] - s[-1]
    tildeA[0, 1] = -q[0]
    tildeA[-1, -2] = s[-1] - q[-1]

    return A, tildeA


def compute_cc_vectors(num_points, s, q, f_end):
    """
    Compute the Ucc and tildeUcc vectors for the Crank-Nicolson scheme.

    Args:
        num_points (int): Number of spatial points.
        s (array): Array of s coefficients.
        q (array): Array of q coefficients.

    Returns:
        tuple: Two vectors (Ucc, tildeUcc) for the Crank-Nicolson scheme.
    """
    Ucc = np.zeros(num_points + 1)
    tildeUcc = np.zeros(num_points + 1)

    # Boundary conditions
    Ucc[-1] = (s[-1] + q[-1]) * f_end
    tildeUcc[-1] = -(s[-1] + q[-1]) * f_end

    return Ucc, tildeUcc


def compute_rhs(f_values, Ucc, tildeUcc, A):
    """
    Compute the right-hand side of the Crank-Nicolson equation.

    Args:
        f_values (array): Current values of the function f at spatial points.
        Ucc (array): Ucc vector for the Crank-Nicolson scheme.
        tildeUcc (array): tildeUcc vector for the Crank-Nicolson scheme.
        A (array): Matrix A for the Crank-Nicolson scheme.

    Returns:
        array: Right-hand side vector for the Crank-Nicolson equation.
    """

    return np.dot(A, f_values) + Ucc - tildeUcc + delta_t * Q


# MAIN FOR THE TEST1 SIMULATION
# if __name__ == "__main__":
#     # Define simulation parameters
#     delta_t = 0.01  # Time step size
#     delta_r = (r_end - r_0) / num_points  # Spatial step size

#     # Prepare for plotting
#     import matplotlib as mpl

#     fig, ax = plt.subplots(figsize=(5, 3))
#     num_steps = 20
#     colors = plt.cm.rainbow(np.linspace(0, 1, num_steps + 1))  # FIX: +1 here

#     # Store initial values for plotting
#     f_values_all = []
#     f_analitic_all = []

#     # Add initial condition (t=0)
#     f_values_all.append(np.append(f_values, 0))
#     f_analitic = np.zeros(num_points + 2)
#     for i in range(num_points + 2):
#         if r[i] == 0:
#             f_analitic[i] = math.pi / 2
#         else:
#             f_analitic[i] = 1 / (2 * r[i]) * np.sin(math.pi * r[i])
#     f_analitic_all.append(f_analitic)

#     # Time-stepping loop
#     for n in range(1, num_steps + 1):
#         # Compute Crank-Nicolson coefficients
#         q, s = compute_crank_nicolson_coefficients(
#             delta_t, delta_r, r_0, D_values, num_points
#         )

#         # Compute Crank-Nicolson matrices
#         A, tildeA = compute_crank_nicolson_matrix(num_points, s, q)
#         Ucc, tildeUcc = compute_cc_vectors(num_points, s, q, f_end)
#         rhs = compute_rhs(f_values, Ucc, tildeUcc, A)
#         f_new = solve(tildeA, rhs, assume_a="tridiagonal")
#         f_values = f_new

#         # Analytical solution for comparison
#         f_analitic = np.zeros(num_points + 2)
#         for i in range(num_points + 2):
#             if r[i] == 0:
#                 f_analitic[i] = (
#                     math.pi / 2 * np.exp(-math.pi**2 * D_values[i] * n * delta_t)
#                 )
#             else:
#                 f_analitic[i] = (
#                     1
#                     / (2 * r[i])
#                     * np.sin(math.pi * r[i])
#                     * np.exp(-math.pi**2 * D_values[i] * n * delta_t)
#                 )

#         # Store for plotting
#         f_values_all.append(np.append(f_values, 0))
#         f_analitic_all.append(f_analitic)

#     # Plot all time steps with fading color
#     for idx in range(num_steps + 1):  # +1 to include initial condition
#         alpha = 0.2 + 0.8 * (idx + 1) / (num_steps + 1)
#         ax.plot(
#             r,
#             f_analitic_all[idx],
#             color=colors[idx],  # FIX: no modulo, just idx
#             alpha=alpha,
#             linestyle="--",
#             label="Analytical" if idx == num_steps else None,
#         )
#         ax.plot(
#             r,
#             f_values_all[idx],
#             color=colors[idx],  # FIX: no modulo, just idx
#             alpha=alpha,
#             linestyle="-",
#             label="Numerical" if idx == num_steps else None,
#         )

#     ax.set_xlabel("$r$")
#     ax.set_ylabel("Solution $f(t,r)$")
#     # ax.set_title("Evolution of Numerical and Analytical Solutions")
#     from matplotlib.lines import Line2D

#     legend_elements = [
#         Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
#         Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
#     ]
#     ax.legend(handles=legend_elements)
#     # ax.set_ylim(-2, 2)
#     ax.grid()
#     fig.tight_layout()

#     # Add colorbar for time gradient
#     sm = mpl.cm.ScalarMappable(
#         cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n * delta_t)
#     )
#     sm.set_array([])
#     cbar = fig.colorbar(sm, ax=ax, pad=0.02)
#     cbar.set_label("$t$")

#     plt.xlim(r_0, r_end)
#     plt.ylim(0, 1.6)
#     plt.grid(False)
#     plt.show()

# MAIN FOR THE TEST2 SIMULATION
if __name__ == "__main__":
    # Define simulation parameters
    delta_t = 0.01  # Time step size
    delta_r = (r_end - r_0) / num_points  # Spatial step size

    # Prepare for plotting
    import matplotlib as mpl

    fig, ax = plt.subplots(figsize=(6, 3.5))
    num_steps = 20
    colors = plt.cm.rainbow(np.linspace(0, 1, num_steps + 1))  # FIX: +1 here

    # Store initial values for plotting
    f_values_all = []
    f_analitic_all = []

    # Add initial condition (t=0)
    f_values_all.append(np.append(f_values, 0))
    f_analitic = np.zeros(num_points + 2)
    for i in range(num_points + 2):
        if r[i] == 0:
            f_analitic[i] = math.pi / 2
        else:
            f_analitic[i] = 1 / (2 * r[i]) * np.sin(math.pi * r[i])
    f_analitic_all.append(f_analitic)

    # Time-stepping loop
    for n in range(1, num_steps + 1):
        # Compute Crank-Nicolson coefficients
        q, s = compute_crank_nicolson_coefficients(
            delta_t, delta_r, r_0, D_values, num_points
        )

        # Compute Crank-Nicolson matrices
        A, tildeA = compute_crank_nicolson_matrix(num_points, s, q)
        Ucc, tildeUcc = compute_cc_vectors(num_points, s, q, f_end)
        rhs = compute_rhs(f_values, Ucc, tildeUcc, A)
        f_new = solve(tildeA, rhs, assume_a="tridiagonal")
        f_values = f_new

        # Analytical solution for comparison
        f_analitic = np.zeros(num_points + 2)
        for i in range(num_points + 2):
            if r[i] == 0:
                f_analitic[i] = (
                    math.pi / 2 * np.exp(-math.pi**2 * D_values[i] * n * delta_t)
                )
            else:
                f_analitic[i] = (
                    1
                    / (2 * r[i])
                    * np.sin(math.pi * r[i])
                    * np.exp(-math.pi**2 * D_values[i] * n * delta_t)
                )

        # Store for plotting
        f_values_all.append(np.append(f_values, 0))
        f_analitic_all.append(f_analitic)

    # Plot all time steps with fading color
    for idx in range(num_steps + 1):  # +1 to include initial condition
        alpha = 0.2 + 0.8 * (idx + 1) / (num_steps + 1)
        ax.plot(
            r,
            f_analitic_all[idx],
            color=colors[idx],  # FIX: no modulo, just idx
            alpha=alpha,
            linestyle="--",
            label="Analytical" if idx == num_steps else None,
        )
        ax.plot(
            r,
            f_values_all[idx],
            color=colors[idx],  # FIX: no modulo, just idx
            alpha=alpha,
            linestyle="-",
            label="Numerical" if idx == num_steps else None,
        )

    ax.set_xlabel("$r$")
    ax.set_ylabel("Solution $f(t,r)$")
    # ax.set_title("Evolution of Numerical and Analytical Solutions")
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="k", linestyle="-", label="Numerical"),
        Line2D([0], [0], color="k", linestyle="--", label="Analytical"),
    ]
    ax.legend(handles=legend_elements)
    # ax.set_ylim(-2, 2)
    ax.grid()
    fig.tight_layout()

    # Add colorbar for time gradient
    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n * delta_t)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$t$")

    plt.xlim(r_0, r_end)
    plt.ylim(0, 1.6)
    plt.grid(False)
    plt.show()
