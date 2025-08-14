import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.sparse import diags
import math
import matplotlib as mpl

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


# Parameters
r_0 = 0.0  # Start of the domain
r_end = 3.0  # End of the domain
num_points = 1000  # Number of points in the spatial grid


# Create spatial grid
r = np.linspace(r_0, r_end, num_points)


# Define the initial test profile
# Example: steady state
# Parameters for the test profile
def f_0(r):
    return np.exp(-0.5 * ((r - 0.3) / 0.1) ** 2)


f_values = f_0(r)

# Create v_field: v_field = 1/r, but constant for r < 0.5
# v_field = np.zeros_like(r)
# v_field[r >= 0.5] = 1.0 / r[r >= 0.5]
# v_field[r < 0.5] = 1.0 / 0.5  # constant value for r < 0.5
# v_field_n = v_field.copy()
# v_field_n1 = v_field.copy()

v_field = np.ones_like(r)
v_field_n = np.ones_like(r)  # Initialize v_field_n as ones
v_field_n1 = np.ones_like(r)  # Initialize v_field_n1 as ones

Q = np.zeros(len(r) + 1)  # Initialize Q as a zero array

# Plot the initial profile
plt.plot(r, f_values, label="Initial Profile")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.xlabel("r")
plt.ylabel("f")
plt.legend()
plt.show()


plt.plot(r, v_field, label="v_field")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.xlabel("r")
plt.ylabel("v_field")
plt.legend()
plt.show()

# Save the initial profile to a file (optional)
# np.savetxt("initial_profile.txt", np.column_stack((r, f_values)), header="r f_values")


##


def compute_crank_nicolson_coefficients(
    delta_t, delta_r, timestep, num_points, v_field_n, v_field_n1
):
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
    a_n = np.zeros(num_points)
    b_n = np.zeros(num_points)
    a_n1 = np.zeros(num_points)
    b_n1 = np.zeros(num_points)

    for i in range(num_points):
        a_n[i] = 4 * i**2 * v_field_n[i] * delta_t / ((2 * i - 1) ** 2 * delta_r)
        b_n[i] = 4 * i**2 * v_field_n[i] * delta_t / ((2 * i + 1) ** 2 * delta_r)
        a_n1[i] = 4 * i**2 * v_field_n1[i] * delta_t / ((2 * i - 1) ** 2 * delta_r)
        b_n1[i] = 4 * i**2 * v_field_n1[i] * delta_t / ((2 * i + 1) ** 2 * delta_r)

    return (a_n, b_n, a_n1, b_n1)


def compute_crank_nicolson_matrix(n, a_n, b_n, a_n1, b_n1):
    """
    Compute the Crank-Nicolson matrix for the given timestep, s, and q coefficients.

    Args:
        timestep (float): Time step size.
        s (array): Array of s coefficients.
        q (array): Array of q coefficients.

    Returns:
        tuple: Two matrices (B, tildeB) for the Crank-Nicolson scheme.
    """

    # Diagonals for B
    main_diag_C = 1 - a_n
    lower_diag_C = 1 + b_n[:-1]

    # Diagonals for B_tilde
    main_diag_C_tilde = 1 + a_n1
    lower_diag_C_tilde = 1 - b_n1[:-1]

    # Construct sparse matrices for C and C_tilde (lower triangular)
    C = diags([main_diag_C, lower_diag_C], offsets=[0, -1], format="csr")
    C_tilde = diags(
        [main_diag_C_tilde, lower_diag_C_tilde], offsets=[0, -1], format="csr"
    )

    return C.toarray(), C_tilde.toarray()


def compute_rhs(f_values, delta_t, C, Q):
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

    return np.dot(C, f_values) + delta_t * (
        Q[1:] + Q[:-1]
    )  # NEED TO CHANGE THE Q PART to include time dependence


if __name__ == "__main__":
    # Define simulation parameters
    delta_t = 0.0001  # Time step size
    delta_r = (r[-1] - r[0]) / (len(r) - 1)  # Spatial step size

    # --- Store solutions for summary plot ---
    saved_f_values = []
    saved_times = []

    # Add initial condition
    saved_f_values.append(f_values.copy())
    saved_times.append(0.0)

    # Loop over time steps
    for n in range(1, 20000):
        print(f"Time step {n}\n")
        # Compute Crank-Nicolson coefficients
        a_n, b_n, a_n1, b_n1 = compute_crank_nicolson_coefficients(
            delta_t, delta_r, n, num_points, v_field_n, v_field_n1
        )

        # Print coefficients for debugging
        # print("Crank-Nicolson Coefficients:")
        # print("d:", d)

        # Compute Crank-Nicolson matrices
        C, tildeC = compute_crank_nicolson_matrix(n, a_n, b_n, a_n1, b_n1)

        # Compute the right-hand side of the equation
        rhs = compute_rhs(f_values, delta_t, C, Q)

        # Solve the linear system A * f_new = rhs
        f_new = solve(tildeC, rhs, assume_a="tridiagonal")

        # Update f_values for the next time step
        f_values = f_new

        # --- Store solutions for summary plot ---
        if n % 1000 == 0:
            saved_f_values.append(f_values.copy())
            saved_times.append(n * delta_t)

        # Visualize the simulation every 200 time steps
        if n % 1000 == 0:
            t = n * delta_t
            u0 = 1.0  # constant speed
            r_shifted = r - u0 * t
            # Avoid division by zero and negative arguments for f0
            mask = r_shifted > 0
            f_analytical = np.zeros_like(r)
            f_analytical[mask] = ((r[mask] - u0 * t) / r[mask]) ** 2 * f_0(
                r_shifted[mask]
            )

            plt.clf()
            plt.plot(r, f_values, label=f"Numerical (Time step {n})")
            plt.plot(
                r, f_analytical, "--", label="Analytical", color="black", linewidth=2
            )
            plt.xlabel("r")
            plt.ylabel("f")
            plt.title(f"Profile at time step {n}")
            plt.legend()
            plt.pause(0.1)

    plt.show()

    # --- Summary plot of solutions ---
    plot_steps = 20
    plot_stride = (
        len(saved_f_values) // plot_steps if len(saved_f_values) > plot_steps else 1
    )
    indices = np.arange(0, len(saved_f_values), plot_stride)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))

    fig, ax = plt.subplots(figsize=(6, 3.5))

    for i, idx in enumerate(indices):
        t = saved_times[idx]
        u0 = 1.0
        r_shifted = r - u0 * t
        mask = r_shifted > 0
        f_analytical = np.zeros_like(r)
        f_analytical[mask] = ((r[mask] - u0 * t) / r[mask]) ** 2 * f_0(r_shifted[mask])

        alpha = 0.2 + 0.8 * (i + 1) / len(indices)
        ax.plot(
            r,
            saved_f_values[idx],
            color=colors[i],
            alpha=alpha,
            linestyle="-",
            label="Numerical" if i == len(indices) - 1 else None,
        )
        ax.plot(
            r,
            f_analytical,
            color=colors[i],
            alpha=alpha,
            linestyle="--",
            label="Analytical" if i == len(indices) - 1 else None,
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

    # Add colorbar for time gradient
    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.rainbow,
        norm=plt.Normalize(vmin=0, vmax=2),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$t$")

    plt.xlim(r_0, r_end)
    plt.ylim(0, 1.1 * np.max([np.max(f) for f in saved_f_values]))
    plt.grid(False)
    plt.show()
