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
p_0 = 1.0  # Start of the domain
p_end = 1000.0  # End of the domain
num_points = 2000  # Number of points in the spatial grid
f_end = 0.0  # End value for f(p) at p_end

# Create spatial grid
p = np.linspace(p_0, p_end, num_points + 1)

# Define the initial test profile
# Example: steady state
beta = 2
alpha = 4
b0 = 1
Q0 = 1
p0 = 1
# Parameters for the test profile
f_values = 1 * np.zeros(num_points)

Q = Q0 * (p / p0) ** (-alpha)  # Q function
P_dot = -b0 * (p / p0) ** beta  # P_dot function


# Plot the initial profile
plt.plot(p, np.append(f_values, 0), label="Initial Profile")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.xlabel("p")
plt.ylabel("f")
plt.legend()
plt.show()

plt.loglog(p, Q, label="Q")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.xlabel("p")
plt.ylabel("Q")
plt.legend()
plt.show()

plt.loglog(p, np.abs(P_dot), label="P_dot")
# plt.semilogy(r, np.append(f_values, 0), label="Initial Profile")
plt.xlabel("p")
plt.ylabel("|P_dot|")
plt.legend()
plt.show()

# Save the initial profile to a file (optional)
# np.savetxt("initial_profile.txt", np.column_stack((r, f_values)), header="r f_values")


##


def compute_crank_nicolson_coefficients(delta_t, delta_p, timestep, P_dot):
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

    return np.delete(-P_dot * delta_t / delta_p, len(P_dot) - 1)


def compute_crank_nicolson_matrix(n, d):
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
    main_diag_B = 1 - d
    upper_diag_B = 1 + d[1:]

    # Diagonals for B_tilde
    main_diag_B_tilde = 1 + d
    upper_diag_B_tilde = 1 - d[1:]

    # Construct sparse matrices
    B = diags([main_diag_B, upper_diag_B], offsets=[0, 1], format="csr")
    B_tilde = diags(
        [main_diag_B_tilde, upper_diag_B_tilde], offsets=[0, 1], format="csr"
    )

    return B.toarray(), B_tilde.toarray()


def compute_rhs(f_values, delta_t, B, Q):
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

    return np.dot(B, f_values) + delta_t * (
        Q[1:] + Q[:-1]
    )  # NEED TO CHANGE THE Q PART to include time dependence


if __name__ == "__main__":
    # Define simulation parameters
    delta_t = 0.001  # Time step size
    delta_p = (p[-1] - p[0]) / (len(p) - 1)  # Spatial step size

    saved_f_values = []  # <-- Add this line to initialize the list

    # Loop over time steps
    for n in range(1, 400):  # Example: 10 time steps
        print(f"Time step {n}\n")
        # Compute Crank-Nicolson coefficients
        d = compute_crank_nicolson_coefficients(delta_t, delta_p, n, P_dot)

        # Print coefficients for debugging
        # print("Crank-Nicolson Coefficients:")
        # print("d:", d)

        # Compute Crank-Nicolson matrices
        B, tildeB = compute_crank_nicolson_matrix(n, d)

        # Compute the right-hand side of the equation
        rhs = compute_rhs(f_values, delta_t, B, Q)

        # Solve the linear system A * f_new = rhs
        f_new = solve(tildeB, rhs, assume_a="tridiagonal")

        # Update f_values for the next time step
        f_values = f_new

        # Save the solution at each step (or every N steps)
        if n % 10 == 0:
            saved_f_values.append(f_values.copy())  # <-- Save a copy

            # Analytical solution for comparison
            f_analitic = (
                Q0
                * p0
                / (1 - alpha)
                / b0
                * ((p_end / p0) ** (1 - alpha) - (p / p0) ** (1 - alpha))
                * (p / p0) ** (-beta)
            )  # Example analytical solution

            # Plot the updated profile (optional, can be removed)
            # plt.loglog(p, p**2 * np.append(f_values, 0), label=f"Time Step {n}")
            # plt.loglog(
            #     p,
            #     p**2 * f_analitic,
            #     label="Analytical Solution",
            # )
            # plt.ylim(10**-13, 10**0)
            # plt.legend()
            # plt.xlabel("p")
            # plt.ylabel("p^2 * f(p))")
            # plt.show()

    # Choose how many time steps to plot (e.g., every 500 steps)
    plot_steps = 40
    step_indices = np.linspace(0, len(saved_f_values) - 1, plot_steps, dtype=int)
    colors = plt.cm.rainbow(np.linspace(0, 1, plot_steps))

    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot the analytical solution (static, bold black)

    # Plot numerical solutions with fading color
    for i, idx in enumerate(step_indices):
        alpha = 0.2 + 0.8 * (i + 1) / plot_steps
        ax.loglog(
            p,
            p**5 * np.append(saved_f_values[idx], 0),
            color=colors[i],
            alpha=alpha,
            linewidth=1.5,
            label="Numerical" if i == plot_steps - 1 else None,
        )

    ax.loglog(
        p,
        p**5 * f_analitic,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Analytical",
    )

    ax.set_xlabel("$p$")
    ax.set_ylabel("$p^2 f(t,p)$")
    ax.set_ylim(1e-13, 1e0)
    ax.set_xlim(p_0, p_end)
    ax.grid(False)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=1.5, label="Numerical "
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=2, label="Analytical"
        ),
    ]
    ax.legend(handles=legend_elements)

    # Add colorbar for time gradient
    sm = mpl.cm.ScalarMappable(
        cmap=plt.cm.rainbow, norm=plt.Normalize(vmin=0, vmax=(n + 1) * delta_t)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("$t$")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
