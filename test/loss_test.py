import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import logging

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Grid import Grid
from State import State
from Solver import Solver

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("loss_test")

# Set plot style
# plt.rcParams.update(
#     {
#         "text.usetex": True,
#         "font.family": "serif",
#     }
# )


def analytical_solution(p, p_end, p0, Q0, b0, alpha, beta):
    """
    Analytical solution for the steady-state case.

    For the equation: df/dt + d(P_dot*f)/dp = Q, the steady-state solution
    (df/dt = 0) with P_dot = -b0*(p/p0)^beta and Q = Q0*(p/p0)^(-alpha) is:

    f(p) = (Q0*p0)/(b0*(alpha-1)) * ((p_end/p0)^(1-alpha)-(p/p0)^(1-alpha)) * (p/p0)^(-beta)

    This is valid when alpha != 1.
    """
    if abs(alpha - 1) < 1e-10:
        # Handle the special case alpha = 1
        return (Q0 * p0 / b0) * np.log(p_end / p) * (p / p0) ** (-beta)
    else:
        return (
            (Q0 * p0 / ((1 - alpha) * b0))
            * ((p_end / p0) ** (1 - alpha) - (p / p0) ** (1 - alpha))
            * (p / p0) ** (-beta)
        )


def run_loss_source_test():
    """
    Test the momentum loss equation with source term using the modern framework.

    This test solves: df/dt + d(P_dot*f)/dp = Q
    Where:
    - P_dot = -b0 * (p/p0)^beta (loss term)
    - Q = Q0 * (p/p0)^(-alpha) (source term)

    We initialize with zero everywhere and observe evolution toward steady state.
    """
    # Parameters
    p_min = 1.0  # Start of the domain
    p_max = 1000.0  # End of the domain
    num_points = 4000  # Number of points in the momentum grid
    t_max = 0.2  # Maximum simulation time
    num_timesteps = 20000  # Number of time steps

    # Physics parameters
    beta = 2.0  # Exponent for momentum loss rate
    alpha = 4.0  # Exponent for injection source
    b0 = 1.0  # Coefficient for loss rate
    Q0 = 1.0  # Coefficient for source
    p0 = 1.0  # Reference momentum

    # Create grid (logarithmic spacing for momentum)
    grid = Grid.log_spaced(
        p_min=p_min,
        p_max=p_max,
        num_p_cells=num_points,
        t_min=0.0,
        t_max=t_max,
        num_timesteps=num_timesteps,
    )

    # Initial condition: zero everywhere
    f_init = np.zeros(num_points)
    state = State(f_init)

    p_centers = grid._p_centers_phys

    # Create the momentum loss rate function: P_dot = -b0 * (p/p0)^beta
    P_dot = -b0 * (p_centers / p0) ** beta

    # Create the source function: Q = Q0 * (p/p0)^(-alpha)
    Q_values = Q0 * (p_centers / p0) ** (-alpha)

    # Configure loss operator parameters
    loss_params = {
        "P_dot": P_dot,
        "limiter": "minmod",
        "cfl": 0.8,
        "inflow_value_f": 0.0,
        "order": 2,  # Second-order scheme
    }

    # Configure source operator parameters
    source_params = {
        "source": Q_values,  # Source term values
    }

    # Set up the combined operator parameters
    operator_params = {
        "loss": loss_params,
        "source": source_params,
    }

    # Create the solver with both loss and source operators
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="loss-source",
        operator_params=operator_params,
    )

    # Storage for plotting
    saved_states = []
    saved_times = []
    saved_integrals = []
    saved_total_flux = []

    # Define steps to save (more points at beginning to capture rapid changes)
    time_points = np.concatenate(
        [
            np.linspace(0, 0.05, 5),  # More detail at beginning
            np.linspace(0.05, t_max, 15)[1:],  # Rest of evolution
        ]
    )

    # Convert to steps
    save_steps = np.unique((time_points / t_max * num_timesteps).astype(int))

    # Add initial state
    saved_states.append(state.f.copy())
    saved_times.append(0.0)
    saved_integrals.append(np.trapz(state.f * p_centers**2, p_centers))
    saved_total_flux.append(np.trapz(Q_values * p_centers**2, p_centers))

    plt.title("Initial parameter setup for Loss and Source Test")
    plt.loglog(p_centers, Q_values, label="Source Q(p)")
    plt.loglog(p_centers, -P_dot, label="Loss Rate |P_dot(p)|")
    plt.xlabel("Momentum p")
    plt.ylabel("Values")
    plt.legend()
    plt.show()

    # Run the simulation
    current_step = 0
    for step in save_steps[1:]:
        steps_to_advance = step - current_step
        if steps_to_advance > 0:
            solver.step(steps_to_advance)
            current_step = step
            saved_states.append(state.f.copy())
            saved_times.append(step * t_max / num_timesteps)
            saved_integrals.append(np.trapz(state.f * p_centers**2, p_centers))
            saved_total_flux.append(np.trapz(Q_values * p_centers**2, p_centers))

    # Compute analytical solution for comparison
    p_values = grid._p_centers_phys
    f_analytical = analytical_solution(p_values, p_max, p0, Q0, b0, alpha, beta)

    # Create plot with numerical and analytical solutions
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a colormap for the different time steps
    colors = plt.cm.viridis(np.linspace(0, 1, len(saved_states)))

    # Plot numerical solutions
    for i, (f_values, time) in enumerate(zip(saved_states, saved_times)):
        alpha_value = 0.3 + 0.7 * (i + 1) / len(saved_states)

        # For the first and last state, add a label
        if i == 0 or i == len(saved_states) - 1:
            label = f"t = {time:.3f}"
        else:
            label = None

        ax.loglog(
            p_values,
            p_values**5 * f_values[0],  # p^5 * f for better visualization
            color=colors[i],
            alpha=alpha_value,
            linewidth=1.5,
            label=label,
        )

    # Plot analytical steady-state solution
    ax.loglog(
        p_values,
        p_values**5 * f_analytical,
        "k--",
        linewidth=2.5,
        label="Analytical (steady state)",
    )

    # Set plot labels and limits
    ax.set_xlabel("Momentum $p$")
    ax.set_ylabel("$p^5 f(p)$")
    ax.set_xlim(p_min, p_max)
    ax.set_ylim(1e-13, 1e0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add a colorbar for time
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=t_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Time")

    plt.title(
        f"Evolution of Momentum Distribution with Losses and Sources\n"
        f"$P_{{dot}}=-b_0(p/p_0)^{{{beta}}}, Q=Q_0(p/p_0)^{{-{alpha}}}$"
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig("loss_source_test.png", dpi=150)
    plt.show()

    # Plot relative error between final state and analytical solution
    fig, ax = plt.subplots(figsize=(10, 6))
    final_state = saved_states[-1][0]
    relative_error = np.abs((final_state - f_analytical) / (f_analytical + 1e-15))

    ax.loglog(p_values, relative_error, "b-", linewidth=2)
    ax.set_xlabel("Momentum $p$")
    ax.set_ylabel("Relative Error")
    ax.set_xlim(p_min, p_max)
    ax.grid(True, alpha=0.3)

    plt.title("Relative Error: Numerical vs. Analytical Solution")
    plt.tight_layout()
    plt.savefig("loss_source_error.png", dpi=150)
    plt.show()

    # Also create a 2D visualization showing time evolution on a colormap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create mesh grid for 2D plot
    p_mesh, t_mesh = np.meshgrid(p_values, saved_times)

    # Prepare data for 2D plot
    f_2d = np.array([f[0] for f in saved_states])

    # Create 2D color plot with p^5*f
    cax = ax.pcolormesh(
        p_mesh,
        t_mesh,
        p_mesh**5 * f_2d,
        norm=mpl.colors.LogNorm(vmin=1e-6, vmax=1e-1),
        cmap="viridis",
        shading="auto",
    )

    # Set axes to logarithmic for momentum
    ax.set_xscale("log")
    ax.set_xlabel("Momentum $p$")
    ax.set_ylabel("Time $t$")

    # Add colorbar
    cbar = fig.colorbar(cax, label="$p^2 f(p,t)$")

    plt.title("Time Evolution of Momentum Distribution")
    plt.tight_layout()
    plt.savefig("loss_source_evolution.png", dpi=150)
    plt.show()

    # plot integral and total flux over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(saved_times, saved_integrals, "b-", label="Integral of p^2 * f")
    ax.plot(saved_times, saved_total_flux, "r--", label="Total Flux of Q")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Values")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.title("Integral of p^2 * f and Total Flux of Q over Time")
    plt.tight_layout()
    plt.savefig("loss_source_integral_flux.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_loss_source_test()
