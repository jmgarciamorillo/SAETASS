import numpy as np
import pytest

from saetass import Grid, Solver, State


def test_dynamic_advection_equivalence():
    """Test that a callable advection velocity gives the same result as a static array."""
    num_r, r_end, t_final = 500, 10.0, 1.0
    v_const = 5.0
    r_initial_peak = 2.0
    sigma = 0.5
    r_grid = np.linspace(0.0, r_end, num_r)
    t_grid = np.linspace(0, t_final, 500)

    f_init = np.exp(-((r_grid - r_initial_peak) ** 2) / (2 * sigma**2))

    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    # 1. Static solver
    state_static = State(f_init.copy())
    solver_static = Solver(
        grid=grid,
        state=state_static,
        problem_type="advection",
        operator_params={
            "advection": {
                "v_centers": np.full(num_r, v_const),
                "order": 1,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            }
        },
        substeps={"advection": 1},
        splitting_scheme="strang",
    )
    solver_static.step(len(t_grid) - 1)

    # 2. Dynamic solver (callable)
    def v_callable(t):
        return np.full(num_r, v_const)

    state_dynamic = State(f_init.copy())
    solver_dynamic = Solver(
        grid=grid,
        state=state_dynamic,
        problem_type="advection",
        operator_params={
            "advection": {
                "v_centers": v_callable,
                "order": 1,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            }
        },
        substeps={"advection": 1},
        splitting_scheme="strang",
    )
    solver_dynamic.step(len(t_grid) - 1)

    assert np.allclose(solver_static.state.f, solver_dynamic.state.f, atol=1e-12)


def test_dynamic_diffusion_equivalence():
    """Test that a callable diffusion coeff gives the same result as a static array."""
    num_r, r_end, t_final = 200, 10.0, 0.5
    D_const = 1.0
    r_grid = np.linspace(0.0, r_end, num_r)
    t_grid = np.linspace(0, t_final, 500)

    f_init = np.exp(-((r_grid - 5.0) ** 2) / 0.5)

    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    # 1. Static solver
    state_static = State(f_init.copy())
    solver_static = Solver(
        grid=grid,
        state=state_static,
        problem_type="diffusion",
        operator_params={
            "diffusion": {
                "D_values": np.full(num_r, D_const),
                "f_end": 0.0,
            }
        },
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )
    solver_static.step(len(t_grid) - 1)

    # 2. Dynamic solver
    def D_callable(t):
        return np.full(num_r, D_const)

    state_dynamic = State(f_init.copy())
    solver_dynamic = Solver(
        grid=grid,
        state=state_dynamic,
        problem_type="diffusion",
        operator_params={
            "diffusion": {
                "D_values": D_callable,
                "f_end": 0.0,
            }
        },
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )
    solver_dynamic.step(len(t_grid) - 1)

    assert np.allclose(solver_static.state.f, solver_dynamic.state.f, atol=1e-12)


def test_dynamic_source_equivalence():
    """Test that a callable source gives the same result as a static array."""
    num_r, r_end, t_final = 100, 10.0, 2.0
    Q_const = 2.0
    r_grid = np.linspace(0.0, r_end, num_r)
    t_grid = np.linspace(0, t_final, 100)

    f_init = np.zeros(num_r)

    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    # 1. Static solver
    state_static = State(f_init.copy())
    solver_static = Solver(
        grid=grid,
        state=state_static,
        problem_type="source",
        operator_params={
            "source": {
                "source": np.full(num_r, Q_const),
            }
        },
        substeps={"source": 1},
        splitting_scheme="strang",
    )
    solver_static.step(len(t_grid) - 1)

    # 2. Dynamic solver (callable taking 3 args r, p, t)
    def Q_callable(r, p, t):
        return np.full_like(r, Q_const)

    state_dynamic = State(f_init.copy())
    solver_dynamic = Solver(
        grid=grid,
        state=state_dynamic,
        problem_type="source",
        operator_params={
            "source": {
                "source": Q_callable,
            }
        },
        substeps={"source": 1},
        splitting_scheme="strang",
    )
    solver_dynamic.step(len(t_grid) - 1)

    assert np.allclose(solver_static.state.f, solver_dynamic.state.f, atol=1e-12)
    # Analytically: f(t=2) = f(0) + 2*2 = 4
    assert np.allclose(solver_static.state.f, 4.0, atol=1e-12)


def test_genuine_time_dependent_source():
    """Test a genuinely time-varying source Q(t) = t."""
    num_r, r_end, t_final = 100, 10.0, 2.0
    r_grid = np.linspace(0.0, r_end, num_r)
    t_grid = np.linspace(0, t_final, 200)

    f_init = np.zeros(num_r)
    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    def Q_callable(r, p, t):
        # Q(t) = t
        return np.full_like(r, t)

    state_dynamic = State(f_init.copy())
    solver_dynamic = Solver(
        grid=grid,
        state=state_dynamic,
        problem_type="source",
        operator_params={
            "source": {
                "source": Q_callable,
            }
        },
        substeps={"source": 1},
        splitting_scheme="strang",
    )
    solver_dynamic.step(len(t_grid) - 1)

    # Analytical: int_0^2 t dt = 2.0
    # Because SourceSolver uses an explicit left Riemann sum update f_new = f + S*dt evaluated at state.t:
    expected_sum = sum(
        t_grid[i] * (t_grid[i + 1] - t_grid[i]) for i in range(len(t_grid) - 1)
    )
    assert np.allclose(solver_dynamic.state.f, expected_sum, atol=1e-5)


def test_analytical_time_dependent_advection(plot_results):
    """Verifies that a time-dependent advection velocity u_w(t) = cos(t) matches the analytical solution."""
    num_r = 1000
    r_end = 10.0
    r_grid = np.linspace(0.0, r_end, num_r)

    t_checkpoints = [0.0, np.pi / 2, np.pi]
    t_final = np.pi

    # Ensure our time grid hits the checkpoints exactly
    t_grid = np.linspace(0.0, t_final, 2000)
    t_grid = np.unique(np.sort(np.append(t_grid, t_checkpoints)))

    r_c = 5.0
    sigma = 0.5
    f_init = np.exp(-((r_grid - r_c) ** 2) / (sigma**2))

    def u_w(t):
        return np.full_like(r_grid, np.cos(t))

    grid = Grid(r_centers=r_grid, t_grid=t_grid)
    state = State(f_init.copy())

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advection",
        operator_params={
            "advection": {
                "v_centers": u_w,
                "order": 2,  # Use higher order to reduce numerical diffusion
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            }
        },
        substeps={"advection": 1},
        splitting_scheme="strang",
    )

    current_step = 0
    saved_states = [f_init.copy()]

    for i in range(1, len(t_checkpoints)):
        target_t = t_checkpoints[i]
        # Find index in t_grid
        target_idx = np.where(t_grid == target_t)[0][0]
        steps_to_take = target_idx - current_step
        if steps_to_take > 0:
            solver.step(steps_to_take)
            current_step = target_idx

        saved_states.append(solver.state.f.copy())

    def analytical_solution(r, t):
        V = np.sin(t)
        # Avoid division by zero at r=0
        r_safe = r.copy()
        r_safe[r == 0] = 1e-10

        factor = ((r - V) ** 2) / (r_safe**2)
        f0 = np.exp(-((r - V - r_c) ** 2) / (sigma**2))
        ans = factor * f0
        ans[r == 0] = 0.0
        return ans

    if plot_results:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

    for i, t in enumerate(t_checkpoints):
        num_f = saved_states[i].flatten()
        ana_f = analytical_solution(r_grid, t)

        if plot_results:
            import matplotlib.pyplot as plt

            if i == 0:
                plt.plot(r_grid, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
            else:
                p = plt.plot(r_grid, num_f, label=f"t={t:.2f} Numerical")
                plt.plot(
                    r_grid,
                    ana_f,
                    "--",
                    color=p[0].get_color(),
                    label=f"t={t:.2f} Analytical",
                )

        if i > 0:
            # We use a loose absolute tolerance (0.05) to allow for the small peak suppression
            # inherent to FVM numerical diffusion with MUSCL scheme.
            assert np.allclose(num_f, ana_f, atol=0.08), f"Failed at t={t}"

    if plot_results:
        import matplotlib.pyplot as plt

        plt.title("Time-Dependent Advection: u_w(t) = cos(t)")
        plt.xlabel("r")
        plt.ylabel("f(r,t)")
        plt.legend()
        plt.grid(True)
        plt.show()


def test_analytical_time_dependent_diffusion(plot_results):
    """Verifies that a time-dependent diffusion coeff D(t) = D0*(1+sin(t)) matches the spherical analytical solution."""
    num_r = 1000
    r_end = 2.0
    r_grid = np.linspace(0.0, r_end, num_r)

    t_checkpoints = [0.0, 2.0, 5.0, 10.0]
    t_final = 10.0

    # Ensure our time grid hits the checkpoints exactly
    t_grid = np.linspace(0.0, t_final, 5000)
    t_grid = np.unique(np.sort(np.append(t_grid, t_checkpoints)))

    D0 = 0.01
    k = np.pi

    def analytical_solution(r, t):
        tau = D0 * (t + 1.0 - np.cos(t))
        ans = np.zeros_like(r)

        # Avoid division by zero
        mask = r > 0
        ans[mask] = (np.sin(k * r[mask]) / r[mask]) * np.exp(-(k**2) * tau)

        # Limit r->0 is k * exp(-k^2 * tau)
        ans[~mask] = k * np.exp(-(k**2) * tau)
        return ans

    f_init = analytical_solution(r_grid, 0.0)

    def D_callable(t):
        return np.full_like(r_grid, D0 * (1.0 + np.sin(t)))

    grid = Grid(r_centers=r_grid, t_grid=t_grid)
    state = State(f_init.copy())

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="diffusion",
        operator_params={
            "diffusion": {
                "D_values": D_callable,
                "f_end": 0.0,
            }
        },
        substeps={"diffusion": 1},
        splitting_scheme="strang",
    )

    current_step = 0
    saved_states = [f_init.copy()]

    for i in range(1, len(t_checkpoints)):
        target_t = t_checkpoints[i]
        target_idx = np.where(t_grid == target_t)[0][0]
        steps_to_take = target_idx - current_step
        if steps_to_take > 0:
            solver.step(steps_to_take)
            current_step = target_idx

        saved_states.append(solver.state.f.copy())

    if plot_results:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

    for i, t in enumerate(t_checkpoints):
        num_f = saved_states[i].flatten()
        ana_f = analytical_solution(r_grid, t)

        if plot_results:
            import matplotlib.pyplot as plt

            if i == 0:
                plt.plot(r_grid, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
            else:
                p = plt.plot(r_grid, num_f, label=f"t={t:.2f} Numerical")
                plt.plot(
                    r_grid,
                    ana_f,
                    "--",
                    color=p[0].get_color(),
                    label=f"t={t:.2f} Analytical",
                )

        if i > 0:
            # We use a very tight tolerance since Crank-Nicolson diffusion is very stable and accurate
            assert np.allclose(num_f, ana_f, atol=1e-3), f"Failed at t={t}"

    if plot_results:
        import matplotlib.pyplot as plt

        plt.title("Time-Dependent Diffusion: D(t) = D0*(1+sin(t))")
        plt.xlabel("r")
        plt.ylabel("f(r,t)")
        plt.legend()
        plt.grid(True)
        plt.show()


def test_manufactured_advection_source(plot_results):
    """
    Test a manufactured solution with space and time dependence for Advection + Source.
    Eq: df/dt + (1/r^2)*d/dr(r^2 * u_w * f) = Q(r,t)
    u_w(r,t) = r/(1+t)
    f_exact(r,t) = (2+cos(t))*exp(-r)
    Q(r,t) = exp(-r) * [ -sin(t) + ((2+cos(t))/(1+t)) * (3-r) ]
    Domain: r in [0, 10], t in [0, 2pi]
    """
    num_r = 1000
    r_end = 10.0
    r_grid = np.linspace(0.0, r_end, num_r)

    t_checkpoints = [0.0, np.pi, 2 * np.pi]
    t_final = 2 * np.pi

    t_grid = np.linspace(0.0, t_final, 5000)
    t_grid = np.unique(np.sort(np.append(t_grid, t_checkpoints)))

    def f_exact(r, t):
        return (2.0 + np.cos(t)) * np.exp(-r)

    f_init = f_exact(r_grid, 0.0)

    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    def u_w(t):
        return grid.r_centers / (1.0 + t)

    def Q_src(r, p, t):
        term1 = -np.sin(t)
        term2 = ((2.0 + np.cos(t)) / (1.0 + t)) * (3.0 - r)
        return np.exp(-r) * (term1 + term2)

    state = State(f_init.copy())

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="advection-source",
        operator_params={
            "advection": {
                "v_centers": u_w,
                "order": 2,
                "limiter": "minmod",
                "cfl": 0.8,
                "inflow_value_U": 0.0,
            },
            "source": {
                "source": Q_src,
            },
        },
        substeps={"advection": 1, "source": 1},
        splitting_scheme="strang",
    )

    current_step = 0
    saved_states = [f_init.copy()]

    for i in range(1, len(t_checkpoints)):
        target_t = t_checkpoints[i]
        target_idx = np.where(t_grid == target_t)[0][0]
        steps_to_take = target_idx - current_step
        if steps_to_take > 0:
            solver.step(steps_to_take)
            current_step = target_idx

        saved_states.append(solver.state.f.copy())

    if plot_results:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for i, t in enumerate(t_checkpoints):
        num_f = saved_states[i].flatten()
        ana_f = f_exact(r_grid, t)

        if plot_results:
            if i == 0:
                ax1.plot(r_grid, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
            else:
                p = ax1.plot(r_grid, num_f, label=f"t={t:.2f} Numerical")
                ax1.plot(
                    r_grid,
                    ana_f,
                    "--",
                    color=p[0].get_color(),
                    label=f"t={t:.2f} Analytical",
                )

            err = np.abs(num_f - ana_f)
            err = np.where(err < 1e-15, 1e-15, err)
            ax2.plot(r_grid, err, label=f"t={t:.2f} Abs Error")

        if i > 0:
            # Validate using relative L2 norm over the full domain.
            rel_l2 = np.sqrt(np.mean((num_f - ana_f) ** 2)) / (
                np.sqrt(np.mean(ana_f**2)) + 1e-300
            )
            assert rel_l2 < 0.02, (
                f"Advection manufactured test failed at t={t:.3f}: "
                f"relative_L2={rel_l2:.4f} > 0.02"
            )

    if plot_results:
        ax1.set_title("Manufactured Advection+Source: f(r,t)")
        ax1.set_xlabel("r")
        ax1.set_ylabel("f(r,t)")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Absolute Error |f_num - f_ana|")
        ax2.set_xlabel("r")
        ax2.set_ylabel("Error")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


def test_manufactured_diffusion_source(plot_results):
    """
    Test a manufactured solution with space and time dependence for Diffusion + Source.
    Eq: df/dt = (1/r^2)*d/dr(r^2 * D * df/dr) + Q(r,t)
    D(t) = D_0 * (1+t), D_0 = 0.01
    f_exact(r,t) = (2+cos(t))*exp(-r)
    Q(r,t) = exp(-r) * [ -sin(t) + D_0*(1+t)*(2+cos(t))*(2/r - 1) ]
    Domain: r in [0, 5.0]
    """
    num_r = 1000
    r_end = 5.0
    r_grid = np.linspace(0.0, r_end, num_r)

    t_checkpoints = [0.0, 1.0, 3.0, 5.0]
    t_final = 5.0

    t_grid = np.linspace(0.0, t_final, 5000)
    t_grid = np.unique(np.sort(np.append(t_grid, t_checkpoints)))

    D0 = 0.01

    def f_exact(r, t):
        return (2.0 + np.cos(t)) * np.exp(-(r**2))

    f_init = f_exact(r_grid, 0.0)

    grid = Grid(r_centers=r_grid, t_grid=t_grid)

    def D_callable(t):
        return np.full_like(grid.r_centers, D0 * (1.0 + t))

    def f_end_callable(t):
        return (2.0 + np.cos(t)) * np.exp(-(r_end**2))

    def Q_src(r, p, t):
        term1 = -np.sin(t)
        # Correct formula for Q = df/dt - ∇·(D∇f) with f = (2+cos t)*exp(-r^2)
        # ∇·(D∇f) = D * (2+cos t) * exp(-r^2) * 2(2r^2 - 3)
        # -∇·(D∇f) = D * (2+cos t) * exp(-r^2) * 2(3 - 2r^2)
        term2 = 2.0 * D0 * (1.0 + t) * (2.0 + np.cos(t)) * (3.0 - 2.0 * r**2)
        Q = np.exp(-(r**2)) * (term1 + term2)
        return Q

    state = State(f_init.copy())

    solver = Solver(
        grid=grid,
        state=state,
        problem_type="diffusion-source",
        operator_params={
            "diffusion": {
                "boundary_condition": "dirichlet",
                "D_values": D_callable,
                "f_end": f_end_callable,
            },
            "source": {
                "source": Q_src,
            },
        },
        substeps={"diffusion": 1, "source": 1},
        splitting_scheme="strang",
    )

    current_step = 0
    saved_states = [f_init.copy()]

    for i in range(1, len(t_checkpoints)):
        target_t = t_checkpoints[i]
        target_idx = np.where(t_grid == target_t)[0][0]
        steps_to_take = target_idx - current_step
        if steps_to_take > 0:
            solver.step(steps_to_take)
            current_step = target_idx

        saved_states.append(solver.state.f.copy())

    if plot_results:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for i, t in enumerate(t_checkpoints):
        num_f = saved_states[i].flatten()
        ana_f = f_exact(r_grid, t)

        if plot_results:
            if i == 0:
                ax1.plot(r_grid, num_f, "k--", label=f"t={t:.2f} (Num/Ana)")
            else:
                p = ax1.plot(r_grid, num_f, label=f"t={t:.2f} Numerical")
                ax1.plot(
                    r_grid,
                    ana_f,
                    "--",
                    color=p[0].get_color(),
                    label=f"t={t:.2f} Analytical",
                )

            err = np.abs(num_f - ana_f)
            err = np.where(err < 1e-15, 1e-15, err)
            ax2.plot(r_grid, err, label=f"t={t:.2f} Abs Error")

        if i > 0:
            # Validate using relative L2 norm over the full domain.
            rel_l2 = np.sqrt(np.mean((num_f - ana_f) ** 2)) / (
                np.sqrt(np.mean(ana_f**2)) + 1e-300
            )
            assert rel_l2 < 0.05, (
                f"Diffusion manufactured test failed at t={t:.3f}: "
                f"relative_L2={rel_l2:.4f} > 0.05"
            )

    if plot_results:
        ax1.set_title("Manufactured Diffusion+Source: f(r,t)")
        ax1.set_xlabel("r")
        ax1.set_ylabel("f(r,t)")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Absolute Error |f_num - f_ana|")
        ax2.set_xlabel("r")
        ax2.set_ylabel("Error")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Running tests with plotting enabled...")
    pytest.main([__file__, "--plot"])
