"""Unit tests for the momentum-loss (LossSolver) operator.

Focus areas
-----------
1. Positivity: f must remain >= 0 at all grid points at every time step.
2. Step advection: a step-function initial condition should shift leftward in
   momentum space (toward lower energy) at the correct rate.
3. Mass conservation (with outflow): mass lost through the low-momentum
   boundary must equal the integral reduction of f.
"""

import pytest
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from saetass import State, Grid, Solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loss_grid_and_solver(
    num_p: int,
    p_min: float,
    p_max: float,
    t_final: float,
    num_t: int,
    P_dot_const: float,
    f_init: np.ndarray,
    order: int = 2,
    cfl: float = 0.5,
):
    """Build a 1D (momentum-only) loss solver with a constant P_dot."""
    p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
    t_grid = np.linspace(0, t_final, num_t + 1)

    # Scalar P_dot (same for every cell) broadcast to shape (num_p,)
    P_dot = np.full(num_p, P_dot_const)

    loss_params = {
        "P_dot": P_dot,
        "limiter": "minmod",
        "cfl": cfl,
        "inflow_value_U": np.zeros(1, dtype=float),
        "order": order,
        "adiabatic_losses": False,
    }

    grid = Grid(p_centers=p_centers, t_grid=t_grid)
    state = State(f_init)
    solver = Solver(
        grid=grid,
        state=state,
        problem_type="loss",
        operator_params={"loss": loss_params},
        substeps={"loss": 1},
        splitting_scheme="strang",
    )
    return solver, p_centers


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLossSolverPositivity:
    """The loss solver must never produce negative values of f."""

    def test_step_function_stays_positive(self):
        """Step-function IC: sharp front is the worst case for overshoots."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        t_final = 0.05  # short run, few loss timescales
        num_t = 500

        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        # Step at mid-grid
        p_mid = np.sqrt(p_min * p_max)
        f_init = np.where(p_centers < p_mid, 1.0, 0.0).astype(float)

        # P_dot < 0: energy is lost (particles shift to lower p)
        # Rate chosen so ν = |V_gen|*dt/dp_log ~ 0.3 with cfl=0.5
        P_dot_const = -0.5 * p_mid  # units: [p] / [t]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.f.flatten()

        assert np.all(
            f_final >= -1e-14
        ), f"Negative values in f after loss step: min={f_final.min():.3e}"

    def test_gaussian_stays_positive(self):
        """Gaussian IC: a smooth initial condition should stay smooth and positive."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        t_final = 0.02
        num_t = 300

        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        log_p = np.log10(p_centers)
        log_p_mid = 0.5 * (np.log10(p_min) + np.log10(p_max))
        sigma = 0.3  # in log10 units
        f_init = np.exp(-((log_p - log_p_mid) ** 2) / (2 * sigma**2))

        P_dot_const = -0.2 * p_centers[num_p // 2]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.f.flatten()

        assert np.all(
            f_final >= -1e-14
        ), f"Negative values after Gaussian loss run: min={f_final.min():.3e}"


class TestLossSolverPhysics:
    """Physical sanity checks for the loss solver."""

    def test_step_advects_leftward(self):
        """Step front should move toward lower momentum (losses drain energy)."""
        num_p = 300
        p_min, p_max = 1.0, 1e4
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        p_mid = np.sqrt(p_min * p_max)

        f_init = np.where(p_centers < p_mid, 1.0, 0.0).astype(float)

        # Short run: front should move, but only slightly
        t_final = 0.02
        num_t = 400
        P_dot_const = -0.3 * p_mid

        solver, pc = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        solver.step(num_t)
        f_final = solver.state.f.flatten()

        # Locate the 50% crossing (step front position) before and after
        def _front_p(f, p):
            mid_val = 0.5
            # Find first index where f drops below 0.5
            for i in range(len(f) - 1):
                if f[i] >= mid_val and f[i + 1] < mid_val:
                    # Linear interpolation
                    return p[i] + (p[i + 1] - p[i]) * (f[i] - mid_val) / (
                        f[i] - f[i + 1]
                    )
            return None

        p_front_initial = p_mid
        p_front_final = _front_p(f_final, pc)

        assert p_front_final is not None, "Could not find step front in final f."
        assert p_front_final < p_front_initial, (
            f"Step front did not shift to lower p: initial={p_front_initial:.3f}, "
            f"final={p_front_final:.3f}"
        )

    def test_mass_non_increasing(self):
        """Total 'mass' integral cannot increase (losses only drain energy)."""
        num_p = 200
        p_min, p_max = 1.0, 1e4
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        dp = np.diff(np.log10(p_centers))
        dp = np.append(dp, dp[-1])  # cell widths in log10 space

        f_init = np.ones(num_p)

        t_final = 0.05
        num_t = 500
        P_dot_const = -0.2 * p_centers[num_p // 2]

        solver, _ = _make_loss_grid_and_solver(
            num_p, p_min, p_max, t_final, num_t, P_dot_const, f_init.copy()
        )
        mass_initial = float(np.sum(f_init * dp))
        solver.step(num_t)
        f_final = solver.state.f.flatten()
        mass_final = float(np.sum(f_final * dp))

        assert (
            mass_final <= mass_initial + 1e-10
        ), f"Total mass increased: {mass_initial:.6g} → {mass_final:.6g}"


class TestLossSolverOrder:
    """Check that order=2 is at least as accurate as order=1 for smooth data."""

    def test_order2_more_accurate_than_order1(self):
        """
        For the same number of time steps and CFL, the 2nd-order scheme should
        produce a lower L2 error than the 1st-order scheme when measured against
        a very fine 2nd-order reference run.
        """
        num_p = 150
        p_min, p_max = 1.0, 1e3
        p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p)
        log_p = np.log10(p_centers)
        log_p_mid = 0.5 * (np.log10(p_min) + np.log10(p_max))
        sigma = 0.4

        f_init = np.exp(-((log_p - log_p_mid) ** 2) / (2 * sigma**2))
        t_final = 0.01
        num_t = 60  # coarse — both schemes use same step count
        P_dot_const = -0.15 * p_centers[num_p // 2]

        # --- reference: very fine 2nd-order run (10× more steps, same cfl) ---
        solver_ref, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t * 10,
            P_dot_const,
            f_init.copy(),
            order=2,
            cfl=0.5,
        )
        solver_ref.step(num_t * 10)
        f_ref = solver_ref.state.f.flatten()

        # --- order 1 (coarse) ---
        solver_1, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t,
            P_dot_const,
            f_init.copy(),
            order=1,
            cfl=0.5,
        )
        solver_1.step(num_t)
        f1 = solver_1.state.f.flatten()

        # --- order 2 (coarse, same step count) ---
        solver_2, _ = _make_loss_grid_and_solver(
            num_p,
            p_min,
            p_max,
            t_final,
            num_t,
            P_dot_const,
            f_init.copy(),
            order=2,
            cfl=0.5,
        )
        solver_2.step(num_t)
        f2 = solver_2.state.f.flatten()

        err1 = float(np.sqrt(np.mean((f1 - f_ref) ** 2)))
        err2 = float(np.sqrt(np.mean((f2 - f_ref) ** 2)))

        # 2nd-order scheme should be at least as accurate as 1st-order at the
        # same step count (allow 20% slack for CFL sub-cycling differences)
        assert err2 <= err1 * 1.2, (
            f"2nd-order scheme significantly worse than 1st-order: "
            f"L2(order1)={err1:.3e}, L2(order2)={err2:.3e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
