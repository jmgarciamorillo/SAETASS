"""
Operator protocol module

Provides small, focused typing.Protocol to represent the minimal interface an
"operator" (subsolver) must provide to be used by the top-level Solver / TimeIntegrator.
The goal is to be permissive enough to support both several possible numerical
schemes (finite-difference / matrix-based or finite-volume).

Contents
- OperatorProtocol: base minimal protocol.
- helper is_stateful_operator(obj) -> bool
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from state import State


@runtime_checkable
class Operator(Protocol):
    """
    Minimal operator protocol describing metadata every operator should expose.

    Implementations should provide:
      - x_grid: np.ndarray  (spatial grid used by the operator)
      - t_grid: np.ndarray  (time grid used by the operator; may be coarse or refined)

    This protocol intentionally does not force particular time-stepping methods
    so it can be implemented by both FD/SubproblemSolver-based solvers and FV solvers.
    """

    x_grid: np.ndarray
    t_grid: np.ndarray
    state: State
    operator_params: dict

    def advance(self, n_steps: int) -> np.ndarray: ...

    def run(self) -> np.ndarray: ...


# ---------------------------
# Utilities
# ---------------------------


def is_operator(obj) -> bool:
    """Return True if object presents a stateful operator interface (duck-typed)."""
    return all(hasattr(obj, a) for a in ("x_grid", "t_grid", "step", "run"))
