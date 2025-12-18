from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

"""
/Users/jmorillo/SolverAlpha/SolverAlpha/State.py

A lightweight State class to track solution and substep history for
a finite-volume method (FVM) with operator splitting.

Design goals:
- Hold conserved variables U as a numpy array (shape: (n_p, n_r))
- Track time, current timestep, and stage index/name
- Record and restore substep snapshots (history) for operator-split stages
- Small, well-documented API for typical FVM routines
"""


@dataclass
class State:
    """
    Container for the solution state used in FVM operator splitting.

    Attributes:
        U (np.ndarray): Conserved variables array, shape (n_p, n_r).
        t (float): Current physical time.
        dt (float): Current timestep size (most recent).
        stage (int): Current operator-splitting stage index.
        stage_name (str): Optional name/label for the current stage.
        history (List[Dict[str, Any]]): Saved snapshots for substeps.
    """

    f: np.ndarray
    t: float = 0.0
    dt: float = 0.0
    stage: int = 0
    stage_name: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Ensure f is a writable float64 numpy array with shape (n_p, n_r)
        self.f = np.array(self.f, dtype=float, copy=True)
        if self.f.ndim == 1:
            self.ndim = 1
            self.f = self.f.reshape((1, self.f.size))  # make it 2D with n_p=1
            logger.debug("Reshaped f to 2D array with n_p=1")
        elif self.f.ndim != 2:
            raise NotImplementedError(
                "f must be a 1D or 2D array at this version of the code"
            )
        else:
            self.ndim = 2
            logger.debug(f"Initialized State with f shape: {self.f.shape}")
            logger.debug(f"This is a multi-species problem with n_p={self.f.shape[0]}")

    @property
    def n_p(self) -> int:
        return self.f.shape[0]

    @property
    def n_r(self) -> int:
        return self.f.shape[1]

    @property
    def grid_shape(self) -> tuple:
        return self.f.shape

    def clone(self, copy_history: bool = False) -> "State":
        """Return a deep copy of this State. By default history is not copied."""
        new_state = State(
            f=self.f.copy(),
            t=float(self.t),
            dt=float(self.dt),
            stage=int(self.stage),
            stage_name=str(self.stage_name),
        )
        if copy_history:
            # deep-copy each snapshot
            new_state.history = [
                {
                    "t": snap["t"],
                    "dt": snap["dt"],
                    "stage": snap["stage"],
                    "stage_name": snap.get("stage_name", ""),
                    "f": snap["f"].copy(),
                }
                for snap in self.history
            ]
        return new_state

    def get_f(self) -> np.ndarray:
        """Return a proper np.array for computation."""
        if self.ndim == 1:
            return self.f[0]  # return 1D array
        return self.f

    # def to_vector(self, order: str = "C") -> np.ndarray:
    #     """Flatten f to a 1D vector (useful for linear algebra ops)."""
    #     return self.f.ravel(order=order)

    # def from_vector(self, vec: Sequence[float], order: str = "C"):
    #     """Load flattened data into f (must match size)."""
    #     arr = np.asarray(vec, dtype=float)
    #     if arr.size != self.f.size:
    #         raise ValueError("Vector size does not match f.size")
    #     self.f = arr.reshape(self.f.shape, order=order)

    def update_f(self, new_f: np.ndarray):
        """Replace f with new_f (must have same shape)."""
        new_f_arr = np.asarray(new_f, dtype=float)
        if new_f_arr.shape != self.f.shape:
            # Handle the case of reshaping 1D to 2D if needed
            if self.ndim == 1 and new_f_arr.ndim == 1 and new_f_arr.size == self.f.size:
                new_f_arr = new_f_arr.reshape((1, new_f_arr.size))
            else:
                raise ValueError(
                    f"new_f must have shape {self.f.shape}, got {new_f_arr.shape}"
                )
        self.f = new_f_arr

    def advance_time(self, dt: float):
        """Advance time by dt and store the last dt."""
        self.t += float(dt)
        self.dt = float(dt)

    def record_substep(self, stage_name: Optional[str] = None):
        """
        Save a snapshot of the current state for later restore/inspection.
        Each snapshot contains a copy of f and metadata: t, dt, stage, stage_name.
        """
        entry = {
            "t": float(self.t),
            "dt": float(self.dt),
            "stage": int(self.stage),
            "stage_name": (
                stage_name if stage_name is not None else str(self.stage_name)
            ),
            "f": self.f.copy(),
        }
        self.history.append(entry)

    def restore_substep(self, identifier: Union[int, str]):
        """
        Restore a saved snapshot. identifier may be:
          - int: index into history (0-based)
          - str: stage_name to search for (first match)
        Raises IndexError/ValueError if not found.
        """
        if isinstance(identifier, int):
            snap = self.history[identifier]
        else:
            # search by name
            matches = [h for h in self.history if h.get("stage_name") == identifier]
            if not matches:
                raise ValueError(f"No snapshot found with stage_name={identifier!r}")
            snap = matches[0]
        # restore
        self.f = snap["f"].copy()
        self.t = float(snap["t"])
        self.dt = float(snap["dt"])
        self.stage = int(snap["stage"])
        self.stage_name = str(snap.get("stage_name", ""))
        return self

    def get_substep(self, index: int) -> Dict[str, Any]:
        """Return a copy of the snapshot at index (does not modify current state)."""
        snap = self.history[index]
        return {
            "t": float(snap["t"]),
            "dt": float(snap["dt"]),
            "stage": int(snap["stage"]),
            "stage_name": str(snap.get("stage_name", "")),
            "f": snap["f"].copy(),
        }

    def clear_history(self):
        """Remove all saved substeps."""
        self.history.clear()

    def step_stage(self, stage_name: Optional[str] = None):
        """Increment stage index and optionally set a new stage_name."""
        self.stage += 1
        if stage_name is not None:
            self.stage_name = stage_name

    def __repr__(self) -> str:
        return (
            f"State(n_r={self.n_r}, n_p={self.n_p}, t={self.t:.6g}, "
            f"dt={self.dt:.6g}, stage={self.stage}, snapshots={len(self.history)})"
        )


class SliceState:
    """TEMPORARY SOLUTION: Helper class to represent a single slice of a 2D State."""

    def __init__(self, full_state: State, idx: int, axis: int = 1):
        self.full_state = full_state
        self.idx = idx
        self.axis = axis
        if self.axis == 1:
            self.f = (
                full_state.f[idx].copy()
                if full_state.f.ndim > 1
                else full_state.f.copy()
            )
        elif self.axis == 0:
            self.f = (
                full_state.f[:, idx].copy()
                if full_state.f.ndim > 1
                else full_state.f.copy()
            )
        else:
            raise ValueError("axis must be 0 or 1")

    def get_f(self) -> np.ndarray:
        """Return a proper np.array for computation."""
        return self.f  # return 1D array

    def update_f(self, new_f):
        """Update only the slice of the full state."""
        new_f_arr = np.asarray(new_f, dtype=float)
        if self.axis == 1:
            # Space problem: update row idx
            expected_shape = (self.full_state.n_r,)
            if new_f_arr.shape != expected_shape:
                raise ValueError(
                    f"new_f must have shape {expected_shape}, got {new_f_arr.shape}"
                )
            if self.full_state.ndim > 1:
                self.full_state.f[self.idx] = new_f_arr
            else:
                self.full_state.f[0] = new_f_arr
        elif self.axis == 0:
            # Momentum problem: update column idx
            expected_shape = (self.full_state.n_p,)
            if new_f_arr.shape != expected_shape:
                raise ValueError(
                    f"new_f must have shape {expected_shape}, got {new_f_arr.shape}"
                )
            if self.full_state.ndim > 1:
                self.full_state.f[:, self.idx] = new_f_arr
            else:
                self.full_state.f[0] = new_f_arr
        else:
            raise ValueError("axis must be 0 or 1")
        self.f = new_f_arr
