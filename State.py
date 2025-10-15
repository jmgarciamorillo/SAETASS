from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

"""
/Users/jmorillo/SolverAlpha/SolverAlpha/State.py

A lightweight State class to track solution and substep history for
a finite-volume method (FVM) with operator splitting.

Design goals:
- Hold conserved variables U as a numpy array (shape: (nvars, ncells))
- Track time, current timestep, and stage index/name
- Record and restore substep snapshots (history) for operator-split stages
- Small, well-documented API for typical FVM routines
"""


@dataclass
class State:
    """
    Container for the solution state used in FVM operator splitting.

    Attributes:
        U (np.ndarray): Conserved variables array, shape (nvars, ncells).
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
        # Ensure f is a writable float64 numpy array with shape (nvars, ncells)
        self.f = np.array(self.f, dtype=float, copy=True)
        # if self.f.ndim != 2:
        #    raise ValueError("f must be a 2D array with shape (nvars, ncells)")

    @property
    def nvars(self) -> int:
        return self.f.shape[0]

    @property
    def ncells(self) -> int:
        return self.f.shape[1]

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

    def to_vector(self, order: str = "C") -> np.ndarray:
        """Flatten f to a 1D vector (useful for linear algebra ops)."""
        return self.f.ravel(order=order)

    def from_vector(self, vec: Sequence[float], order: str = "C"):
        """Load flattened data into f (must match size)."""
        arr = np.asarray(vec, dtype=float)
        if arr.size != self.f.size:
            raise ValueError("Vector size does not match f.size")
        self.f = arr.reshape(self.f.shape, order=order)

    # def apply_update(
    #     self, df: Union[np.ndarray, Sequence[float]], inplace: bool = True
    # ) -> "State":
    #     """
    #     Apply an update to f. df must have same shape as f (or be a flattened sequence).
    #     If inplace is False, returns a new State with updated f.
    #     """
    #     df_arr = np.asarray(df, dtype=float)
    #     if df_arr.shape != self.f.shape:
    #         # try flattened
    #         if df_arr.size == self.f.size:
    #             df_arr = df_arr.reshape(self.f.shape)
    #         else:
    #             raise ValueError("df must have the same shape as f")
    #     if inplace:
    #         self.f += df_arr
    #         return self
    #     else:
    #         new = self.clone(copy_history=False)
    #         new.f = new.f + df_arr
    #         return new

    def update_f(self, new_f: np.ndarray):
        """Replace f with new_f (must have same shape)."""
        new_f_arr = np.asarray(new_f, dtype=float)
        if new_f_arr.shape != self.f.shape:
            raise ValueError("new_f must have the same shape as f")
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
            f"State(nvars={self.nvars}, ncells={self.ncells}, t={self.t:.6g}, "
            f"dt={self.dt:.6g}, stage={self.stage}, snapshots={len(self.history)})"
        )
