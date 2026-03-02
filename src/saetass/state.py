"""
This module provides a class and auxiliary methods to track the state of a solution
in a finite-volume method with operator splitting solver such as SAETASS. The :class:`State`
class is designed to support the following features:

- Hold the conserved variables (e.g., particle distribution function) as a ``Numpy`` array.
- Encapsulate the update and retrieval of the solution state as an API used by finite-volume
method routines.
- Track the current time of the solution.
- Record and restore snapshots of the solution at different stages of the operator splitting,
allowing for detailed analysis and debugging.

________________
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    Main class to serve as a container and tracker of solution state.

    A :class:`State` instance is initialized for specific domain parameters, which are implicit in the shape of the
    distribution function array f. The array is expected to match the size and shape of the :class:`Grid` object
    used in the :class:`Solver`. The class provides methods to update, advance time and record snapshots of the state
    during the operator splitting steps.

    Parameters
    ----------
    f : np.ndarray
        The initial distribution function, expected to be a 1D or 2D array.
    t : float, optional
        The initial time of the solution (default is 0.0).
    dt : float, optional
        The time step used in the last update (default is 0.0).
    stage : int, optional
        The current stage index in the simulation (default is 0).
    stage_name : str, optional
        A descriptive name for the current stage (default is an empty string).
    history : List[Dict[str, Any]], optional
        A list to store snapshots of the state at different stages, where each snapshot is a dictionary
        containing the time, dt, stage index, stage name and a copy of the distribution function at that point
        (default is an empty list).
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
        """Number of momentum bins.

        Returns
        -------
            int: Number of momentum bins.
        """
        return self.f.shape[0]

    @property
    def n_r(self) -> int:
        """Number of spatial bins.

        Returns
        -------
            int: Number of spatial bins.
        """
        return self.f.shape[1]

    @property
    def grid_shape(self) -> tuple:
        """Shape of the grid (n_p, n_r) as defined by the shape of f.

        Returns
        -------
            tuple: Shape of the grid (n_p, n_r).
        """
        return self.f.shape

    def clone(self, copy_history: bool = False) -> State:
        """Creates a deep copy of the current state, including f and metadata. Optionally includes history.

        Parameters
        ----------
        copy_history : bool, optional
            If True, also deep-copy the history of snapshots (default is False).

        Returns
        -------
            State: A new State instance with copied data.
        """
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
        """Returns f as a numpy array with appropriate dimensions for computation.

        Returns
        -------
            np.ndarray: The distribution function array,.
        """
        if self.ndim == 1:
            return self.f[0]  # return 1D array
        return self.f

    def update_f(self, new_f: np.ndarray):
        """Replaces the current f with new_f, ensuring it has the correct shape and type.

        Parameters
        ----------
        new_f : np.ndarray
            The new distribution function array to replace the current f. Must have the same shape as the current f.
        """
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
        """Advances the current time by dt and updates the dt attribute.

        Parameters
        ----------
        dt : float
            The time step to advance the current time by.
        """
        self.t += float(dt)
        self.dt = float(dt)

    def record_substep(self, stage_name: Optional[str] = None):
        """Records a snapshot of the current state, including time, dt, stage index, optional stage_name and a copy of f.

        Parameters
        ----------
        stage_name : str, optional
            An optional descriptive name for the current stage to include in the snapshot (default is None).
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
        """Restores the state to a previously recorded snapshot identified by either its index in the history list or its stage_name.

        Parameters
        ----------
        identifier : int or str
            The index of the snapshot in the history list or the stage_name of the snapshot to restore.
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
        """Returns a copy of the snapshot at the given index without modifying the current state.

        Parameters
        ----------
        index : int
            The index of the snapshot in the history list to retrieve.

        Returns
        -------
            dict: A dictionary containing the snapshot data (t, dt, stage, stage_name, f).
        """
        snap = self.history[index]
        return {
            "t": float(snap["t"]),
            "dt": float(snap["dt"]),
            "stage": int(snap["stage"]),
            "stage_name": str(snap.get("stage_name", "")),
            "f": snap["f"].copy(),
        }

    def clear_history(self):
        """Clears all recorded snapshots from the history list, leaving the current state intact."""
        self.history.clear()

    def step_stage(self, stage_name: Optional[str] = None):
        """Increments the stage index by 1 and optionally updates the stage_name.

        Parameters
        ----------
        stage_name : str, optional
            An optional descriptive name for the new stage to set (default is None, which resets the stage_name to an empty string).
        """
        self.stage += 1
        if stage_name is not None:
            self.stage_name = stage_name
        else:
            self.stage_name = ""

    def __repr__(self) -> str:
        return (
            f"State(t={self.t:.3f}, dt={self.dt:.3f}, stage={self.stage}, "
            f"stage_name={self.stage_name!r}, f_shape={self.f.shape}, history_len={len(self.history)})"
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
