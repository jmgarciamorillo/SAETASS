"""
This module provides the :py:class:`~saetass.state.State` dataclass, which serves as the single source of truth for the particle distribution function :math:`f` and its associated metadata (current time, operator-splitting stage and a snapshot history) throughout a simulation.

The :py:class:`~saetass.state.State` object is passed by reference between operator subsolvers during an operator-splitting step; each subsolver reads and writes ``f`` through the public :py:meth:`~saetass.state.State.get_f` / :py:meth:`~saetass.state.State.update_f` API, so that the rest of the solver infrastructure never needs to access the raw array directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    Container and tracker for the particle distribution function and simulation metadata.

    A :py:class:`State` instance wraps the distribution function array ``f`` together with the current simulation time, time step, operator-splitting stage information and an optional snapshot history.  It is the central object passed between all subsolvers during an operator-splitting step.

    The array ``f`` is always stored internally as a 2D float64 array of shape ``(n_p, n_r)`` regardless of whether the problem is 1D or 2D.
    When a 1D array is provided at construction it is automatically promoted to shape ``(1, n_r)``; :py:meth:`get_f` transparently returns the appropriate shape to the caller.

    Parameters
    ----------
    f : ndarray
        Initial distribution function.
        Must be a 1D array of shape ``(n,)`` or a 2D array of shape ``(n_p, n_r)``.
    t : float, optional
        Initial simulation time (default: ``0.0``).
    dt : float, optional
        Time step used in the last update (default: ``0.0``).
    stage : int, optional
        Current operator-splitting stage index (default: ``0``).
    stage_name : str, optional
        Descriptive label for the current operator-splitting stage
        (default: ``''``).
    history : list of dict, optional
        Pre-populated snapshot history (default: empty list).  Each element
        is a dictionary with keys ``'t'``, ``'dt'``, ``'stage'``,
        ``'stage_name'`` and ``'f'``.

    Attributes
    ----------
    ndim : ``{1, 2}``
        Dimensionality of the problem as inferred from the initial ``f``:
        ``1`` for a 1D spatial or momentum problem,
        ``2`` for a full 2D spatial-momentum problem.
    """

    f: np.ndarray
    t: float = 0.0
    dt: float = 0.0
    stage: int = 0
    stage_name: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

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
        """Number of momentum bins (rows of ``f``)."""
        return self.f.shape[0]

    @property
    def n_r(self) -> int:
        """Number of spatial bins (columns of ``f``)."""
        return self.f.shape[1]

    @property
    def grid_shape(self) -> tuple:
        """Shape of the internal ``f`` array, ``(n_p, n_r)``."""
        return self.f.shape

    def clone(self, copy_history: bool = False) -> State:
        """
        Create a deep copy of the current state.

        Parameters
        ----------
        copy_history : bool, optional
            If ``True``, deep-copy the full snapshot history as well (default: ``False``).

        Returns
        -------
        State
            A new :py:class:`State` with copied ``f`` and metadata.
            The ``history`` list of the clone is empty unless ``copy_history`` is ``True``.
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
        """
        Return the distribution function in its natural dimensionality.

        For 1D problems (``ndim == 1``) returns a 1D array of shape ``(n,)``;
        for 2D problems returns the full 2D array of shape ``(n_p, n_r)``.

        Returns
        -------
        ndarray
            The current distribution function.
        """
        if self.ndim == 1:
            return self.f[0]  # return 1D array
        return self.f

    def update_f(self, new_f: np.ndarray) -> None:
        """
        Replace the current distribution function with ``new_f``.

        The new array must be shape-compatible with the current ``f``.
        For 1D states a 1D input of length ``n_r`` is automatically promoted to shape ``(1, n_r)`` before storing.

        Parameters
        ----------
        new_f : ndarray
            New distribution function.  Must have shape ``(n_p, n_r)`` or, for 1D states, ``(n_r,)``.

        Raises
        ------
        ValueError
            If ``new_f`` has an incompatible shape.
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

    def set_time(self, t: float) -> None:
        """
        Set the simulation clock to an exact value.

        This method assigns ``t`` directly rather than accumulating increments, preventing floating-point drift when snapping to a canonical grid point.
        :py:attr:`dt` is updated to reflect the elapsed interval since the previous time.

        Parameters
        ----------
        t : float
            Exact time value to assign.
        """
        self.dt = float(t) - self.t
        self.t = float(t)

    def record_substep(self, stage_name: str | None = None) -> None:
        """
        Append a snapshot of the current state to the history.

        Each snapshot captures the current values of :py:attr:`t`, :py:attr:`dt`, :py:attr:`stage`, the ``stage_name`` (if provided) and a copy of :py:attr:`f`.
        Use :py:meth:`restore_substep` or :py:meth:`get_substep` to retrieve a saved snapshot later.

        Parameters
        ----------
        stage_name : str, optional
            Descriptive label to attach to this snapshot.
            If ``None``, the current :py:attr:`stage_name` attribute is used instead.
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

    def restore_substep(self, identifier: int | str) -> State:
        """
        Restore the state to a previously recorded snapshot.

        The snapshot to restore can be identified either by its numeric position in the history list or by its ``stage_name`` string.
        When identified by name, the *first* matching snapshot is used.

        Parameters
        ----------
        identifier : int or str
            Integer index into the :py:attr:`history` list, or a ``stage_name`` string to search for.

        Returns
        -------
        State
            The current instance (``self``) after restoration, for convenience.

        Raises
        ------
        ValueError
            If ``identifier`` is a string and no matching snapshot is found.
        IndexError
            If ``identifier`` is an integer that is out of range.
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

    def get_substep(self, index: int) -> dict[str, Any]:
        """
        Retrieve a copy of a snapshot without modifying the current state.

        Parameters
        ----------
        index : int
            Index of the snapshot in :py:attr:`history`.

        Returns
        -------
        dict
            A copy of the snapshot dictionary with keys ``'t'`` (float), ``'dt'`` (float), ``'stage'`` (int), ``'stage_name'`` (str) and ``'f'`` (ndarray copy).

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        snap = self.history[index]
        return {
            "t": float(snap["t"]),
            "dt": float(snap["dt"]),
            "stage": int(snap["stage"]),
            "stage_name": str(snap.get("stage_name", "")),
            "f": snap["f"].copy(),
        }

    def clear_history(self) -> None:
        """Remove all recorded snapshots from :py:attr:`history`, leaving the current state intact."""
        self.history.clear()

    def step_stage(self, stage_name: str | None = None) -> None:
        """
        Increment the operator-splitting stage counter by one.

        Optionally updates :py:attr:`stage_name` to the given label.
        If no label is supplied, :py:attr:`stage_name` is reset to an empty string.

        Parameters
        ----------
        stage_name : str, optional
            Descriptive label for the new stage (default: ``None``, which resets :py:attr:`stage_name` to ``''``).
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
