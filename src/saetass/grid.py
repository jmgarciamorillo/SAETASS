"""
The grid module defines the :py:class:`~saetass.grid.Grid` class, which manages spatial, momentum and temporal discretization for simulations initialization.
The :py:class:`~saetass.grid.Grid` serves as the foundational data structure describing the discrete domain upon which the distribution function :math:`f(r, p, t)` evolves.
It is natively responsible for caching and providing access to the geometrical properties of the space and momentum domains, as well as the discrete time steps.

In the context of the greater solver pipeline, the instantiated :py:class:`~saetass.grid.Grid` object is shared globally between the parent orchestrator, :py:class:`~saetass.solver.Solver`, and the individual physics operators, i.e. any instance of :py:class:`~saetass.solver.SubSolver`.

Furthermore, :py:class:`~saetass.grid.Grid` provides ``@classmethods`` to instantiate uniform, non-uniform and logarithmically spaced grids.
"""

import logging
from functools import cached_property

import numpy as np

logger = logging.getLogger(__name__)


class Grid:
    r"""
    The :py:class:`~saetass.grid.Grid` class can be instantiated in several ways:

    - By providing the spatial and/or momentum faces grid coordinates.
    - By providing the spatial and/or momentum centers grid coordinates.
    - By providing the spatial and/or momentum faces and centers grid coordinates.
    - By instantiating any of the ``@classmethods`` that it provides.

    When centers or faces are not provided, the class will attempt to construct them from the provided faces or centers, respectively.
    Moreover, :py:class:`~saetass.grid.Grid` performs the coordinate mapping between physical and computational space for the case of the momentum grid in logarithmic scale (:math:`y = \log_{10}(p)`).

    Parameters
    ----------
    r_centers : np.ndarray, optional
        1D array containing the centroid coordinates of the spatial grid cells.
    r_faces : np.ndarray, optional
        1D array containing the boundary (interface) coordinates of the spatial grid cells. Length is two greater than ``r_centers``.
    p_centers : np.ndarray, optional
        1D array containing the centroid coordinates of the momentum grid cells.
    p_faces : np.ndarray, optional
        1D array containing the boundary (interface) coordinates of the momentum grid cells. Length is two greater than ``p_centers``.
    t_grid : np.ndarray, optional
        1D array defining the discrete macro-timesteps over which the simulation will globally advance.
    is_p_log : bool, optional
        Flag dictating whether numerical operations inside solvers should treat the momentum grid internally in logarithmic spacing :math:`\log_{10}(p)`. Default is ``True``.
    """

    def __init__(
        self,
        r_centers: np.ndarray = None,
        r_faces: np.ndarray = None,
        p_centers: np.ndarray = None,
        p_faces: np.ndarray = None,
        t_grid: np.ndarray = None,
        is_p_log: bool = True,
    ):
        # Check that at least one grid type is provided
        if (
            r_faces is None
            and r_centers is None
            and p_faces is None
            and p_centers is None
        ):
            raise ValueError("At least one grid (spatial or momentum) must be provided")

        self.is_log_p = is_p_log

        # Initialize spatial grid if provided
        if r_faces is not None or r_centers is not None:
            self._init_spatial_grid(r_faces, r_centers)
        else:
            self.r_faces = None
            self.r_centers = None

        # Initialize momentum grid if provided
        if p_faces is not None or p_centers is not None:
            self._init_momentum_grid(p_faces, p_centers)
        else:
            self.p_faces = None
            self.p_centers = None

        # Temporal grid
        self.t_grid = np.asarray(t_grid) if t_grid is not None else None
        if self.t_grid is not None and len(self.t_grid) > 1:
            self.dt = np.diff(self.t_grid)

        if self.is_log_p and self.p_centers is not None:
            self._p_centers_phys = self.p_centers.copy()
            self._p_faces_phys = self.p_faces.copy()
            self.p_centers = self._p_to_y(self.p_centers)
            self.p_faces = self._p_to_y(self.p_faces)

    def _init_spatial_grid(self, r_faces, r_centers):
        """Initialize the spatial grid from faces or centers."""
        if r_faces is not None:
            self.r_faces = np.asarray(r_faces)
            # Calculate cell centers as midpoints between faces
            self.r_centers = 0.5 * (self.r_faces[:-1] + self.r_faces[1:])
        elif r_centers is not None:
            self.r_centers = np.asarray(r_centers)
            # Approximate face positions for non-uniform grid
            if len(self.r_centers) > 1:
                # For internal faces: midpoint between cell centers
                internal_faces = 0.5 * (self.r_centers[:-1] + self.r_centers[1:])
                # For boundary faces: extrapolate
                left_face = self.r_centers[0] - 0.5 * (
                    self.r_centers[1] - self.r_centers[0]
                )
                right_face = self.r_centers[-1] + 0.5 * (
                    self.r_centers[-1] - self.r_centers[-2]
                )
                self.r_faces = np.concatenate(
                    [[left_face], internal_faces, [right_face]]
                )
            else:
                # Single cell case
                dr = 1.0  # Default width for a single cell
                self.r_faces = np.array(
                    [self.r_centers[0] - 0.5 * dr, self.r_centers[0] + 0.5 * dr]
                )

    def _init_momentum_grid(self, p_faces, p_centers):
        """Initialize the momentum grid from faces or centers."""
        if p_faces is not None:
            self.p_faces = np.asarray(p_faces)
            # Calculate cell centers as midpoints between faces
            self.p_centers = 0.5 * (self.p_faces[:-1] + self.p_faces[1:])
        elif p_centers is not None:
            self.p_centers = np.asarray(p_centers)
            # Approximate face positions for non-uniform grid
            if len(self.p_centers) > 1:
                # For internal faces: midpoint between cell centers
                internal_faces = 0.5 * (self.p_centers[:-1] + self.p_centers[1:])
                # For boundary faces: extrapolate
                left_face = self.p_centers[0] - 0.5 * (
                    self.p_centers[1] - self.p_centers[0]
                )
                right_face = self.p_centers[-1] + 0.5 * (
                    self.p_centers[-1] - self.p_centers[-2]
                )
                if left_face <= 0:
                    ValueError("Momentum faces must be positive values.")
                else:
                    left_face = left_face  # np.finfo(float).tiny
                self.p_faces = np.concatenate(
                    [[left_face], internal_faces, [right_face]]
                )
            else:
                # Single cell case
                dp = 1.0  # Default width for a single cell
                self.p_faces = np.array(
                    [self.p_centers[0] - 0.5 * dp, self.p_centers[0] + 0.5 * dp]
                )

    @cached_property
    def dr(self) -> np.ndarray:
        """
        Spatial cell widths.

        Returns
        -------
        np.ndarray or None
            Array of spatial cell widths, or None if the spatial grid is not initialized.
        """
        if self.r_faces is not None:
            return self.r_faces[1:] - self.r_faces[:-1]
        return None

    @cached_property
    def dp(self) -> np.ndarray:
        """
        Momentum cell widths.

        Returns
        -------
        np.ndarray or None
            Array of momentum cell widths, or None if the momentum grid is not initialized.
        """
        if self.p_faces is not None:
            return self.p_faces[1:] - self.p_faces[:-1]
        return None

    @cached_property
    def volumes(self) -> np.ndarray:
        """
        Cell volumes based on spherical geometry.

        Returns
        -------
        np.ndarray or None
            Array of cell volumes assuming spherical shells, or None if the spatial grid is not initialized.
        """
        if self.r_faces is not None:
            volumes = (4 * np.pi / 3) * (self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3)
            return volumes
        return None

    @cached_property
    def face_areas(self) -> np.ndarray:
        """
        Face areas based on spherical geometry.

        Returns
        -------
        np.ndarray or None
            Array of face areas assuming spherical shells, or None if the spatial grid is not initialized.
        """
        if self.r_faces is not None:
            return 4 * np.pi * self.r_faces**2
        return None

    @cached_property
    def num_timesteps(self) -> int:
        """
        Number of global timesteps.

        Returns
        -------
        int
            Number of recorded timesteps (length of temporal grid minus 1).

        Raises
        ------
        ValueError
            If the temporal grid is not properly defined with at least 2 points.
        """
        if self.t_grid is not None and len(self.t_grid) > 1:
            return len(self.t_grid) - 1
        else:
            raise ValueError("Temporal grid is not properly defined.")

    @cached_property
    def num_cells_r(self) -> int:
        """
        Number of spatial cells.

        Returns
        -------
        int
            Number of spatial cells in the grid.
        """
        if self.r_centers is not None:
            return self.r_centers.size
        return 0

    @cached_property
    def num_cells_p(self) -> int:
        """
        Number of momentum cells.

        Returns
        -------
        int
            Number of momentum cells in the grid.
        """
        if self.p_centers is not None:
            return self.p_centers.size
        return 0

    @cached_property
    def shape(self) -> tuple:
        """
        Shape of the grid.

        Returns
        -------
        tuple
            The shape of the grid as ``(n_p, n_r)`` if 2D, or a 1D tuple if only one dimension is defined.

        Raises
        ------
        ValueError
            If no grid dimensions are defined.
        """
        if self.p_centers is not None and self.r_centers is not None:
            return (self.num_cells_p, self.num_cells_r)
        elif self.r_centers is not None:
            return (self.num_cells_r,)
        elif self.p_centers is not None:
            return (self.num_cells_p,)
        else:
            raise ValueError("No grid dimensions are defined.")

    def _p_to_y(self, p: np.ndarray) -> np.ndarray:
        """Convert momentum p to logarithmic variable y = log10(p)."""
        return np.log10(p)

    def _y_to_p(self, y: np.ndarray) -> np.ndarray:
        """Convert logarithmic variable y = log10(p) back to momentum p."""
        return 10**y

    def post_process_calculations(self):
        """
        Perform post-processing calculations after grid exit of :py:class:`~saetass.solver.Solver` pipeline.

        This function converts the logarithmic momentum grid tracking back into standard momentum physical
        values. It is intended to be called after the :py:class:`~saetass.solver.Solver` has completed its operations, and not by the user directly.
        """
        if self.is_log_p:
            self._p_centers_calc = self._y_to_p(self.p_centers)
            self._p_faces_calc = self._y_to_p(self.p_faces)
            self.dp_calc = self.dp
            self.p_centers = self._p_centers_calc
            self.p_faces = self._p_faces_calc
        else:
            logger.warning(
                "Post-processing calculations skipped as momentum grid is not logarithmic."
            )

    def is_compatible_array(self, array: np.ndarray) -> bool:
        """
        Check if the given array is compatible with this :py:class:`~saetass.grid.Grid`.

        Parameters
        ----------
        array : np.ndarray
            The array to check for shape compatibility.

        Returns
        -------
        bool
            ``True`` if the array shape matches the grid shape, ``False`` otherwise.
        """
        expected_shape = self.shape
        return array.shape == expected_shape

    @staticmethod
    def _validate_grid_params(
        r_min: float,
        r_max: float,
        num_r_cells: int,
        p_min: float,
        p_max: float,
        num_p_cells: int,
        t_min: float,
        t_max: float,
        num_timesteps: int,
        req_r_pos: bool = False,
        req_p_pos: bool = False,
    ) -> tuple[bool, bool, bool]:
        """
        Validates the grid generation boundaries.

        Parameters
        ----------
        r_min, r_max, num_r_cells : float, float, int
            Spatial grid definitions.
        p_min, p_max, num_p_cells : float, float, int
            Momentum grid definitions.
        t_min, t_max, num_timesteps : float, float, int
            Temporal grid definitions.
        req_r_pos : bool, optional
            Whether r_min must be strictly positive (for log spaces).
        req_p_pos : bool, optional
            Whether p_min must be strictly positive (for log spaces).

        Returns
        -------
        has_r, has_p, has_t : Tuple[bool, bool, bool]
            Boolean flags indicating which configurations are correctly fully specified.

        Raises
        ------
        ``ValueError``
            If invalid parameter sets are given, such as min bounding over max,
            or required groupings of values are incompletely filled.
        """
        has_r = False
        if not (r_min is None and r_max is None and num_r_cells is None):
            if r_min is None or r_max is None or num_r_cells is None:
                raise ValueError(
                    "r_min, r_max, and num_r_cells must all be specified together."
                )
            if r_min < 0:
                raise ValueError("r_min cannot be negative.")
            if req_r_pos and r_min <= 0:
                raise ValueError("r_min must be strictly positive for this grid type.")
            if r_max <= r_min:
                raise ValueError("r_max must be strictly greater than r_min.")
            if num_r_cells <= 0:
                raise ValueError("num_r_cells must be a positive integer.")
            has_r = True

        has_p = False
        if not (p_min is None and p_max is None and num_p_cells is None):
            if p_min is None or p_max is None or num_p_cells is None:
                raise ValueError(
                    "p_min, p_max, and num_p_cells must all be specified together."
                )
            if p_min < 0:
                raise ValueError("p_min cannot be negative.")
            if req_p_pos and p_min <= 0:
                raise ValueError("p_min must be strictly positive for this grid type.")
            if p_max <= p_min:
                raise ValueError("p_max must be strictly greater than p_min.")
            if num_p_cells <= 0:
                raise ValueError("num_p_cells must be a positive integer.")
            has_p = True

        if not has_r and not has_p:
            raise ValueError(
                "At least one spatial (r) or momentum (p) grid must be fully specified."
            )

        has_t = False
        if not (t_min is None and t_max is None and num_timesteps is None):
            if t_min is None or t_max is None or num_timesteps is None:
                raise ValueError(
                    "t_min, t_max, and num_timesteps must all be specified together."
                )
            if t_max <= t_min:
                raise ValueError("t_max must be strictly greater than t_min.")
            if num_timesteps <= 0:
                raise ValueError("num_timesteps must be a positive integer.")
            has_t = True

        return has_r, has_p, has_t

    @classmethod
    def uniform(
        cls,
        r_min: float = None,
        r_max: float = None,
        num_r_cells: int = None,
        p_min: float = None,
        p_max: float = None,
        num_p_cells: int = None,
        t_min: float = None,
        t_max: float = None,
        num_timesteps: int = None,
    ):
        """
        Create a uniform grid linearly spaced.

        At least one space (:math:`r`) or momentum (:math:`p`) grid must be fully specified with
        its respective (min, max, cells) triplet.

        Parameters
        ----------
        r_min : float, optional
            Minimum radius (must be >= 0).
        r_max : float, optional
            Maximum radius.
        num_r_cells : int, optional
            Number of spatial cells.
        p_min : float, optional
            Minimum momentum (must be >= 0).
        p_max : float, optional
            Maximum momentum.
        num_p_cells : int, optional
            Number of momentum cells.
        t_min : float, optional
            Minimum time.
        t_max : float, optional
            Maximum time.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A uniform :py:class:`~saetass.grid.Grid` object linearly spaced.

        Raises
        ------
        ``ValueError``
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, or incomplete parameter sets are given.
        """
        has_r, has_p, has_t = cls._validate_grid_params(
            r_min,
            r_max,
            num_r_cells,
            p_min,
            p_max,
            num_p_cells,
            t_min,
            t_max,
            num_timesteps,
        )

        r_centers = np.linspace(r_min, r_max, num_r_cells) if has_r else None
        p_centers = np.linspace(p_min, p_max, num_p_cells) if has_p else None
        t_grid = np.linspace(t_min, t_max, num_timesteps + 1) if has_t else None

        return cls(
            r_centers=r_centers, p_centers=p_centers, t_grid=t_grid, is_p_log=False
        )

    @classmethod
    def non_uniform_clustering(
        cls,
        r_min: float,
        r_max: float,
        num_r_cells: int,
        cluster_center: float,
        cluster_width: float,
        cluster_strength: float = 0.9,
        t_min: float = None,
        t_max: float = None,
        num_timesteps: int = None,
    ):
        """
        Create a non-uniform grid with clustering around a specific spatial point.

        Requires full specification of the spatial parameters. Momentum grid construction
        is explicitly unused in this classmethod.

        Parameters
        ----------
        r_min : float
            Minimum radius (must be >= 0).
        r_max : float
            Maximum radius.
        num_r_cells : int
            Number of spatial cells.
        cluster_center : float
            Center of clustering region.
        cluster_width : float
            Width of the clustering region.
        cluster_strength : float, optional
            Strength of clustering between 0 and 1. Defaults to 0.9.
        t_min : float, optional
            Minimum time.
        t_max : float, optional
            Maximum time.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A non-uniform :py:class:`~saetass.grid.Grid` object with cell clustering.

        Raises
        ------
        ``ValueError``
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, or incomplete parameter sets are given.
        """
        # Validate spatial and temporal bounds explicitly
        has_r, _, has_t = cls._validate_grid_params(
            r_min, r_max, num_r_cells, None, None, None, t_min, t_max, num_timesteps
        )

        # Normalize to [0, 1]
        x_c = (cluster_center - r_min) / (r_max - r_min)
        width = cluster_width / (r_max - r_min)

        # Generate initial uniform grid in [0, 1]
        xi = np.linspace(0, 1, num_r_cells + 1)

        # Apply tanh clustering
        s = (xi - x_c) / (0.5 * width)
        xi = xi - cluster_strength * np.tanh(s) * (0.5 * width)

        # Ensure bounds and monotonicity
        xi = np.clip(xi, 0, 1)
        xi[0] = 0
        xi[-1] = 1
        xi = np.sort(xi)

        # Map back to original domain
        r_faces = r_min + xi * (r_max - r_min)

        t_grid = np.linspace(t_min, t_max, num_timesteps + 1) if has_t else None

        return cls(r_faces=r_faces, t_grid=t_grid, is_p_log=False)

    @classmethod
    def log_spaced(
        cls,
        r_min: float = None,
        r_max: float = None,
        num_r_cells: int = None,
        p_min: float = None,
        p_max: float = None,
        num_p_cells: int = None,
        t_min: float = None,
        t_max: float = None,
        num_timesteps: int = None,
    ):
        """
        Create a logarithmically spaced grid.

        At least one space (:math:`r`) or momentum (:math:`p`) grid must be fully specified with its respective (min, max, cells) triplet.

        Parameters
        ----------
        r_min : float, optional
            Minimum radius (must be > 0 if provided).
        r_max : float, optional
            Maximum radius.
        num_r_cells : int, optional
            Number of spatial cells.
        p_min : float, optional
            Minimum momentum (must be > 0 if provided).
        p_max : float, optional
            Maximum momentum.
        num_p_cells : int, optional
            Number of momentum cells.
        t_min : float, optional
            Minimum time.
        t_max : float, optional
            Maximum time.
        num_timesteps : int, optional
            Number of timesteps.

        Returns
        -------
        Grid
            A log-spaced :py:class:`~saetass.grid.Grid` object.

        Raises
        ------
        ValueError
            If bounds are invalid, negative minimums are provided, maximums are lower than minimums, 0 is supplied for min values, or incomplete parameter sets are given.
        """
        has_r, has_p, has_t = cls._validate_grid_params(
            r_min,
            r_max,
            num_r_cells,
            p_min,
            p_max,
            num_p_cells,
            t_min,
            t_max,
            num_timesteps,
            req_r_pos=True,
            req_p_pos=True,
        )

        r_centers = (
            np.logspace(np.log10(r_min), np.log10(r_max), num_r_cells)
            if has_r
            else None
        )
        p_centers = (
            np.logspace(np.log10(p_min), np.log10(p_max), num_p_cells)
            if has_p
            else None
        )
        t_grid = np.linspace(t_min, t_max, num_timesteps + 1) if has_t else None

        return cls(
            r_centers=r_centers, p_centers=p_centers, t_grid=t_grid, is_p_log=True
        )

    def __str__(self) -> str:
        """String representation of the Grid."""
        info = ["Grid:"]

        if self.r_faces is not None:
            info.append(
                f"  Spatial range: {self.r_faces[0]:.4e} to {self.r_faces[-1]:.4e}"
            )
            info.append(f"  Number of spatial cells: {len(self.r_centers)}")

        if self.p_faces is not None:
            if hasattr(self, "_p_faces_phys"):
                p_min_val, p_max_val = self._p_faces_phys[0], self._p_faces_phys[-1]
            else:
                p_min_val, p_max_val = self.p_faces[0], self.p_faces[-1]
            info.append(f"  Momentum range: {p_min_val:.4e} to {p_max_val:.4e}")
            info.append(f"  Number of momentum cells: {len(self.p_centers)}")

        if self.t_grid is not None:
            info.append(
                f"  Temporal range: {self.t_grid[0]:.4e} to {self.t_grid[-1]:.4e}"
            )
            info.append(f"  Number of timesteps: {len(self.t_grid) - 1}")

        return "\n".join(info)
