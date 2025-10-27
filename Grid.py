import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
import logging
from functools import cached_property

logger = logging.getLogger(__name__)


class Grid:
    """
    Grid class for 1D and 2D simulations.

    Handles spatial, momentum, and temporal grids with support for:
    - Uniform and non-uniform spacing
    - Cell centers and face positions
    - Grid metadata (cell widths, volumes, etc.)
    - Different coordinate systems (cartesian, spherical, cylindrical)
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
        """
        Initialize a Grid object.

        Args:
            r_faces: Array of face positions for spatial grid
            r_centers: Array of cell centers for spatial grid
            p_faces: Array of face positions for momentum grid
            p_centers: Array of cell centers for momentum grid
            t_grid: Array of temporal grid points
        """
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
                self.p_faces = np.concatenate(
                    [[left_face], internal_faces, [right_face]]
                )
            else:
                # Single cell case
                dp = 1.0  # Default width for a single cell
                self.p_faces = np.array(
                    [self.p_centers[0] - 0.5 * dp, self.p_centers[0] + 0.5 * dp]
                )
        if np.any(self.p_centers <= 0) or np.any(self.p_faces <= 0):
            raise ValueError("Momentum centers and faces must be positive values.")

    @cached_property
    def dr(self) -> np.ndarray:
        """Compute spatial cell widths."""
        if self.r_faces is not None:
            return self.r_faces[1:] - self.r_faces[:-1]
        return None

    @cached_property
    def dp(self) -> np.ndarray:
        """Compute momentum cell widths."""
        if self.p_faces is not None:
            return self.p_faces[1:] - self.p_faces[:-1]
        return None

    @cached_property
    def volumes(self) -> np.ndarray:
        """Compute cell volumes based on geometry."""
        if self.r_faces is not None:
            volumes = (4 * np.pi / 3) * (self.r_faces[1:] ** 3 - self.r_faces[:-1] ** 3)
            return volumes
        return None

    @cached_property
    def face_areas(self) -> np.ndarray:
        """Compute face areas based on geometry."""
        if self.r_faces is not None:
            return 4 * np.pi * self.r_faces**2
        return None

    @cached_property
    def num_timesteps(self) -> int:
        """Return the number of timesteps."""
        if self.t_grid is not None and len(self.t_grid) > 1:
            return len(self.t_grid) - 1
        else:
            raise ValueError("Temporal grid is not properly defined.")

    @cached_property
    def num_cells_r(self) -> int:
        """Return the number of spatial cells."""
        if self.r_centers is not None:
            return self.r_centers.size
        return 0

    @cached_property
    def num_cells_p(self) -> int:
        """Return the number of momentum cells."""
        if self.p_centers is not None:
            return self.p_centers.size
        return 0

    @cached_property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the grid as (n_p, n_r)."""
        return (self.num_cells_p, self.num_cells_r)

    def _p_to_y(self, p: np.ndarray) -> np.ndarray:
        """Convert momentum p to logarithmic variable y = log10(p)."""
        return np.log10(p)

    def _y_to_p(self, y: np.ndarray) -> np.ndarray:
        """Convert logarithmic variable y = log10(p) back to momentum p."""
        return 10**y

    def post_process_calculations(self):
        """Perform post-processing calculations after grid exit of solver pipeline.

        This function is intended to be called after the solver has completed its operations,
        not by the user directly.
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
        Create a uniform grid.

        Args:
            r_min: Minimum radius (optional)
            r_max: Maximum radius (optional)
            num_r_cells: Number of spatial cells (optional)
            p_min: Minimum momentum (optional)
            p_max: Maximum momentum (optional)
            num_p_cells: Number of momentum cells (optional)
            t_min: Minimum time (optional)
            t_max: Maximum time (optional)
            num_timesteps: Number of timesteps (optional)

        Returns:
            Grid: A uniform Grid object
        """
        # Check that at least one grid type is specified
        if (r_min is None or r_max is None or num_r_cells is None) and (
            p_min is None or p_max is None or num_p_cells is None
        ):
            raise ValueError(
                "At least one grid type (spatial or momentum) must be fully specified"
            )

        # Create uniform face-centered spatial grid if specified
        r_faces = None
        if r_min is not None and r_max is not None and num_r_cells is not None:
            r_faces = np.linspace(r_min, r_max, num_r_cells + 1)

        # Create uniform face-centered momentum grid if specified
        p_faces = None
        if p_min is not None and p_max is not None and num_p_cells is not None:
            p_faces = np.linspace(p_min, p_max, num_p_cells + 1)

        # Create temporal grid if specified
        t_grid = None
        if t_min is not None and t_max is not None and num_timesteps is not None:
            t_grid = np.linspace(t_min, t_max, num_timesteps + 1)

        return cls(r_faces=r_faces, p_faces=p_faces, t_grid=t_grid)

    @classmethod
    def non_uniform_clustering(
        cls,
        r_min: float,
        r_max: float,
        num_cells: int,
        cluster_center: float,
        cluster_width: float,
        cluster_strength: float = 0.9,
        t_min: float = None,
        t_max: float = None,
        num_timesteps: int = None,
    ):
        """
        Create a non-uniform grid with clustering around a specific point.

        Args:
            r_min: Minimum radius
            r_max: Maximum radius
            num_cells: Number of cells
            cluster_center: Center of clustering
            cluster_width: Width of clustering region
            cluster_strength: Strength of clustering (0-1)
            t_min: Minimum time (optional)
            t_max: Maximum time (optional)
            num_timesteps: Number of timesteps (optional)

        Returns:
            Grid: A non-uniform Grid object with clustering
        """
        # Normalize to [0, 1]
        x_c = (cluster_center - r_min) / (r_max - r_min)
        width = cluster_width / (r_max - r_min)

        # Generate initial uniform grid in [0, 1]
        xi = np.linspace(0, 1, num_cells + 1)

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

        # Create temporal grid if specified
        t_grid = None
        if t_min is not None and t_max is not None and num_timesteps is not None:
            t_grid = np.linspace(t_min, t_max, num_timesteps + 1)

        return cls(r_faces=r_faces, t_grid=t_grid)

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

        Args:
            r_min: Minimum radius (optional, must be > 0 if provided)
            r_max: Maximum radius (optional, must be > 0 if provided)
            num_r_cells: Number of spatial cells (optional)
            p_min: Minimum momentum (optional, must be > 0 if provided)
            p_max: Maximum momentum (optional, must be > 0 if provided)
            num_p_cells: Number of momentum cells (optional)
            t_min: Minimum time (optional)
            t_max: Maximum time (optional)
            num_timesteps: Number of timesteps (optional)

        Returns:
            Grid: A log-spaced Grid object
        """
        # Check that at least one grid type is specified
        if (r_min is None or r_max is None or num_r_cells is None) and (
            p_min is None or p_max is None or num_p_cells is None
        ):
            raise ValueError(
                "At least one grid type (spatial or momentum) must be fully specified"
            )

        # Create log-spaced spatial grid if specified
        r_centers = None
        if r_min is not None and r_max is not None and num_r_cells is not None:
            if r_min <= 0:
                raise ValueError("r_min must be positive for log-spaced grid")
            r_centers = np.logspace(np.log10(r_min), np.log10(r_max), num_r_cells)

        # Create log-spaced momentum grid if specified
        p_centers = None
        if p_min is not None and p_max is not None and num_p_cells is not None:
            if p_min <= 0:
                raise ValueError("p_min must be positive for log-spaced grid")
            p_centers = np.logspace(np.log10(p_min), np.log10(p_max), num_p_cells)

        # Create temporal grid if specified
        t_grid = None
        if t_min is not None and t_max is not None and num_timesteps is not None:
            t_grid = np.linspace(t_min, t_max, num_timesteps + 1)

        return cls(
            r_centers=r_centers, p_centers=p_centers, t_grid=t_grid, is_p_log=True
        )

    def __str__(self) -> str:
        """String representation of the Grid."""
        info = ["Grid:"]

        if self.r_faces is not None:
            info.append(f"  Spatial range: {self.r_faces[0]} to {self.r_faces[-1]}")
            info.append(f"  Number of spatial cells: {len(self.r_centers)}")

        if self.p_faces is not None:
            info.append(f"  Momentum range: {self.p_faces[0]} to {self.p_faces[-1]}")
            info.append(f"  Number of momentum cells: {len(self.p_centers)}")

        if self.t_grid is not None:
            info.append(f"  Temporal range: {self.t_grid[0]} to {self.t_grid[-1]}")
            info.append(f"  Number of timesteps: {len(self.t_grid) - 1}")

        return "\n".join(info)


if __name__ == "__main__":
    # Example usage
    # 1. Create a uniform spatial grid only
    spatial_grid = Grid.uniform(r_min=0.0, r_max=10.0, num_r_cells=100)
    print("Spatial Grid:")
    print(spatial_grid)

    # 2. Create a uniform momentum grid only
    momentum_grid = Grid.uniform(p_min=0.1, p_max=100.0, num_p_cells=50)
    print("\nMomentum Grid:")
    print(momentum_grid)

    # 3. Create a combined spatial and momentum grid
    combined_grid = Grid.uniform(
        r_min=0.0,
        r_max=10.0,
        num_r_cells=100,
        p_min=0.1,
        p_max=100.0,
        num_p_cells=50,
        t_min=0.0,
        t_max=1.0,
        num_timesteps=100,
    )
    print("\nCombined Grid:")
    print(combined_grid)

    # 4. Create a log-spaced momentum grid
    log_p_grid = Grid.log_spaced(p_min=0.1, p_max=1000.0, num_p_cells=50)
    print("\nLog-spaced Momentum Grid:")
    print(log_p_grid)

    # Plotting example
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    # Plot spatial grid faces
    if spatial_grid.r_faces is not None:
        plt.subplot(2, 1, 1)
        plt.plot(
            spatial_grid.r_faces,
            np.zeros_like(spatial_grid.r_faces),
            "bo",
            label="Uniform spatial grid",
        )
        plt.yticks([0], ["Spatial"])
        plt.xlabel("r")
        plt.legend()
        plt.grid(True)

    # Plot momentum grid faces
    if log_p_grid.p_faces is not None:
        plt.subplot(2, 1, 2)
        plt.semilogx(
            log_p_grid.p_faces,
            np.zeros_like(log_p_grid.p_faces),
            "ro",
            label="Log momentum grid",
        )
        plt.yticks([0], ["Momentum"])
        plt.xlabel("p")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
