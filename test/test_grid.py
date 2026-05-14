import numpy as np
import pytest

from saetass.grid import Grid


class TestGrid:
    def test_missing_initialization(self):
        with pytest.raises(ValueError):
            Grid()

    def test_init_from_centers_and_faces(self):
        # Only r_centers
        r_c = np.array([1.0, 2.0, 3.0])
        g1 = Grid(r_centers=r_c)
        assert g1.shape == (3,)
        assert g1.r_faces is not None
        assert len(g1.r_faces) == 4

        # Only p_centers
        p_c = np.array([5.0, 10.0])
        g2 = Grid(p_centers=p_c, is_p_log=False)
        assert g2.shape == (2,)

        # Both
        g3 = Grid(r_centers=r_c, p_centers=p_c, is_p_log=False)
        assert g3.shape == (2, 3)

        # Null cases for properties
        assert g2.dr is None
        assert g1.dp is None
        assert g2.volumes is None
        assert g2.face_areas is None

    def test_single_cell_initialization(self):
        g = Grid(r_centers=[1.0], p_centers=[2.0], is_p_log=False)
        assert g.dr[0] == 1.0
        assert g.dp[0] == 1.0

    def test_negative_p_faces_approximation(self):
        # p=0 center leads to negative left face extrapolation, and should ValueError
        with pytest.raises(ValueError):
            Grid(p_centers=[0.0, 1.0], is_p_log=False)

    def test_uniform_grid_validations(self):
        with pytest.raises(ValueError, match="together"):
            Grid.uniform(r_min=1.0, r_max=2.0)  # Missing num_cells

        with pytest.raises(ValueError, match="negative"):
            Grid.uniform(r_min=-1.0, r_max=2.0, num_r_cells=10)

        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(r_min=2.0, r_max=1.0, num_r_cells=10)

        with pytest.raises(ValueError, match="positive integer"):
            Grid.uniform(r_min=1.0, r_max=2.0, num_r_cells=0)

        # Same for p
        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(p_min=2.0, p_max=1.0, num_p_cells=10)

        # Missing both r and p
        with pytest.raises(ValueError, match="fully specified"):
            Grid.uniform(t_min=0, t_max=1, num_timesteps=10)

        # Wrong temp bounds
        with pytest.raises(ValueError, match="together"):
            Grid.uniform(r_min=1.0, r_max=2.0, num_r_cells=10, t_min=1)

        with pytest.raises(ValueError, match="positive integer"):
            Grid.uniform(
                r_min=1.0, r_max=2.0, num_r_cells=10, t_min=0, t_max=1, num_timesteps=-1
            )

        with pytest.raises(ValueError, match="strictly greater"):
            Grid.uniform(
                r_min=1.0, r_max=2.0, num_r_cells=10, t_min=1, t_max=0, num_timesteps=10
            )

    def test_log_grid(self):
        g = Grid.log_spaced(
            r_min=1.0,
            r_max=100.0,
            num_r_cells=3,
            p_min=50.0,
            p_max=100.0,
            num_p_cells=3,
        )
        assert g.is_log_p
        assert np.isclose(g.p_centers[0], np.log10(50.0))

        with pytest.raises(ValueError, match="strictly positive"):
            Grid.log_spaced(r_min=0.0, r_max=10.0, num_r_cells=10)

        with pytest.raises(ValueError, match="strictly positive"):
            Grid.log_spaced(p_min=0.0, p_max=10.0, num_p_cells=10)

    def test_non_uniform_clustering(self):
        g = Grid.non_uniform_clustering(
            r_min=0, r_max=10, num_r_cells=100, cluster_center=5.0, cluster_width=1.0
        )
        assert g.r_faces[0] == 0.0
        assert g.r_faces[-1] == 10.0

    def test_properties(self):
        t_g = np.linspace(0, 1, 11)
        g = Grid(r_centers=[1.0, 2.0], t_grid=t_g)
        assert g.num_timesteps == 10

        with pytest.raises(ValueError):
            g_bad = Grid(r_centers=[1.0], t_grid=[0])
            _ = g_bad.num_timesteps

        assert g.num_cells_p == 0
        assert g.num_cells_r == 2
        assert g.volumes is not None
        assert g.face_areas is not None

    def test_post_process(self):
        g = Grid(p_centers=[50.0, 100.0], is_p_log=True)
        assert np.isclose(g.p_centers[0], np.log10(50.0))
        g.post_process_calculations()
        assert np.isclose(g.p_centers[0], 50.0)

        # Non-log doesn't mutate
        g2 = Grid(p_centers=[50.0, 100.0], is_p_log=False)
        g2.post_process_calculations()
        assert g2.p_centers[0] == 50.0

    def test_compatible_array(self):
        g = Grid(r_centers=[1.0, 2.0, 3.0])
        arr = np.zeros(3)
        assert g.is_compatible_array(arr)
        arr2 = np.zeros((2, 3))
        assert not g.is_compatible_array(arr2)

    def test_str_representation(self):
        g = Grid(
            r_centers=[1.0, 2.0], p_centers=[1.0, 2.0], is_p_log=True, t_grid=[0, 1]
        )
        s = str(g)
        assert "Grid:" in s
        assert "Spatial range:" in s
        assert "Momentum range:" in s
        assert "Temporal range:" in s


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    print("Running tests...")
    pytest.main([__file__])
