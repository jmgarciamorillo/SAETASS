import astropy.units as u
import numpy as np
import pytest

from saetass.utils.bubble_profiles import BubbleModel, BubbleProfileCalculator


@pytest.fixture
def base_kwargs():
    return {
        "L_wind": 1e38 * u.erg / u.s,
        "M_dot": 1e-5 * u.M_sun / u.yr,
        "rho_0": 1e-24 * u.g / u.cm**3,
        "t_b": 1e6 * u.yr,
    }


@pytest.fixture
def r_grid():
    return np.linspace(0.1, 100, 1000) * u.pc


@pytest.fixture
def array_r_grid():
    return np.linspace(0.1, 100, 1000)


class TestBubbleProfileCalculator:
    def test_initialization(self, r_grid, array_r_grid, base_kwargs):
        # Test Quantity grid
        calc1 = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        assert calc1.R_TS is not None
        assert calc1.R_b is not None
        assert calc1.v_w is not None

        # Test ndarray grid
        calc2 = BubbleProfileCalculator(array_r_grid, model="Morlino21", **base_kwargs)
        assert calc2.r_grid.unit.is_equivalent(u.pc)

        # Check invalid units
        with pytest.raises(ValueError):
            BubbleProfileCalculator(np.linspace(0.1, 100, 100) * u.K, **base_kwargs)

        # Check invalid model mapping
        with pytest.raises(ValueError):
            BubbleProfileCalculator(r_grid, model="InvalidModel", **base_kwargs)

    def test_missing_kwargs(self, r_grid):
        with pytest.raises(ValueError):
            BubbleProfileCalculator(r_grid, model="Weaver77", L_wind=1e38 * u.erg / u.s)

    def test_density_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        density = calc.compute_density_profile()
        assert density.unit.is_equivalent(u.cm**-3)
        assert len(density) == len(r_grid)

    def test_temperature_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        temp = calc.compute_temperature_profile()
        assert temp.unit.is_equivalent(u.K)
        assert len(temp) == len(r_grid)

    def test_velocity_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        v_field = calc.compute_velocity_profile()
        assert v_field.unit.is_equivalent(u.km / u.s)
        assert len(v_field) == len(r_grid)

    def test_magnetic_field_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        b_field = calc.compute_magnetic_field_profile()
        assert b_field.unit.is_equivalent(u.G)
        assert len(b_field) == len(r_grid)

    def test_diffusion_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        E_k = 1 * u.TeV

        for diff_model in ["kolmogorov", "kraichnan", "bohm"]:
            D = calc.compute_diffusion_profile(E_k, diffusion_model=diff_model)
            assert D.unit.is_equivalent(u.cm**2 / u.s)
            assert len(D) == len(r_grid)

        with pytest.raises(ValueError):
            calc.compute_diffusion_profile(E_k, diffusion_model="invalid")

    def test_source_term(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        E_k = 1 * u.TeV
        Q = calc.compute_source_term(E_k)
        assert isinstance(Q, np.ndarray)
        assert len(Q) == len(r_grid)

    def test_analytical_CR_profile(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        E_k = 1 * u.TeV
        D = calc.compute_diffusion_profile(E_k)
        CR = calc.compute_analytical_CR_profile(D)
        assert len(CR) == len(r_grid)

    def test_get_all_profiles(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        E_k = 1 * u.TeV
        profiles = calc.get_all_profiles(E_k)
        assert "n_gas" in profiles
        assert "T_gas" in profiles
        assert "v_field" in profiles
        assert "B_field" in profiles
        assert "D_values" in profiles
        assert "Q" in profiles

    def test_not_implemented_errors(self, r_grid, base_kwargs):
        # Weaver77 model does not support magnetic/diffusion methods natively as Morlino21
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)

        with pytest.raises(NotImplementedError):
            calc.compute_magnetic_field_profile()

        with pytest.raises(NotImplementedError):
            E_k = 1 * u.TeV
            calc.compute_diffusion_profile(E_k)

    def test_bubble_model_enum(self):
        assert BubbleModel.WEAVER77 == "Weaver77"
        assert BubbleModel.MORLINO21 == "Morlino21"


if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    print("Running tests...")
    pytest.main([__file__])
