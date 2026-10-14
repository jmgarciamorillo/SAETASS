import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass.utils.bubble_profiles import BubbleModel, BubbleProfileCalculator

FINE = np.linspace(0.1, 100, 1000) * u.pc


@pytest.fixture
def base_kwargs():
    # R_TS = 13.5 pc and R_b = 78.1 pc
    return {
        "L_wind": 1e38 * u.erg / u.s,
        "M_dot": 1e-5 * u.M_sun / u.yr,
        "rho_0": 1e-24 * u.g / u.cm**3,
        "t_b": 1e6 * u.yr,
    }


@pytest.fixture
def r_grid():
    return FINE


class TestBubbleProfileCalculator:
    def test_initialization(self, r_grid, base_kwargs):
        calc = BubbleProfileCalculator(r_grid, model="Weaver77", **base_kwargs)
        assert calc.R_TS is not None and calc.R_b is not None and calc.v_w is not None

    @pytest.mark.parametrize(
        "args, kwargs, error, match",
        [
            ((FINE.value,), {}, TypeError, None),
            ((np.linspace(0.1, 100, 100) * u.K,), {}, u.UnitsError, None),
            ((FINE,), {"model": "InvalidModel"}, ValueError, None),
            ((FINE,), {"eta_m": 0.0}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 1.5}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 0.5 * u.one}, ValueError, "eta_m"),
        ],
    )
    def test_invalid_arguments(self, base_kwargs, args, kwargs, error, match):
        with pytest.raises(error, match=match):
            BubbleProfileCalculator(*args, **kwargs, **base_kwargs)

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


class TestBubbleStructure:
    """Radii and gas of the bubble in its intermediate stage (Weaver et al. 1977)."""

    def test_radii(self, r_grid, base_kwargs):
        L, rho_0, t = base_kwargs["L_wind"], base_kwargs["rho_0"], base_kwargs["t_b"]
        calc = BubbleProfileCalculator(r_grid, **base_kwargs)
        # R_b = 0.7628 (L t^3 / rho_0)^(1/5)
        expected = 0.762865 * (L * t**3 / rho_0) ** (1 / 5)
        assert calc.R_b.to_value(u.pc) == pytest.approx(expected.to_value(u.pc))
        # Wind density just upstream of the termination shock
        rho_w = base_kwargs["M_dot"] / (4 * np.pi * calc.R_TS**2 * calc.v_w)
        assert calc.rho_w.to_value(u.g / u.cm**3) == pytest.approx(
            rho_w.to_value(u.g / u.cm**3)
        )
        # Energy losses reduce the luminosity of the bubble, not that of the wind
        lossy = BubbleProfileCalculator(r_grid, eta_m=0.032, **base_kwargs)
        factor = 0.032 ** (1 / 5)  # 0.5025
        assert (lossy.R_b / calc.R_b).to_value(u.one) == pytest.approx(factor)
        assert (lossy.R_TS / calc.R_TS).to_value(u.one) == pytest.approx(1 / factor)
        assert lossy.v_w == calc.v_w

    def test_gas_density(self, base_kwargs, r_grid):
        calc = BubbleProfileCalculator(r_grid, R_c=5 * u.pc, **base_kwargs)
        n = calc.compute_density_profile()
        n0 = (base_kwargs["rho_0"] / const.m_p).to(u.cm**-3)

        # Core and cold wind, from the mass loss rate
        M_dot, v_w = base_kwargs["M_dot"], calc.v_w
        n_core = M_dot / (4 * np.pi * (5 * u.pc) ** 2 * v_w * const.m_p)
        assert n[calc.masks["r_core"]].to_value(u.cm**-3) == pytest.approx(
            n_core.to_value(u.cm**-3)
        )

        # Hot bubble, filled by the mass evaporated from the shell: 2e-4 M_sun/yr for
        # L = 1e37 erg/s, n0 = 10 cm^-3 and 1 Myr (Menchiari et al. 2024, Eq. 15)
        M_dot_s = (
            2e-4
            * (base_kwargs["L_wind"] / (1e37 * u.erg / u.s)).to_value(u.one)
            ** (27 / 35)
            * (n0 / (10 / u.cm**3)).to_value(u.one) ** (-2 / 35)
            * u.M_sun
            / u.yr
        )
        V_hot = 4 / 3 * np.pi * (calc.R_cd**3 - calc.R_TS**3)
        n_b = (M_dot_s * base_kwargs["t_b"] / (V_hot * const.m_p)).to(u.cm**-3)
        hot = calc.masks["r_hot"]
        if np.any(hot):
            np.testing.assert_allclose(n[hot].to_value(u.cm**-3), n_b.value)
