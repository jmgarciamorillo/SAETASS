import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from saetass import Grid
from saetass.utils.bubble_profiles import BubbleModel, BubbleProfileCalculator

FINE = np.linspace(0.1, 100, 1000) * u.pc
COARSE = np.linspace(0.0, 100.0, 11) * u.pc  # 10 pc cells, wider than the 3.9 pc shell
FRAGMENTED = {
    "shell": "fragmented",
    "shell_width": 0.5 * u.pc,
    "shell_penetration": 2.0 * u.pc,
}


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


def _overlap(faces, r_in, r_out):
    """Volume of each cell between the radii r_in and r_out."""
    lower, upper = np.clip(faces[:-1], r_in, r_out), np.clip(faces[1:], r_in, r_out)
    return 4 / 3 * np.pi * (upper**3 - lower**3)


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
            ((FINE,), {"shell": "thin"}, ValueError, "'uniform' or 'fragmented'"),
            (
                (FINE,),
                {"shell": "fragmented", "shell_width": 0.5 * u.pc},
                ValueError,
                "shell_penetration",
            ),
            (
                (FINE,),
                {"shell": "fragmented", "shell_penetration": 2.0 * u.pc},
                ValueError,
                "shell_width",
            ),
            ((FINE,), {"eta_m": 0.0}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 1.5}, ValueError, "eta_m"),
            ((FINE,), {"eta_m": 0.5 * u.one}, ValueError, "eta_m"),
            ((FINE,), {"delta_shell": 1.0}, ValueError, "delta_shell"),
            ((FINE,), {"delta_shell": [0.05]}, ValueError, "delta_shell"),
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

    @pytest.mark.parametrize("R_c", [0.0, 5.0] * u.pc)
    def test_magnetic_field(self, r_grid, base_kwargs, R_c):
        calc = BubbleProfileCalculator(r_grid, R_c=R_c, **base_kwargs)
        B = calc.compute_magnetic_field_profile(eta_B=0.1).to_value(u.G)
        # r^2 v_w B^2 = eta_B M_dot v_w^2 / 2 in the wind, in Gaussian units
        flux = np.sqrt(
            0.05
            * base_kwargs["M_dot"].to_value(u.g / u.s)
            * calc.v_w.to_value(u.cm / u.s)
        )
        r = r_grid.to_value(u.cm)
        wind = calc.masks["r_wind"]
        np.testing.assert_allclose(B[wind] * r[wind], flux)
        # constant inside the core, as the density, and finite at r = 0 without it
        core = calc.masks["r_core"]
        if R_c > 0:
            np.testing.assert_allclose(B[core], flux / R_c.to_value(u.cm))
        assert np.all(np.isfinite(B))
        # compressed by sqrt(11) downstream of the shock, and zero outside the bubble
        bubble = calc.masks["r_bubble"]
        np.testing.assert_allclose(
            B[bubble], np.sqrt(11) * flux / calc.R_TS.to_value(u.cm)
        )
        assert np.all(B[calc.masks["r_ISM"]] == 0.0)

    @pytest.mark.parametrize("r_grid", [FINE, COARSE], ids=["fine", "coarse"])
    @pytest.mark.parametrize(
        "shell",
        [
            {"shell": "uniform"},
            {"shell": "uniform", "delta_shell": 0.1, "M_shell": 1e4 * u.M_sun},
            {**FRAGMENTED, "M_shell": 2e4 * u.M_sun},
        ],
        ids=["uniform", "uniform-custom", "fragmented"],
    )
    def test_gas_mass_budget(self, base_kwargs, r_grid, shell):
        calc = BubbleProfileCalculator(r_grid, R_c=5 * u.pc, **shell, **base_kwargs)
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

        # Swept-up shell, exactly the requested mass whatever the grid: the cells it
        # touches hold it on top of their hot-bubble and ambient parts
        g = Grid(r_centers=r_grid)
        shell_cells = calc.masks["r_shell"]
        background = n_b * _overlap(g.r_faces, 0 * u.pc, calc.R_cd) + n0 * _overlap(
            g.r_faces, calc.R_b, g.r_faces[-1]
        )
        mass = np.sum((n * g.volumes - background)[shell_cells]) * const.m_p
        expected = shell.get(
            "M_shell", 4 / 3 * np.pi * calc.R_b**3 * base_kwargs["rho_0"]
        )
        assert mass.to_value(u.M_sun) == pytest.approx(expected.to_value(u.M_sun))

        # Uniform between R_cd = (1 - delta_shell) R_b and R_b
        assert calc.R_cd == (1 - shell.get("delta_shell", 0.05)) * calc.R_b
        full = (g.r_faces[:-1] >= calc.R_cd) & (g.r_faces[1:] <= calc.R_b)
        V_shell = 4 / 3 * np.pi * (calc.R_b**3 - calc.R_cd**3)
        n_uniform = (expected / (V_shell * const.m_p)).to(u.cm**-3)
        if np.any(full):
            np.testing.assert_allclose(n[full].to_value(u.cm**-3), n[full][0].value)
        if shell["shell"] == "uniform":
            if np.any(full):
                np.testing.assert_allclose(n[full].to_value(u.cm**-3), n_uniform.value)
        else:
            # Fragmented: part of the mass penetrates 2 pc into the hot bubble, with a
            # density that decreases inwards from R_cd and peaks in the shell
            within = (g.r_faces[:-1] >= calc.R_cd - 2 * u.pc) & (
                g.r_faces[1:] <= calc.R_cd
            )
            assert np.all(shell_cells[within])
            assert np.all(np.diff(n[within]) >= 0)
            assert np.argmax(n) in np.flatnonzero(
                _overlap(g.r_faces, calc.R_cd, calc.R_b)
            )
            assert n.max() < n_uniform

    def test_shell_does_not_change_the_transport(self, r_grid, base_kwargs):
        uniform = BubbleProfileCalculator(r_grid, model="Morlino21", **base_kwargs)
        fragmented = BubbleProfileCalculator(
            r_grid, model="Morlino21", **FRAGMENTED, **base_kwargs
        )
        for method, kwargs in [
            ("compute_velocity_profile", {}),
            ("compute_magnetic_field_profile", {}),
            ("compute_diffusion_profile", {"E_k": 1 * u.TeV}),
        ]:
            np.testing.assert_array_equal(
                getattr(fragmented, method)(**kwargs),
                getattr(uniform, method)(**kwargs),
            )
