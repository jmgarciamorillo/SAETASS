"""
Canonical internal unit definitions for SAETASS.

This module provides immutable, static Astropy units representing the canonical
internal standard for all physical dimensions used in the package:
- Length: pc (parsec)
- Time: Myr (megayears)
- Mass: M_sun (solar mass)
- Energy: GeV (gigaelectronvolts)
"""

from typing import Final

import astropy.constants as const
import astropy.units as u

# =============================================================================
# 1. Base Canonical Physical Units (Astrophysical Standard)
# =============================================================================
LENGTH: Final[u.Unit] = u.pc
TIME: Final[u.Unit] = u.Myr
MASS: Final[u.Unit] = u.M_sun
ENERGY: Final[u.Unit] = u.GeV

# =============================================================================
# 2. Kinematic and Dynamic Units
# =============================================================================
VELOCITY: Final[u.Unit] = LENGTH / TIME  # pc / Myr

# Named unit for GeV/c. It cannot be written as ``(u.GeV / const.c).unit``: that
# keeps only the unit of c and silently drops its value, yielding GeV s / m, where
# 1 GeV s / m = 2.998e8 GeV/c. It is registered so that unit strings containing
# it (e.g. ``"1 / (GeV_c pc3)"``) can be parsed back by Astropy.
GEV_C: Final[u.Unit] = u.def_unit(
    ["GeV_c"],
    u.GeV / const.c,
    format={"latex": r"GeV/c"},
    doc="Momentum unit GeV/c",
)
u.add_enabled_units([GEV_C])

MOMENTUM: Final[u.Unit] = GEV_C  # GeV / c
DIFFUSION_COEFFICIENT: Final[u.Unit] = LENGTH**2 / TIME  # pc^2 / Myr
DIFFUSION: Final[u.Unit] = DIFFUSION_COEFFICIENT  # Alias

# Loss and Evolution Rates
ENERGY_LOSS_RATE: Final[u.Unit] = ENERGY / TIME  # GeV / Myr
MOMENTUM_LOSS_RATE: Final[u.Unit] = MOMENTUM / TIME  # (GeV / c) / Myr

# =============================================================================
# 3. Geometric and Environmental Units
# =============================================================================
AREA: Final[u.Unit] = LENGTH**2  # pc^2
VOLUME: Final[u.Unit] = LENGTH**3  # pc^3
NUMBER_DENSITY: Final[u.Unit] = LENGTH**-3  # pc^-3
MASS_DENSITY: Final[u.Unit] = MASS / LENGTH**3  # M_sun / pc^3

MAGNETIC_FIELD: Final[u.Unit] = u.uG  # microGauss
LUMINOSITY: Final[u.Unit] = ENERGY / TIME  # GeV / Myr (convertible to erg / s)
MASS_LOSS_RATE: Final[u.Unit] = MASS / TIME  # M_sun / Myr (convertible to M_sun / yr)

CROSS_SECTION: Final[u.Unit] = LENGTH**2  # pc^2 (convertible to cm^2 / mbarn)
DIFFERENTIAL_CROSS_SECTION: Final[u.Unit] = LENGTH**2 / ENERGY  # pc^2 / GeV

# =============================================================================
# 4. Particle Distribution Representations (State)
# =============================================================================
# Phase space distribution: f_ps(r, p) -> [dn / (d^3r d^3p)]
F_PS: Final[u.Unit] = NUMBER_DENSITY / (MOMENTUM**3)

# Momentum differential density: psi_p(r, p) = dn / dp = 4*pi*p^2 * f_ps
PSI_P: Final[u.Unit] = NUMBER_DENSITY / MOMENTUM

# Energy differential density: psi_E(r, E) = dn / dE
PSI_E: Final[u.Unit] = NUMBER_DENSITY / ENERGY

# Source terms: Q(r, p) -> d(psi_p)/dt or d(psi_E)/dt
SOURCE_PSI_P: Final[u.Unit] = PSI_P / TIME
SOURCE_PSI_E: Final[u.Unit] = PSI_E / TIME


__all__ = [
    "LENGTH",
    "TIME",
    "MASS",
    "ENERGY",
    "VELOCITY",
    "GEV_C",
    "MOMENTUM",
    "DIFFUSION_COEFFICIENT",
    "DIFFUSION",
    "ENERGY_LOSS_RATE",
    "MOMENTUM_LOSS_RATE",
    "AREA",
    "VOLUME",
    "NUMBER_DENSITY",
    "MASS_DENSITY",
    "MAGNETIC_FIELD",
    "LUMINOSITY",
    "MASS_LOSS_RATE",
    "CROSS_SECTION",
    "DIFFERENTIAL_CROSS_SECTION",
    "F_PS",
    "PSI_P",
    "PSI_E",
    "SOURCE_PSI_P",
    "SOURCE_PSI_E",
]
