<div align="center">

<img src="docs/assets/saetass_logo_horizontal.svg" alt="SAETASS Logo" width="500">

<br/>
<br/>

**Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry**

A Python library for simulating cosmic-ray transport and acceleration in astrophysical environments using finite volume methods and operator splitting.

<br/>

<!-- ─────────────────────── Badges ─────────────────────── -->

<!-- CI & Code Quality -->
[![CI tests](https://img.shields.io/github/actions/workflow/status/jmgarciamorillo/SAETASS/tests.yml?logo=github&logoColor=white&label=CI%20tests&color=F3AC4B)](https://github.com/jmgarciamorillo/SAETASS/actions)
[![Codecov Placeholder](https://img.shields.io/badge/coverage-100%25-F3AC4B?logo=codecov&logoColor=white)](https://codecov.io/gh/jmgarciamorillo/SAETASS) <!-- [![codecov](https://img.shields.io/codecov/c/github/jmgarciamorillo/SAETASS?logo=codecov&logoColor=white&color=F3AC4B)](https://codecov.io/gh/jmgarciamorillo/SAETASS) -->
[![Docs Placeholder](https://img.shields.io/badge/docs-passing-F3AC4B?logo=readthedocs&logoColor=white)](YOUR_DOCS_URL_HERE) <!-- [![Docs](https://img.shields.io/readthedocs/saetass?logo=readthedocs&logoColor=white&color=F3AC4B)](DOCS_URL) -->
[![PyPI Version Placeholder](https://img.shields.io/badge/PyPI-v0.1.0-F3AC4B?logo=pypi&logoColor=white)](https://pypi.org/project/saetass/) <!-- [![PyPI version](https://img.shields.io/pypi/v/saetass?logo=pypi&logoColor=white&color=F3AC4B)](https://pypi.org/project/saetass/) -->
[![Python Versions Placeholder](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-F3AC4B?logo=python&logoColor=white)](https://pypi.org/project/saetass/) <!-- [![Python versions](https://img.shields.io/pypi/pyversions/saetass?logo=python&logoColor=white&color=F3AC4B)](https://pypi.org/project/saetass/) -->
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome!-F3AC4B.svg?logo=github&logoColor=white)](CONTRIBUTING.md)
[![License: BSD-3](https://img.shields.io/badge/license-BSD_3--Clause-F3AC4B.svg?logo=open-source-initiative&logoColor=white)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/jmgarciamorillo/SAETASS?color=F3AC4B&logo=git&logoColor=white)](https://github.com/jmgarciamorillo/SAETASS/commits/main)
[![Issues](https://img.shields.io/github/issues/jmgarciamorillo/SAETASS?color=F3AC4B&logo=github&logoColor=white)](https://github.com/jmgarciamorillo/SAETASS/issues)
[![Stars](https://img.shields.io/github/stars/jmgarciamorillo/SAETASS?color=F3AC4B&logo=github&logoColor=white)](https://github.com/jmgarciamorillo/SAETASS/stargazers)


</div>

## Overview

**SAETASS** numerically solves the **astroparticle transport equation** in one-dimensional spherical symmetry; this is, the fundamental equation governing the propagation and energy losses of energetic particles within astrophysical environments such as stellar wind bubbles.

The package decomposes the full transport equation into independent physical operators: **diffusion**, **advection**, **energy losses** and **source**. It evolves them via mathematically robust **operator-splitting schemes**. Each operator is implemented as a dedicated finite-volume solver, ensuring modularity, testability and physical transparency.

> **For comprehensive documentation**, visit the [SAETASS Documentation](DOCS_URL_PLACEHOLDER).


<div align="center">

<img src="docs/assets/evolution.gif" alt="SAETASS Multi-Energy Diffusion Simulation Evolution" width="900">

</div>


## Key Features

| Feature | Description |
|---------|-------------|
| **Modular solvers** | Independent numerical schemes for advection, diffusion energy losses, and source injection; combined via operator splitting |
| **Flexible grids** | Dedicated grid object with support for diverse spatial and momentum grid configurations |
| **Physical accuracy** | Deep integration with `astropy.units` and `astropy.constants` for strict dimensional analysis |
| **Extensibility** | Plug in custom diffusion coefficients, velocity fields, loss functions and source terms |

## Installation

A quick simple installation to get started using SAETASS can be simply done via `pip`:

```bash
pip install saetass
```

This will install the latest stable version of SAETASS from [PyPI](PYPI_URL_PLACEHOLDER).

> See the full [Installation Guide](INSTALLATION_GUIDE_URL_PLACEHOLDER) for additional installation options and troubleshooting.

## Quickstart

For step-by-step guides to get started with SAETASS, check the [Tutorials](TUTORIALS_URL_PLACEHOLDER) section of the [documentation](DOCS_URL_PLACEHOLDER).

For complete, more advanced examples, check the [Examples](EXAMPLES_URL_PLACEHOLDER) section of the [documentation](DOCS_URL_PLACEHOLDER).


## Mathematical background

SAETASS solves the following **astroparticle transport equation** for in spherically symmetric geometry:

$$\frac{\partial f}{\partial t} 
    + \frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2 u_\mathrm{w}f\right)
    + \frac{\partial}{\partial p}\left( \dot{p} f \right)
    = \frac{1}{r^2} \frac{\partial}{\partial r}
    \left(r^2 D\frac{\partial f}{\partial r}\right)
    + Q.$$

Where:
- $f(t,r, p)$ is the particle distribution function,
- $u_\mathrm{w}(t,r,p)$ is the advection (wind) velocity,
- $\dot{p}(t, r, p)$ is the rate of energy loss,
- $D(t, r, p)$ is the spatial diffusion coefficient,
- $Q(t, r, p)$ is the source term.

The equation is discretized using a **finite volume scheme** and solved by an **operator-splitting routine**, which decomposes the full PDE into simpler, independently solvable subproblems.

### References

<!-- TODO: Replace PLACEHOLDER_FOR_TECHNICAL_PAPER with the actual URL of the technical paper -->
> For a detailed derivation of the numerical schemes, see the [technical paper](PLACEHOLDER_FOR_TECHNICAL_PAPER).


## Citation

If you use SAETASS in your research, **please cite the technical paper** to acknowledge the development effort:

```bibtex
@article{PLACEHOLDER_FOR_TECHNICAL_PAPER,
  author  = {Garcia-Morillo, J. M.},
  title   = {SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry},
  journal = {XXXX},
  year    = {2026},
  volume  = {X},
  doi     = {10.XXXX/XXXXX}
}
```

<!-- TODO: Uncomment when a Zenodo DOI is assigned -->
For version-specific citations, you may use the **Zenodo DOI** [10.5281/zenodo.XXXXX](https://doi.org/10.5281/zenodo.XXXXX) citation:

```bibtex
@software{PLACEHOLDER_FOR_ZENODO_CITATION,
  author    = {Garcia-Morillo, J. M.},
  title     = {SAETASS: Solver for Astroparticle Equation of Transport Analysis in Spherical Symmetry},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXX}
}
```

## Contributing

Contributions, bug reports and feature suggestions are welcome! Whether you want to fix a typo, improve a solver or add a new physical module: we'd love your help!

Please read the [Contributing Guidelines](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

## License

SAETASS is released under the [BSD 3-Clause License](LICENSE).

## Acknowledgements

<div align="center">

SAETASS is developed within the [**VHEGA**](https://vhega.iaa.es) research group at the [**Instituto de Astrofísica de Andalucía (IAA-CSIC)**](https://www.iaa.csic.es), Granada, Spain.

<br/>

[![Author](https://img.shields.io/badge/Author-J.%20M.%20Garc%C3%ADa%20Morillo-F3AC4B?logo=github&logoColor=white)](https://github.com/jmgarciamorillo)
[![Email](https://img.shields.io/badge/Email-jmorillo%40iaa.es-D14836?logo=minutemailer&logoColor=white)](mailto:jmorillo@iaa.es)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5232--349X-a6ce39?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-5232-349X)

</div>
