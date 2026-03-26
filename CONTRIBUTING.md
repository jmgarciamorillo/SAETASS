# Contributing to SAETASS

Thank you for your interest in contributing to **SAETASS**! Whether it's a bug report, a feature suggestion, documentation improvements, or a new physical module: every contribution helps the project grow.

Please take a moment to review this guide before submitting your contribution.

## Table of contents

- [Code of conduct](#-code-of-conduct)
- [How can I contribute?](#-how-can-i-contribute)
  - [Reporting bugs](#-reporting-bugs)
  - [Suggesting features](#-suggesting-features)
  - [Contributing Code](#-contributing-code)
- [Development Setup](#-development-setup)
- [Code Style and Standards](#-code-style-and-standards)
- [Pull Request Process](#-pull-request-process)

## Code of Conduct

This project adheres to the [**Contributor Covenant Code of Conduct**](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [**jmorillo@iaa.es**](mailto:jmorillo@iaa.es).

## How can I contribute?

### Reporting bugs

If you find a bug, please [**open an issue**](https://github.com/jmgarciamorillo/SAETASS/issues/new) and include:

- A clear and descriptive title
- Steps to reproduce the behavior
- Expected versus actual results
- Your Python version and OS
- The version of SAETASS and its dependencies (`pip list`)
- If applicable, a minimal code snippet that reproduces the issue

### Suggesting features

Feature requests **are welcome**! When suggesting a new feature:

- Describe the **problem** your feature would solve
- Explain the **proposed solution** and any alternatives you've considered
- Indicate whether you'd be willing to implement it yourself

### Contributing code

We welcome pull requests for:

- Bug fixes
- New solvers or operator implementations
- Performance improvements (e.g., Numba optimizations, parallelization strategies, etc.)
- Documentation improvements
- Additional examples or validation scripts
- Test coverage improvements

## Development setup

1. **Fork the repository** at https://github.com/jmgarciamorillo/SAETASS.git in your own GitHub account.

2. **Clone the forked repository** to your local machine:

   ```bash
   git clone https://github.com/<your-username>/SAETASS.git
   cd SAETASS
   ```

2. **Create a virtual environment** and install in editable mode with dev dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run the test suite** to verify everything works:

   ```bash
   pytest
   ```

4. **Create a feature branch** for your work:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code style and standards

- **Dependencies versions**: See [`pyproject.toml`](pyproject.toml).
- **Docstrings**: Use the [NumPy/SciPy docstring convention](https://numpydoc.readthedocs.io/en/latest/format.html) for all public classes and functions.
- **Type hints**: Encouraged for function signatures where they improve clarity.
- **Testing**: Every new feature or fix should include corresponding tests in the `test/` directory. Use `pytest` to run them.
- **Naming**: Follow PEP 8 conventions. Use descriptive variable names — prefer `diffusion_coefficient` over `D` in public APIs.

## Pull request process

1. **Ensure tests pass**: Run `pytest` before submitting.
2. **Write descriptive commit messages**: Use clear and concise commit messages.
3. **Keep PRs focused**: One feature or fix per pull request. Large changes should be discussed in an issue first.
4. **Update documentation**: If your change adds or modifies public API, update the relevant docstrings and Sphinx `.rst` files.
5. **Reference related issues**: Link to any relevant issues in your PR description (e.g., `Fixes #42`).

After submitting:

- A maintainer will review your PR and may request changes
- Once approved, your contribution will be merged into `main`
- You will be credited in the project. Thank you!!

## Thank you!

Every contribution, no matter how small, makes SAETASS better for the entire astroparticle physics community. **We truly appreciate your time and effort!**
