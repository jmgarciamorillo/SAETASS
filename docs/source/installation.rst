.. _installation:

Installing SAETASS
==================

SAETASS is distributed as a standard Python package, designed to integrate with the existing scientific ecosystem. This section outlines the installation procedures for two distinct use cases: **deployment** for running simulations and **editable installation** for codebase development.

Prerequisites
-------------

SAETASS has been tested for **Python 3.10** or higher, so previous versions might show compatibility issues.

The core numerical engine relies on the standard PyData stack. While the package manager resolves these dependencies automatically, users integrating SAETASS into existing pipelines should be aware of the following requirements:

- :code:`NumPy`: For dense array operations and vectorization.
- :code:`SciPy`: For sparse linear algebra solvers and special functions.
- :code:`Astropy`: For physical units management and constants.

Environment Isolation
---------------------

To ensure reproducibility of simulation results and prevent version conflicts with system-level libraries, we strictly recommend installing SAETASS within an isolated virtual environment.

Whether you utilize standard :code:`venv`, :code:`uv` or :code:`conda`, the goal is to encapsulate the solver and its specific dependency tree from other projects.

Installation Workflows
----------------------

Select the installation method that aligns with your intended usage.

.. tabs::

   .. tab:: Standard installation (User)

      This workflow is intended for users or researchers who consume SAETASS as a library within a broader analysis pipeline and do not need to modify the source code. In this mode, the package is installed as a read-only dependency, ensuring that the solver's core logic remains immutable and reproducible.

      **1. Environment management**

      Before installation, you must initialize an isolated environment. The choice of tool typically depends on your local infrastructure or personal preference. There are the typical standard options:

      .. code-block:: bash

         # Using standard venv
         python -m venv saetass_env
         source saetass_env/bin/activate

         # Using uv
         uv venv
         source .venv/bin/activate

         # Using Conda
         conda create -n saetass_env python=3.10
         conda activate saetass_env

      **2. Installing the package**

      Once the environment is active, install SAETASS from the Python Package Index (PyPI). This will automatically resolve and download compatible versions of the dependencies.

      .. code-block:: bash

         # Standard install
         python -m pip install saetass

         # High-speed install with uv
         uv pip install saetass

      .. tip::
         SAETASS follows standard PEP 517 conventions and can be distributed as a pure-Python wheel. This makes the package itself compatible with offline installation workflows using :code:`python -m pip install --no-index`.

         In restricted environments without internet access, offline installation is possible provided that SAETASS and all its runtime dependencies are available locally as pre-built wheels.


   .. tab:: Editable installation (Developer)

      This workflow is designed for users who need to modify the solver's numerical schemes, add new physical modules or wish to run the test suite. An **editable installation** (or "development mode") allows you to modify the source code and have those changes take effect immediately without re-running the installation command.

      **1. Source acquisition**

      Clone the repository from the official source through standard git commands:

      .. code-block:: bash

         git clone https://github.com/jmgarciamorillo/SAETASS.git
         cd SAETASS

      **2. Environment and dependency resolution**

      In development mode, you require more than just the runtime engine. You need the "extra" dependencies for testing (e.g. :code:`pytest`).

      .. code-block:: bash

         # Create the environment
         python -m venv .venv
         source .venv/bin/activate

         # Perform the editable install with 'dev' extras and optionally with standard PyData plotting tools
         python -m pip install -e ".[dev,plotting]"

      .. warning::
         The `-e` flag (short for `--editable`) creates a link to your current directory. If you move or rename the `SAETASS` folder, the Python interpreter will no longer be able to find the package, and you will encounter an ``ImportError``.

      **3. Verification of the development path**

      To ensure your environment is indeed pointing to your local source code and not a previously installed version, you can check the location of the package:

      .. code-block:: bash

         python -c "import saetass; print(saetass.__file__)"

      The output should point to your local git directory.

.. note::
   Throughout this documentation we invoke pip via ``python -m pip`` instead of the standalone ``pip`` executable.
   This guarantees that the package manager is bound to the currently active interpreter and avoids edge cases in environments where multiple Python installations coexist.

Post-Installation Verification
------------------------------

A successful exit code from a package manager does not inherently guarantee that the numerical environment is correctly configured for scientific computation. We recommend a two-tier verification process: a structural check and a functional validation.

Structural verification
~~~~~~~~~~~~~~~~~~~~~~~

First, confirm that the Python interpreter can resolve the package and that the version matches your expectations. This is particularly important when managing multiple environments for different research projects.

.. code-block:: python

   import saetass
   import numpy
   import scipy

   print(f"SAETASS version: {saetass.__version__}")
   print(f"Location:        {saetass.__file__}")
   print(f"NumPy version:   {numpy.__version__}")
   print(f"SciPy version:   {scipy.__version__}")

.. note::
   If the `Location` points to a system directory (e.g., `/usr/lib/python...`) while you intended to use a local virtual environment, deactivate your current shell and re-verify your environment activation.

Functional validation (Developer/Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For users who have installed SAETASS in **editable mode**, running the integrated test suite is highly-recommended. SAETASS uses :code:`pytest` to verify the mathematical consistency of the solver, including conservation of particle flux and the accuracy of the finite difference schemes.

From the root of the source directory, execute:

.. code-block:: bash

   # Run the full suite with performance benchmarks
   pytest

A successful run will output a series of dots (representing passed tests). If any test fails with an ``AssertionError``, it typically indicates a mismatch in the underlying numerical libraries or an architecture-specific precision issue.

Maintenance and Updates
-----------------------

Scientific software evolves to incorporate more efficient algorithms and stability fixes for complex physical parameters. Keeping your installation synchronized with the upstream source is essential for the validity and reproducibility of your numerical results.

.. tabs::

   .. tab:: Standard installation (User)

      If you installed SAETASS as a **library** via a package manager, synchronize with the latest stable release using the command corresponding to your environment manager:

      .. code-block:: bash

         # Using standard pip
         python -m pip install --upgrade saetass

         # Using uv (High-speed resolution)
         uv pip install --upgrade saetass

         # Using Conda
         conda update saetass

   .. tab:: Editable installation (Developer)

      In **editable mode**, the update process requires both synchronizing the local source code and refreshing the environment's dependency metadata.

      **1. Synchronize source**

      Pull the latest commits from the primary branch of the repository:

      .. code-block:: bash

         git checkout main
         git pull origin main

      **2. Refresh dependencies**

      If the new version introduces new internal dependencies or metadata changes, re-run the editable installation to ensure your environment remains compliant:

      .. code-block:: bash

         python -m pip install -e ".[dev,plotting]"

.. warning::
   If you have active simulations running in a persistent environment—such as a **Jupyter kernel** or a long-running **Screen** session—they will continue to execute the *old* code loaded in memory until the interpreter is restarted.

   To avoid "ghost bugs" caused by a mismatch between the code on disk and the code in memory, always restart your interactive kernels or scripts immediately after an update.

Troubleshooting
---------------

If you encounter a ``ModuleNotFoundError`` despite a successful installation, check the following:

1. **Interpreter mismatch:** Ensure that the command `which python` (Linux/macOS) or `where python` (Windows) points to the interpreter inside your virtual environment.
2. **Path conflicts:** Ensure the environment variable ``PYTHONPATH`` is not being overridden by your `.bashrc` or `.zshrc`, as this can force Python to look for libraries in the wrong locations.

For further assistance, please refer to the GitHub issues page or contact the maintainer directly.
