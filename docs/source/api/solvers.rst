Solvers
=======

SAETASS is designed to be modular and extensible, allowing users to implement and integrate their own
solvers for any physical process in the spherical astroparticle transport equation. The solvers are
responsible for advancing the state of the system in time according to the governing equations and numerical methods.
SAETASS includes a set of built-in solvers for common astrophysical processes, such as a :class:`HyperbolicSolver`
for advection and loss terms, a :class:`DiffusionSolver` for diffusion term and a :class:`SourceSolver`
for source terms.

Users can also create custom solvers by providing their own implementations with the correct API interface called by the
general :class:`Solver` framework.

The Solver Module
------------------

.. automodule:: saetass.solver
    :members:

Physical solvers
----------------

Hyperbolic solver
~~~~~~~~~~~~~~~~~

.. automodule:: saetass.solvers.hyperbolic_solver
    :members:

Advection Solver
^^^^^^^^^^^^^^^^
.. automodule:: saetass.solvers.advection_solver
    :members:
    :undoc-members:
    :show-inheritance:

Loss Solver
^^^^^^^^^^^
.. automodule:: saetass.solvers.loss_solver
    :members:
    :undoc-members:
    :show-inheritance:

Diffusion solver
~~~~~~~~~~~~~~~~

.. automodule:: saetass.solvers.diffusion_solver
    :members:
    :undoc-members:

Source solver
~~~~~~~~~~~~~

.. automodule:: saetass.solvers.source_solver
    :members:
    :undoc-members: