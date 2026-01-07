Users Guide
===========

This guide explains how to use SAETASS modules.

Example usage:

.. code-block:: python

    from saetass import Grid, State, Solver
    from saetass.solvers import DiffusionSolver

    grid = Grid(nx=100)
    state = State(grid)
    solver = DiffusionSolver(grid, state)
    solver.run()

Refer to the API documentation for detailed class descriptions.
