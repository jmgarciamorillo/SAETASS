SAETASS Tutorials
=================

Este tutorial explica cómo usar SAETASS paso a paso.

1. Crear un objeto Grid
2. Crear un objeto State
3. Inicializar un Solver
4. Ejecutar la simulación

.. code-block:: python

    from saetass import Grid, State, Solver
    grid = Grid(nx=100)
    state = State(grid)
    solver = Solver(grid, state)
