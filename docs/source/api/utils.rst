Utils
=======

The utility module in SAETASS provides supplementary physical calculators to easily compute complex backgrounds and processes required by the transport solvers. These tools are designed to be independent, robust and extensible for a wide range of astrophysical environments.

SAETASS currently includes built-in utilities to compute particle loss timescales through the :py:class:`~saetass.utils.energy_losses.EnergyLossCalculator`, as well as a flexible framework for determining the spatial profiles of stellar wind bubbles via the :py:class:`~saetass.utils.bubble_profiles.BubbleProfileCalculator`.

Users can make use of these utilities to build physical setups and pass the generated data directly to the SAETASS solvers.

------------------

Energy losses module
---------------------

.. automodule:: saetass.utils.energy_losses
    :members:

Bubble profiles module
---------------------

.. automodule:: saetass.utils.bubble_profiles
    :members:
