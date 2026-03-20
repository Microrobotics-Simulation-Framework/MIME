"""MIME visualisation layer — USD stage bridge and viewport implementations.

Optional dependency: requires `usd-core` and optionally `pyvista`.
Install with: pip install mime-microrobotics[viz]

The viz layer is decoupled from the simulation — it reads state via
the StepObserver callback and writes to a USD stage. The viewport
reads the stage and produces pixels. Simulation code never imports
from this module.
"""
