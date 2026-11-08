"""The constants of the study folder the plot, recipe, comparison and overlay tests share (``conftest.py`` writes it). They sit
in a module of their own name, since tests import them and a second ``conftest`` in the tree (``tests/sharing``) takes that
module name where two directories are collected in one session."""

from __future__ import annotations

SPEED = "swim_speed:umr.speed"
FREQUENCIES = (10.0, 20.0, 30.0)
STAGE_EVERY = 0.005
