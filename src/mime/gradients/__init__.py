"""Gradient statuses, their composition and the gradient map's checks."""

from enum import Enum


class GradientStatus(Enum):
    """How a derivative passes through an entry; every registered entry declares one.

    ``EXACT``: the derivative of the computed value. ``FIXED_KEY``: exact with every random key held fixed.
    ``PIECEWISE``: exact away from declared switches, where it jumps. ``UNVERIFIED``: computed but not yet checked.
    ``CUT``: no derivative passes, for a stated reason.
    """

    EXACT = "exact"
    FIXED_KEY = "fixed_key"
    PIECEWISE = "piecewise"
    UNVERIFIED = "unverified"
    CUT = "cut"


__all__ = ["GradientStatus"]
