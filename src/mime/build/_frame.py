"""The world frame, as a stand-in until the scene lands: the experiment declares ``up`` once and its gravity's
magnitude, and the build derives the gravity vector as ``-g * up``. There are no default axes. When the scene arrives
the frame is read from the stage instead, once, and this module goes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mime.units import ACCELERATION, UnitError, si


@dataclass(frozen=True)
class WorldFrame:
    up: tuple[float, float, float]
    gravity: tuple[float, float, float]


def world_frame(up: Any, gravity: Any) -> tuple[WorldFrame | None, list[tuple[str, str, str]]]:
    """The frame, or the problems with ``up`` and ``gravity`` as ``(code, message, action)``."""
    problems: list[tuple[str, str, str]] = []
    axis: tuple[float, float, float] | None = None
    if up is None:
        problems.append(("frame.up-missing", "the experiment declares no up direction, and MIME assumes no axis",
                         "pass up=(0, 0, 1) (or whichever direction is up in your world) to Experiment"))
    else:
        try:
            components = tuple(float(c) for c in up)
        except (TypeError, ValueError):
            components = ()
        norm = math.sqrt(sum(c * c for c in components)) if len(components) == 3 else 0.0
        if len(components) != 3 or not math.isfinite(norm) or norm == 0.0:
            problems.append(("frame.up-invalid", f"up={up!r} is not a finite, non-zero 3-vector",
                             "pass up as three numbers, such as (0, 0, 1)"))
        else:
            axis = (components[0] / norm, components[1] / norm, components[2] / norm)
    magnitude: float | None = None
    if gravity is None:
        problems.append(("frame.gravity-missing", "the experiment declares no gravity",
                         "pass gravity=standard_gravity(), or m_per_s2(0) for none"))
    else:
        try:
            value = si(gravity, ACCELERATION, what="gravity")
        except UnitError as error:
            problems.append(("frame.gravity-units", str(error), "write gravity as an acceleration, such as "
                                                                  "standard_gravity() or m_per_s2(9.81)"))
        else:
            if isinstance(value, tuple) or not (math.isfinite(value) and value >= 0.0):
                problems.append(("frame.gravity-invalid", f"gravity {gravity} is not a finite magnitude >= 0; "
                                                          "its direction is -up",
                                 "pass the magnitude of gravity, such as standard_gravity()"))
            else:
                magnitude = value
    if axis is None or magnitude is None:
        return None, problems
    return WorldFrame(axis, (-magnitude * axis[0], -magnitude * axis[1], -magnitude * axis[2])), problems
