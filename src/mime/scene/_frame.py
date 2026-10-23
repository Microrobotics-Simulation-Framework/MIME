"""The world frame, declared once on the scene: ``up``, the length unit the stage is written in, and gravity derived as
``-g * up`` from the experiment's magnitude. There are no default axes. Up is +Y or +Z, the two axes USD's ``upAxis``
names, so the stage's own metadata says which way is up."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mime.units import ACCELERATION, UnitError, si

#: The length units a scene may be written in, as USD's ``metersPerUnit``. MIME computes in SI whatever the scene's
#: unit; the stage's points and translations are written in it.
METERS_PER_UNIT = {"m": 1.0, "cm": 0.01, "mm": 0.001, "um": 1e-6}

#: The directions a scene's up may take, each with the token USD's ``upAxis`` names it by.
UP_AXES: dict[tuple[float, float, float], str] = {(0.0, 1.0, 0.0): "Y", (0.0, 0.0, 1.0): "Z"}

type Problem = tuple[str, str, str]


@dataclass(frozen=True)
class WorldFrame:
    up: tuple[float, float, float]
    gravity: tuple[float, float, float]


def unit_vector(value: Any) -> tuple[float, float, float] | None:
    """``value`` as a unit 3-vector, or ``None`` when it is not three finite numbers, not all zero."""
    try:
        components = tuple(float(c) for c in value)
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(c * c for c in components)) if len(components) == 3 else 0.0
    if len(components) != 3 or not math.isfinite(norm) or norm == 0.0:
        return None
    return (components[0] / norm, components[1] / norm, components[2] / norm)


def scene_up(up: Any) -> tuple[tuple[float, float, float] | None, list[Problem]]:
    """The scene's up as a unit vector, or the problem with it as ``(code, message, action)``."""
    if up is None:
        return None, [("frame.up-missing", "the scene declares no up direction, and MIME assumes no axis",
                       "pass up=+Z (or whichever direction is up in your world) to Scene.new")]
    axis = unit_vector(up)
    if axis is None:
        return None, [("frame.up-invalid", f"the scene's up={up!r} is not a finite, non-zero 3-vector",
                       "pass up as a signed axis such as +Z, or as three numbers")]
    problem = up_axis_problem(up)
    return (None, [problem]) if problem is not None else (axis, [])


def up_axis_problem(up: Any) -> Problem | None:
    """The problem with ``up`` when it is a direction other than +Y or +Z; ``None`` for those two, and for what is no
    direction at all (``scene_up`` names that)."""
    axis = unit_vector(up)
    if axis is None or axis in UP_AXES:
        return None
    return ("frame.up-not-y-or-z",
            f"the scene's up={up!r} is neither +Y nor +Z, and a scene's up is +Y or +Z only, the two axes USD's "
            "upAxis names",
            "up is a choice of coordinates, so re-express the scene with up along +Y or +Z (for example by swapping "
            "two axes); importing data in another frame will be a declared frame transform")


def meters_per_unit(units: Any) -> tuple[float | None, list[Problem]]:
    if isinstance(units, str) and units in METERS_PER_UNIT:
        return METERS_PER_UNIT[units], []
    return None, [("scene.units", f"the scene's units={units!r} is not a length unit a stage is written in",
                   f"pass units= one of {', '.join(repr(u) for u in METERS_PER_UNIT)}")]


def world_frame(up: tuple[float, float, float] | None, gravity: Any) -> tuple[WorldFrame | None, list[Problem]]:
    """The frame from the scene's up and the experiment's gravity, or the problems with gravity."""
    problems: list[Problem] = []
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
    if up is None or magnitude is None:
        return None, problems
    return WorldFrame(up, (-magnitude * up[0], -magnitude * up[1], -magnitude * up[2])), problems
