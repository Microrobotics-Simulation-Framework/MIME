"""The ``plug`` geometry kind: a solid circular cylinder placed in the world, the region a porous material fills in a
vessel (a clot). Kept in a module of its own while its place is decided (a kind of its own, or a placement and a region
on ``cylinder``): the flow reads a structure's cylinder through one adapter (``mime.dynamics.flow``'s
``porous_cylinder``), so either choice changes only that."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, cast, override

from mime.gradients import GradientStatus
from mime.registry import PHYSICAL, Direction, DirectionLike, Geometry, Native, convention, register_geometry
from mime.units import Length, Magnitude, Quantity, m


@register_geometry
@dataclass(frozen=True, kw_only=True)
class Plug(Geometry):
    """A solid circular cylinder placed in the world: ``radius`` and ``length`` along ``axis`` through ``centre``, its
    midpoint. A structure of this kind is a region of a vessel that a porous material fills, a clot
    (``exp.attach(Darcy(), medium=blood, structure=clot)``), its one region its ``solid``. Its distance is no shape
    a law reads by name: a flow that holds a porous region samples the solid's exact distance on its own grid
    (``mime.nodes.flow.fvm.plug_depth``), from these settings. On a stage it is USD's own ``Cylinder``."""

    name = "plug"
    version = "0.1.0"
    setting_roles = {
        **dict.fromkeys(("axis", "radius", "length"), PHYSICAL),
        "centre": convention(PHYSICAL, "a plug is centred on the world's origin unless placed"),
    }
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": ()}
    regions = ("solid",)
    directions = ("axis",)
    placement = {"centre": "origin", "axis": "z_axis"}
    native = Native("Cylinder", {"radius": "radius", "height": "length"}, {"axis": "Z"})
    axis: Annotated[DirectionLike, Direction]
    radius: Annotated[Quantity, Length]
    length: Annotated[Quantity, Length]
    centre: Annotated[Quantity, Length] = m((0.0, 0.0, 0.0))

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        found = [f"{name} is {settings[name]!r} m, not a length above zero" for name in ("radius", "length")
                 if not (isinstance(value := settings[name], float) and math.isfinite(value) and value > 0)]
        centre = settings["centre"]
        if not (isinstance(centre, tuple) and len(centre) == 3):
            found.append(f"centre is {centre!r}, not a point (three lengths)")
        return found

    @override
    def direction(self, name: str, settings: Mapping[str, Magnitude]) -> tuple[float, float, float]:
        if name != "axis":
            raise KeyError(name)
        axis = cast(tuple[float, float, float], settings["axis"])
        return (float(axis[0]), float(axis[1]), float(axis[2]))


__all__ = ["Plug"]
