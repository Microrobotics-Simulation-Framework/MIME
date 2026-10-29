"""Wall kinds: how a confining wall reaches a solver of the flow (``mime.registry.WallKind``), each declared once for
every solver family: the geometry it matches exactly, the parameters it reads from it, its validity about the wall and
its gradient status. A solver registers its own use of a kind under the kind's name and version (the
boundary-element method's correction, ``mime.nodes.hydro.bem``).

In 0.3 the boundary-element method is the one solver, and ``free_space``, a medium that fills no region, its one wall
kind. The straight circular tube's and the plane's are reserved (``RESERVED``), each with its reason."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mime.gradients import GradientStatus
from mime.registry import WallForm, WallKind, register_wall_kind

#: Wall kinds whose names and meanings are fixed but which are not built, with why: a script naming one is refused
#: with the reason.
RESERVED: Mapping[str, str] = MappingProxyType({
    "tube": "the straight circular tube's tables of its image system, the v0.2 line's of 2026-05-21, fail the "
            "Fourier-Bessel check (MIME-ANO-117); the corrected table is built and shipped as the reference "
            "unit_tube_R0.997, and the wall kind is reserved until it is registered on it; the boundary-element "
            "method's read of the table is in place (mime.nodes.hydro.bem.tube_correction)",
    "plane": "a plane wall has no correction in 0.3: its analytic kind (Blake's image system) and the meshed form, "
             "which samples any wall a geometry kind can, are reserved for later",
})


@register_wall_kind
@dataclass(frozen=True, kw_only=True)
class FreeSpaceWall(WallKind):
    """An unbounded medium: no wall, so nothing is added to the free-space kernel. It matches a medium that fills no
    region; a medium that fills one is confined, and leaving its wall out would change the flow."""

    name = "free_space"
    version = "0.1.0"
    form = WallForm.ANALYTIC
    matches = None
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": ()}
    validity = ("an unbounded medium: no wall within the reach of the body's flow",)


__all__ = ["RESERVED", "FreeSpaceWall"]
