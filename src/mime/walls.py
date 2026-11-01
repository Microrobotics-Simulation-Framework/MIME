"""Wall kinds: how a confining wall reaches a solver of the flow (``mime.registry.WallKind``), each declared once for
every solver family: the geometry it matches exactly, the parameters it reads from it, its validity about the wall and
its gradient status. A solver registers its own use of a kind under the kind's name and version (the
MRS's correction, ``mime.nodes.hydro.mrs``).

In 0.3 the method of regularised Stokeslets is the one solver. Its wall kinds are ``free_space``, a medium that fills no region,
and ``tube``, the straight circular tube's, read from the corrected table of its image system (MIME-ANO-117), a file
fetched into the assets cache (``mime assets fetch unit_tube_R0.997``). The plane's is reserved (``RESERVED``), with
its reason."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mime.core import switch
from mime.gradients import GradientStatus
from mime.registry import GradientRecord, WallForm, WallKind, register_wall_kind

#: Wall kinds whose names and meanings are fixed but which are not built, with why: a script naming one is refused
#: with the reason.
RESERVED: Mapping[str, str] = MappingProxyType({
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


#: The tube's read of its table, the switches it evaluates (``mime.nodes.hydro.mrs._tube``): which interval of a
#: table's axis a query is in; the fold of the angle between two samples and of their axial offset, with the sign of
#: the components odd in them; and the rule on the axis.
TUBE_SWITCHES = (
    switch.Declaration(
        owner="tube.table",
        indicator="a query's offset from its nearest knot on a table axis, in that axis's units (the tube's radius "
                  "for a radius or an axial offset, radians for the angle)",
        orders={"resistance": 1},
        route="none needed: the table is read by a cubic, continuously differentiable across every knot inside the "
              "readable range; a read past the range stops the run (wall_accepted false)"),
    switch.Declaration(
        owner="tube.angle",
        indicator="the sine of the angle from one sample to another about the tube's axis",
        orders={"resistance": 1},
        route="none needed: the table is mirrored past its first and last angle knots with the slope of the symmetric "
              "extension, and the odd components are zero there, so the folded read is continuously differentiable "
              "across them",
        unmeasured="the table is mirrored past its angle knots with its components' parities, so the read at the "
                   "negative fold with the sign of the odd components is the read at the positive one: the two "
                   "branches agree to rounding"),
    switch.Declaration(
        owner="tube.axial",
        indicator="the axial offset between two samples, in the tube's radius",
        orders={"resistance": 1},
        route="none needed: as tube.angle's, the table is mirrored past its first axial knot, a zero offset",
        unmeasured="as tube.angle's: the table is mirrored past a zero offset with its components' parities, so the "
                   "two branches agree to rounding"),
    switch.Declaration(
        owner="tube.axis",
        indicator="a sample's distance from the tube's axis, in the tube's radius (a pair's: the root of the mean of "
                  "its two samples' squared distances)",
        orders={"resistance": 0},
        route="a table read that is smooth in the square of the distance from the axis, for instance one mirrored "
              "past the axis knot with the slope of the symmetric extension; the read is continuous through the axis "
              "and, being cubic in the distance, has a different slope on its two sides"),
)


@register_wall_kind
@dataclass(frozen=True, kw_only=True)
class TubeWall(WallKind):
    """A straight circular tube with a no-slip wall, the medium filling its lumen: the image system of a Stokeslet in
    it (Liron and Shahar 1978), read from the corrected table of the unit tube, which the radius and the viscosity
    scale to any tube. It matches a cylinder region and reads its centre, axis and radius."""

    name = "tube"
    version = "0.1.0"
    form = WallForm.ANALYTIC
    matches = "cylinder"
    reads = ("centre", "axis", "radius")
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=TUBE_SWITCHES)
    references = ("LironShahar1978",)
    builds = {"": ()}
    validity = (
        "a straight circular tube of no-slip wall, infinitely long, the medium filling its lumen; the body's samples "
        "read within the table's readable range, from the axis to its last radial knot and as far apart along the "
        "axis as its file reaches, a read outside refused (wall_accepted false, and the plan refuses a body longer "
        "than the table reaches)",
        "the read's accuracy against the builder's image, and the table's gate, are MIME-ANO-117's",
        "from several radii along the axis the read's error exceeds the tube's whole Green's function: use it for a "
        "body's drag, not for interactions between bodies that far apart",
    )


__all__ = ["RESERVED", "TUBE_SWITCHES", "FreeSpaceWall", "TubeWall"]
