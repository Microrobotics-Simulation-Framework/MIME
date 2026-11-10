"""Wall kinds: how a confining wall reaches a solver of the flow (``mime.registry.WallKind``), each declared once for
every solver family: the geometry it matches exactly, the parameters it reads from it, its validity about the wall and
its gradient status. A solver registers its own use of a kind under the kind's name and version (the
MRS's correction, ``mime.nodes.hydro.mrs``).

In 0.3 the method of regularised Stokeslets is the one solver. Its wall kinds are ``free_space``, a medium that fills no
region; ``tube``, the straight circular tube's, read from the corrected table of its image system (MIME-ANO-117), a file
fetched into the assets cache (``mime assets fetch unit_tube_R0.997``); and ``plane``, an unbounded plane's, the image
system in closed form, regularised by the drag's own blob. No name is reserved now (``RESERVED``).

Each answers where its wall is (``WallKind.distance``): its signed distance from a point of the medium and its normal
into it, minus the matched region's shape's, so the tube's is ``R`` less the point's distance from its axis and the
plane's the point's height above it; free space has no wall, and gives ``None``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mime.core import switch
from mime.gradients import GradientStatus
from mime.registry import GradientRecord, WallForm, WallKind, register_wall_kind

#: Wall kinds whose names and meanings are fixed but which are not built, with why: a script naming one is refused
#: with the reason.
RESERVED: Mapping[str, str] = MappingProxyType({})


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


@register_wall_kind
@dataclass(frozen=True, kw_only=True)
class PlaneWall(WallKind):
    """An unbounded plane with a no-slip wall, the medium filling the half-space its normal points into: the image
    system of a Stokeslet above it (Blake 1971), regularised by the drag's own blob (Ainley et al. 2008), in closed form,
    so no table and no switch. It matches a half_space region and reads its normal and offset; the offset is the drag
    node's leaf ``wall_offset`` (a gradient, a caller of the node and an override or a study of the plane's offset
    reach it), and the normal is held as built."""

    name = "plane"
    version = "0.1.0"
    form = WallForm.ANALYTIC
    matches = "half_space"
    reads = ("normal", "offset")
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord()
    references = ("Blake1971", "Ainley2008")
    builds = {"": ()}
    validity = (
        "an unbounded plane of no-slip wall, the medium on the side its normal points into; one plane alone: two walls' "
        "images do not sum to a two-wall image (each reflects the other's flow)",
        "every sample of the body in the medium, above the plane: a sample through it is outside the image's domain, "
        "which nothing refuses while the run goes on (no contact holds a body off a plane yet), so the node reports the "
        "body's height above it (wall_gap, below zero through it) and raises near_wall within a force spacing",
        "the regularised image of the drag's blob, exact for it (no slip and no divergence to rounding); the drag near "
        "the plane is as accurate as its force points resolve the gap, so the error is a function of the gap in force "
        "spacings, not radii. Estimates, not bounds: one sphere against Brenner's exact series for the normal entry "
        "(and the method of fundamental solutions with Blake's image for the others), force spacings of 0.3 and 0.2 "
        "radii, blob 0.1, float64. With the image at the force points the normal entry reads +3.6% at one spacing from "
        "the plane, +5.4% at 0.75, +10.3% at 0.55 (the drag's clear line), +14% at 0.5, +23% at 0.45 and +70% at 0.4 "
        "(0.2 radii: +11.8% at 0.55, +16.5% at 0.5); at 0.36 spacings one pose was indefinite (the run stops there, "
        "drag.mrs-indefinite) and at 0.33 a definite resistance read the normal entry 34% low, so definiteness does not "
        "mean accuracy here; at the default blob 0.5 the same image reads +30% at 0.75 spacings and +60% at 0.55 and is "
        "indefinite from 0.33",
        "with the image integrated over the cells (wall_refinement=2) the worst entry at 0.5 spacings is 2.6% to 14.8% "
        "off, depending on the force spacing and on which way the sphere's sampling grid faces the wall (0.3 radii: "
        "2.6% unturned, 4.6% turned; 0.2 radii: 14.8% with a grid pole facing the wall, 5.0% turned); the drag raises "
        "near_wall within one spacing, where the point image's normal entry is already +3.6%, and lubrication's "
        "hand-over, not this image, is what closes the last spacing",
        "float32 against float64 holds to 5e-6 of an entry's diagonal scale for a plane through the origin (measured "
        "at the gate gaps); the plane's height is a difference, y . n less the offset, in the working dtype, so with the "
        "plane displaced the error grows with the offset over the body's size (a 1 mm sphere half a radius from the "
        "plane, entries on their diagonal scale: 8e-7 at offset 0, 2e-5 at 0.5 m, 1.7e-4 at 5 m, 5,000 sizes, past "
        "1e-4): nothing flags or refuses it, so where the offset passes about a thousand body sizes "
        "in float32, run in float64 or place the world's origin on the plane",
    )


__all__ = ["RESERVED", "TUBE_SWITCHES", "FreeSpaceWall", "PlaneWall", "TubeWall"]
