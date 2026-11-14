"""The switches of the camera's renderer: each owner the pixel integration, the closed forms and the corner choice pass
to ``mime.core.switch``, declared once here and carried by the ``camera`` entry (``mime.observations.Camera``).

Every order is of ``expected``, the camera's differentiable output (``image`` is a rounding and carries none, and the
counters are counts). A row of ``tests/saved/test_switch_harness.py`` measures each owner at the renderer's function
that evaluates it, on one pixel or one ray: the forced-branch difference of the function's value, which the pixel's
coverage and transmission read, is the only thing the harness can see, so an order is that of the function's value at
the switch, a jump where its branches differ there and a kink where they meet. A switch that is a jump or a kink in a
function the pixel's value reads only through a saturating step (a coverage clipped to zero or one) is measured at a
point where it is not saturated.

The renderer's non-smooth operations that are not declared here stay raw in ``docs/ratchets/raw_switches.txt``, each
with its reason: an operation inside the undistortion's solve (whose derivative is the implicit function's, taken after
the loop), a rounding or a count, a flag's, or a floor that is active only where the quantity it keeps off zero is zero."""

from __future__ import annotations

from mime.core import switch

#: The pixel integration's box edge: the share of a pixel's square on the body's side of a straight line, a piecewise
#: function of the line's signed distance ``c`` from the pixel's centre (positive on the body's side, in pixels): zero
#: until the line reaches the near corner, a triangle until it reaches the second, linear to the centre, and the same
#: mirrored. Written for the line's side of the centre (``outside``: the centre outside the body, ``inside``: inside),
#: each side evaluates both, one on the line and one on the centre.
EDGE_SWITCHES = (
    switch.Declaration(
        owner="camera.edge.out_far",
        indicator="the distance of the line from the far corner of the pixel's square on the side of the centre "
                  "outside the body, less the share's start: minus the line's distance from the centre less the "
                  "square's corner reach, in pixels (positive where no part of the square is covered)",
        orders={"expected": 1},
        route="none needed: the share starts as a triangle with the slope zero, so it is continuously differentiable "
              "across the corner (a second derivative jumps)"),
    switch.Declaration(
        owner="camera.edge.out_near",
        indicator="the distance of the line from the pixel's near side on the side of the centre outside the body, "
                  "in pixels (positive where the share is the triangle)",
        orders={"expected": 1},
        route="none needed: the triangle meets the linear share with the same value and slope (a second derivative "
              "jumps)"),
    switch.Declaration(
        owner="camera.edge.in_far",
        indicator="as camera.edge.out_far, for the line mirrored through the centre (the centre inside the body), "
                  "in pixels",
        orders={"expected": 1},
        route="none needed: as camera.edge.out_far"),
    switch.Declaration(
        owner="camera.edge.in_near",
        indicator="as camera.edge.out_near, for the line mirrored through the centre (the centre inside the body), "
                  "in pixels",
        orders={"expected": 1},
        route="none needed: as camera.edge.out_near"),
    switch.Declaration(
        owner="camera.edge.centre",
        indicator="minus the line's signed distance from the pixel's centre, in pixels (positive where the centre is "
                  "outside the body)",
        orders={"expected": 1},
        route="none needed: the share is symmetric about the centre and its slope is the same either side (a second "
              "derivative jumps where the normal is diagonal)"),
)

#: Where two pieces of a body's outline meet in a pixel, the line that composes them is the nearer piece's: the least
#: of two distances for a union, the greatest for an intersection, each carrying its own normal.
PIECES_SWITCHES = (
    switch.Declaration(
        owner="camera.pieces.least",
        indicator="the second piece's line's distance from the pixel's centre less the first's, in pixels (positive "
                  "where the first is nearer-inside and taken)",
        orders={"expected": -1},
        route="the corner sub-samples take the pieces' lines at the sub-pixels' centres and so narrow the jump to a "
              "sub-pixel's share (camera.corner); a smooth union of the pieces (a softened least) would remove it"),
    switch.Declaration(
        owner="camera.pieces.greatest",
        indicator="the first piece's line's distance from the pixel's centre less the second's, in pixels (positive "
                  "where the first is farther-out and taken)",
        orders={"expected": -1},
        route="as camera.pieces.least, for an intersection"),
)

#: The corner choice: a pixel through which two pieces of a body's outline pass within half its diagonal, or a turning
#: piece's outline, takes its sub-samples' mean in place of one straight edge.
CORNER_SWITCHES = (
    switch.Declaration(
        owner="camera.corner",
        indicator="the second of the frame's lines to come within the corner reach (half a pixel's diagonal), how far "
                  "inside it that line's distance from its pixel's centre lies, or a turning piece's, the greater, "
                  "the greatest over the frame's pixels, in pixels (positive where the frame takes the sub-samples)",
        orders={"expected": 0},
        route="none needed: the choice is continuous where it is made",
        unmeasured="at its threshold the lines that make a pixel a corner graze the pixel's corner and cut no area, "
                   "and the pixel's two estimates (one straight edge, the sub-samples' mean) are each the share the "
                   "other lines cut: they differ only inside the reach, by 15 % of a pixel at a rod's corner, "
                   "where the choice is the sub-samples, and a pixel with a straight edge has the same value either way"),
)

#: The exact disc coverage: the area of a pixel's quadrilateral inside the unit disc, a closed chain of pieces on each
#: side (``_edge_in_disc``), the pixel's sides taken as straight between the undistorted corners.
DISC_SWITCHES = (
    switch.Declaration(
        owner="camera.disc.series",
        indicator="the bound of the angle's series less the absolute angle a chord subtends, in radians (positive "
                  "where the segment's area is taken from the series)",
        orders={"expected": -1},
        route="a longer series: it stops at its ninth power, 6e-12 of a pixel's area short of the closed form at the "
              "threshold (0.5 rad), and one more term brings that to 1e-14"),
    switch.Declaration(
        owner="camera.disc.start",
        indicator="one minus the squared distance of a side's first end from the disc's centre, in the disc's units "
                  "squared (positive where the end is inside the circle)",
        orders={"expected": 0},
        route="none: an end outside the circle enters the chain as its projection onto it, which meets the end on the "
              "circle and differs in its slope across it"),
    switch.Declaration(
        owner="camera.disc.end",
        indicator="one minus the squared distance of a side's second end from the disc's centre, in the disc's units "
                  "squared (positive where the end is inside the circle)",
        orders={"expected": 0},
        route="none: as camera.disc.start"),
    switch.Declaration(
        owner="camera.disc.crossing",
        indicator="the least of three margins in the segment's own fraction: the half-chord where the line meets the "
                  "circle (minus the root of how far it misses it where it does not), the chord's end past the "
                  "segment's start, and the segment's end past the chord's start (positive where the segment "
                  "crosses the circle)",
        orders={"expected": 0},
        route="none: where a segment reaches the circle (tangent, or ending on it) the chain's pieces change and "
              "their sum, the area, is continuous; its slope is not"),
    switch.Declaration(
        owner="camera.disc.enter.low",
        indicator="the fraction of a side at which its line enters the circle, in the segment's own fraction "
                  "(positive past the segment's start)",
        orders={"expected": 0},
        route="none: a side that starts inside the circle enters it at its start"),
    switch.Declaration(
        owner="camera.disc.enter.high",
        indicator="the segment's end less the fraction at which its line enters the circle, in the segment's own "
                  "fraction",
        orders={"expected": 0},
        route="none: a side that ends before it reaches the circle does not cross it, which camera.disc.crossing "
              "selects"),
    switch.Declaration(
        owner="camera.disc.exit.low",
        indicator="the fraction of a side at which its line leaves the circle, in the segment's own fraction "
                  "(positive past the segment's start)",
        orders={"expected": 0},
        route="none: a side whose line leaves the circle before the segment starts does not cross it, which "
              "camera.disc.crossing selects"),
    switch.Declaration(
        owner="camera.disc.exit.high",
        indicator="the segment's end less the fraction at which its line leaves the circle, in the segment's own "
                  "fraction",
        orders={"expected": 0},
        route="none: a side that ends inside the circle leaves it at its end"),
    switch.Declaration(
        owner="camera.disc.full",
        indicator="the least of the four corners' margins inside the circle, in the disc's units squared (positive "
                  "where all four are inside)",
        orders={"expected": 0},
        route="none: a pixel wholly inside the circle is covered exactly, as the chain's sum is to rounding",
        unmeasured="the chain's sum is the pixel's area to rounding, so the pixel's share is one by either branch"),
    switch.Declaration(
        owner="camera.disc.clear",
        indicator="the least of the four corners' distances outside the circle, in the disc's units squared, and "
                  "one less the count of sides crossing it (positive where the pixel is clear of the circle)",
        orders={"expected": 0},
        route="none: a pixel the circle does not touch holds none of the disc or all of it, as the chain gives",
        unmeasured="the chain's sum is zero (or the disc's whole area) to rounding, so the pixel's share is the same "
                   "by either branch"),
    switch.Declaration(
        owner="camera.disc.holds",
        indicator="how far inside the pixel's quadrilateral the origin lies, in the cross products of its sides "
                  "(positive where the pixel holds the disc's centre)",
        orders={"expected": -1},
        route="none: a pixel clear of the circle that holds its centre holds the whole disc, and one that does not "
              "holds none of it; the choice is reached only where no side crosses the circle",
        unmeasured="reached only for a pixel clear of the circle (no side crossing it), which holds the origin "
                   "or not as the whole disc lies in it or outside it, never near a side: no point the harness can "
                   "place has the origin at a side and no crossing"),
)

#: The closed forms: the chords and edge functions along a ray.
FORM_SWITCHES = (
    switch.Declaration(
        owner="camera.ahead",
        indicator="the ray parameter of the line's nearest point to the piece, in metres (positive where it is ahead "
                  "of the pinhole)",
        orders={"expected": 0},
        route="none: a piece's value on the ray is its value on the line where the line's nearest point is ahead, "
              "and at the pinhole where it is behind: the two meet at the pinhole, with a different slope"),
    switch.Declaration(
        owner="camera.cap_side",
        indicator="the cap plane's height above the pinhole, in metres (positive where the pinhole is above the "
                  "plane: the side the cone faces)",
        orders={"expected": -1},
        route="none: a pinhole in a cap's plane sees the cap edge-on, and the cone's side flips there, so its edge "
              "function changes by twice its facing term"),
    switch.Declaration(
        owner="camera.span.low",
        indicator="the ray parameter of the farther plane of a cylinder's span less the nearer's, in metres "
                  "(positive where the first is the least)",
        orders={"expected": 0},
        route="none: the planes' order is the order of the parameters the ray meets them at; the pair is the same "
              "whichever is first, and the interval's ends meet where the parameters do"),
    switch.Declaration(
        owner="camera.span.high",
        indicator="the ray parameter of the first plane of a cylinder's span less the second's, in metres (positive "
                  "where the first is the greatest)",
        orders={"expected": 0},
        route="none: as camera.span.low"),
    switch.Declaration(
        owner="camera.span.axial",
        indicator="the axial coordinate of the ray's nearest approach to a cylinder's axis, in metres (positive "
                  "where it is above the middle)",
        orders={"expected": 0},
        route="none: the span piece's value is the distance beyond the half-length, a V at the middle plane, where a "
              "smooth piece would be a hyperbola"),
    switch.Declaration(
        owner="camera.interval.enter",
        indicator="the ray parameter where its line enters a cylinder's side, less where it enters its span, in "
                  "metres (positive where the side is the later)",
        orders={"expected": 0},
        route="none: an interval's start is the later of the side's and the span's"),
    switch.Declaration(
        owner="camera.interval.leave",
        indicator="the ray parameter where its line leaves a cylinder's span, less where it leaves its side, in metres "
                  "(positive where the side is the earlier)",
        orders={"expected": 0},
        route="none: an interval's end is the earlier of the side's and the span's"),
    switch.Declaration(
        owner="camera.interval.start",
        indicator="the interval's start along the ray, in metres (positive where it is ahead of the pinhole)",
        orders={"expected": 0},
        route="none: a ray starts at the pinhole, so nothing behind it is imaged"),
    switch.Declaration(
        owner="camera.interval.stop",
        indicator="the interval's end along the ray, in metres (positive where it is ahead of the pinhole)",
        orders={"expected": 0},
        route="none: as camera.interval.start"),
    switch.Declaration(
        owner="camera.interval.chord",
        indicator="the interval's length, in metres (positive where the ray meets the body)",
        orders={"expected": 0},
        route="none: the chord is the interval's length where it is not empty, and zero where it is"),
)

SWITCHES = (*EDGE_SWITCHES, *PIECES_SWITCHES, *CORNER_SWITCHES, *DISC_SWITCHES, *FORM_SWITCHES)

__all__ = ["CORNER_SWITCHES", "DISC_SWITCHES", "EDGE_SWITCHES", "FORM_SWITCHES", "PIECES_SWITCHES", "SWITCHES"]
