"""What the build and the run report of a flow's porous regions (``mime.interactions.Darcy``): each region's
permeability against the grid, where its Brinkman layer (about ``sqrt K`` thick) is under a cell, the route its
throughflow takes, where it lies outside what was measured, and the vessel's topology the hand-over reads. The node
refuses what that cannot decide (``flow.porous-series``).

**The hand-over** (the record's item 54, row #369). Above :data:`HANDOVER_LINE` (``sqrt(K)/h``, by the box type, the
stiff interior and the D/h, :class:`Handover`) the penalty resolves a region's throughflow two-way. Below it the solve holds the region at the line's permeability
(:func:`held_permeability`): a wall for the outer flow, to within the flux the held region passes, and never stiffer
than the solve was measured converging at. The throughflow is then read one way from the solved face pressures, Darcy's
law in Brinkman's form (:func:`darcy_reading`). The reading needs the solved drop to be the plug's, which holds where
the flow can go elsewhere; where a law fixes the flux through plugs closing every path out of a part of the lumen (an
inflow, a valve, with no opening of a pressure law beside it), the solved drop is the held regions', and the run raises
``forced_flagged`` (:func:`forced_through`, :func:`forced_by`, every such region cut at once). The test is structural:
a held region obeys its own Darcy relation in both cases, so no comparison of its flux with the reading can tell them
apart, and the comparison runs only where the vessel's topology says the flux is fixed. A held region in one flow path
with another region passes the line's flux, which the other carries (:func:`series_pairs`, :func:`in_series`): the
node refuses it."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from jax import Array
from jax.scipy.special import i0e, i1e

from mime.core import jnp

#: Where ``sqrt(K)`` is at least this many cells the plug's Brinkman profile is resolved (within 1.3 % of its
#: centreline from one cell, POROUS_MEASURED); under it the layer is sub-cell, and its bulk Darcy flux is what holds.
RESOLVED_CELLS = 1.0
#: The hand-over's line, ``sqrt(K)/h``, by box type (a node kind's :attr:`HANDOVER_BOX`), stiff interior and D/h: rows
#: ``(D/h, line)``, each line the least of 0.003, 0.01, 0.03 and 0.1 at which the held cold solve converged on every draw
#: at the CG's default cap (5,000, which an experiment cannot raise) at that D/h and below it, down to
#: :data:`HANDOVER_FROM`, and the wall's leak (the intercept of the throughflow against K) was at most 1 % of the
#: throughflow (the line rule, the record's item 54). A box's D/h takes the row of the least D/h at or above it; outside
#: :data:`HANDOVER_FROM` to the last row's D/h a region below the line is flagged (``porous_flagged``). The solve's count
#: rises with D/h at a fixed line (the held term's stiffness against the Laplacian's least eigenvalue), so the line rises
#: with it. Measured with the release's CG as it stands (a change to its stagnation stop may move the counts), float64,
#: run 43247885625 (10 draws a row, 5 at D/h 64). **Periodic** (a plug closing a straight periodic tube, its period 2 to 4
#: diameters, its length 0.4 to 0.9): with 0.1h at 0.01 every draw converged at D/h 8 and 16 (at most 1,488 and 3,368
#: iterations), 0.003 on 5 of 10 and 0 of 10 (stagnated or at the cap); at 0.03 at D/h 32 (at most 2,287), 0.01 on none
#: (at the cap); at 0.1 at D/h 64 (at most 1,405), where 0.03 converged on 5 of 5 draws whose periods reached only 3.05
#: diameters, its count rising with the period to 4,632 [SUPPLEMENT]; with 0.01h the same lines (at most 1,379 / 3,031 /
#: 2,249 / 2,026), 0.003 on none at D/h 8 and 16, 0.01 on none at 32. **Open** (an open vessel two
#: diameters long: a fixed inflow, a given pressure, a coaxial plug leaving a gap, and at D/h 8 and 12 two parallel
#: tubes from one inlet plane): at 0.01 with either interior every step of 50 of 50 draws converged in each case at D/h
#: 8 to 16 (runs 43213789033, 43201390806, 43207166164), where 0.003 stagnated or went not finite (run 43201390806); at
#: D/h 32 at 0.03 every step of every draw in each single-tube case at both interiors (at most 2,040 iterations), 0.01 on
#: 2 / 5 of 10 fixed-inflow draws (0.1h / 0.01h). The leak at these lines: under 3.5e-10 / 3.8e-11 / 4e-12 of the open
#: tube's flux at D/h 16 / 32 / 64 in the periodic tube whatever the line (runs 43123280055, 43159332258), against a held
#: throughflow that grows as the line's square (5e-5 of it at 0.01, D/h 16); under 2.7e-3 / 1.4e-3 of the throughflow
#: at D/h 8 / 16 in the open vessel at 0.01 (run 43213789033), and at D/h 32 unmeasured.
HANDOVER_LINE: dict[str, dict[str, tuple[tuple[float, float], ...]]] = {
    "periodic": {"0.1h": ((16.0, 0.01), (32.0, 0.03), (64.0, 0.1)), "0.01h": ((16.0, 0.01), (32.0, 0.03), (64.0, 0.1))},
    "open": {"0.1h": ((16.0, 0.01), (32.0, 0.03)), "0.01h": ((16.0, 0.01), (32.0, 0.03))}}
#: The least D/h each box type's line was measured at.
HANDOVER_FROM = {"periodic": 8.0, "open": 8.0}
#: Where each box type's line was measured, as the build report names it.
HANDOVER_SOURCE = {"periodic": "the hand-over's line probe at the default cap, run 43247885625",
                   "open": "the hand-over's open-vessel probes, runs 43213789033, 43201390806, 43207166164 and "
                           "43247885625"}
#: The box types by name, as the texts state them.
BOX_NAMES = {"periodic": "a periodic box", "open": "an open vessel"}


@dataclass(frozen=True)
class Handover:
    """A box's hand-over at its build: its type, its stiff interior, its lumen's D/h, the line there (``sqrt(K)/h``,
    :data:`HANDOVER_LINE`) and whether that D/h lies outside the range the line was measured at."""

    box: str
    interior: str
    across: float
    line: float
    outside: bool

    def stated(self) -> str:
        rows = HANDOVER_LINE[self.box][self.interior]
        measured = f"measured at D/h {HANDOVER_FROM[self.box]:g} to {rows[-1][0]:g}"
        return (f"the hand-over's line, sqrt(K)/h {self.line:g} with the {self.interior} interior in "
                f"{BOX_NAMES[self.box]} at D/h {self.across:.3g} (the least at which the held solve converged on every "
                f"draw at the CG's default cap, the wall's leak at most 1 % of the throughflow, {measured}"
                f"{': outside it, flagged' if self.outside else ''}; {HANDOVER_SOURCE[self.box]})")


def handover_of(box: str, interior: str, across: float) -> Handover | None:
    """The hand-over at a box of type ``box`` whose lumen is ``across`` cells across (D/h): the line of the least D/h
    measured at or above it, the largest's above them all; none for a box type the line table does not list (a tube
    tree's, refused when planned: no line was measured there)."""
    rows = HANDOVER_LINE.get(box, {}).get(interior)
    if not rows:
        return None
    fits = 1.0 + 1e-6
    line = next((line for dh, line in rows if across <= dh * fits), rows[-1][1])
    outside = across * fits < HANDOVER_FROM[box] or across > rows[-1][0] * fits
    return Handover(box, interior, across, line, outside)


#: The least ``sqrt(K)/h`` at which a region leaving a gap to the wall (a channel past a clot that does not fill the
#: vessel) was measured within the annulus gate's line: within 2.1 % at sqrt(K)/h 1 and 3; at 0.1 to 0.01 its flux
#: read 12 to 16 % low (a coaxial core of half the lumen's radius, D/h 16 and 32), its face against the free fluid a
#: wall the band smears.
GAP_MEASURED_DOWN_TO = 1.0
#: The state field a flow holding porous regions raises at each step a region lies outside what was measured.
POROUS_FLAG = "porous_flagged"


def lines_stated() -> str:
    """The hand-over's lines as the texts state them, every box type's at both interiors by D/h."""
    def rows(held: tuple[tuple[float, float], ...]) -> str:
        return ", ".join(f"{line:g} to D/h {dh:g}" for dh, line in held)

    return "; ".join(f"{BOX_NAMES[box]} (from D/h {HANDOVER_FROM[box]:g}): {rows(lines['0.1h'])} with 0.1h, "
                     f"{rows(lines['0.01h'])} with 0.01h" for box, lines in HANDOVER_LINE.items())


POROUS_FLAG_MEANING = (
    f"a porous region (Darcy) lay outside what was measured: it left a gap to the wall (it does not close the lumen, or "
    f"the vessel is a tube tree, where none was measured) with sqrt(K)/h under {GAP_MEASURED_DOWN_TO:g}, where a "
    f"coaxial core's gap flux read 12 to 16 % low at sqrt(K)/h 0.1 to 0.01; or it lay below the hand-over's line "
    f"({lines_stated()}) in a box holding a resolved body, whose displacement the forced-flow test does not see, or at "
    f"a D/h outside the range that box type's line was measured at: the run's flux and pressure drop there are "
    f"unmeasured. A finer spacing raises sqrt(K)/h against the gap's line")
#: How far the one-way reading may fall short of the solve's flux through a plug a law forces flow through before
#: ``forced_flagged`` fires: the reading's own error at the held permeability (its face drop against Darcy's at the
#: inflow within 3.9 % in the open vessel at D/h 16, the porous plug's gate), with room. A forced plug within this share
#: of the line's permeability is not told from its reading, whose error there is under this share and that one.
FORCED_LINE = 0.1
#: A drop is compared where it is over this many solve tolerances of the face pressures' magnitude; under it the drop,
#: the reading and the solve's flux are all rounding (a closed valve's dead end, through which nothing flows).
DROP_RESOLVED = 100.0
#: The state field a flow holding porous regions raises at each step a law forces flow through a plug below the line.
FORCED_FLAG = "forced_flagged"
FORCED_FLAG_MEANING = (
    "flow forced through a plug below the line: a law fixes the flux (an inflow, a valve) out of a part of the lumen "
    "whose every other way out is through plugs below the line closing it (no opening of a pressure law in that part), "
    "and the plug's one-way reading fell short of the solve's flux through it by more than the reading's error: the "
    "pressure drop is the held plug's, not the plug's, so porous_flux, lumen_flux and the drop are not the plug's. "
    "Give the flow somewhere else to go (an opening's law of pressure, a Windkessel), or resolve the plug: a spacing "
    "at which its sqrt(K)/h is at or above the line there (the line rises with D/h)")

#: The loop past a closing plug in a box driven by a body force (the fix round's loop and lumen probes): the lumen's own
#: mean less the box's flux, over the open tube's flux, the same at every K.
LOOP_MEASURED = ("the lumen's own mean flux exceeds lumen_flux by 2.51e-3 / 3.70e-4 of the open tube's flux at D/h 16 / "
                 "32 with the 0.1h interior and 2.80e-4 / 6.2e-5 with 0.01h, the same to 0.1 % at sqrt(K)/h 0.03 to "
                 "0.003: some 6 / 3 times a plug's throughflow at sqrt(K)/h 0.03 with 0.1h, about half of it with 0.01h, "
                 "and more at a smaller K")
#: What was measured of a porous region (the porous plug's gates and probes, GitHub runners, float64), as the flow's
#: validity line states it.
POROUS_MEASURED = (
    "Measured on straight periodic tubes, steady Stokes, float64, aligned plugs: lumen_flux, which with a porous region "
    "is the box's whole axial flux (the flux through every section, so the region's throughflow), for a plug closing "
    "the tube, one diameter long in a period of four, gives the series of Darcy's resistance (with Brinkman's wall "
    "factor) and the open tube's within -1.4 to +1.6 % at D/h 16, -0.3 to +0.9 % at 32 and 0.0 to +0.6 % at 64 for "
    "sqrt(K)/h 0.01 to 1 with the 0.01h interior (-1.7 to +2.0 % and -0.3 to +1.4 % at 16 and 32 with 0.1h), its face "
    "pressures' drop Darcy's at the measured flux within 3.5 %; with the 0.1h interior at sqrt(K)/h 0.003 within "
    "-1.8 / -0.4 / -0.005 % at D/h 16 / 32 / 64. Those reached sqrt(K)/h 0.003 with the 0.1h interior and 0.01 with "
    "0.01h only with the CG's cap raised past its default (the cold solve took 7,129 / 16,736 iterations at 0.003 at "
    "D/h 16 / 32, and 6,703 at 0.01 with 0.01h at D/h 32): at the default cap of 5,000, which an experiment cannot "
    "raise, the two-way solve converged on every draw only down to the hand-over's line, which rises with D/h (0.01 at "
    "D/h 8 and 16, 0.03 at 32, 0.1 at 64, either interior), and below it the region is held and read one way; the "
    "floor, the flux the wall lets "
    "past a plug closing the tube whatever its K (the intercept of lumen_flux against K), under 3.5e-10 / 3.8e-11 / "
    "4e-12 of the open tube's at D/h 16 / 32 / 64 with 0.1h and under 7e-8 / 1e-8 / 1.4e-9 with 0.01h (bounds: no leak "
    "above them); a plug tilted 10 degrees bounded only near 2e-7 at D/h 16. With a body force driving the box, the "
    "field in the open part past a closing plug carries a loop the region does not cross, the lumen forward and the "
    "wall's band and solid back (the pressure that holds the drive back pushes the wall's porous solid, which the drive "
    f"does not reach, and the region closes the solid in its own span): {LOOP_MEASURED}; lumen_flux does not carry it, a "
    "reading of the velocity next to the plug does, and bypass_ratio near minus one shows it. Brinkman's profile in a plug "
    "filling the tube within 1.3 % of its centreline at sqrt(K)/h 1 and 3 (D/h 16 to 64); a coaxial plug of half the "
    "lumen's radius the Brinkman core in a Stokes annulus within 2.1 % at sqrt(K)/h 1 and 3, and 12 to 16 % low at 0.1 "
    "to 0.01 (D/h 16 and 32), flagged (porous_flagged); a plug driven by a body force alone, no pressure drop across "
    "it, reads up to 5.2 % low at D/h 16, 2.7 % at 32 and 1.4 % at 64, against Brinkman; float32 measured at sqrt(K)/h "
    "0.3 and 1 only; a tube tree holding a plug is refused (flow.porous-tree)")

#: What was measured of the hand-over (its gates and probe, GitHub runners, float64 unless named), as the flow's validity
#: line states it.
HANDOVER_MEASURED = (
    "Below the line the solve holds a region at the line's permeability, so the outer flow carries the flux the held "
    "region passes (about 32 K_line / R^2 of the open tube's for a plug a quarter of its period long: 5e-5 at the line "
    "0.01 at D/h 16, 1.1e-4 at 0.03 at 32, 3.1e-4 at 0.1 at 64), and the reading's error is that region's feedback, "
    "about R_rest / R_held: every resistance in series with the held region outside it (the open lumen's, and in an "
    "open vessel its outlets' laws) over the held region's, beside the face pressures' own error (in an open vessel "
    "with a distal resistance of a 1 mm artery's, 1e11 Pa s/m^3, beside a 0.5 mm plug held at the line at D/h 16, about "
    "6 % low [arithmetic], unflagged). lumen_flux counts a region below the line at its reading in place of the held "
    "flux through its mid-section, so it is the throughflow on either side of the line; the velocity field next to "
    "the region is the held solve's. Where both routes hold (a plug closing the tube, a quarter of its period long, the "
    "hand-over's arm holding it at nine times its K), the reading against the resolved lumen_flux: +0.4 to -0.5 % at "
    "sqrt(K)/h 0.003 to 0.03 and -4.3 % at 0.1 (D/h 16), +0.03 to -0.4 % and -1.8 % (D/h 32), the held region's "
    "feedback 4 % and 1 % at 0.1. Below the line at the default cap's lines (the line probe, run 43247885625: K 1e-4 "
    "to 0.5 of the line's, periods 2 to 4 diameters, plugs 0.4 to 0.9 long, the solve converged) the reading against "
    "the series closed form: -4.3 to -6.8 % / -3.9 to -5.3 % at D/h 8 (0.1h / 0.01h), -1.5 to -1.9 % / -1.3 to -1.6 % "
    "at 16, -0.4 to -0.5 % / -0.3 to -0.5 % at 32 and -0.1 to -0.2 % at 64; far below the line (sqrt(K)/h 1e-3 to "
    "1e-4) the same to 1e-6 at every K, not the carried region's share of the lumen (4 h / R); float32 read within 2e-4 "
    "of float64 at the earlier lines (0.003 with 0.1h, 0.01 with 0.01h), its steady cold solve converged at D/h 16 with "
    "0.01h and stopped stagnated at 1.2e-3 to 2.6e-3 at 0.003 and near 5e-5 at D/h 32 with 0.01h, flagged, and at the "
    "default cap's lines above D/h 16 it is unmeasured; a body's displacement is not in the forced-flow test, a thin "
    "gap whose conductance is under the held region's acts as forced, which only the gap rule flags, and a forced plug "
    f"within {FORCED_LINE:g} of the line's K reads up to {100 * FORCED_LINE:g} % low, unflagged; a solve whose step "
    "went not finite from a finite start, drive and resistivity in a flow holding a porous region is refused "
    "(flow.porous-solve-not-finite; a NaN given to the step is the run's own refusal, run.non-finite), and a region "
    "below the line closing the lumen in one flow path with another porous region is refused (flow.porous-series): two "
    "plugs in series, one held, read 88 and 0.88 times the series closed form (run 43241909199)")


def leaves_a_gap(radius: Any, offset_squared: Any, lumen_radius: Any) -> Any:
    """Whether a region leaves a gap between itself and a straight lumen's wall: its radius under the lumen's, or its
    axis farther off the lumen's (``offset_squared``, the square of its centre's distance across it) than their
    difference. Squares, so the step takes no root."""
    return (radius < lumen_radius) | (offset_squared > (radius - lumen_radius) ** 2)


def line_permeability(spacing: Any, line: float) -> Any:
    """The permeability at the hand-over's line ``line`` (``sqrt(K)/h``, :attr:`Handover.line`), ``(line h)^2``
    (m^2)."""
    return (line * spacing) ** 2


def handed_over(permeability: Any, spacing: Any, line: float | None) -> Any:
    """Whether a region's permeability lies below the hand-over's line (:func:`line_permeability`); never where the box
    has no line (``None``)."""
    return jnp.asarray(False) if line is None else permeability < line_permeability(spacing, line)


def held_permeability(permeability: Any, spacing: Any, line: float | None) -> Any:
    """The permeability the solve holds a region at: its own at or above the hand-over's line, the line's below it, so
    the solve is never stiffer than at the line, where it was measured converging. Below the line the solve's flux
    through the region is the held one's, and its own is the reading (:func:`darcy_reading`). Its own where the box
    has no line."""
    return permeability if line is None else jnp.maximum(permeability, line_permeability(spacing, line))


def wall_factor(permeability: Any, radius: Any) -> Any:
    """Brinkman's wall factor of a porous medium filling a tube of ``radius``, ``1 - 2 (sqrt K / a) I1(a / sqrt K) /
    I0(a / sqrt K)``: the share of Darcy's flux its no-slip wall leaves. One where ``K`` or ``a`` is not above zero
    (the region is flagged or holds no lumen)."""
    held = (permeability > 0.0) & (radius > 0.0)
    root = jnp.sqrt(jnp.where(held, permeability, 1.0))
    reach = jnp.where(held, radius, 1.0) / root
    return jnp.where(held, 1.0 - 2.0 * i1e(reach) / (reach * i0e(reach)), 1.0)


def section_area(radius: Any, offset_squared: Any, lumen_radius: Any) -> Any:
    """The area a region's disc of ``radius`` covers of a straight lumen's (``lumen_radius``), its axis
    ``sqrt(offset_squared)`` off the lumen's: the lesser disc where one holds the other, nothing where they are apart,
    their lens between."""
    r, big = radius, lumen_radius
    inside = offset_squared <= (big - r) ** 2
    apart = offset_squared >= (big + r) ** 2
    # The lens's formula taken at a separation where it is defined, wherever it is not the one read.
    squared = jnp.where(inside | apart, jnp.maximum(r, big) ** 2, offset_squared)
    d = jnp.sqrt(squared)
    own = r * r * jnp.arccos(jnp.clip((squared + r * r - big * big) / (2.0 * d * r), -1.0, 1.0))
    lumen = big * big * jnp.arccos(jnp.clip((squared + big * big - r * r) / (2.0 * d * big), -1.0, 1.0))
    kite = 0.5 * jnp.sqrt(jnp.maximum((r + big - d) * (d + r - big) * (d - r + big) * (d + r + big), 0.0))
    return jnp.where(inside, jnp.pi * jnp.minimum(r, big) ** 2, jnp.where(apart, 0.0, own + lumen - kite))


def darcy_reading(permeability: Any, area: Any, length: Any, drop: Any, viscosity: Any) -> Any:
    """A region's throughflow read one way from the drop across it (m^3/s, along its axis): Darcy's law in Brinkman's
    form, ``F_B K A dp / (mu L)``, ``F_B`` the wall factor (:func:`wall_factor`) of the disc of the region's section
    ``A``, and ``dp`` the drive across its length: its face pressures' difference and the body force on its own fluid
    between them. Differentiable in ``K`` and in the drop."""
    radius = jnp.sqrt(jnp.maximum(area, 0.0) / jnp.pi)
    return wall_factor(permeability, radius) * permeability * area * drop / (viscosity * length)


def forced_through(through: Any, handed: Any, flux: Any, reading: Any, drop: Any, scale: Any, tolerance: Any) -> Any:
    """Whether flow was forced through a region below the line: the vessel's topology says a law fixes the flux through
    it (``through``, ``OpenFlowNode.through_plug``), it lies below the line (``handed``), its drop is not provably
    unresolved (at most :data:`DROP_RESOLVED` solve tolerances of the pressures' magnitude ``scale``), and its reading
    is not provably within :data:`FORCED_LINE` of the solve's flux through its mid-section: a value not finite there
    is flagged, never read as agreement."""
    resolved = ~(jnp.abs(drop) <= DROP_RESOLVED * tolerance * scale)
    short = ~(jnp.abs(reading) >= (1.0 - FORCED_LINE) * jnp.abs(flux))
    return through & handed & resolved & short


def _labelled(mask: np.ndarray) -> np.ndarray:
    """``mask``'s connected parts, each its own label (zero outside), cells joined through their faces."""
    from scipy import ndimage

    return np.asarray(cast(Any, ndimage).label(mask)[0], dtype=np.int64)


def _bordering(cut: np.ndarray, labels: np.ndarray) -> set[int]:
    """The labels of the parts next to ``cut`` (through a face)."""
    from scipy import ndimage

    near = np.asarray(cast(Any, ndimage).binary_dilation(cut)) & ~cut & (labels > 0)
    return {int(label) for label in np.unique(labels[near])}


def _reached(labels: np.ndarray, touches: Sequence[np.ndarray]) -> list[set[int]]:
    """Per opening, the labels of the parts its cells (``touches``) lie in."""
    return [{int(label) for label in np.unique(labels[touch & (labels > 0)])} for touch in touches]


def branch_cut(lumen: np.ndarray, disc: np.ndarray, layer: int) -> np.ndarray:
    """The cells of the lumen's branch a region sits in, in the cell layer ``layer`` across the box's axis: the
    layer's connected parts of the lumen (``lumen``, cells, its faces' neighbours) that meet the region's ``disc``."""
    labels = _labelled(lumen[:, :, layer])
    met = np.unique(labels[disc & (labels > 0)])
    cut = np.zeros_like(lumen)
    cut[:, :, layer] = np.isin(labels, met) & (labels > 0)
    return cut


def forced_by(lumen: np.ndarray, touches: Sequence[np.ndarray], fixes: Sequence[bool],
              cuts: Sequence[np.ndarray]) -> list[bool]:
    """Per cut, whether a law fixes the flux through it: it borders a part of the lumen less every cut (``lumen``,
    cells joined through their faces) that holds an opening whose law fixes its flux (``fixes``, per opening: an inflow,
    a valve, open or closed) and none whose law states its pressure, so the flux through the cuts around that part
    together is the laws'. Every cut at once: two plugs in parallel branches below one fixed inflow are forced
    together."""
    union = np.logical_or.reduce(list(cuts)) if cuts else np.zeros_like(lumen)
    labels = _labelled(lumen & ~union)
    reached = _reached(labels, touches)
    fixed = set[int]().union(*(r for r, fixing in zip(reached, fixes, strict=True) if fixing))
    stated = set[int]().union(*(r for r, fixing in zip(reached, fixes, strict=True) if not fixing))
    forced = fixed - stated
    return [bool(_bordering(cut, labels) & forced) for cut in cuts]


def forced_table(lumen: np.ndarray, touches: Sequence[np.ndarray], fixes: Sequence[bool],
                 cuts: Sequence[np.ndarray]) -> np.ndarray:
    """:func:`forced_by` for every set of regions that may lie below the line closing the lumen, which the program
    decides: row ``sum 2^i`` over the set's regions ``i``, a column a region (``2^n`` labellings for ``n`` regions)."""
    count = len(cuts)
    table = np.zeros((2 ** count, count), dtype=bool)
    for index in range(1, 2 ** count):
        chosen = [k for k in range(count) if index >> k & 1]
        table[index, chosen] = forced_by(lumen, touches, fixes, [cuts[k] for k in chosen])
    return table


def series_pairs(lumen: np.ndarray, touches: Sequence[np.ndarray], cuts: Sequence[np.ndarray]) -> np.ndarray:
    """Which regions share a flow path, pairwise (``(n, n)``): two whose cuts leave a part of the lumen between them,
    next to both, that reaches no opening (``touches``), so every flux through one crosses the other."""
    count = len(cuts)
    pairs = np.zeros((count, count), dtype=bool)
    for i in range(count):
        for j in range(i + 1, count):
            labels = _labelled(lumen & ~(cuts[i] | cuts[j]))
            reached = set[int]().union(*_reached(labels, touches))
            pairs[i, j] = pairs[j, i] = bool((_bordering(cuts[i], labels) & _bordering(cuts[j], labels)) - reached)
    return pairs


def in_series(pairs: np.ndarray, closing: Sequence[Any]) -> Any:
    """Whether a region below the line closing the lumen (``closing``, per region, in the program) shares a flow path
    with another porous region (``pairs``, :func:`series_pairs`): the held region passes the line's flux, and every
    region in its path carries it."""
    partnered = [bool(np.any(np.delete(pairs[i], i))) for i in range(len(closing))]
    held = [jnp.asarray(c) for c, p in zip(closing, partnered, strict=True) if p]
    return jnp.any(jnp.stack(held)) if held else jnp.asarray(False)


def outside_measured(permeability: Any, gap: Any, spacing: Any, handover: Handover | None,
                     bodies: bool = False) -> Any:
    """Whether a region lies outside what was measured: a gap (:func:`leaves_a_gap`) with ``sqrt(K)/h`` under
    :data:`GAP_MEASURED_DOWN_TO`, or below the hand-over's line in a box holding resolved bodies (``bodies``) or at a
    D/h outside the line's measured range (``handover.outside``); compared as ``K`` against ``(line h)^2``. A plug
    closing the lumen below the line is the hand-over's, not outside."""
    gapped = gap & (permeability < (GAP_MEASURED_DOWN_TO * spacing) ** 2)
    if handover is None or not (bodies or handover.outside):
        return gapped
    return gapped | handed_over(permeability, spacing, handover.line)


def porous_flagged(permeabilities: Sequence[Array], gaps: Sequence[Array], spacing: Any, handover: Handover | None,
                   bodies: bool = False) -> Array:
    """The step's :data:`POROUS_FLAG`: any region outside what was measured (:func:`outside_measured`)."""
    return jnp.any(jnp.stack([outside_measured(k, gap, spacing, handover, bodies)
                              for k, gap in zip(permeabilities, gaps, strict=True)]))


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], gaps: Sequence[bool], spacing: float,
                 handover: Handover | None) -> list[str]:
    """Each region's line where its Brinkman layer is under :data:`RESOLVED_CELLS`: its profile across the layer is
    not resolved; its route (resolved two-way at or above the box's line, :class:`Handover`, read one way below it,
    with the line, its range and its source, and the measured D/h at which it would be resolved, if any); one leaving a
    gap (``gaps``) is outside what was measured under :data:`GAP_MEASURED_DOWN_TO`, which the run flags."""
    lines: list[str] = []
    for law, region, gap in zip(laws, regions, gaps, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        cells = root / spacing
        if cells >= RESOLVED_CELLS:
            continue
        if handover is None:
            route = ("in a box the hand-over's line table does not list (no line was measured there), its throughflow "
                     "resolved two-way at its own permeability")
            below = False
        else:
            below = not cells >= handover.line
            route = (f"below {handover.stated()}, the flow holds it as a wall at the line's permeability, "
                     f"{line_permeability(spacing, handover.line):.3g} m^2, and reads its throughflow one way from its "
                     f"face pressures (porous_flux, porous_seepage, and lumen_flux counting it at its reading), the run "
                     f"raising forced_flagged where a law forces flow through it and refusing it in series with another "
                     f"porous region (flow.porous-series)"
                     if below else f"at or above {handover.stated()}, its throughflow is resolved two-way")
        if gap:
            held = ("it leaves a gap to the wall (it does not close the lumen, or the vessel is a tube tree), where a "
                    "coaxial core's gap flux was measured within 2.1 % at sqrt(K)/h 1 and 3 and 12 to 16 % low at 0.1 "
                    f"to 0.01 (D/h 16 and 32): outside the measured line here, flagged at run time (porous_flagged); "
                    f"{route}")
        elif not below:
            held = (f"closing the lumen, {route}: its throughflow (lumen_flux, the box's whole flux, and porous_flux) "
                    f"and pressure drop measured down to the line on aligned plugs closing a straight tube; with a body "
                    f"force the field next to it carries the loop the validity line states")
        else:
            assert handover is not None
            diameter = handover.across * spacing
            reach = [f"{dh:g}" for dh, line in HANDOVER_LINE[handover.box][handover.interior]
                     if root * dh / diameter >= line]
            held = (f"closing the lumen, {route} (the line rises with D/h, so a coarser spacing lifts it toward the "
                    + (f"line: at D/h {', '.join(reach)} its sqrt(K)/h is at or above the line there)" if reach else
                       "line: at no D/h measured is its sqrt(K)/h at or above the line)"))
        lines.append(f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                     f"{cells:.3g} cells: under a cell, so its profile across the layer is not resolved; {held} (no "
                     f"OpenMaterica record gives a permeability yet: it is declared)")
    return lines


__all__ = ["BOX_NAMES", "DROP_RESOLVED", "FORCED_FLAG", "FORCED_FLAG_MEANING", "FORCED_LINE", "GAP_MEASURED_DOWN_TO",
           "HANDOVER_FROM", "HANDOVER_LINE", "HANDOVER_MEASURED", "HANDOVER_SOURCE", "Handover", "LOOP_MEASURED",
           "POROUS_FLAG", "POROUS_FLAG_MEANING", "POROUS_MEASURED", "RESOLVED_CELLS", "branch_cut", "darcy_reading",
           "forced_by", "forced_table", "forced_through", "handed_over", "handover_of", "held_permeability",
           "in_series", "leaves_a_gap", "line_permeability", "lines_stated", "outside_measured", "porous_flagged",
           "porous_lines", "section_area", "series_pairs", "wall_factor"]
