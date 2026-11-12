"""What the build and the run report of a flow's porous regions (``mime.interactions.Darcy``): each region's
permeability against the grid, where its Brinkman layer (about ``sqrt K`` thick) is under a cell, the route its
throughflow takes, and where it lies outside what was measured. None is a refusal.

**The hand-over** (the record's item 54, row #369). Above :data:`HANDOVER_LINE` (``sqrt(K)/h``, by the stiff interior)
the penalty resolves a region's throughflow two-way. Below it the solve holds the region at the line's permeability
(:func:`held_permeability`): a wall for the outer flow, to within the flux the held region passes, and never stiffer
than the solve was measured converging at. The throughflow is then read one way from the solved face pressures, Darcy's
law in Brinkman's form (:func:`darcy_reading`). The reading needs the solved drop to be the plug's, which holds where
the flow can go elsewhere; where a law fixes the flux through a plug closing the only path (an inflow or a closed valve
on a single tube), the solved drop is the held region's, and the run raises ``forced_flagged``
(:func:`forced_through`). The test is structural: a held region obeys its own Darcy relation in both cases, so no
comparison of its flux with the reading can tell them apart, and the comparison runs only where the vessel's topology
says the flux is fixed."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from jax import Array
from jax.scipy.special import i0e, i1e

from mime.core import jnp

#: Where ``sqrt(K)`` is at least this many cells the plug's Brinkman profile is resolved (within 1.3 % of its
#: centreline from one cell, POROUS_MEASURED); under it the layer is sub-cell, and its bulk Darcy flux is what holds.
RESOLVED_CELLS = 1.0
#: The hand-over's line, ``sqrt(K)/h`` by the stiff interior: the least at which a plug closing a straight tube was
#: resolved with the solve converged, and the wall's leak (the intercept of the throughflow against K) at most 1 % of
#: its throughflow (the line rule, the record's item 54): at 0.003 with 0.1h the leak is under 3.5e-10 / 3.8e-11 / 4e-12
#: of the open tube's flux at D/h 16 / 32 / 64 against a throughflow of 4.3e-6 / 1.1e-6 / 2.8e-7, and at 0.01 with
#: 0.01h under 7.3e-8 / 9.9e-9 against about 5e-5 / 1.3e-5 (the porous plug's intercept and lumen probes, runs
#: 43123280055 and 43159332258); below them the cold solve stagnated or reached its cap. Measured at D/h 16 to 64.
HANDOVER_LINE = {"0.1h": 0.003, "0.01h": 0.01}
#: The least ``sqrt(K)/h`` at which a region leaving a gap to the wall (a channel past a clot that does not fill the
#: vessel) was measured within the annulus gate's line: within 2.1 % at sqrt(K)/h 1 and 3; at 0.1 to 0.01 its flux
#: read 12 to 16 % low (a coaxial core of half the lumen's radius, D/h 16 and 32), its face against the free fluid a
#: wall the band smears.
GAP_MEASURED_DOWN_TO = 1.0
#: The state field a flow holding porous regions raises at each step a region lies outside what was measured.
POROUS_FLAG = "porous_flagged"
POROUS_FLAG_MEANING = (
    f"a porous region (Darcy) lay outside what was measured: it left a gap to the wall (it does not close the lumen, or "
    f"the vessel is a tube tree, where none was measured) with sqrt(K)/h under {GAP_MEASURED_DOWN_TO:g}, where a "
    f"coaxial core's gap flux read 12 to 16 % low at sqrt(K)/h 0.1 to 0.01; or it lay below the hand-over's line "
    f"(sqrt(K)/h {HANDOVER_LINE['0.1h']:g} with 0.1h, {HANDOVER_LINE['0.01h']:g} with 0.01h) in a box holding a "
    f"resolved body, whose displacement the forced-flow test does not see: the run's flux and pressure drop there are "
    f"unmeasured. A finer spacing raises sqrt(K)/h")
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
    "flow forced through a plug below the line: a law fixes the flux (an inflow, a closed valve) on every path through "
    "the lumen from its opening to the others, the plug closes that path, and its one-way reading fell short of the "
    "solve's flux through it by more than the reading's error: the pressure drop is the held plug's, not the plug's, "
    "so porous_flux and the drop are not the plug's. Give the flow somewhere else to go (an opening's law of pressure, "
    "a Windkessel), or resolve the plug: a spacing of sqrt(K) / line or finer")

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
    "-1.8 / -0.4 / -0.005 % at D/h 16 / 32 / 64: resolved down to sqrt(K)/h 0.003 with the 0.1h interior (a K of about "
    "1e-14 m^2 in a 1 mm vessel at D/h 32) and 0.01 with 0.01h, unmeasured below (the first step's cold solve stopped "
    "there at the cap or stagnated, flagged; a warm-started later step may converge); the floor, the flux the wall lets "
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
    "0.3 and 1 only; a tube tree holding a plug runs, unmeasured (flagged where sqrt(K)/h is under 1)")

#: What was measured of the hand-over (its gates and probe, GitHub runners, float64 unless named), as the flow's validity
#: line states it.
HANDOVER_MEASURED = (
    "Below the line the solve holds a region at the line's permeability, so the outer flow carries the flux the held "
    "region passes (about 32 K_line / R^2 of the open tube's for a plug a quarter of its period long: 4.3e-6 / 1.1e-6 "
    "at D/h 16 / 32 with 0.1h, 5e-5 / 1.3e-5 with 0.01h), and the reading's error is that region's feedback, about its "
    "flux over the outer flow's, beside the face pressures' own error. Where both routes hold (a plug closing the tube, "
    "a quarter of its period long, the hand-over's arm holding it at nine times its K), the reading against the "
    "resolved lumen_flux: +0.4 to -0.5 % at sqrt(K)/h 0.003 to 0.03 and -4.3 % at 0.1 (D/h 16), +0.03 to -0.4 % and "
    "-1.8 % (D/h 32), the held region's feedback 4 % and 1 % at 0.1; far below the line (sqrt(K)/h 1e-3 to 1e-4) the "
    "reading against the series closed form -1.5 / -1.2 % at D/h 16 and -0.4 / -0.3 % at 32 (0.1h / 0.01h), the same "
    "to 1e-6 at every K, not the carried region's share of the lumen (4 h / R); float32 reads within 2e-4 of float64 "
    "there, its solve converged at D/h 16 with 0.01h and stopped stagnated at 1.2e-3 to 2.6e-3 at the 0.1h line and near 5e-5 at "
    "D/h 32 with 0.01h, flagged; a body's displacement is not in the forced-flow test, and a thin gap whose conductance "
    "is under the held region's acts as forced, which only the gap rule flags; the cold steady solve at the 0.1h line "
    "takes 7,129 / 16,736 iterations at D/h 16 / 32 (over the default cap of 5,000, flagged), and on a box two "
    "diameters long it stops stagnated at 1,600 (float64 residual 0.82, its reading within 1.7 %; float32 residual 1.0, "
    "its reading 0.31 of the series), where IMEX Euler's warm start converges; the line was measured on periodic tubes, "
    "and in an open vessel between a given pressure and a resistance the solve at the 0.1h line stopped stagnated near "
    "3e-5 on 49 of 50 draws, and in two parallel tubes from one inlet plane (a node's rig) went not finite on 32 of 50, "
    "flagged, where the 0.01h interior's converged")


def leaves_a_gap(radius: Any, offset_squared: Any, lumen_radius: Any) -> Any:
    """Whether a region leaves a gap between itself and a straight lumen's wall: its radius under the lumen's, or its
    axis farther off the lumen's (``offset_squared``, the square of its centre's distance across it) than their
    difference. Squares, so the step takes no root."""
    return (radius < lumen_radius) | (offset_squared > (radius - lumen_radius) ** 2)


def line_permeability(spacing: Any, interior: str) -> Any:
    """The permeability at the hand-over's line, ``(HANDOVER_LINE[interior] h)^2`` (m^2)."""
    return (HANDOVER_LINE[interior] * spacing) ** 2


def handed_over(permeability: Any, spacing: Any, interior: str) -> Any:
    """Whether a region's permeability lies below the hand-over's line (:func:`line_permeability`)."""
    return permeability < line_permeability(spacing, interior)


def held_permeability(permeability: Any, spacing: Any, interior: str) -> Any:
    """The permeability the solve holds a region at: its own at or above the hand-over's line, the line's below it, so
    the solve is never stiffer than at the line, where it was measured converging. Below the line the solve's flux
    through the region is the held one's, and its own is the reading (:func:`darcy_reading`)."""
    return jnp.maximum(permeability, line_permeability(spacing, interior))


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


def branch_cut(lumen: np.ndarray, disc: np.ndarray, layer: int) -> np.ndarray:
    """The cells of the lumen's branch a region sits in, in the cell layer ``layer`` across the box's axis: the
    layer's connected parts of the lumen (``lumen``, cells, its faces' neighbours) that meet the region's ``disc``."""
    labels = _labelled(lumen[:, :, layer])
    met = np.unique(labels[disc & (labels > 0)])
    cut = np.zeros_like(lumen)
    cut[:, :, layer] = np.isin(labels, met) & (labels > 0)
    return cut


def isolated(lumen: np.ndarray, touches: Sequence[np.ndarray], fixing: Sequence[int], cut: np.ndarray) -> bool:
    """Whether every path through the lumen from some opening whose law fixes its flux (``fixing``, by index) to every
    other opening crosses ``cut``: the lumen's cells less the cut, joined through their faces, leave that opening's
    cells (``touches``, each opening's lumen cells on its patch) in no part with another's."""
    labels = _labelled(lumen & ~cut)
    reached = [{int(label) for label in np.unique(labels[touch & (labels > 0)])} for touch in touches]
    for k in fixing:
        others: set[int] = set[int]().union(*(reached[j] for j in range(len(touches)) if j != k))
        if not reached[k] & others:
            return True
    return False


def outside_measured(permeability: Any, gap: Any, spacing: Any, interior: str, bodies: bool = False) -> Any:
    """Whether a region lies outside what was measured: a gap (:func:`leaves_a_gap`) with ``sqrt(K)/h`` under
    :data:`GAP_MEASURED_DOWN_TO`, or below the hand-over's line in a box holding resolved bodies (``bodies``);
    compared as ``K`` against ``(line h)^2``. A plug closing the lumen below the line is the hand-over's, not outside."""
    gapped = gap & (permeability < (GAP_MEASURED_DOWN_TO * spacing) ** 2)
    return gapped | (handed_over(permeability, spacing, interior) & bodies) if bodies else gapped


def porous_flagged(permeabilities: Sequence[Array], gaps: Sequence[Array], spacing: Any, interior: str,
                   bodies: bool = False) -> Array:
    """The step's :data:`POROUS_FLAG`: any region outside what was measured (:func:`outside_measured`)."""
    return jnp.any(jnp.stack([outside_measured(k, gap, spacing, interior, bodies)
                              for k, gap in zip(permeabilities, gaps, strict=True)]))


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], gaps: Sequence[bool], spacing: float,
                 interior: str) -> list[str]:
    """Each region's line where its Brinkman layer is under :data:`RESOLVED_CELLS`: its profile across the layer is
    not resolved; its route (resolved two-way at or above the interior's :data:`HANDOVER_LINE`, read one way below it,
    with the line and its source); one leaving a gap (``gaps``) is outside what was measured under
    :data:`GAP_MEASURED_DOWN_TO`, which the run flags."""
    least = HANDOVER_LINE[interior]
    line = (f"the hand-over's line, sqrt(K)/h {least:g} with the {interior} interior (the least at which a plug "
            f"closing a straight tube was resolved, the wall's leak at most 1 % of its throughflow: the porous plug's "
            f"intercept and lumen probes, runs 43123280055 and 43159332258)")
    lines: list[str] = []
    for law, region, gap in zip(laws, regions, gaps, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        cells = root / spacing
        if cells >= RESOLVED_CELLS:
            continue
        below = not cells >= least
        route = (f"below {line}, the flow holds it as a wall at the line's permeability, "
                 f"{(least * spacing) ** 2:.3g} m^2, and reads its throughflow one way from its face pressures "
                 f"(porous_flux, porous_seepage), the run raising forced_flagged where a law forces flow through it"
                 if below else f"at or above {line}, its throughflow is resolved two-way")
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
            held = (f"closing the lumen, {route} (a spacing of {root / least:.3g} m or finer resolves it two-way)")
        lines.append(f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                     f"{cells:.3g} cells: under a cell, so its profile across the layer is not resolved; {held} (no "
                     f"OpenMaterica record gives a permeability yet: it is declared)")
    return lines


__all__ = ["DROP_RESOLVED", "FORCED_FLAG", "FORCED_FLAG_MEANING", "FORCED_LINE", "GAP_MEASURED_DOWN_TO",
           "HANDOVER_LINE", "HANDOVER_MEASURED", "LOOP_MEASURED", "POROUS_FLAG", "POROUS_FLAG_MEANING", "POROUS_MEASURED", "RESOLVED_CELLS",
           "branch_cut", "darcy_reading", "forced_through", "handed_over", "held_permeability", "isolated",
           "leaves_a_gap", "line_permeability", "outside_measured", "porous_flagged", "porous_lines", "section_area",
           "wall_factor"]
