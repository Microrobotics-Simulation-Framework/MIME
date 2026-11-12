"""What the build and the run report of a flow's porous regions (``mime.interactions.Darcy``): each region's
permeability against the grid, where its Brinkman layer (about ``sqrt K`` thick) is under a cell, and where its flux
lies outside what was measured: below the least ``sqrt(K)/h`` measured, or leaving a gap to the wall below the gap's
measured line. The build's lines are validity lines; the run raises ``porous_flagged`` at each step a region lies
outside, with a warning. None is a refusal."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from jax import Array

from mime.core import jnp

#: Where ``sqrt(K)`` is at least this many cells the plug's Brinkman profile is resolved (within 1.3 % of its
#: centreline from one cell, POROUS_MEASURED); under it the layer is sub-cell, and its bulk Darcy flux is what holds.
RESOLVED_CELLS = 1.0
#: The least ``sqrt(K)/h`` at which a plug's Darcy flux was measured, by the stiff interior (a plug closing a straight
#: tube, POROUS_MEASURED): resolved down to it, unmeasured below, where the first step's cold solve stopped.
FLUX_MEASURED_DOWN_TO = {"0.1h": 0.003, "0.01h": 0.01}
#: The least ``sqrt(K)/h`` at which a region leaving a gap to the wall (a channel past a clot that does not fill the
#: vessel) was measured within the annulus gate's line: within 2.1 % at sqrt(K)/h 1 and 3; at 0.1 to 0.01 its flux
#: read 12 to 16 % low (a coaxial core of half the lumen's radius, D/h 16 and 32), its face against the free fluid a
#: wall the band smears.
GAP_MEASURED_DOWN_TO = 1.0
#: The state field a flow holding porous regions raises at each step a region lies outside what was measured.
POROUS_FLAG = "porous_flagged"
POROUS_FLAG_MEANING = (
    f"a porous region (Darcy) lay outside what was measured: its sqrt(K)/h under the least measured with the run's "
    f"interior ({FLUX_MEASURED_DOWN_TO['0.1h']:g} with 0.1h, {FLUX_MEASURED_DOWN_TO['0.01h']:g} with 0.01h), or it "
    f"left a gap to the wall (it does not close the lumen, or the vessel is a tube tree, where none was measured) with "
    f"sqrt(K)/h under {GAP_MEASURED_DOWN_TO:g}, where a coaxial core's gap flux read 12 to 16 % low at sqrt(K)/h 0.1 to "
    f"0.01: the run's flux and pressure drop there are unmeasured. A finer spacing raises sqrt(K)/h")
#: The loop past a closing plug in a box driven by a body force (the loop probe, run LOOP_RUN): (lumen - box) over the
#: open tube's flux, the same at every K.
LOOP_MEASURED = "LOOP_MEASURED"
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
    "reading of the velocity next to the plug does, and bypass_ratio (about -1) shows it. Brinkman's profile in a plug "
    "filling the tube within 1.3 % of its centreline at sqrt(K)/h 1 and 3 (D/h 16 to 64); a coaxial plug of half the "
    "lumen's radius the Brinkman core in a Stokes annulus within 2.1 % at sqrt(K)/h 1 and 3, and 12 to 16 % low at 0.1 "
    "to 0.01 (D/h 16 and 32), flagged (porous_flagged); a plug driven by a body force alone, no pressure drop across "
    "it, reads up to 5.2 % low at D/h 16, 2.7 % at 32 and 1.4 % at 64, against Brinkman; float32 measured at sqrt(K)/h "
    "0.3 and 1 only; a tube tree holding a plug runs, unmeasured (flagged where sqrt(K)/h is under 1)")


def leaves_a_gap(radius: Any, offset_squared: Any, lumen_radius: Any) -> Any:
    """Whether a region leaves a gap between itself and a straight lumen's wall: its radius under the lumen's, or its
    axis farther off the lumen's (``offset_squared``, the square of its centre's distance across it) than their
    difference. Squares, so the step takes no root."""
    return (radius < lumen_radius) | (offset_squared > (radius - lumen_radius) ** 2)


def outside_measured(permeability: Any, gap: Any, spacing: Any, interior: str) -> Any:
    """Whether a region lies outside what was measured: ``sqrt(K)/h`` under the interior's
    :data:`FLUX_MEASURED_DOWN_TO`, or a gap (:func:`leaves_a_gap`) with ``sqrt(K)/h`` under
    :data:`GAP_MEASURED_DOWN_TO`; compared as ``K`` against ``(line h)^2``."""
    least = FLUX_MEASURED_DOWN_TO[interior]
    return (permeability < (least * spacing) ** 2) | (gap & (permeability < (GAP_MEASURED_DOWN_TO * spacing) ** 2))


def porous_flagged(permeabilities: Sequence[Array], gaps: Sequence[Array], spacing: Any, interior: str) -> Array:
    """The step's :data:`POROUS_FLAG`: any region outside what was measured (:func:`outside_measured`)."""
    return jnp.any(jnp.stack([outside_measured(k, gap, spacing, interior)
                              for k, gap in zip(permeabilities, gaps, strict=True)]))


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], gaps: Sequence[bool], spacing: float,
                 interior: str) -> list[str]:
    """Each region's line where its Brinkman layer is under :data:`RESOLVED_CELLS`: its profile across the layer is
    not resolved; a region closing the lumen has its throughflow resolved down to the interior's
    :data:`FLUX_MEASURED_DOWN_TO` and unmeasured below; one leaving a gap (``gaps``) is outside what was measured under
    :data:`GAP_MEASURED_DOWN_TO`, which the run flags."""
    least = FLUX_MEASURED_DOWN_TO[interior]
    lines: list[str] = []
    for law, region, gap in zip(laws, regions, gaps, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        cells = root / spacing
        if cells >= RESOLVED_CELLS:
            continue
        if gap:
            held = ("it leaves a gap to the wall (it does not close the lumen, or the vessel is a tube tree), where a "
                    "coaxial core's gap flux was measured within 2.1 % at sqrt(K)/h 1 and 3 and 12 to 16 % low at 0.1 "
                    "to 0.01 (D/h 16 and 32): outside the measured line here, flagged at run time (porous_flagged)")
        elif cells >= least:
            held = (f"closing the lumen, its throughflow (lumen_flux, the box's whole flux) and pressure drop are "
                    f"resolved, measured down to sqrt(K)/h {least:g} with the {interior} interior on aligned plugs "
                    f"closing a straight tube; with a body force the field next to it carries the loop the validity "
                    f"line states")
        else:
            held = (f"closing the lumen, its throughflow is resolved down to sqrt(K)/h {least:g} with the {interior} "
                    f"interior and unmeasured below, so unmeasured here, flagged at run time (porous_flagged; a "
                    f"spacing of {root / least:.3g} m or finer reaches the measured range)")
        lines.append(f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                     f"{cells:.3g} cells: under a cell, so its profile across the layer is not resolved; {held} (no "
                     f"OpenMaterica record gives a permeability yet: it is declared)")
    return lines


__all__ = ["FLUX_MEASURED_DOWN_TO", "GAP_MEASURED_DOWN_TO", "LOOP_MEASURED", "POROUS_FLAG", "POROUS_FLAG_MEANING",
           "POROUS_MEASURED", "RESOLVED_CELLS", "leaves_a_gap", "outside_measured", "porous_flagged", "porous_lines"]
