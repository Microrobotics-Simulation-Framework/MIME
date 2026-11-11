"""What the build reports of a flow's porous regions (``mime.interactions.Darcy``): each region's permeability against
the grid, where its Brinkman layer (about ``sqrt K`` thick) is under a cell, and where its Darcy flux is below what was
measured. Each line is a validity line the build report shows; none is a refusal."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: Where ``sqrt(K)`` is at least this many cells the plug's Brinkman profile is resolved (within 1.3 % of its
#: centreline from one cell, POROUS_MEASURED); under it the layer is sub-cell, and its bulk Darcy flux is what holds.
RESOLVED_CELLS = 1.0
#: The least ``sqrt(K)/h`` at which a plug's Darcy flux was measured (a plug closing a straight tube, POROUS_MEASURED):
#: resolved down to it, unmeasured below.
FLUX_MEASURED_DOWN_TO = 0.01
#: What was measured of a porous region (the porous plug's gates and probes, GitHub runners, float64), as the flow's
#: validity line states it.
POROUS_MEASURED = (
    "Measured on straight periodic tubes, steady Stokes, float64: a plug closing the tube, one diameter long in a period "
    "of four, gives the series of Darcy's resistance (with Brinkman's wall factor) and the open tube's within -1.4 to "
    "+1.6 % at D/h 16, -0.3 to +0.9 % at 32 and 0.0 to +0.6 % at 64 for sqrt(K)/h 0.01 to 1 with the 0.01h interior "
    "(-1.7 to +2.0 % and -0.3 to +1.4 % at 16 and 32 with 0.1h), its face pressures' drop Darcy's at the measured flux "
    "within 3.5 %: resolved down to sqrt(K)/h 0.01 (within 0.3 % at D/h 32, a K of about 1e-13 m^2 in a 1 mm vessel), "
    "unmeasured below, where at sqrt(K)/h 0.001 the solve stopped stagnated at every D/h; Brinkman's profile in a plug "
    "filling the tube within 1.3 % of its centreline at sqrt(K)/h 1 and 3 (D/h 16 to 64); a coaxial plug of half the "
    "lumen's radius the Brinkman core in a Stokes annulus within 2.1 % at sqrt(K)/h 1 and 3, its face against the free "
    "fluid a wall the band smears where its layer is sub-cell (at sqrt(K)/h 0.1, 16 % low at D/h 16 and 12 % at 32); a "
    "plug driven by a body force alone, no pressure drop across it, reads up to 5.2 % low at D/h 16, 2.7 % at 32 and "
    "1.4 % at 64, against Brinkman")


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], spacing: float) -> list[str]:
    """Each region's line where its Brinkman layer is under :data:`RESOLVED_CELLS`: its profile across the layer is
    not resolved, and its bulk Darcy flux is resolved down to :data:`FLUX_MEASURED_DOWN_TO` and unmeasured below."""
    lines: list[str] = []
    for law, region in zip(laws, regions, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        cells = root / spacing
        if cells >= RESOLVED_CELLS:
            continue
        held = (f"its bulk Darcy flux and pressure drop are resolved (measured down to sqrt(K)/h "
                f"{FLUX_MEASURED_DOWN_TO:g})" if cells >= FLUX_MEASURED_DOWN_TO else
                f"its Darcy flux is resolved down to sqrt(K)/h {FLUX_MEASURED_DOWN_TO:g} and unmeasured below, so "
                f"unmeasured here (a spacing of {root / FLUX_MEASURED_DOWN_TO:.3g} m or finer reaches the measured "
                f"range)")
        lines.append(f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                     f"{cells:.3g} cells: under a cell, so its profile across the layer is not resolved; {held} (no "
                     f"OpenMaterica record gives a permeability yet: it is declared)")
    return lines


__all__ = ["FLUX_MEASURED_DOWN_TO", "POROUS_MEASURED", "RESOLVED_CELLS", "porous_lines"]
