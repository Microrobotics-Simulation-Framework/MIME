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
#: The least ``sqrt(K)/h`` at which a plug's Darcy flux was measured, by the stiff interior (a plug closing a straight
#: tube, POROUS_MEASURED): resolved down to it, unmeasured below, where the solve stopped stagnated.
FLUX_MEASURED_DOWN_TO = {"0.1h": 0.003, "0.01h": 0.01}
#: What was measured of a porous region (the porous plug's gates and probes, GitHub runners, float64), as the flow's
#: validity line states it.
POROUS_MEASURED = (
    "Measured on straight periodic tubes, steady Stokes, float64: a plug closing the tube, one diameter long in a period "
    "of four, gives the series of Darcy's resistance (with Brinkman's wall factor) and the open tube's within -1.4 to "
    "+1.6 % at D/h 16, -0.3 to +0.9 % at 32 and 0.0 to +0.6 % at 64 for sqrt(K)/h 0.01 to 1 with the 0.01h interior "
    "(-1.7 to +2.0 % and -0.3 to +1.4 % at 16 and 32 with 0.1h), its face pressures' drop Darcy's at the measured flux "
    "within 3.5 %; with the 0.1h interior at sqrt(K)/h 0.003 within -1.8 / -0.4 / -0.005 % at D/h 16 / 32 / 64: resolved "
    "down to sqrt(K)/h 0.003 with the 0.1h interior (a K of about 1e-14 m^2 in a 1 mm vessel at D/h 32) and 0.01 with "
    "0.01h, unmeasured below, where the solve stopped stagnated (0.003 with 0.01h and 0.001 with either, at every D/h); "
    "the floor, the flux the wall lets past a plug closing the tube whatever its K (the intercept of the flux against K), "
    "under 3.5e-10 / 3.8e-11 / 4e-12 of the open tube's at D/h 16 / 32 / 64 with 0.1h and under 7e-8 / 1e-8 / 1.4e-9 "
    "with 0.01h (bounds: no leak above them); Brinkman's profile in a plug filling the tube within 1.3 % of its "
    "centreline at sqrt(K)/h 1 and 3 (D/h 16 to 64); a coaxial plug of half the lumen's radius the Brinkman core in a "
    "Stokes annulus within 2.1 % at sqrt(K)/h 1 and 3, its face against the free fluid a wall the band smears where its "
    "layer is sub-cell (at sqrt(K)/h 0.1, 16 % low at D/h 16 and 12 % at 32); a plug driven by a body force alone, no "
    "pressure drop across it, reads up to 5.2 % low at D/h 16, 2.7 % at 32 and 1.4 % at 64, against Brinkman")


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], spacing: float, interior: str) -> list[str]:
    """Each region's line where its Brinkman layer is under :data:`RESOLVED_CELLS`: its profile across the layer is
    not resolved, and its bulk Darcy flux is resolved down to the interior's :data:`FLUX_MEASURED_DOWN_TO` and
    unmeasured below."""
    least = FLUX_MEASURED_DOWN_TO[interior]
    lines: list[str] = []
    for law, region in zip(laws, regions, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        cells = root / spacing
        if cells >= RESOLVED_CELLS:
            continue
        held = (f"its bulk Darcy flux and pressure drop are resolved (measured down to sqrt(K)/h {least:g} with the "
                f"{interior} interior)" if cells >= least else
                f"its Darcy flux is resolved down to sqrt(K)/h {least:g} with the {interior} interior and unmeasured "
                f"below, so unmeasured here (a spacing of {root / least:.3g} m or finer reaches the measured range)")
        lines.append(f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                     f"{cells:.3g} cells: under a cell, so its profile across the layer is not resolved; {held} (no "
                     f"OpenMaterica record gives a permeability yet: it is declared)")
    return lines


__all__ = ["FLUX_MEASURED_DOWN_TO", "POROUS_MEASURED", "RESOLVED_CELLS", "porous_lines"]
