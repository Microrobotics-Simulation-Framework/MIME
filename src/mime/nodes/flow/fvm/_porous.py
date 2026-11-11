"""What the build reports of a flow's porous regions (``mime.interactions.Darcy``): each region's permeability against
the grid (where its Brinkman layer, about ``sqrt K`` thick, is under a cell), and, for a plug that closes a straight
tube's lumen, its own Darcy flux against the floor, the flux the wall's porous solid leaks around a plug that lets
nothing through, measured. Each line is a validity line the build report shows; none is a refusal."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: Where ``sqrt(K)`` is at least this many cells the plug's Brinkman profile is resolved (gate 2: within 2 % of the
#: centreline from 2 cells at D/h 16); under it the layer is sub-cell, and its bulk Darcy flux is what holds.
RESOLVED_CELLS = 1.0
#: The floor: the flux an impermeable plug (``sqrt(K)/h`` 1e-3) filling a straight periodic tube, one diameter long in a
#: period of four, lets through the wall's solid around it, over the open tube's, by the stiff interior's depth and the
#: lumen's diameter in cells. Measured on the parallel form's plug (FLOOR_MEASURED).
FLOOR: Mapping[str, tuple[tuple[float, float], ...]] = {
    "0.1h": ((16.0, 0.0), (32.0, 0.0), (64.0, 0.0)),
    "0.01h": ((16.0, 0.0), (32.0, 0.0), (64.0, 0.0)),
}
FLOOR_MEASURED = ""
#: What was measured of a porous region (the porous plug's gates and probes), as the flow's validity line states it.
POROUS_MEASURED = (
    "Measured on straight periodic tubes, steady Stokes, float64: a plug filling the tube, one diameter long in a "
    "period of four, gives the series of Darcy's resistance (with Brinkman's wall factor) and the open tube's within "
    "0.1 to 1.6 % at D/h 16 and 0.3 to 0.9 % at 32 for sqrt(K)/h 0.1 to 1 with the 0.01h interior, its face pressures "
    "the drop at the measured flux; Brinkman's profile in a plug filling the tube within 2 % of its centreline from "
    "sqrt(K)/h 2 at D/h 16; a coaxial plug of half the lumen's radius the Brinkman core in a Stokes annulus within 3 % "
    "at sqrt(K)/h 1 and 3 (D/h 16), its face against the free fluid a wall the band smears where its layer is "
    "sub-cell (at sqrt(K)/h 0.3, 10 % low); a plug driven by a body force alone, no pressure drop across it, reads up to "
    "3.8 % low at D/h 16 and 2.0 % at 32, against Brinkman")
#: The leak's share of a plug's own flux above which its line says the flux through it is not resolved to 1 %.
LEAK_SHARE_LINE = 1e-2


def floor_at(interior: str, cells: float) -> tuple[float, str]:
    """The measured floor at ``cells`` across the lumen, interpolated log-log between the measured diameters, and
    whether it was read beyond them (then the nearest measurement, with a word)."""
    held = FLOOR[interior]
    xs, ys = np.log([d for d, _ in held]), np.log([f for _, f in held])
    if cells < held[0][0] or cells > held[-1][0]:
        nearest = held[0] if cells < held[0][0] else held[-1]
        return nearest[1], f" (measured at D/h {held[0][0]:g} to {held[-1][0]:g}; this one beyond, read at the nearest)"
    return float(np.exp(np.interp(math.log(cells), xs, ys))), ""


def darcy_share(permeability: float, radius: float, length: float, period: float) -> float:
    """A plug's flux closing the tube over the open tube's, by Darcy's resistance in series with the open part's
    Poiseuille's: ``R_open(P) / (R_open(P - L) + mu L / (K pi R^2))`` (``mu`` cancels)."""
    open_whole = 8.0 * period / (math.pi * radius ** 4)
    open_part = 8.0 * max(period - length, 0.0) / (math.pi * radius ** 4)
    return open_whole / (open_part + length / (permeability * math.pi * radius ** 2))


def porous_lines(laws: Sequence[str], regions: Sequence[Mapping[str, Any]], spacing: float, interior: str,
                 tube: Mapping[str, Any] | None, period: float) -> list[str]:
    """Each region's lines: its permeability against the grid where its layer is under :data:`RESOLVED_CELLS`, and, for
    a plug closing a straight tube's lumen (``tube``: its ``centre``, ``axis`` and ``radius`` in the box's frame;
    ``None`` for a tube tree's), its Darcy flux against the measured floor at the run's resolution and stiff
    interior."""
    lines: list[str] = []
    for law, region in zip(laws, regions, strict=True):
        permeability = float(region["permeability"])
        root = math.sqrt(permeability) if permeability > 0.0 else float("nan")
        if not root / spacing >= RESOLVED_CELLS:
            lines.append(
                f"the porous region {law}: K = {permeability:.3g} m^2, its Brinkman layer sqrt(K) = {root:.3g} m, "
                f"{root / spacing:.3g} cells: under a cell, so its profile across the layer is not resolved; its bulk "
                f"Darcy flux and pressure drop hold (no OpenMaterica record gives a permeability yet: it is declared)")
        if tube is None:
            lines.append(f"the porous region {law}: its leak floor is measured on straight tubes only, so its flux "
                         f"against the wall's leak around it is not stated on a tube tree's wall")
            continue
        radius = float(tube["radius"])
        axis = np.asarray(tube["axis"], dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        offset = np.asarray(region["centre"], dtype=np.float64) - np.asarray(tube["centre"], dtype=np.float64)
        across = float(np.linalg.norm(offset - float(np.dot(offset, axis)) * axis))
        if float(region["radius"]) < radius + across - 0.5 * spacing:
            continue  # a gap is left: the flow passes the plug there, and the floor is no measure of it
        cells = 2.0 * radius / spacing
        floor, beyond = floor_at(interior, cells)
        share = darcy_share(permeability, radius, min(float(region["length"]), period), period)
        leak = floor / share * period / (4.0 * float(region["length"])) if share > 0.0 else float("inf")
        verdict = ("below the floor: the flux through it is the wall's leak, not Darcy's, and it reads as a wall"
                   if leak >= 1.0 else
                   f"the leak about {leak:.2g} of its flux, more than {LEAK_SHARE_LINE:g}: its flux is not resolved to "
                   f"1 %" if leak > LEAK_SHARE_LINE else f"the leak about {leak:.2g} of its flux")
        lines.append(f"the porous region {law} closes the lumen: its Darcy flux, about {share:.3g} of the open tube's "
                     f"(Darcy's resistance in series with the open part's), against the floor, the wall's leak around a "
                     f"plug that lets nothing through, {floor:.3g} of it at {cells:.3g} cells across with the "
                     f"{interior} interior{beyond} (measured, {FLOOR_MEASURED}): {verdict}")
    return lines


__all__ = ["FLOOR", "FLOOR_MEASURED", "LEAK_SHARE_LINE", "POROUS_MEASURED", "RESOLVED_CELLS", "darcy_share", "floor_at", "porous_lines"]
