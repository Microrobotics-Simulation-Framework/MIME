"""What the FVM's build makes of a body a resolved drag registers with it (F4, ``Drag.Penalty``): the body's solid as
the flow's node lifts it (``mime.nodes.flow.fvm._resolved``), and the spacing it asks of the grid (B8.2: the finer of
the lumen's D/h 32 and N cells across the body's smallest feature, B12 (b))."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from mime.nodes.flow.fvm._band import Band
from mime.nodes.flow.fvm._resolved import RESOLVED_SHAPES, window_cells
from mime.nodes.flow.fvm._resolved_node import ResolvedSpec
from mime.registry import Hull
from mime.units import Magnitude

#: N, the cells across the radius of a resolved body's smallest feature by default (B12 (b)): measured on a sphere in
#: Stokes flow, 6.9 % under Haberman and Sayre at 3 cells a radius and 1.0 % at 6 (the coupling spike's R3).
FEATURE_CELLS = 6


def solid_region(hull: Hull) -> tuple[str, dict[str, Magnitude]] | None:
    """The hull's solid as one region with a resolvable shape (:data:`RESOLVED_SHAPES`): its shape's name and its
    parameters in the part's frame (SI); ``None`` for a kind whose solid is several regions or none a registered shape
    evaluates."""
    shapes = dict(hull.kind.region_shapes)
    if len(hull.kind.regions) != 1 or len(shapes) != 1:
        return None
    region, shape = next(iter(shapes.items()))
    if shape not in RESOLVED_SHAPES or region != hull.kind.regions[0]:
        return None
    return shape, dict(hull.kind.region_parameters(region, hull.settings))


def feature_radius(hull: Hull) -> float | None:
    """The radius of the hull's smallest feature (m): its solid's radius (a sphere's, a capsule's); ``None`` for a hull
    that is not resolvable."""
    region = solid_region(hull)
    if region is None:
        return None
    held = region[1]["radius"]
    return float(held) if isinstance(held, (int, float)) else None


def feature_cells(settings: Mapping[str, Magnitude]) -> float:
    """N as the registering drag wrote it (``feature_cells``), or :data:`FEATURE_CELLS`."""
    held = settings.get("feature_cells")
    return float(held) if isinstance(held, (int, float)) and not isinstance(held, bool) else float(FEATURE_CELLS)


def finest_spacing(hull: Hull | None, settings: Mapping[str, Magnitude]) -> float | None:
    """The spacing (m) at which the hull's smallest feature holds N cells across its radius, or ``None``."""
    radius = feature_radius(hull) if hull is not None else None
    return None if radius is None else radius / feature_cells(settings)


def body_reach(hull: Hull) -> float:
    """How far the hull's solid reaches from the body's centre (m), in any orientation: its part's offset from the
    centre plus the kind's enclosing radius about the part. The window about the body's centre is sized from it."""
    return float(hull.kind.enclosing_radius(hull.settings)) + math.dist(hull.centre, (0.0, 0.0, 0.0))


def haberman_sayre(ratio: float) -> float:
    """Haberman and Sayre's (1958) wall factor of a sphere translating along a tube's axis at a radius ratio ``ratio``
    (its drag over Stokes's), held at 0.9 past it."""
    r = min(ratio, 0.9)
    return (1 - 0.75857 * r ** 5) / (1 - 2.1050 * r + 2.0865 * r ** 3 - 1.7068 * r ** 5 + 0.72603 * r ** 6)


@dataclass(frozen=True)
class Relaxation:
    """A resolved body's relaxation times at the run's values (the lagged coupling's edge is measured against them):
    its translation's ``tau_t = m / (6 pi mu a K)``, ``K`` Haberman and Sayre's axial factor at ``a/R``, and its
    rotation's ``tau_r = I / (8 pi mu a^3)`` (``I`` the least principal moment), ``a`` its part's enclosing radius
    (exact for a sphere; for a capsule an enclosing sphere's, whose drag bounds the capsule's from above, so its times
    are under the capsule's); its density ratio ``m / (rho V)`` and its inertia ratio ``I / (0.4 rho V a^2)`` (a
    uniform sphere's equals its density ratio)."""

    tau_t: float
    tau_r: float
    density_ratio: float
    inertia_ratio: float
    radius: float
    factor: float


def relaxation(hull: Hull, mass: float, inertia: Sequence[float], viscosity: float, density: float,
               lumen_radius: float | None) -> Relaxation:
    """The body's :class:`Relaxation` (``inertia`` three principal moments or nine numbers, row-major)."""
    import numpy as np

    radius = float(hull.kind.enclosing_radius(hull.settings))
    volume = float(hull.kind.volume(hull.settings))
    held = np.asarray(inertia, dtype=np.float64)
    least = float(np.min(np.linalg.eigvalsh(np.diag(held) if held.size == 3 else held.reshape(3, 3))))
    factor = haberman_sayre(radius / lumen_radius) if lumen_radius else 1.0
    return Relaxation(mass / (6.0 * math.pi * viscosity * radius * factor),
                      least / (8.0 * math.pi * viscosity * radius ** 3), mass / (density * volume),
                      least / (0.4 * density * volume * radius ** 2), radius, factor)


#: The lagged coupling's measured edges (B1, PR 259's fix round, run 43137909114): by the mode (``translation``: x =
#: dt / tau_t; ``rotation``: x = dt / tau_r) and whether the scheme stores the fluid's inertia (IMEX Euler, convection
#: on or off alike) or not (its steady limit), for each N measured, the largest x at which no run grew (the stable end
#: of a bisection bracket 1.06 wide) at each ratio measured: the density ratio for the translation, the inertia ratio
#: ``I / (0.4 rho V a^2)`` for the rotation (a ratio of 0 a row the ratio does not move). Measured on R3's sphere (a/R
#: 0.375) at rest in a quiescent fluid, one mode perturbed, judged by its kinetic energy against its start's, float64,
#: both ends of each bracket again in float32 (every one agreeing); the steady limit's the same at ratios 0.3 and 8.
LAGGED_EDGES: Mapping[tuple[str, bool], Mapping[int, tuple[tuple[float, float], ...]]] = {
    ("translation", True): {3: ((0.3, 0.533), (0.6, 0.671), (1.0, 0.845), (1.4, 1.00), (3.0, 1.42), (8.0, 2.01),
                                (20.0, 2.38)),
                            6: ((0.3, 0.119), (0.6, 0.224), (1.0, 0.399), (1.4, 0.598), (3.0, 1.19), (8.0, 1.79),
                                (20.0, 2.25))},
    ("rotation", True): {3: ((0.3, 0.711), (1.0, 0.895), (8.0, 1.42)), 6: ((0.3, 0.475), (1.0, 0.753), (8.0, 1.42))},
    ("translation", False): {3: ((0.0, 2.53),), 6: ((0.0, 2.53),)},
    ("rotation", False): {3: ((0.0, 1.79),), 6: ((0.0, 1.79),)},
}
#: The share of the measured edge the refusal admits (the margin it stands under the edge by).
LAGGED_MARGIN = 0.8


def lagged_text(mode: str, inertial: bool) -> str:
    """The measured edges of ``mode`` as the validity line states them."""
    table = LAGGED_EDGES[(mode, inertial)]
    if all(len(rows) == 1 and rows[0][0] == 0.0 for rows in table.values()):
        return "the " + mode + "'s " + ", ".join(f"{rows[0][1]:g} at N = {n}" for n, rows in sorted(table.items()))
    return "; ".join(f"at N = {n} " + ", ".join(f"{x:g} at {r:g}" for r, x in rows) for n, rows in sorted(table.items()))


def lagged_edge(mode: str, inertial: bool, cells: float, ratio: float) -> float | None:
    """The measured edge of x for ``mode`` at N ``cells`` and ``ratio``; ``None`` below the least ratio measured
    (unmeasured: refused). At each measured N the lower of the measured ratios either side (the last's past it); for N
    between the measured ones the lower of their edges; past the finest the finest's, falling as (N_finest / N)^p, p
    the fall measured between the two finest N at that ratio, with the fluid's inertia stored at least 2 (a light
    body's edge is held at a viscous number nu dt / h^2, so its x falls as N^-2; in the steady limit the explicit
    drag's edge does not fall): an extrapolation, under the margin."""
    table = LAGGED_EDGES[(mode, inertial)]
    measured = sorted(table)

    def at(n: int) -> float | None:
        rows = table[n]
        if ratio < rows[0][0] - 1e-12:
            return None
        below = [x for r, x in rows if r <= ratio + 1e-12][-1:]
        above = [x for r, x in rows if r >= ratio - 1e-12][:1]
        return min(below + above)

    edges = {n: at(n) for n in measured}
    if any(edge is None for edge in edges.values()):
        return None
    held = cast(dict[int, float], edges)
    if cells <= measured[-1] + 1e-9:
        near = [n for n in measured if n <= cells + 1e-9][-1:] + [n for n in measured if n >= cells - 1e-9][:1]
        return min(held[n] for n in (near or measured[:1]))
    finest, least = measured[-1], 2.0 if inertial else 0.0
    fall = max(least, math.log(held[measured[-2]] / held[finest]) / math.log(finest / measured[-2])) \
        if len(measured) > 1 and held[finest] > 0.0 else least
    return held[finest] * (finest / cells) ** fall


def resolved_spec(hull: Hull, law: str, spacing: float, band: Band) -> ResolvedSpec:
    """The node's description of the hull's body, resolved by the drag ``law``, on a grid of ``spacing`` (its window
    sized for the band and the part's offset from the body's centre)."""
    region = solid_region(hull)
    assert region is not None, "the drag's plan refuses a hull that is not resolvable"
    shape, parameters = region
    rotation = tuple(float(value) for row in hull.rotation for value in row)
    return ResolvedSpec(hull.body, law, shape, {key: _number(value) for key, value in parameters.items()},
                        (float(hull.centre[0]), float(hull.centre[1]), float(hull.centre[2])), rotation,
                        window_cells(body_reach(hull), spacing, band),
                        float(hull.kind.volume(hull.settings)))


def _number(value: Magnitude) -> float | tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    return float(value)


__all__ = ["FEATURE_CELLS", "LAGGED_EDGES", "LAGGED_MARGIN", "Relaxation", "body_reach", "feature_cells",
           "feature_radius", "finest_spacing", "haberman_sayre", "lagged_edge", "lagged_text", "relaxation", "resolved_spec",
           "solid_region"]
