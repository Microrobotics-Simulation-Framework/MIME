"""What the FVM's build makes of a body a resolved drag registers with it (F4, ``Drag.Penalty``): the body's solid as
the flow's node lifts it (``mime.nodes.flow.fvm._resolved``), and the spacing it asks of the grid (B8.2: the finer of
the lumen's D/h 32 and N cells across the body's smallest feature, B12 (b))."""

from __future__ import annotations

from collections.abc import Mapping

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


def resolved_spec(hull: Hull, law: str, spacing: float, band: Band) -> ResolvedSpec:
    """The node's description of the hull's body, resolved by the drag ``law``, on a grid of ``spacing`` (its window
    sized for the band)."""
    region = solid_region(hull)
    assert region is not None, "the drag's plan refuses a hull that is not resolvable"
    shape, parameters = region
    rotation = tuple(float(value) for row in hull.rotation for value in row)
    return ResolvedSpec(hull.body, law, shape, {key: _number(value) for key, value in parameters.items()},
                        (float(hull.centre[0]), float(hull.centre[1]), float(hull.centre[2])), rotation,
                        window_cells(hull.kind.enclosing_radius(hull.settings), spacing, band),
                        float(hull.kind.volume(hull.settings)))


def _number(value: Magnitude) -> float | tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
    return float(value)  # type: ignore[arg-type]


__all__ = ["FEATURE_CELLS", "feature_cells", "feature_radius", "finest_spacing", "resolved_spec", "solid_region"]
