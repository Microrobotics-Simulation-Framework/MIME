"""The periodic box's layout: every choice the build makes from a value (the cell counts, the spacing, where the
vessel sits, which periodic images can reach a face) made in one function, :func:`periodic_tube_layout`, so that a later
declared range can make each choice from the range instead of the value, here and nowhere else.

A periodic box holds a straight tube whose period along its axis is declared (``length``): the spacing is the
largest that fits a whole number of cells along that period and is no coarser than the one asked for (D/h cells across
the lumen, 32 by default, the band's sweet-spot resolution), to within the count's slack.

**The count's slack.** Every count is a whole number of cells taken up from a ratio of the inputs, and at round inputs
the ratio is itself a whole number (``2R/h`` is the cells asked for across; ``L/h`` is whole), so a bare ceiling there
would sit on its own boundary: a radius or a period moved up by a part in a million would add a cell, or refine the
spacing by one cell in the period. So every count is ``mime.scene._sampling.counted``'s, the rule the MRS's sample
counts take: the ceiling of the ratio less a relative ``COUNT_SLACK`` (1e-4), so a whole-number ratio keeps its count
and the next count starts a relative 1e-4 above it (a hundred times the two-builds walk's move). A central difference in
a value the layout is made from (the radius's, when the gradient gate takes it) takes its step inside one
discretisation: a relative step under the slack (5e-5, say) up from a round input, and both sides checked to lay out
the same box (:func:`same_discretisation`); elsewhere it takes the counts from a declared range (S-F0-2's). The box's other extents are not physical: they hold the
lumen, its band and a margin of solid between the tube and its images, an odd count of cells across (so the axis sits
at a cell's centre unless offset). A tilted tube's axis is a lattice direction ``(a, 0, b)`` of the box, so that the tube
wraps once per box; the box then holds one image of it, its neighbours a box's period away across x and y.

**A tube tree** (``mime.scene.TubeTree``) periodic along z is held the same way (:func:`periodic_tree_layout`): its
period along z is declared, the spacing the largest that fits a whole number of cells in it and no coarser than the
narrowest lumen's diameter over the cells asked for across it, and across x and y the box holds the tree's extent (its
balls' bounding box) with the same reach and margin. The tree reaches past the box's two ends along z, so each face
reads the tree itself and no image; the build checks that the tree repeats over the period where the fields read it
(:func:`period_mismatch`)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mime.scene._sampling import counted

#: The default resolution: cells across the narrowest lumen (B8.2 on a box with no openings; round 4, item 6).
DEFAULT_CELLS_ACROSS = 32
#: How far the wall's lift reaches into the solid, in cells (the band's outer half and the stiff interior's ramp for
#: the default band), and the cells of solid kept beside it, in all across the box (S7's tube box: D/h + 3 + 8).
DEFAULT_REACH = 3.0
DEFAULT_MARGIN = 8


@dataclass(frozen=True)
class TubeLayout:
    """A periodic box holding one straight tube: its cells, spacing (m), the tube's centre, axis and radius (SI), the
    axial length of the tube per box (m), and the box periods that can bring an image of the tube within a face."""

    cells: tuple[int, int, int]
    spacing: float
    centre: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    axial_length: float
    images: tuple[tuple[float, float, float], ...]

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.cells[0] * self.spacing, self.cells[1] * self.spacing, self.cells[2] * self.spacing)


def _odd(count: int) -> int:
    return count + (count + 1) % 2


def periodic_tube_layout(radius: float, length: float, *, cells_across: int = DEFAULT_CELLS_ACROSS,
                         spacing: float | None = None, reach_cells: float = DEFAULT_REACH,
                         margin_cells: int = DEFAULT_MARGIN,
                         lattice: tuple[int, int] = (0, 1), offset_cells: tuple[float, float] = (0.0, 0.0)
                         ) -> TubeLayout:
    """The box for a tube of ``radius`` (m) whose period along its axis is ``length`` (m), the axis along the lattice
    direction ``(a, 0, b)`` (``(0, 1)``: along z). ``spacing`` (m) is the finest the box may take; by default
    ``2 radius / cells_across``. ``offset_cells`` moves the axis off the cell centre across x and y, in cells (a
    sub-cell offset)."""
    if radius <= 0.0 or length <= 0.0:
        raise ValueError(f"a tube's radius and period are positive, not {radius} and {length}")
    a, b = lattice
    if b <= 0 or a < 0 or math.gcd(a, b) != 1:
        raise ValueError(f"a tilted tube's axis is a lattice direction (a, 0, b) with b > 0, a >= 0 and no common "
                         f"factor, not {lattice}")
    target = spacing if spacing is not None else 2.0 * radius / cells_across
    clear = 2.0 * radius / target + reach_cells + margin_cells
    if a == 0:
        along = max(1, counted(length / target))
        h = length / along
        across = _odd(counted(2.0 * radius / h + reach_cells + margin_cells))
        cells = (across, across, along)
        centre = ((across / 2 + offset_cells[0]) * h, (across / 2 + offset_cells[1]) * h, 0.0)
        return TubeLayout(cells, h, centre, (0.0, 0.0, 1.0), radius, along * h, ((0.0, 0.0, 0.0),))
    norm = math.hypot(a, b)
    # The tube's images sit a box's x period apart; their distance across the axis is that period times b / |(a, b)|.
    along_x = _odd(counted(clear * norm / b))
    h = target
    cells_z = along_x * b // a if (along_x * b) % a == 0 else None
    while cells_z is None:
        along_x += 2
        cells_z = along_x * b // a if (along_x * b) % a == 0 else None
    across_y = _odd(counted(clear))
    cells = (along_x, across_y, cells_z)
    extents = (along_x * h, across_y * h, cells_z * h)
    centre = ((along_x / 2 + offset_cells[0]) * h, (across_y / 2 + offset_cells[1]) * h, extents[2] / 2)
    axis = (a / norm, 0.0, b / norm)
    axial_length = extents[0] / a * norm
    images = tuple((m * extents[0], n * extents[1], 0.0) for m in (-1, 0, 1) for n in (-1, 0, 1))
    return TubeLayout(cells, h, centre, axis, radius, axial_length, images)


def same_discretisation(a: TubeLayout, b: TubeLayout) -> bool:
    """Whether two layouts are one discretisation (the same cells and the same spacing), as the two sides of a central
    difference in a value the layout is made from must be."""
    return a.cells == b.cells and a.spacing == b.spacing


@dataclass(frozen=True)
class TreeLayout:
    """A periodic box holding a tube tree periodic along z: its cells, spacing (m), the box's corner in the tree's
    frame (``origin``, m), and the tree's points in the box's frame (m)."""

    cells: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]
    points: tuple[tuple[float, float, float], ...]

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.cells[0] * self.spacing, self.cells[1] * self.spacing, self.cells[2] * self.spacing)


def periodic_tree_layout(points: Sequence[Sequence[float]], segments: Sequence[Sequence[int]],
                         radii: Sequence[Sequence[float]], length: float, *, cells_across: int = DEFAULT_CELLS_ACROSS,
                         spacing: float | None = None, reach_cells: float = DEFAULT_REACH,
                         margin_cells: int = DEFAULT_MARGIN, offset_cells: Sequence[float] = (0.0, 0.0, 0.0)
                         ) -> TreeLayout:
    """The box for a tube tree (``points`` in m, ``segments`` and their ``radii`` at each end, m) periodic along z with
    period ``length`` (m): ``spacing`` (m) the finest the box may take, by default the narrowest lumen's diameter over
    ``cells_across``. The tree's balls' bounding box is centred across x and y, its middle along z at the period's
    middle, each then moved by ``offset_cells`` (a sub-cell offset)."""
    held = np.asarray(points, dtype=np.float64)
    radius = np.asarray(radii, dtype=np.float64)
    if length <= 0.0 or held.ndim != 2 or held.shape[1] != 3:
        raise ValueError(f"a tree's period is positive and its points three numbers each, not {length} and "
                         f"{held.shape}")
    ends = np.concatenate([held[[a for a, _ in segments]] - radius[:, :1], held[[b for _, b in segments]] - radius[:, 1:]])
    tops = np.concatenate([held[[a for a, _ in segments]] + radius[:, :1], held[[b for _, b in segments]] + radius[:, 1:]])
    low, high = ends.min(axis=0), tops.max(axis=0)
    target = spacing if spacing is not None else 2.0 * float(radius.min()) / cells_across
    along = max(1, counted(length / target))
    h = length / along
    cells = tuple(_odd(counted(float(high[k] - low[k]) / h + reach_cells + margin_cells)) for k in range(2))
    middle = 0.5 * (low + high)
    origin = (float(middle[0] - (cells[0] / 2 + offset_cells[0]) * h),
              float(middle[1] - (cells[1] / 2 + offset_cells[1]) * h),
              float(middle[2] - (along / 2 + offset_cells[2]) * h))
    moved = held - np.asarray(origin)
    return TreeLayout((cells[0], cells[1], along), h, origin,
                      tuple((float(p[0]), float(p[1]), float(p[2])) for p in moved))


def period_mismatch(minimum: Any, layout: TreeLayout, reach: float) -> float:
    """How far the tree fails to repeat over its period where the fields read it: the largest difference, over the
    cell centres of the box's first layer along z within ``reach`` (m) of the wall, between the tree's distance there
    (``minimum``, a function of points ``(N, 3)`` in the box's frame) and one period on (m)."""
    h = layout.spacing
    nx, ny, nz = layout.cells
    x, y = np.meshgrid((np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h, indexing="ij")
    first = np.stack([x.ravel(), y.ravel(), np.full(x.size, 0.5 * h)], axis=-1)
    again = first + np.asarray([0.0, 0.0, nz * h])
    a, b = np.asarray(minimum(first), np.float64), np.asarray(minimum(again), np.float64)
    near = (np.abs(a) <= reach) | (np.abs(b) <= reach)
    return float(np.max(np.abs(a - b)[near])) if near.any() else 0.0


__all__ = ["DEFAULT_CELLS_ACROSS", "DEFAULT_MARGIN", "DEFAULT_REACH", "TreeLayout", "TubeLayout", "period_mismatch",
           "periodic_tree_layout", "periodic_tube_layout", "same_discretisation"]
