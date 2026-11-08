"""The periodic box's layout: every choice the build makes from a value (the cell counts, the spacing, where the
vessel sits, which periodic images can reach a face) made in one function, :func:`periodic_tube_layout`, so that a later
declared range can make each choice from the range instead of the value, here and nowhere else.

A periodic box holds a straight tube whose period along its axis is declared (``length``): the spacing is the
largest that fits a whole number of cells along that period and is no coarser than the one asked for (D/h cells across
the lumen, 32 by default, the band's sweet-spot resolution). The box's other extents are not physical: they hold the
lumen, its band and a margin of solid between the tube and its images, an odd count of cells across (so the axis sits
at a cell's centre unless offset). A tilted tube's axis is a lattice direction ``(a, 0, b)`` of the box, so that the tube
wraps once per box; the box then holds one image of it, its neighbours a box's period away across x and y."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
        along = max(1, math.ceil(length / target - 1e-9))
        h = length / along
        across = _odd(math.ceil(2.0 * radius / h + reach_cells + margin_cells))
        cells = (across, across, along)
        centre = ((across / 2 + offset_cells[0]) * h, (across / 2 + offset_cells[1]) * h, 0.0)
        return TubeLayout(cells, h, centre, (0.0, 0.0, 1.0), radius, along * h, ((0.0, 0.0, 0.0),))
    norm = math.hypot(a, b)
    # The tube's images sit a box's x period apart; their distance across the axis is that period times b / |(a, b)|.
    along_x = _odd(math.ceil(clear * norm / b))
    h = target
    cells_z = along_x * b // a if (along_x * b) % a == 0 else None
    while cells_z is None:
        along_x += 2
        cells_z = along_x * b // a if (along_x * b) % a == 0 else None
    across_y = _odd(math.ceil(clear))
    cells = (along_x, across_y, cells_z)
    extents = (along_x * h, across_y * h, cells_z * h)
    centre = ((along_x / 2 + offset_cells[0]) * h, (across_y / 2 + offset_cells[1]) * h, extents[2] / 2)
    axis = (a / norm, 0.0, b / norm)
    axial_length = extents[0] / a * norm
    images = tuple((m * extents[0], n * extents[1], 0.0) for m in (-1, 0, 1) for n in (-1, 0, 1))
    return TubeLayout(cells, h, centre, axis, radius, axial_length, images)


__all__ = ["DEFAULT_CELLS_ACROSS", "DEFAULT_MARGIN", "DEFAULT_REACH", "TubeLayout", "periodic_tube_layout"]
