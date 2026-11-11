"""The exchange law's kernel as a MADDENING mapping kind on geometry edges (F3; the record's B8, body half point 2): a
gather (``direction="gather"``: one face set of a field on the grid, in the grid's own shape, read at a partner's
points) and a scatter (``"scatter"``: a value at each point, spread onto one face set as a force density), by the grid
kind's weights (``MACGrid.exchange_stencil``) and any of the law's modes (``mime.nodes.flow.fvm._exchange``): the
couple's (``"curl"``) reads and delivers a vector of three at each point. Each edge reads its points from the
partner's node (``geometry=(anchor, "points")``), in the grid's coordinates."""

from __future__ import annotations

from typing import Any

from jax import Array
from maddening.core.coupling.mapping import register_mapping
from maddening.core.coupling.mapping_spec import MappingSpec

from mime.nodes.flow.fvm._exchange import MODES, VECTOR_MODES, gather, scatter
from mime.nodes.flow.fvm._grid import MACGrid

#: The kind's registered name: a Gaussian on the Cartesian MAC grid kind's face sets.
KIND = "mime.gaussian_mac"
DIRECTIONS = ("gather", "scatter")


class GaussianMAC:
    """One exchange edge: a gather or a scatter of the ``component``-th face set of a ``cells`` grid of ``spacing``
    (m), a Gaussian ``width_cells`` cells wide, in ``mode``, for ``points`` points."""

    kind = KIND
    needs_geometry = True

    def __init__(self, *, nx: int, ny: int, nz: int, spacing: float, width_cells: float, component: int, mode: str,
                 direction: str, count: int) -> None:
        if mode not in MODES or direction not in DIRECTIONS or component not in (0, 1, 2):
            raise ValueError(f"an exchange edge's mode is one of {MODES}, its direction one of {DIRECTIONS} and its "
                             f"component 0, 1 or 2, not {mode!r}, {direction!r}, {component!r}")
        self.grid = MACGrid((nx, ny, nz), spacing)
        self.width_cells, self.component, self.mode, self.direction = width_cells, component, mode, direction
        self.points = count
        self.geometry_shape = (count, 3)
        self.spec = MappingSpec(KIND, dict(nx=nx, ny=ny, nz=nz, spacing=spacing, width_cells=width_cells,
                                           component=component, mode=mode, direction=direction, count=count), {})

    @property
    def n_source(self) -> int:
        return self.grid.cell_count if self.direction == "gather" else self.points

    @property
    def n_target(self) -> int:
        return self.points if self.direction == "gather" else self.grid.cell_count

    def field_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The grid's field in its own shape, the points' as one value each (three for the couple's mode)."""
        grid = tuple(self.grid.cells)
        points = (self.points, 3) if self.mode in VECTOR_MODES else (self.points,)
        return (grid, points) if self.direction == "gather" else (points, grid)

    def params_pytree(self) -> dict[str, Array]:
        return {}

    def apply(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        del weights
        assert geom is not None, "an exchange edge reads its points"
        stencil = self.grid.exchange_stencil(self.component, geom.astype(field.dtype), self.width_cells)
        if self.direction == "gather":
            return gather(self.grid, self.component, field, stencil, self.mode)
        return scatter(self.grid, self.component, field, stencil, self.mode)

    def apply_T(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        """The transpose: a gather's is the cell volume times the scatter, a scatter's the gather over it."""
        del weights
        assert geom is not None, "an exchange edge reads its points"
        stencil = self.grid.exchange_stencil(self.component, geom.astype(field.dtype), self.width_cells)
        if self.direction == "gather":
            return self.grid.cell_volume * scatter(self.grid, self.component, field, stencil, self.mode)
        return gather(self.grid, self.component, field, stencil, self.mode) / self.grid.cell_volume


@register_mapping(KIND, arrays=(), hyperparameters={"nx": int, "ny": int, "nz": int, "spacing": float,
                                                    "width_cells": float, "component": int, "mode": str,
                                                    "direction": str, "count": int}, needs_geometry=True)
def gaussian_mac(*, nx: int, ny: int, nz: int, spacing: float, width_cells: float, component: int, mode: str,
                 direction: str, count: int) -> GaussianMAC:
    """The exchange edge's mapping (the kind's factory, by which a saved graph rebuilds it)."""
    return GaussianMAC(nx=nx, ny=ny, nz=nz, spacing=spacing, width_cells=width_cells, component=component, mode=mode,
                       direction=direction, count=count)


__all__ = ["DIRECTIONS", "KIND", "GaussianMAC", "gaussian_mac"]
