"""The refinement patch's two transfers as MADDENING mapping kinds (the patch phase; item 59's Q2: the one-pass route
calls the same functions inside the Krylov loop, these kinds serve the lagged edge route and a saved graph's rebuild).

- ``mime.patch_boundary``: one face set of a coarse MAC field read trilinearly at the patch's given values of the same
  component (``mime.nodes.flow.fvm._patch_transfers.patch_boundary``), anchored at the patch's origin (whole fine cells,
  held as a float, the target's geometry).
- ``mime.patch_read``: one face set of the patch's padded field read trilinearly at a body's points held in the patch's
  frame (``_workarounds.in_patch_frame``), anchored there, since an edge has one anchor.

Each is linear in its field for a fixed geometry, its transpose the same stencil's spread."""

from __future__ import annotations

from typing import Any

import numpy as np
from jax import Array
from maddening.core.coupling.mapping import register_mapping
from maddening.core.coupling.mapping_spec import MappingSpec

from mime.core import jnp
from mime.nodes.flow.fvm._grid import MACGrid
from mime.nodes.flow.fvm._nested import NestedMACGrid
from mime.nodes.flow.fvm._open import OpenMACGrid
from mime.nodes.flow.fvm._patch_transfers import boundary_stencils, read_stencils, sampled, spread

#: The kinds' registered names.
BOUNDARY_KIND = "mime.patch_boundary"
READ_KIND = "mime.patch_read"


def _coarse(nx: int, ny: int, nz: int, spacing: float, open_z: bool) -> MACGrid:
    return OpenMACGrid((nx, ny, nz), spacing) if open_z else MACGrid((nx, ny, nz), spacing)


def _coarse_shape(coarse: MACGrid, component: int) -> tuple[int, int, int]:
    return coarse.shape(component) if isinstance(coarse, OpenMACGrid) else coarse.cells


class PatchBoundary:
    """One component's coarse face set (``nx, ny, nz`` cells of ``spacing``, open along z or periodic, its corner at
    ``corner``) read at a ``cells`` patch's given values (fine cell ``fine``) of the same component."""

    kind = BOUNDARY_KIND
    #: A read of values (preCICE's term): the given values interpolate the coarse field.
    mode = "consistent"
    needs_geometry = True
    geometry_shape = (1, 3)

    def __init__(self, *, nx: int, ny: int, nz: int, spacing: float, open_z: bool, corner_x: float, corner_y: float,
                 corner_z: float, px: int, py: int, pz: int, fine: float, component: int) -> None:
        self.coarse = _coarse(nx, ny, nz, spacing, open_z)
        self.patch = NestedMACGrid((px, py, pz), fine)
        self.corner = (corner_x, corner_y, corner_z)
        self.component = component
        self.spec = MappingSpec(BOUNDARY_KIND, dict(nx=nx, ny=ny, nz=nz, spacing=spacing, open_z=open_z,
                                                    corner_x=corner_x, corner_y=corner_y, corner_z=corner_z, px=px,
                                                    py=py, pz=pz, fine=fine, component=component), {})

    @property
    def n_source(self) -> int:
        return int(np.prod(_coarse_shape(self.coarse, self.component)))

    @property
    def n_target(self) -> int:
        return self.patch.given_count(self.component)

    def params_pytree(self) -> dict[str, Array]:
        return {}

    def _stencil(self, geom: Array, dtype: Any) -> Any:
        shapes = (_coarse_shape(self.coarse, 0), _coarse_shape(self.coarse, 1), _coarse_shape(self.coarse, 2))
        origin = jnp.round(geom.reshape(3)).astype(jnp.int32)
        return boundary_stencils(self.patch, origin, self.coarse, shapes, jnp.asarray(self.corner, dtype=dtype),
                                 dtype)[self.component]

    def apply(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        del weights
        assert geom is not None, "the patch's boundary edge reads its origin"
        shape = _coarse_shape(self.coarse, self.component)
        return sampled(field.reshape(shape), self._stencil(geom, field.dtype))

    def apply_T(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        del weights
        assert geom is not None, "the patch's boundary edge reads its origin"
        shape = _coarse_shape(self.coarse, self.component)
        return spread(field, self._stencil(geom, field.dtype), shape).reshape(-1)


class PatchRead:
    """One component of a ``cells`` patch's padded field (fine cell ``fine``) read at ``count`` points in the patch's
    frame (m from its corner)."""

    kind = READ_KIND
    mode = "consistent"
    needs_geometry = True

    def __init__(self, *, px: int, py: int, pz: int, fine: float, component: int, count: int) -> None:
        self.patch = NestedMACGrid((px, py, pz), fine)
        self.component = component
        self.points = count
        self.geometry_shape = (count, 3)
        self.spec = MappingSpec(READ_KIND, dict(px=px, py=py, pz=pz, fine=fine, component=component, count=count), {})

    @property
    def n_source(self) -> int:
        return int(np.prod(self.patch.padded_shape(self.component)))

    @property
    def n_target(self) -> int:
        return self.points

    def params_pytree(self) -> dict[str, Array]:
        return {}

    def _stencil(self, geom: Array, dtype: Any) -> Any:
        return read_stencils(self.patch, jnp.zeros(3, jnp.int32), geom.astype(dtype))[self.component]

    def apply(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        del weights
        assert geom is not None, "the patch's read edge reads its points"
        return sampled(field.reshape(self.patch.padded_shape(self.component)), self._stencil(geom, field.dtype))

    def apply_T(self, field: Array, weights: Any = None, geom: Array | None = None) -> Array:
        del weights
        assert geom is not None, "the patch's read edge reads its points"
        return spread(field, self._stencil(geom, field.dtype), self.patch.padded_shape(self.component)).reshape(-1)


@register_mapping(BOUNDARY_KIND, arrays=(), hyperparameters={
    "nx": int, "ny": int, "nz": int, "spacing": float, "open_z": bool, "corner_x": float, "corner_y": float,
    "corner_z": float, "px": int, "py": int, "pz": int, "fine": float, "component": int}, needs_geometry=True)
def patch_boundary_kind(*, nx: int, ny: int, nz: int, spacing: float, open_z: bool, corner_x: float, corner_y: float,
                        corner_z: float, px: int, py: int, pz: int, fine: float, component: int) -> PatchBoundary:
    """The patch's boundary edge's mapping (the kind's factory, by which a saved graph rebuilds it)."""
    return PatchBoundary(nx=nx, ny=ny, nz=nz, spacing=spacing, open_z=open_z, corner_x=corner_x, corner_y=corner_y,
                         corner_z=corner_z, px=px, py=py, pz=pz, fine=fine, component=component)


@register_mapping(READ_KIND, arrays=(), hyperparameters={
    "px": int, "py": int, "pz": int, "fine": float, "component": int, "count": int}, needs_geometry=True)
def patch_read_kind(*, px: int, py: int, pz: int, fine: float, component: int, count: int) -> PatchRead:
    """The patch's read edge's mapping (the kind's factory)."""
    return PatchRead(px=px, py=py, pz=pz, fine=fine, component=component, count=count)


__all__ = ["BOUNDARY_KIND", "READ_KIND", "PatchBoundary", "PatchRead", "patch_boundary_kind", "patch_read_kind"]
