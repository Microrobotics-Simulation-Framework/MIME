"""The Cartesian MAC grid kind nested in another grid: a box whose every face is given (the patch phase; F0's C4, the
patch spike's ``patch.py``).

**Every face given.** The velocity on the box's six planes (the normal faces) and half a cell outside them (the
tangential components' ghosts) is given, read from the grid the box is nested in; the unknowns are the interior faces,
``u`` on the ``(nx - 1, ny, nz)`` faces normal to x inside the box, ``v`` and ``w`` alike. The operators of this kind act on
the interior with the given values zero (the homogeneous problem the solve iterates on); :meth:`NestedMACGrid.padded`
holds the interior and the given values together (each component's own axis all ``n + 1`` faces, the other two the
``n`` centres and a ghost each side), and :meth:`NestedMACGrid.laplacian_given` is the given values' share of the
interior's Laplacian, which a solve moves to its right-hand side.

**The projector.** The divergence's adjoint on the interior faces is the gradient with the pressure's normal derivative
zero on every plane (the given normal faces carry no unknown), so ``D G`` is the cells' Neumann Laplacian on all three
axes, diagonal in the orthonormal cosine basis (the DCT-II), whose constant mode is the pressure's free constant, set to
zero. ``P = I - G (D G)^-1 D`` is then exact and orthogonal under :meth:`inner`, and the operator ``c - L + Sigma`` on the
interior is self-adjoint: the projected CG applies unchanged. The fast solve runs in the grids' Poisson site
(``fvm_poisson``), a transform.

**Compatibility.** A divergence-free interior needs the given normal faces' net outward flux to be zero; values read
from another grid miss it by that grid's interpolation error. :meth:`NestedMACGrid.compatible` removes the net flux
uniformly over the boundary's normal faces (the patch spike's: 1e-5 to 1e-3 of the boundary's absolute flux, measured)
and returns what it removed over that absolute flux, which the patch reports. :meth:`NestedMACGrid.lifted` gives the
given values a divergence-free interior, ``-G (D G)^-1 D`` of them, which the Neumann solve makes without touching
them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, override

import numpy as np
from jax import Array
from jax.scipy import fft as _fft

from mime.core import jnp
from mime.core.solves import solving
from mime.nodes.flow.fvm._grid import POISSON_SITE, MACGrid, Velocity

#: The interior's place in each padded component: every axis without its first and last entry.
INTERIOR = (slice(1, -1), slice(1, -1), slice(1, -1))


def _axes(component: int, cells: tuple[int, int, int], spacing: float) -> list[np.ndarray]:
    """Each axis's coordinates (m, from the box's corner) of a padded component: its own axis the ``n + 1`` faces, the
    others the ``n`` centres with a ghost half a cell outside each end."""
    axes: list[np.ndarray] = []
    for axis, count in enumerate(cells):
        if axis == component:
            axes.append(np.arange(count + 1) * spacing)
        else:
            axes.append((np.arange(count + 2) - 0.5) * spacing)
    return axes


@dataclass(frozen=True)
class NestedMACGrid(MACGrid):
    """The Cartesian MAC grid kind on a box of ``cells`` cubic cells of side ``spacing`` (m), every face given (the
    module states the unknowns, the given values, the projector and the compatibility)."""

    kind: ClassVar[str] = "cartesian_mac_nested"
    version: ClassVar[str] = "0.1.0"
    #: Exact under the interior faces' inner product: the Neumann solve inverts ``D G`` to rounding.
    projector_exact: ClassVar[bool] = True
    periodic: ClassVar[tuple[bool, bool, bool]] = (False, False, False)

    # -- shapes and places ----------------------------------------------------------------------------------------
    def interior_shape(self, component: int) -> tuple[int, int, int]:
        shape = list(self.cells)
        shape[component] -= 1
        return (shape[0], shape[1], shape[2])

    def padded_shape(self, component: int) -> tuple[int, int, int]:
        shape = [count + 2 for count in self.cells]
        shape[component] -= 1
        return (shape[0], shape[1], shape[2])

    @override
    def zeros(self, dtype: Any) -> Velocity:
        return (jnp.zeros(self.interior_shape(0), dtype=dtype), jnp.zeros(self.interior_shape(1), dtype=dtype),
                jnp.zeros(self.interior_shape(2), dtype=dtype))

    def padded_zeros(self, dtype: Any) -> Velocity:
        return (jnp.zeros(self.padded_shape(0), dtype=dtype), jnp.zeros(self.padded_shape(1), dtype=dtype),
                jnp.zeros(self.padded_shape(2), dtype=dtype))

    def padded_points(self, component: int, dtype: Any) -> Array:
        """Where each entry of the ``component``-th padded array sits (m, from the box's corner), ``(*shape, 3)``."""
        axes = [jnp.asarray(a, dtype=dtype) for a in _axes(component, self.cells, float(self.spacing))]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    @override
    def face_points(self, component: int, dtype: Any) -> Array:
        """The interior faces' points (m, from the box's corner)."""
        return self.padded_points(component, dtype)[INTERIOR]

    def given_mask(self, component: int) -> np.ndarray:
        """Which entries of the ``component``-th padded array are given values the interior's operators read: the
        normal faces on the two planes across its own axis and the ghosts beyond the other four, each over the
        interior's range of the other two axes (an edge's or a corner's entry is read by nothing)."""
        shape = self.padded_shape(component)
        mask = np.zeros(shape, dtype=bool)
        for axis in range(3):
            for end in (0, shape[axis] - 1):
                index: list[Any] = [slice(1, -1)] * 3
                index[axis] = end
                mask[tuple(index)] = True
        return mask

    def given_points(self, component: int, dtype: Any) -> Array:
        """The given values' points (m, from the box's corner), ``(m, 3)``, in :meth:`given_mask`'s order."""
        return self.padded_points(component, dtype)[np.nonzero(self.given_mask(component))]

    def given_count(self, component: int) -> int:
        return int(np.count_nonzero(self.given_mask(component)))

    def given_values(self, padded: Velocity) -> tuple[Array, Array, Array]:
        """Each component's given values, flat in :meth:`given_mask`'s order."""
        return (padded[0][np.nonzero(self.given_mask(0))], padded[1][np.nonzero(self.given_mask(1))],
                padded[2][np.nonzero(self.given_mask(2))])

    def with_given(self, values: tuple[Array, Array, Array], interior: Velocity | None = None) -> Velocity:
        """Padded arrays holding ``values`` (each component's, flat in :meth:`given_mask`'s order) and ``interior``
        (zero where not given; an edge's entry zero)."""
        dtype = values[0].dtype
        padded: list[Array] = []
        for component in range(3):
            held = jnp.zeros(self.padded_shape(component), dtype=dtype)
            held = held.at[np.nonzero(self.given_mask(component))].set(values[component], unique_indices=True)
            if interior is not None:
                held = held.at[INTERIOR].set(interior[component])
            padded.append(held)
        return (padded[0], padded[1], padded[2])

    @staticmethod
    def interior_of(padded: Velocity) -> Velocity:
        return (padded[0][INTERIOR], padded[1][INTERIOR], padded[2][INTERIOR])

    def padded(self, interior: Velocity) -> Velocity:
        """The interior with every given value zero."""
        dtype = interior[0].dtype
        held = [jnp.zeros(self.padded_shape(c), dtype=dtype).at[INTERIOR].set(interior[c]) for c in range(3)]
        return (held[0], held[1], held[2])

    # -- the operators on the interior (given values zero) --------------------------------------------------------
    def divergence_padded(self, padded: Velocity) -> Array:
        """``D`` at the cells of a padded field (its normal given faces read; its ghosts not)."""
        u, v, w = padded
        h = self.spacing
        return ((u[1:, 1:-1, 1:-1] - u[:-1, 1:-1, 1:-1]) + (v[1:-1, 1:, 1:-1] - v[1:-1, :-1, 1:-1])
                + (w[1:-1, 1:-1, 1:] - w[1:-1, 1:-1, :-1])) / h

    @override
    def divergence(self, velocity: Velocity) -> Array:
        return self.divergence_padded(self.padded(velocity))

    @override
    def gradient(self, pressure: Array) -> Velocity:
        """``G p = -D^T p`` on the interior faces."""
        h = self.spacing
        return ((pressure[1:] - pressure[:-1]) / h, (pressure[:, 1:] - pressure[:, :-1]) / h,
                (pressure[:, :, 1:] - pressure[:, :, :-1]) / h)

    def _seven_point(self, padded: Array, sign: float) -> Array:
        return (padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1] + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
                + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2] + sign * 6.0 * padded[1:-1, 1:-1, 1:-1]
                ) / self.spacing ** 2

    def laplacian_padded(self, padded: Velocity) -> Velocity:
        """The seven-point Laplacian on the interior faces of a padded field, its given values read."""
        return (self._seven_point(padded[0], -1.0), self._seven_point(padded[1], -1.0),
                self._seven_point(padded[2], -1.0))

    @override
    def laplacian(self, velocity: Velocity) -> Velocity:
        return self.laplacian_padded(self.padded(velocity))

    def laplacian_given(self, values: tuple[Array, Array, Array], dtype: Any) -> Velocity:
        """The given values' share of the interior's Laplacian: what a solve with them moves to its right-hand side."""
        held = (jnp.asarray(values[0], dtype=dtype), jnp.asarray(values[1], dtype=dtype), jnp.asarray(values[2], dtype=dtype))
        return self.laplacian_padded(self.with_given(held))

    @override
    def laplacian_magnitude(self, velocity: Velocity) -> Velocity:
        padded = self.padded(velocity)
        return (self._seven_point(padded[0], 1.0), self._seven_point(padded[1], 1.0),
                self._seven_point(padded[2], 1.0))

    @override
    def convection(self, velocity: Velocity) -> Velocity:
        raise NotImplementedError("the nested grid kind carries no convection: the patch is Stokes-Brinkman")

    @override
    def inner(self, a: Velocity, b: Velocity) -> Array:
        """The cell volume times the sum over every interior face of each component's product."""
        return self.cell_volume * (jnp.sum(a[0] * b[0]) + jnp.sum(a[1] * b[1]) + jnp.sum(a[2] * b[2]))

    # -- the fast solve and the projector -------------------------------------------------------------------------
    def _symbol(self, dtype: Any) -> Array:
        """The eigenvalues of ``D G`` in the DCT-II's layout (the constant mode's set to one: its value is fixed apart)."""
        h2 = jnp.asarray(self.spacing, dtype=dtype) ** 2
        nx, ny, nz = self.cells
        s = [jnp.sin(jnp.pi * jnp.arange(n, dtype=dtype) / (2.0 * n)) ** 2 for n in (nx, ny, nz)]
        symbol = -(4.0 / h2) * (s[0][:, None, None] + s[1][None, :, None] + s[2][None, None, :])
        return symbol.at[0, 0, 0].set(1.0)

    @override
    def poisson(self, source: Array) -> Array:
        """The mean-free ``phi`` with ``D G phi = source - mean(source)``, by the DCT-II on three axes."""
        with solving(POISSON_SITE):
            modes = _fft.dctn(source, type=2, norm="ortho")
            modes = (modes / self._symbol(source.dtype)).at[0, 0, 0].set(0.0)
            return _fft.idctn(modes, type=2, norm="ortho").astype(source.dtype)

    @override
    def project(self, velocity: Velocity) -> Velocity:
        """``P u = u - G (D G)^-1 D u`` on the interior: orthogonal under :meth:`inner`."""
        correction = self.gradient(self.poisson(self.divergence(velocity)))
        return (velocity[0] - correction[0], velocity[1] - correction[1], velocity[2] - correction[2])

    @override
    def pressure_of(self, remainder: Velocity) -> Array:
        return self.poisson(self.divergence(remainder))

    # -- the given values ---------------------------------------------------------------------------------------
    def outward(self, component: int) -> np.ndarray:
        """Each given value's outward sign as a normal face (``-1`` on the low plane, ``+1`` on the high one), zero
        for a ghost, in :meth:`given_mask`'s order."""
        shape = self.padded_shape(component)
        sign = np.zeros(shape)
        low: list[Any] = [slice(1, -1)] * 3
        high: list[Any] = [slice(1, -1)] * 3
        low[component], high[component] = 0, shape[component] - 1
        sign[tuple(low)] = -1.0
        sign[tuple(high)] = 1.0
        return sign[np.nonzero(self.given_mask(component))]

    def compatible(self, values: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        """The given values with their net outward flux removed uniformly over the boundary's normal faces (linear)."""
        dtype = values[0].dtype
        signs = [jnp.asarray(self.outward(c), dtype=dtype) for c in range(3)]
        flux = sum((jnp.sum(signs[c] * values[c]) for c in range(3)), start=jnp.zeros((), dtype=dtype))
        faces = sum(float(np.count_nonzero(self.outward(c))) for c in range(3))
        speed = flux / faces
        return (values[0] - speed * signs[0], values[1] - speed * signs[1], values[2] - speed * signs[2])

    def flux_share(self, values: tuple[Array, Array, Array]) -> Array:
        """The given values' net outward flux over their absolute flux (zero for none): what :meth:`compatible`
        removes, on the scale of what crosses the boundary."""
        dtype = values[0].dtype
        signs = [jnp.asarray(self.outward(c), dtype=dtype) for c in range(3)]
        flux = sum((jnp.sum(signs[c] * values[c]) for c in range(3)), start=jnp.zeros((), dtype=dtype))
        size = sum((jnp.sum(jnp.abs(signs[c] * values[c])) for c in range(3)), start=jnp.zeros((), dtype=dtype))
        return jnp.where(size > 0.0, flux / jnp.where(size > 0.0, size, 1.0), 0.0)

    def lifted(self, values: tuple[Array, Array, Array]) -> Velocity:
        """The padded field holding ``values`` (compatible) and the divergence-free interior ``-G (D G)^-1 D`` of
        them: the Neumann solve leaves the given values as they are."""
        given = self.with_given(values)
        interior = self.gradient(self.poisson(self.divergence_padded(given)))
        return self.with_given(values, (-interior[0], -interior[1], -interior[2]))


__all__ = ["INTERIOR", "NestedMACGrid"]
