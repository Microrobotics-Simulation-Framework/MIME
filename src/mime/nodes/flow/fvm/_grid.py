"""The grid kind's contract and its first kind, the Cartesian MAC grid on a box periodic on every axis.

**The contract** (F0's B2) is what the time scheme, the solve, the preconditioner and the ledger use of a grid, and
nothing more: the operators (divergence, gradient, Laplacian and its magnitude, convection), the weighted inner product,
a projector with its exactness declared, where each field component is sampled, and the indicator of a solid from a
signed distance (``_band``). A face-graph grid or an open axis fits it because the projector's exactness is declared,
never assumed. Gather and scatter weights (bodies) and port patches (openings) are its later entries.

**The Cartesian MAC kind.** Cubic cells of spacing ``h``; the velocity's three components on the faces normal to their
axis, ``u`` at ``(i h, (j + 1/2) h, (k + 1/2) h)``, ``v`` and ``w`` alike, the pressure at the cells' centres. Its
projector is exact: the fast transform inverts ``D G``, the cells' seven-point Laplacian, on a periodic box, so
``P = I - G (D G)^-1 D`` is the orthogonal projector onto discretely divergence-free fields under :meth:`inner`.
Convection is the divergence form averaged to the faces and edges, whose discrete kinetic energy flux vanishes on a
periodic box for a divergence-free field; it is central, so smooth for derivatives.

The spacing is held as a value, not a Python constant of the operators, so it may arrive traced; F1a holds it fixed
(a structural setting the box's cell counts were chosen with)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from jax import Array

from mime.core import jnp

#: The velocity: its components on the faces normal to x, y and z.
type Velocity = tuple[Array, Array, Array]


def _roll(a: Array, shift: int, axis: int) -> Array:
    return jnp.roll(a, shift, axis)


def _average_to_face(a: Array, axis: int) -> Array:
    """A cell-centred value along ``axis`` averaged onto the face below each cell."""
    return 0.5 * (_roll(a, 1, axis) + a)


def _difference_to_face(a: Array, axis: int, h: Any) -> Array:
    return (a - _roll(a, 1, axis)) / h


def _average_to_centre(a: Array, axis: int) -> Array:
    """A face value along ``axis`` averaged onto the cell centre above each face."""
    return 0.5 * (a + _roll(a, -1, axis))


def _difference_to_centre(a: Array, axis: int, h: Any) -> Array:
    return (_roll(a, -1, axis) - a) / h


@dataclass(frozen=True)
class MACGrid:
    """The Cartesian MAC grid kind on a box of ``cells`` cubic cells of side ``spacing`` (m), periodic on every axis."""

    cells: tuple[int, int, int]
    spacing: Any

    kind: ClassVar[str] = "cartesian_mac"
    version: ClassVar[str] = "0.1.0"
    #: The projector inverts ``D G`` exactly (to rounding), so the ledger books no projection residual.
    projector_exact: ClassVar[bool] = True
    periodic: ClassVar[tuple[bool, bool, bool]] = (True, True, True)

    @property
    def cell_count(self) -> int:
        return int(np.prod(self.cells))

    @property
    def cell_volume(self) -> Any:
        return self.spacing ** 3

    def face_points(self, component: int, dtype: Any) -> Array:
        """The points (m) the ``component``-th velocity component is sampled at, shape ``(*cells, 3)``: on the faces
        normal to that axis, at cell centres along the other two. The origin is the box's corner."""
        h = jnp.asarray(self.spacing, dtype=dtype)
        axes: list[Any] = []
        for axis, count in enumerate(self.cells):
            index = jnp.arange(count, dtype=dtype)
            axes.append(index * h if axis == component else (index + 0.5) * h)
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    def cell_points(self, dtype: Any) -> Array:
        h = jnp.asarray(self.spacing, dtype=dtype)
        axes = [(jnp.arange(count, dtype=dtype) + 0.5) * h for count in self.cells]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    def zeros(self, dtype: Any) -> Velocity:
        return (jnp.zeros(self.cells, dtype=dtype), jnp.zeros(self.cells, dtype=dtype),
                jnp.zeros(self.cells, dtype=dtype))

    def divergence(self, velocity: Velocity) -> Array:
        h = self.spacing
        return sum((_difference_to_centre(component, axis, h) for axis, component in enumerate(velocity)),
                   start=jnp.zeros(self.cells, dtype=velocity[0].dtype))

    def gradient(self, pressure: Array) -> Velocity:
        h = self.spacing
        return (_difference_to_face(pressure, 0, h), _difference_to_face(pressure, 1, h),
                _difference_to_face(pressure, 2, h))

    def laplacian(self, velocity: Velocity) -> Velocity:
        """The seven-point Laplacian of each component on its own face lattice."""
        h2 = self.spacing ** 2

        def of(a: Array) -> Array:
            return sum(((_roll(a, 1, axis) + _roll(a, -1, axis) - 2.0 * a) for axis in range(3)),
                       start=jnp.zeros_like(a)) / h2

        return (of(velocity[0]), of(velocity[1]), of(velocity[2]))

    def laplacian_magnitude(self, velocity: Velocity) -> Velocity:
        """:meth:`laplacian` with every coefficient in magnitude, ``(sum of the six neighbours + 6 a) / h^2``: the
        scale its rounding is judged on (a solve's rounding floor)."""
        h2 = self.spacing ** 2

        def of(a: Array) -> Array:
            return sum(((_roll(a, 1, axis) + _roll(a, -1, axis) + 2.0 * a) for axis in range(3)),
                       start=jnp.zeros_like(a)) / h2

        return (of(velocity[0]), of(velocity[1]), of(velocity[2]))

    def laplacian_diagonal(self) -> Any:
        """The diagonal of :meth:`laplacian`, the same on every face: ``-6 / h^2``."""
        return -6.0 / self.spacing ** 2

    def convection(self, velocity: Velocity) -> Velocity:
        """``-div(u u)`` on each face set, the right-hand side's convective term: normal fluxes at the cells' centres,
        tangential ones at the edges, each the product of averages (S1's form)."""
        u, v, w = velocity
        h = self.spacing
        centred = [_average_to_centre(component, axis) for axis, component in enumerate(velocity)]
        normal = [_difference_to_face(c * c, axis, h) for axis, c in enumerate(centred)]
        xy = _average_to_face(u, 1) * _average_to_face(v, 0)
        xz = _average_to_face(u, 2) * _average_to_face(w, 0)
        yz = _average_to_face(v, 2) * _average_to_face(w, 1)
        cu = normal[0] + _difference_to_centre(xy, 1, h) + _difference_to_centre(xz, 2, h)
        cv = normal[1] + _difference_to_centre(xy, 0, h) + _difference_to_centre(yz, 2, h)
        cw = normal[2] + _difference_to_centre(xz, 0, h) + _difference_to_centre(yz, 1, h)
        return (-cu, -cv, -cw)

    def inner(self, a: Velocity, b: Velocity) -> Array:
        """The weighted inner product every inner product goes through: the cell volume times the sum over every face
        of each component's product (every face of a periodic box carries a whole cell)."""
        return self.cell_volume * sum((jnp.sum(x * y) for x, y in zip(a, b, strict=True)),
                                      start=jnp.zeros((), dtype=a[0].dtype))

    def inner_magnitude(self, a: Velocity, b: Velocity) -> Array:
        """:meth:`inner` with every product in magnitude: the scale its rounding is judged on."""
        return self.inner((jnp.abs(a[0]), jnp.abs(a[1]), jnp.abs(a[2])), (jnp.abs(b[0]), jnp.abs(b[1]), jnp.abs(b[2])))

    def _poisson_symbol(self, dtype: Any) -> Array:
        """The eigenvalues of ``-D G`` in ``rfftn``'s layout, the mean's set to one (its value is fixed apart)."""
        h2 = jnp.asarray(self.spacing, dtype=dtype) ** 2
        nx, ny, nz = self.cells
        sx = jnp.sin(jnp.pi * jnp.arange(nx, dtype=dtype) / nx) ** 2
        sy = jnp.sin(jnp.pi * jnp.arange(ny, dtype=dtype) / ny) ** 2
        sz = jnp.sin(jnp.pi * jnp.arange(nz // 2 + 1, dtype=dtype) / nz) ** 2
        symbol = (4.0 / h2) * (sx[:, None, None] + sy[None, :, None] + sz[None, None, :])
        return symbol.at[0, 0, 0].set(1.0)

    def poisson(self, source: Array) -> Array:
        """The mean-free ``phi`` with ``D G phi = source - mean(source)``, by the fast transform."""
        symbol = self._poisson_symbol(source.dtype)
        transformed = jnp.fft.rfftn(source)
        solved = -transformed / symbol
        solved = solved.at[0, 0, 0].set(0.0)
        return jnp.fft.irfftn(solved, s=source.shape).astype(source.dtype)

    def project(self, velocity: Velocity) -> Velocity:
        """``P u = u - G (D G)^-1 D u``: the orthogonal projection onto discretely divergence-free fields."""
        phi = self.poisson(self.divergence(velocity))
        correction = self.gradient(phi)
        return (velocity[0] - correction[0], velocity[1] - correction[1], velocity[2] - correction[2])

    def pressure_of(self, remainder: Velocity) -> Array:
        """The pressure (kinematic, mean-free) whose gradient balances ``remainder``, the momentum equation's
        right-hand side less its operator applied to the solution: ``p = (D G)^-1 D (rhs - A u)``. The pressure is
        not a state of the monolithic step; it is recovered by this one fast solve when an output needs it."""
        return self.poisson(self.divergence(remainder))


__all__ = ["MACGrid", "Velocity"]
