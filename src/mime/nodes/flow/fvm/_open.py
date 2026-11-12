"""The Cartesian MAC grid kind open along its z axis: a box periodic across x and y whose two z faces are openings
(F0's B2 and B6/B7, decision 7; the open-ports spike's open axis).

**The port faces are evolved half cells.** ``w`` has ``nz + 1`` faces along z, the first and last on the box's two
z planes; each of those carries half a cell, so the weighted inner product counts it at half weight (``V = 1/2``).
Under that product the gradient on the end faces is the half cell's, ``2 (p_0 - p_b) / h``, the negative adjoint of
the divergence (``G = -V^-1 D^T``), so ``P = I - G (D G)^-1 D`` is the orthogonal projector onto discretely
divergence-free fields and the operator ``c - nu L + Sigma`` is self-adjoint: the projected CG applies unchanged in form
(the spike: the open tube matched the periodic one to 9e-14 and 4e-14 at D/h 16 and 32). An opening's pressure ``p_b``
enters the momentum as the force ``+-2 p_b / (rho h)`` on its half cell (:meth:`OpenMACGrid.port_force`); the projector
itself holds ``p_b = 0`` on both planes.

**The open planes' other conditions:** the normal velocity is evolved; the tangential velocity and the normal one have
zero normal gradient at the plane (the viscous flux through an opening is zero); convection's flux through the plane
is its own face value (``w_b^2`` on the normal half cell, the tangential velocity extrapolated to the plane on the
edges), so the kinetic energy convection carries through the openings is the scheme's own convective power
``<C(u), u>``, which the ledger books as such (B9, decision 14).

**The fast solve.** ``D G`` with homogeneous Dirichlet planes along z and periodic across: a real FFT across x and y
and, along z, the cell-centred Dirichlet Laplacian's own orthonormal sine basis (``sin(pi (m + 1) (j + 1/2) / nz)``),
applied as a dense ``nz x nz`` product at the highest matmul precision. Its symbol has no zero, so no mean is fixed.

**The openings** (:class:`OpeningLayout`): each opening is a patch of one plane, the low or the high; where a plane holds
one opening it is the whole plane, and where it holds several each of the plane's cells belongs to the opening whose
centre is nearest (in cells; the lower index on a tie), so the solid's faces on a plane belong to its openings too and
their Darcy flux is in their fluxes. Its sums (the fluxes, the openings' forces, the kinetic energy carried through each)
are over its patch.

**The kinetic energy convection carries through each opening** (:meth:`OpenMACGrid.opening_kinetic_fluxes`): on a
discretely divergence-free field the scheme's convective power telescopes to its planes, ``<C(u), u> = -sum_k K_k``,
``K_k = h^2 sum_patch (1/2)(avg_x u^2 + avg_y v^2 + w_b^2) w_b n_k`` (the tangential velocity's square from the end
cell averaged onto the cell's centre, ``w_b`` the plane's face, ``n_k`` the outward sign): the divergence form is
conservative given the discrete continuity (Morinishi et al. 1998), so the remainder is the field's divergence, rounding
after the projector."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.core.solves import solving
from mime.nodes.flow.fvm._grid import POISSON_SITE, TRUNCATION, MACGrid, Velocity

#: The weight the ends' half cells carry in the inner product.
HALF = 0.5
_HIGHEST: Any = jax.lax.Precision.HIGHEST


def _sine_basis(count: int) -> np.ndarray:
    """The orthonormal eigenvectors of the cell-centred Laplacian on ``count`` cells with homogeneous Dirichlet planes
    (columns): ``sqrt(2 / n) sin(pi (m + 1) (j + 1/2) / n)``, the last ``sqrt(1 / n) (-1)^j``."""
    j = np.arange(count)[:, None] + 0.5
    m = np.arange(count)[None, :] + 1.0
    basis = np.sqrt(2.0 / count) * np.sin(np.pi * m * j / count)
    basis[:, -1] = np.sqrt(1.0 / count) * np.cos(np.pi * np.arange(count))
    return basis


def _end_weight(shape: tuple[int, ...], dtype: Any) -> Array:
    """``V`` on the z faces: one on every face but the two planes', a half there."""
    weight = np.ones(shape[2])
    weight[0] = weight[-1] = HALF
    return jnp.asarray(weight, dtype=dtype)[None, None, :]


def _average_to_face(a: Array, axis: int) -> Array:
    return 0.5 * (jnp.roll(a, 1, axis) + a)


def _difference_to_face(a: Array, axis: int, h: Any) -> Array:
    return (a - jnp.roll(a, 1, axis)) / h


def _average_to_centre(a: Array, axis: int) -> Array:
    return 0.5 * (a + jnp.roll(a, -1, axis))


def _difference_to_centre(a: Array, axis: int, h: Any) -> Array:
    return (jnp.roll(a, -1, axis) - a) / h


def _z_difference_to_centre(w: Array, h: Any) -> Array:
    return (w[..., 1:] - w[..., :-1]) / h


#: The planes an opening lies on, by index: the low plane (``z = 0``) and the high one.
LOW, HIGH = 0, 1


@dataclass(frozen=True)
class OpeningLayout:
    """Each opening's plane (:data:`LOW` or :data:`HIGH`) and centre on it (in cells along x and y from the box's
    corner; read only where its plane holds another opening): the module's patches."""

    planes: tuple[int, ...] = (LOW, HIGH)
    centres: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if any(plane not in (LOW, HIGH) for plane in self.planes) or len(set(self.planes)) != 2:
            raise ValueError(f"each opening lies on the low plane (0) or the high one (1), each plane holding one at "
                             f"least: not {self.planes}")
        shared = len(self.planes) > 2
        if shared and len(self.centres) != len(self.planes):
            raise ValueError(f"a plane holding several openings splits between their centres: give one an opening, "
                             f"not {len(self.centres)} for {len(self.planes)}")

    def masks(self, cells: tuple[int, int, int]) -> np.ndarray:
        """Each opening's patch, ``(openings, nx, ny)``, ones on its cells: the whole plane where it is alone there,
        else the plane's cells nearest its centre."""
        nx, ny = cells[0], cells[1]
        x, y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
        masks = np.zeros((len(self.planes), nx, ny))
        for plane in (LOW, HIGH):
            members = [k for k, held in enumerate(self.planes) if held == plane]
            if len(members) == 1:
                masks[members[0]] = 1.0
                continue
            distance = np.stack([np.hypot(x - self.centres[k][0], y - self.centres[k][1]) for k in members])
            nearest = np.argmin(distance, axis=0)
            for position, k in enumerate(members):
                masks[k] = (nearest == position).astype(np.float64)
        return masks

    def alone(self, opening: int) -> bool:
        return self.planes.count(self.planes[opening]) == 1


@dataclass(frozen=True)
class OpenMACGrid(MACGrid):
    """The Cartesian MAC grid kind on a box of ``cells`` cubic cells of side ``spacing`` (m), periodic across x and y and
    open at its two z planes, its openings laid on them by ``layout`` (the module states the faces' weights and
    conditions, and the openings' patches)."""

    layout: OpeningLayout = OpeningLayout()

    kind: ClassVar[str] = "cartesian_mac_open_z"
    version: ClassVar[str] = "0.1.0"
    #: Exact under the half-weight inner product: the ledger's projection term is rounding.
    projector_exact: ClassVar[bool] = True
    periodic: ClassVar[tuple[bool, bool, bool]] = (True, True, False)

    @override
    def shape(self, component: int) -> tuple[int, int, int]:
        nx, ny, nz = self.cells
        return (nx, ny, nz + 1) if component == 2 else (nx, ny, nz)

    @override
    def stencil_indices(self, axis: int, component: int, lattice: Array) -> Array:
        """Wrapped across x and y; along the open z axis held inside the face set (``w``'s ``nz + 1`` faces, the
        others' ``nz``): a kernel's weight beyond a plane is folded onto the plane's faces, read and delivered there
        alike, and the delivery takes each face's weight (``mime.nodes.flow.fvm._exchange.scatter``), so the pair is
        a transpose pair in this grid's product up to the planes."""
        if axis != 2:
            return super().stencil_indices(axis, component, lattice)
        return jnp.clip(lattice, 0, self.shape(component)[2] - 1)

    def stencil_reaches_plane(self, points: Array, width_cells: float) -> Array:
        """Whether an exchange kernel of ``width_cells`` at each of ``points`` (``(n, 3)``, m) reaches a plane's faces
        (within its truncated reach and the Laplacian's cell of one): a boolean each."""
        reach = (math.ceil(TRUNCATION * float(width_cells)) + 2) * self.spacing
        z = points[:, 2]
        return (z - reach <= 0.0) | (z + reach >= self.cells[2] * self.spacing)

    @override
    def face_points(self, component: int, dtype: Any) -> Array:
        """The points (m) the ``component``-th velocity component is sampled at, ``(*shape, 3)``; ``w``'s include the
        two planes. The origin is the box's corner, the low plane at ``z = 0``."""
        h = jnp.asarray(self.spacing, dtype=dtype)
        axes: list[Any] = []
        for axis, count in enumerate(self.shape(component)):
            index = jnp.arange(count, dtype=dtype)
            axes.append(index * h if axis == component else (index + 0.5) * h)
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    @override
    def cell_points(self, dtype: Any) -> Array:
        h = jnp.asarray(self.spacing, dtype=dtype)
        axes = [(jnp.arange(count, dtype=dtype) + 0.5) * h for count in self.cells]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)

    @override
    def zeros(self, dtype: Any) -> Velocity:
        return (jnp.zeros(self.shape(0), dtype=dtype), jnp.zeros(self.shape(1), dtype=dtype),
                jnp.zeros(self.shape(2), dtype=dtype))

    def weights(self, dtype: Any) -> Velocity:
        """``V`` on each face set (the module's half weights on the planes' faces)."""
        return (jnp.ones(self.shape(0), dtype=dtype), jnp.ones(self.shape(1), dtype=dtype),
                jnp.broadcast_to(_end_weight(self.shape(2), dtype), self.shape(2)))

    # -- the operators --------------------------------------------------------------------------------------------
    @override
    def divergence(self, velocity: Velocity) -> Array:
        h = self.spacing
        u, v, w = velocity
        return _difference_to_centre(u, 0, h) + _difference_to_centre(v, 1, h) + _z_difference_to_centre(w, h)

    @override
    def gradient(self, pressure: Array) -> Velocity:
        """``G p``, the planes held at zero: the end faces' half-cell gradient ``2 (p_0 - 0) / h``."""
        h = self.spacing
        low = 2.0 * pressure[..., :1] / h
        inside = (pressure[..., 1:] - pressure[..., :-1]) / h
        high = -2.0 * pressure[..., -1:] / h
        return (_difference_to_face(pressure, 0, h), _difference_to_face(pressure, 1, h),
                jnp.concatenate([low, inside, high], axis=-1))

    @staticmethod
    def _across(a: Array, h2: Any, magnitude: bool) -> Array:
        """The x and y part of the seven-point Laplacian (periodic), or its magnitude."""
        sign = 1.0 if magnitude else -1.0
        return sum(((jnp.roll(a, 1, axis) + jnp.roll(a, -1, axis) + sign * 2.0 * a) for axis in (0, 1)),
                   start=jnp.zeros_like(a)) / h2

    @staticmethod
    def _along_centred(a: Array, h2: Any, magnitude: bool) -> Array:
        """The z part on a cell-centred component, zero gradient at both planes (its ghost equals its end value)."""
        sign = 1.0 if magnitude else -1.0
        below = jnp.concatenate([a[..., :1], a[..., :-1]], axis=-1)
        above = jnp.concatenate([a[..., 1:], a[..., -1:]], axis=-1)
        return (below + above + sign * 2.0 * a) / h2

    @staticmethod
    def _along_faces(w: Array, h2: Any, magnitude: bool) -> Array:
        """The z part on ``w``: the interior faces' three points; each plane's half cell the gradient inside it over
        half a cell, its gradient at the plane zero: ``2 (w_1 - w_0) / h^2``."""
        sign = 1.0 if magnitude else -1.0
        inside = (w[..., 2:] + w[..., :-2] + sign * 2.0 * w[..., 1:-1]) / h2
        low = 2.0 * (w[..., 1:2] + sign * w[..., :1]) / h2
        high = 2.0 * (w[..., -2:-1] + sign * w[..., -1:]) / h2
        return jnp.concatenate([low, inside, high], axis=-1)

    @override
    def laplacian(self, velocity: Velocity) -> Velocity:
        h2 = self.spacing ** 2
        u, v, w = velocity
        return (self._across(u, h2, False) + self._along_centred(u, h2, False),
                self._across(v, h2, False) + self._along_centred(v, h2, False),
                self._across(w, h2, False) + self._along_faces(w, h2, False))

    @override
    def laplacian_magnitude(self, velocity: Velocity) -> Velocity:
        h2 = self.spacing ** 2
        u, v, w = velocity
        return (self._across(u, h2, True) + self._along_centred(u, h2, True),
                self._across(v, h2, True) + self._along_centred(v, h2, True),
                self._across(w, h2, True) + self._along_faces(w, h2, True))

    @override
    def laplacian_diagonal(self) -> Any:
        """The interior faces' diagonal, ``-6 / h^2``: Jacobi's (the end cells' differ by one coefficient, which a
        preconditioner need not match)."""
        return -6.0 / self.spacing ** 2

    @override
    def convection(self, velocity: Velocity) -> Velocity:
        """``-div(u u)`` on each face set in divergence form, as the periodic grid's, with each open plane's flux its
        own face value: on ``w``'s half cells ``2 (w_c^2 - w_b^2) / h`` against the plane's ``w_b^2``, and on the
        edges the tangential velocity taken at the plane as its end cell's."""
        u, v, w = velocity
        h = self.spacing
        uc, vc = _average_to_centre(u, 0), _average_to_centre(v, 1)
        wc = 0.5 * (w[..., 1:] + w[..., :-1])
        cu = _difference_to_face(uc * uc, 0, h)
        cv = _difference_to_face(vc * vc, 1, h)
        flux = wc * wc
        cw = jnp.concatenate([2.0 * (flux[..., :1] - w[..., :1] ** 2) / h, (flux[..., 1:] - flux[..., :-1]) / h,
                              2.0 * (w[..., -1:] ** 2 - flux[..., -1:]) / h], axis=-1)
        xy = _average_to_face(u, 1) * _average_to_face(v, 0)
        cu = cu + _difference_to_centre(xy, 1, h)
        cv = cv + _difference_to_centre(xy, 0, h)

        def at_planes(a: Array) -> Array:
            return jnp.concatenate([a[..., :1], 0.5 * (a[..., 1:] + a[..., :-1]), a[..., -1:]], axis=-1)

        xz = at_planes(u) * _average_to_face(w, 0)
        yz = at_planes(v) * _average_to_face(w, 1)
        cu = cu + _z_difference_to_centre(xz, h)
        cw = cw + _difference_to_centre(xz, 0, h)
        cv = cv + _z_difference_to_centre(yz, h)
        cw = cw + _difference_to_centre(yz, 1, h)
        return (-cu, -cv, -cw)

    @override
    def inner(self, a: Velocity, b: Velocity) -> Array:
        """The weighted inner product: the cell volume times each face's product, the planes' faces at half weight."""
        weight = _end_weight(self.shape(2), a[2].dtype)
        return self.cell_volume * (jnp.sum(a[0] * b[0]) + jnp.sum(a[1] * b[1]) + jnp.sum(weight * a[2] * b[2]))

    @override
    def inner_magnitude(self, a: Velocity, b: Velocity) -> Array:
        return self.inner((jnp.abs(a[0]), jnp.abs(a[1]), jnp.abs(a[2])), (jnp.abs(b[0]), jnp.abs(b[1]), jnp.abs(b[2])))

    # -- the fast solve and the projector -------------------------------------------------------------------------
    def _symbol(self, dtype: Any) -> Array:
        """The eigenvalues of ``D G`` in the transform's layout (all negative)."""
        h2 = jnp.asarray(self.spacing, dtype=dtype) ** 2
        nx, ny, nz = self.cells
        sx = jnp.sin(jnp.pi * jnp.arange(nx, dtype=dtype) / nx) ** 2
        sy = jnp.sin(jnp.pi * jnp.arange(ny // 2 + 1, dtype=dtype) / ny) ** 2
        sz = jnp.sin(jnp.pi * (jnp.arange(nz, dtype=dtype) + 1.0) / (2.0 * nz)) ** 2
        return -(4.0 / h2) * (sx[:, None, None] + sy[None, :, None] + sz[None, None, :])

    @override
    def poisson(self, source: Array) -> Array:
        """``phi`` with ``D G phi = source``, the planes held at zero, by the fast transform."""
        nx, ny, nz = self.cells
        with solving(POISSON_SITE):
            basis = jnp.asarray(_sine_basis(nz), dtype=source.dtype)
            across = jnp.fft.rfft2(source, axes=(0, 1))
            modes = jnp.einsum("jm,abj->abm", basis, across, precision=_HIGHEST)
            modes = modes / self._symbol(source.dtype)
            back = jnp.einsum("jm,abm->abj", basis, modes, precision=_HIGHEST)
            return jnp.fft.irfft2(back, s=(nx, ny), axes=(0, 1)).astype(source.dtype)

    @override
    def project(self, velocity: Velocity) -> Velocity:
        """``P u = u - G (D G)^-1 D u``: the projection onto discretely divergence-free fields, orthogonal under
        :meth:`inner`."""
        phi = self.poisson(self.divergence(velocity))
        correction = self.gradient(phi)
        return (velocity[0] - correction[0], velocity[1] - correction[1], velocity[2] - correction[2])

    @override
    def pressure_of(self, remainder: Velocity) -> Array:
        """The pressure's field part (kinematic, zero on the planes, whose values the openings' forces carry) whose
        gradient balances ``remainder``: ``(D G)^-1 D (rhs - A u)``."""
        return self.poisson(self.divergence(remainder))

    # -- the openings ---------------------------------------------------------------------------------------------
    @property
    def opening_count(self) -> int:
        return len(self.layout.planes)

    def _plane(self, a: Array, opening: int) -> Array:
        return a[..., 0] if self.layout.planes[opening] == LOW else a[..., -1]

    def _outward(self, opening: int) -> float:
        return -1.0 if self.layout.planes[opening] == LOW else 1.0

    def _patch_sums(self, at_plane: Callable[[int], Array]) -> Array:
        """Each opening's sum of ``at_plane(opening)`` (an ``nx x ny`` array on its plane) over its patch."""
        masks = self.layout.masks(self.cells)
        return jnp.stack([jnp.sum(at_plane(k)) if self.layout.alone(k) else
                          jnp.sum(jnp.asarray(masks[k], dtype=at_plane(k).dtype) * at_plane(k))
                          for k in range(self.opening_count)])

    def opening_fluxes(self, velocity: Velocity) -> Array:
        """The volume flux (m^3/s) out through each opening (the layout's order), outward positive."""
        w = velocity[2]
        area = self.spacing ** 2
        signs = jnp.asarray([self._outward(k) for k in range(self.opening_count)], dtype=w.dtype)
        return signs * area * self._patch_sums(lambda k: self._plane(w, k))

    def opening_fluxes_magnitude(self, velocity: Velocity) -> Array:
        """:meth:`opening_fluxes` with every face's value in magnitude: the scale their rounding is judged on."""
        w = jnp.abs(velocity[2])
        return self.spacing ** 2 * self._patch_sums(lambda k: self._plane(w, k))

    def opening_faces(self) -> tuple[int, ...]:
        """How many faces each opening's patch holds."""
        masks = self.layout.masks(self.cells)
        return tuple(int(np.count_nonzero(masks[k])) for k in range(masks.shape[0]))

    def opening_kinetic_fluxes(self, velocity: Velocity) -> tuple[Array, Array]:
        """The kinetic energy (per unit density, m^5/s^3) convection carries out through each opening, the scheme's
        own (the module: ``<C(u), u> = -sum_k K_k`` on a discretely divergence-free field), and each in magnitude
        (its terms' absolute values summed: its rounding's scale)."""
        u, v, w = velocity

        def density(k: int) -> Array:
            at = 0 if self.layout.planes[k] == LOW else -1
            su = 0.5 * (u[..., at] ** 2 + jnp.roll(u[..., at] ** 2, -1, 0))
            sv = 0.5 * (v[..., at] ** 2 + jnp.roll(v[..., at] ** 2, -1, 1))
            wb = w[..., at]
            return 0.5 * (su + sv + wb * wb) * wb

        area = self.spacing ** 2
        signs = jnp.asarray([self._outward(k) for k in range(self.opening_count)], dtype=w.dtype)
        flux = signs * area * self._patch_sums(density)
        size = area * self._patch_sums(lambda k: jnp.abs(density(k)))
        return flux, size

    def port_force(self, pressures: Any, density: Any) -> Velocity:
        """The force per unit mass of the openings' pressures (Pa, the layout's order) on the planes' half cells, each
        on its patch: ``+2 p / (rho h)`` on the low plane, ``-2 p / (rho h)`` on the high one (each pushing into the
        box)."""
        p = jnp.asarray(pressures)
        dtype = p.dtype
        zero = self.zeros(dtype)
        scale = 2.0 / (density * self.spacing)
        masks = self.layout.masks(self.cells)
        planes: list[Any] = []
        for plane, sign in ((LOW, 1.0), (HIGH, -1.0)):
            members = [k for k, held in enumerate(self.layout.planes) if held == plane]
            if len(members) == 1:
                planes.append(sign * scale * p[members[0]])
            else:
                planes.append(sign * scale * sum((p[k] * jnp.asarray(masks[k], dtype=dtype) for k in members),
                                                 start=jnp.zeros(self.cells[:2], dtype=dtype)))
        w = zero[2].at[..., 0].set(planes[0]).at[..., -1].set(planes[1])
        return (zero[0], zero[1], w)


__all__ = ["HALF", "HIGH", "LOW", "OpenMACGrid", "OpeningLayout"]
