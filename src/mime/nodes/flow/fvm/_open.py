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

**A prescribed plane** (the maintainer's inflow decision (A), 2026-11-12: a velocity inlet). Where ``prescribed`` names a
plane, its faces are given, not evolved: the normal velocity is a law's profile (``mime.nodes.flow.fvm._inlet``), the
tangential velocity zero at the plane (in the viscous ghost, ``-a_0``, and in convection's edge flux). The projector
holds those faces: their gradient is zero (the pressure Neumann there, Dirichlet at the other plane), the fast solve's
basis along z the mixed one (``sqrt(2 / n) cos(pi (m + 1/2) (j + 1/2) / n)``, mirrored for the high plane), and the
projection zeroes the given faces before it removes the divergence, ``P = P_N M``: orthogonal under the half-weight
product on the fields whose given faces are zero, the subspace the solve works in. A field carrying the given faces is
lifted (:meth:`OpenMACGrid.prescribed_field`: those faces, made divergence-free by ``P_N`` alone, which keeps them).

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, override

import jax
import numpy as np
from jax import Array

from mime.core import jnp
from mime.core.solves import solving
from mime.nodes.flow.fvm._grid import POISSON_SITE, MACGrid, Velocity

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


def _mixed_basis(count: int, low: bool) -> np.ndarray:
    """The orthonormal eigenvectors of the cell-centred Laplacian on ``count`` cells with a Neumann plane (the low one
    where ``low``, else the high) and a homogeneous Dirichlet one (columns): ``sqrt(2 / n) cos(pi (m + 1/2) (j + 1/2) /
    n)`` for a Neumann low plane, its rows reversed for a Neumann high one; eigenvalues ``-(4 / h^2) sin^2(pi (m + 1/2)
    / (2 n))``."""
    j = np.arange(count)[:, None] + 0.5
    m = np.arange(count)[None, :] + 0.5
    basis = np.sqrt(2.0 / count) * np.cos(np.pi * m * j / count)
    return basis if low else basis[::-1, :]


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
    #: The plane whose faces are given (:data:`LOW` or :data:`HIGH`, the module's prescribed plane), or none.
    prescribed: int | None = None

    kind: ClassVar[str] = "cartesian_mac_open_z"
    version: ClassVar[str] = "0.1.0"
    #: Exact under the half-weight inner product: the ledger's projection term is rounding.
    projector_exact: ClassVar[bool] = True
    periodic: ClassVar[tuple[bool, bool, bool]] = (True, True, False)

    def shape(self, component: int) -> tuple[int, int, int]:
        nx, ny, nz = self.cells
        return (nx, ny, nz + 1) if component == 2 else (nx, ny, nz)

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
        """``G p``, the planes held at zero: the end faces' half-cell gradient ``2 (p_0 - 0) / h``; zero on a prescribed
        plane's faces (its pressure Neumann)."""
        h = self.spacing
        low = jnp.zeros_like(pressure[..., :1]) if self.prescribed == LOW else 2.0 * pressure[..., :1] / h
        inside = (pressure[..., 1:] - pressure[..., :-1]) / h
        high = jnp.zeros_like(pressure[..., -1:]) if self.prescribed == HIGH else -2.0 * pressure[..., -1:] / h
        return (_difference_to_face(pressure, 0, h), _difference_to_face(pressure, 1, h),
                jnp.concatenate([low, inside, high], axis=-1))

    @staticmethod
    def _across(a: Array, h2: Any, magnitude: bool) -> Array:
        """The x and y part of the seven-point Laplacian (periodic), or its magnitude."""
        sign = 1.0 if magnitude else -1.0
        return sum(((jnp.roll(a, 1, axis) + jnp.roll(a, -1, axis) + sign * 2.0 * a) for axis in (0, 1)),
                   start=jnp.zeros_like(a)) / h2

    def _along_centred(self, a: Array, h2: Any, magnitude: bool) -> Array:
        """The z part on a cell-centred component, zero gradient at both planes (its ghost equals its end value); zero
        at a prescribed plane (its ghost the end value negated; in magnitude, the same terms)."""
        sign = 1.0 if magnitude else -1.0
        held_low = self.prescribed == LOW and not magnitude
        held_high = self.prescribed == HIGH and not magnitude
        below = jnp.concatenate([-a[..., :1] if held_low else a[..., :1], a[..., :-1]], axis=-1)
        above = jnp.concatenate([a[..., 1:], -a[..., -1:] if held_high else a[..., -1:]], axis=-1)
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
            low = jnp.zeros_like(a[..., :1]) if self.prescribed == LOW else a[..., :1]
            high = jnp.zeros_like(a[..., -1:]) if self.prescribed == HIGH else a[..., -1:]
            return jnp.concatenate([low, 0.5 * (a[..., 1:] + a[..., :-1]), high], axis=-1)

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
        shift = 0.5 if self.prescribed is not None else 1.0
        sz = jnp.sin(jnp.pi * (jnp.arange(nz, dtype=dtype) + shift) / (2.0 * nz)) ** 2
        return -(4.0 / h2) * (sx[:, None, None] + sy[None, :, None] + sz[None, None, :])

    @override
    def poisson(self, source: Array) -> Array:
        """``phi`` with ``D G phi = source``, the planes held at zero (a prescribed one Neumann), by the fast
        transform."""
        nx, ny, nz = self.cells
        with solving(POISSON_SITE):
            held = _sine_basis(nz) if self.prescribed is None else _mixed_basis(nz, self.prescribed == LOW)
            basis = jnp.asarray(held, dtype=source.dtype)
            across = jnp.fft.rfft2(source, axes=(0, 1))
            modes = jnp.einsum("jm,abj->abm", basis, across, precision=_HIGHEST)
            modes = modes / self._symbol(source.dtype)
            back = jnp.einsum("jm,abm->abj", basis, modes, precision=_HIGHEST)
            return jnp.fft.irfft2(back, s=(nx, ny), axes=(0, 1)).astype(source.dtype)

    @override
    def project(self, velocity: Velocity) -> Velocity:
        """``P u = u - G (D G)^-1 D u``: the projection onto discretely divergence-free fields, orthogonal under
        :meth:`inner`; with a prescribed plane, onto those whose given faces are zero (``P_N M``, the module's)."""
        held = self.held_out(velocity)
        phi = self.poisson(self.divergence(held))
        correction = self.gradient(phi)
        return (held[0] - correction[0], held[1] - correction[1], held[2] - correction[2])

    @override
    def pressure_of(self, remainder: Velocity) -> Array:
        """The pressure's field part (kinematic, zero on the planes, whose values the openings' forces carry; Neumann
        at a prescribed plane) whose gradient balances ``remainder``: ``(D G)^-1 D M (rhs - A u)``, the given faces'
        rows (their reaction, :meth:`given_part`) left out."""
        return self.poisson(self.divergence(self.held_out(remainder)))

    # -- a prescribed plane -----------------------------------------------------------------------------------------
    def _given_index(self) -> int:
        assert self.prescribed is not None, "the grid has no prescribed plane"
        return 0 if self.prescribed == LOW else -1

    def held_out(self, velocity: Velocity) -> Velocity:
        """``M u``: ``velocity`` with a prescribed plane's faces zero (``velocity`` itself where none is)."""
        if self.prescribed is None:
            return velocity
        return (velocity[0], velocity[1], velocity[2].at[..., self._given_index()].set(0.0))

    def given_part(self, velocity: Velocity) -> Array:
        """``velocity``'s ``w`` on the prescribed plane's faces (``nx x ny``)."""
        return velocity[2][..., self._given_index()]

    def prescribed_field(self, face: Array) -> Velocity:
        """The divergence-free field whose prescribed plane's faces are ``face`` (``nx x ny``, ``w`` there): those faces
        alone, less the gradient that removes their divergence (``P_N``, which keeps them): the lift a solve in the
        prescribed subspace adds back."""
        zero = self.zeros(face.dtype)
        given = (zero[0], zero[1], zero[2].at[..., self._given_index()].set(face))
        correction = self.gradient(self.poisson(self.divergence(given)))
        return (given[0] - correction[0], given[1] - correction[1], given[2] - correction[2])

    def given_pressure_work(self, pressure: Array, velocity: Velocity) -> Array:
        """The work per unit density and time (m^5/s^3) the field's pressure ``pressure`` (kinematic) does on the fluid
        through the prescribed plane's faces, ``h^2 sum p_c w_b n_in`` (``p_c`` the end cell's, ``n_in`` +1 for the low
        plane, -1 for the high): ``-<G p, u>`` less the pressure's work against ``D u`` (rounding after the
        projector), the term the half-weight product's projection books where no plane is given."""
        at = self._given_index()
        inward = 1.0 if self.prescribed == LOW else -1.0
        return inward * self.spacing ** 2 * jnp.sum(pressure[..., at] * velocity[2][..., at])

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
            if self.layout.planes[k] == self.prescribed:
                return 0.5 * (su + sv + wb * wb) * wb - self._given_edges(u[..., at], v[..., at], wb, k)
            return 0.5 * (su + sv + wb * wb) * wb

        area = self.spacing ** 2
        signs = jnp.asarray([self._outward(k) for k in range(self.opening_count)], dtype=w.dtype)
        flux = signs * area * self._patch_sums(density)
        size = area * self._patch_sums(lambda k: jnp.abs(density(k)))
        return flux, size

    def _given_edges(self, u: Array, v: Array, wb: Array, opening: int) -> Array:
        """At a prescribed plane, what its tangential velocity held at zero in convection's edge flux moves the
        scheme's convective power by, per face, over the opening's outward sign: the end cells' ``a`` (``u``, ``v``)
        carried by the plane's ``w`` averaged to their faces ``W``, ``a^2 W + a W (w_- - w) / (2 n)`` (``w_-`` the
        neighbour's along ``a``'s axis, ``n`` the outward sign), so the fluxes stay the scheme's own power exactly."""
        outward = self._outward(opening)
        total = jnp.zeros_like(wb)
        for a, axis in ((u, 0), (v, 1)):
            behind = jnp.roll(wb, 1, axis)
            carried = 0.5 * (behind + wb)
            total = total + a * a * carried + a * carried * (behind - wb) / (2.0 * outward)
        return total

    def _plane_masks(self, openings: Sequence[int], dtype: Any) -> tuple[Array, Array]:
        """The low and the high plane's faces that belong to one of ``openings``' patches (ones there), ``nx x ny``."""
        masks = self.layout.masks(self.cells)
        planes: list[Array] = []
        for plane in (LOW, HIGH):
            held = np.zeros(self.cells[:2])
            for k in openings:
                if self.layout.planes[k] == plane:
                    held = held + masks[k]
            planes.append(jnp.asarray(held, dtype=dtype))
        return planes[0], planes[1]

    def backflow_coefficient(self, velocity: Velocity, openings: Sequence[int], beta: Any = 1.0,
                             fluid: Array | None = None) -> Array:
        """The directional backflow term's coefficient on ``w``'s faces (F0's B6 and B7, decision 12, in its lagged
        implicit form): on ``openings``' patches' half cells ``beta |min(w_n, 0)| / h``, ``w_n`` the face's outward
        velocity in ``velocity`` (the step's start), weighted by ``fluid`` (on ``w``'s faces) where given; zero
        wherever the flow leaves, and on every other face. The term it makes with the step's end velocity,
        ``f = -c w'``, is a diagonal beside the wall's penalty: its power ``-c w'^2`` is never positive."""
        w = velocity[2]
        coefficient = jnp.zeros_like(w)
        if not openings:
            return coefficient
        low, high = self._plane_masks(openings, w.dtype)
        if fluid is not None:
            low, high = low * fluid[..., 0], high * fluid[..., -1]
        inward_low, inward_high = jnp.maximum(w[..., 0], 0.0), jnp.maximum(-w[..., -1], 0.0)
        return coefficient.at[..., 0].set(beta * low * inward_low / self.spacing) \
            .at[..., -1].set(beta * high * inward_high / self.spacing)

    def opening_reversed_fluxes(self, velocity: Velocity, fluid: Array) -> Array:
        """The volume flux (m^3/s) entering through each opening's patch where its faces' flow is inward, each face
        weighted by ``fluid`` (the faces' ``1 - chi``, on ``w``'s faces): ``h^2 sum_patch (1 - chi) max(-w_n, 0)``."""
        w = velocity[2]

        def inward(k: int) -> Array:
            outward = self._outward(k) * self._plane(w, k)
            return self._plane(fluid, k) * jnp.maximum(-outward, 0.0)

        return self.spacing ** 2 * self._patch_sums(inward)

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
