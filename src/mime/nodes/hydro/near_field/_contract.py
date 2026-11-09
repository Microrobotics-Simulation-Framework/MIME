"""The near-field body-model contract, and the joint system of a medium's interacting bodies built on it.

**The contract** (:class:`BodyOperator`). A body model in the near field gives a small core: the points where it reads
the flow and delivers its traction (its force points, placed by its pose), the cells they stand for (their areas,
centroids and blob), and its response at those points (its own block of the joint matrix, at unit viscosity, in the
world). Beside the core it declares capabilities: a cross-kernel (the velocity its traction makes at any other body's
points), a self-block constant in its own frame (made once per build and turned to the pose, so a rigid body's
free-space operator is never made again in a run), and a preconditioner block (its own block's factors, which the
joint system's GMRES takes). Linear (Stokes) near fields only. Several geometries and body models in one medium are
blocks of different sizes.

**The joint system** (:func:`joint_matrix`, :func:`grand_resistance`). Every block is a pure function of geometry:
the bodies' poses and leaves, the wall's parameters and the viscosity are arguments, never read from a node's state
behind the call, so that a caller may evaluate it with a neighbour or a wall moved (a contact opened). The cluster's
grand resistance ``R = W^T A^-1 M`` (``6K x 6K``) has every body's six rigid motions as right-hand sides: a body's
coupled 6 x 6 is six columns of it, each a whole column block, so a lagged step needs all of it whatever the solver.
It is solved by the dense LU of the joint matrix (:func:`grand_resistance`), whose factors serve every right-hand
side, or, for a cluster past the dense route's size, by GMRES on every column at once, each body's own block's
inverse its preconditioner (:func:`iterative_resistance`), which holds no factors of the whole."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from jax import Array

from mime.core import jnp
from mime.nodes.hydro.mrs._solve import PRECISION, force_sums, rigid_motions
from mime.nodes.hydro.mrs._walls import Corrected, WallCorrection
from mime.nodes.hydro.near_field._gmres import Operator, Stopping, reported_solve

#: The capabilities a body model may declare beside the core.
CROSS_KERNEL, BODY_FRAME_CONSTANT, PRECONDITIONER = "cross_kernel", "body_frame_constant", "preconditioner"


@dataclasses.dataclass(frozen=True)
class Placed:
    """A body model at a pose, in the world: its force points ``(N, 3)``, each cell's centroid ``(N, 3)`` and area
    ``(N,)``, the blob's square, the centre its rigid motions and wrench are taken about (the body's centre of mass),
    and what else the model made at that pose for its own blocks (``own``: an MRS body's quadrature, placed), so that a
    step places each body once; and, where the model gives them, its surface's outward normals at the force points
    ``(N, 3)``, which the gap between bodies is estimated with (:func:`pair_gaps`)."""

    force: Array
    centroid: Array
    area: Array
    eps_squared: Array
    centre: Array
    own: Any = None
    normal: Array | None = None

    @property
    def count(self) -> int:
        return int(self.force.shape[0])

    def motions(self) -> Array:
        """``M``, ``3N x 6``: each force point's velocity for a unit translation and rotation about the centre."""
        return rigid_motions(self.force, self.centre)

    def wrench(self) -> Array:
        """``W``, ``3N x 6``: each cell's force and torque about the centre per unit traction, so that the wrench of
        tractions ``g`` is ``W^T g``."""
        return rigid_motions(self.centroid, self.centre) * jnp.repeat(self.area, 3)[:, None]


class BodyOperator(Protocol):
    """A body model in the near field: the contract the joint node is written against.

    ``leaves`` are the model's own traced quantities by name (its shape's numbers, its placement in the body, its
    blob), ``held`` what it made once per build in its frame (``None`` where it declares no
    :data:`BODY_FRAME_CONSTANT`), ``pose`` the body's ``[x; q]``."""

    @property
    def count(self) -> int: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def placed(self, leaves: Mapping[str, Any], pose: Array) -> Placed: ...

    def self_block(self, placed: Placed, leaves: Mapping[str, Any], held: Array | None, pose: Array, viscosity: Any,
                   wall: WallCorrection, parameters: Mapping[str, Any]) -> tuple[Array, Array, Array]:
        """Its own block of the joint matrix in the world (``3N x 3N``, the velocity at its force points per unit
        traction on its cells, its wall's self-image included), whether every read of the wall's data was accepted,
        and whether a cell within its wall image's reach was left out of its budget."""
        ...

    def cross(self, placed: Placed, targets: Array, reach: Any) -> tuple[Array, Array]:
        """The velocity its traction makes at ``targets`` ``(T, 3)`` (another body's force points), ``3T x 3N`` at unit
        viscosity in free space, from its own placement; and whether a pair within ``reach`` of the targets left a cell
        out of its budget."""
        ...


@dataclasses.dataclass(frozen=True)
class Joint:
    """The joint system at one geometry: the matrix ``A`` (``3 sum N x 3 sum N``), each body's placement, whether every
    wall read was accepted, and per body whether its wall image or a cross-kernel into it ran short of its budget."""

    matrix: Array
    placed: tuple[Placed, ...]
    accepted: Array
    wall_short: Array
    cross_short: Array


def offsets(counts: Sequence[int]) -> list[int]:
    """Where each body's rows start in the joint matrix, and its end: ``3 N`` rows a body."""
    found = [0]
    for count in counts:
        found.append(found[-1] + 3 * count)
    return found


def cross_image(wall: WallCorrection, target: Placed, source: Placed, parameters: Mapping[str, Any],
                viscosity: Any) -> Corrected:
    """The wall's image between two bodies: at ``target``'s force points per unit traction on ``source``'s cells, each
    cell's force placed at its force point (the column times its area), ``3T x 3N``. Zero for a wall that adds
    nothing."""
    if wall.adds_nothing:
        return Corrected(jnp.zeros((3 * target.count, 3 * source.count), dtype=target.force.dtype), jnp.asarray(True))
    image = wall.image
    if image is None:
        raise ValueError(f"the wall correction {wall.name} {wall.version} gives no image between point sets, which "
                         f"bodies in one medium are coupled by")
    found = image(target.force, source.force, parameters, viscosity, eps_squared=source.eps_squared)
    blocks = found.matrix * source.area[None, :, None, None]
    return Corrected(blocks.transpose(0, 2, 1, 3).reshape(3 * target.count, 3 * source.count), found.accepted)


def joint_matrix(members: Sequence[BodyOperator], leaves: Sequence[Mapping[str, Any]], held: Sequence[Array | None],
                 poses: Array, viscosity: Any, wall: WallCorrection, parameters: Mapping[str, Any],
                 reach: Any, coupled: bool = True) -> Joint:
    """The joint matrix of ``members`` at ``poses`` ``(K, 7)``: each body's own block (:meth:`BodyOperator.self_block`)
    on the diagonal, and off it the source's cross-kernel at the target's force points over the viscosity plus the
    wall's image between them (:func:`cross_image`). A pure function of its arguments. ``coupled=False`` leaves the
    off-diagonal blocks zero (the control: bodies that do not see each other)."""
    placed = tuple(member.placed(own, poses[i]) for i, (member, own) in enumerate(zip(members, leaves, strict=True)))
    rows: list[Array] = []
    accepted = jnp.asarray(True)
    wall_short: list[Array] = []
    cross_short: list[Array] = []
    for i, target in enumerate(placed):
        row: list[Array] = []
        short = jnp.asarray(False)
        for j, source in enumerate(placed):
            if i == j:
                block, ok, cut = members[i].self_block(target, leaves[i], held[i], poses[i], viscosity, wall,
                                                       parameters)
                wall_short.append(cut)
            elif not coupled:
                block, ok = jnp.zeros((3 * target.count, 3 * source.count), dtype=target.force.dtype), \
                    jnp.asarray(True)
            else:
                free, cut = members[j].cross(source, target.force, reach)
                image = cross_image(wall, target, source, parameters, viscosity)
                block, ok = free / viscosity + image.matrix, image.accepted
                short = short | cut
            row.append(block)
            accepted = accepted & ok
        cross_short.append(short)
        rows.append(jnp.concatenate(row, axis=1))
    return Joint(jnp.concatenate(rows, axis=0), placed, accepted, jnp.stack(wall_short), jnp.stack(cross_short))


def joint_motions(joint: Joint) -> Array:
    """``M``, ``3 sum N x 6K``: each body's six rigid motions at its own force points, zero at the others'."""
    counts = [p.count for p in joint.placed]
    at = offsets(counts)
    size, bodies = at[-1], len(counts)
    motions = jnp.zeros((size, 6 * bodies), dtype=joint.matrix.dtype)
    for i, p in enumerate(joint.placed):
        motions = motions.at[at[i]:at[i + 1], 6 * i:6 * i + 6].set(p.motions())
    return motions


def _wrenches(joint: Joint, tractions: Array) -> Array:
    at = offsets([p.count for p in joint.placed])
    rows = [force_sums(p.wrench(), tractions[at[i]:at[i + 1]]) for i, p in enumerate(joint.placed)]
    return jnp.concatenate(rows, axis=0)


def grand_resistance(joint: Joint) -> Array:
    """``R = W^T A^-1 M``, ``6K x 6K``: the force and torque on each body (about its centre) for each unit rigid motion
    of each body, the others held, by one dense LU of the joint matrix with every body's six motions as right-hand
    sides (``jnp.linalg.solve``: its derivative reuses the factors). Not symmetrised: its asymmetry is the
    discretisation's."""
    return _wrenches(joint, jnp.linalg.solve(joint.matrix, joint_motions(joint)))


def iterative_resistance(joint: Joint, blocks: Sequence[Operator | None], transposed: Sequence[Operator | None],
                         stopping: Stopping, tap: Array) -> tuple[Array, Array]:
    """:func:`grand_resistance`'s ``R`` by GMRES on its ``6K`` columns at once, right-preconditioned by each body's own
    block (``blocks[i]`` on its rows; ``None``, the identity, for a model that declares no :data:`PRECONDITIONER`),
    ``transposed`` the transpose solve's, through ``custom_linear_solve`` with the report block ``tap`` on its
    right-hand side (``mime.nodes.hydro.near_field._gmres.reported_solve``); with the primal solve's report. Its
    answer is the dense route's to the solve's tolerance."""
    at = offsets([p.count for p in joint.placed])

    def joined(parts: Sequence[Operator | None]) -> Operator:
        def apply(v: Array) -> Array:
            return jnp.concatenate([v[at[i]:at[i + 1]] if part is None else part(v[at[i]:at[i + 1]])
                                    for i, part in enumerate(parts)])

        return apply

    tractions, report = reported_solve(joint.matrix, joint_motions(joint), tap, joined(blocks), joined(transposed),
                                       stopping)
    return _wrenches(joint, tractions), report


def blocks_of(resistance: Array, bodies: int) -> tuple[Array, Array]:
    """The block-diagonal part of a ``6K x 6K`` resistance (each body's own ``6 x 6``, ``(K, 6, 6)``) and the rest,
    the cross blocks, as a ``6K x 6K`` with zeros on its diagonal blocks."""
    own = jnp.stack([resistance[6 * i:6 * i + 6, 6 * i:6 * i + 6] for i in range(bodies)])
    mask = jnp.kron(jnp.eye(bodies, dtype=resistance.dtype), jnp.ones((6, 6), dtype=resistance.dtype))
    return own, resistance * (1.0 - mask)


def cross_loads(off_diagonal: Array, twists: Array) -> Array:
    """Each body's lagged cross load, ``-sum_b R_ab V_b`` at the twists ``(K, 6)``, ``(K, 6)``."""
    bodies = twists.shape[0]
    return -jnp.matmul(off_diagonal, twists.reshape(-1), precision=PRECISION).reshape(bodies, 6)


def lagged_gain(own: Array, off_diagonal: Array, masses: Array, dt: float) -> Array:
    """The lagged scheme's gain, the spectral radius of ``(M/dt + R_d)^-1 R_off``: each body's step implicit in its own
    ``R_aa``, the cross load from the step before. ``M/dt + R_d`` is block-diagonal and symmetric positive definite
    and ``R_off`` symmetric, so the pencil's eigenvalues are those of the symmetric ``L^-1 R_off L^-T`` (``L L^T = M/dt
    + R_d``, a Cholesky factor per body): real, found by ``eigvalsh``, on any device. ``masses`` are the bodies' mass
    matrices in the world ``(K, 6, 6)``; ``own`` and ``off_diagonal`` symmetric. Not a number where a body's ``M/dt +
    R_aa`` is not positive definite."""
    bodies = own.shape[0]
    factors = jnp.linalg.cholesky(masses / dt + own)
    inverse = jnp.linalg.inv(factors)
    spread = jnp.zeros_like(off_diagonal)
    for i in range(bodies):
        for j in range(bodies):
            if i != j:
                block = off_diagonal[6 * i:6 * i + 6, 6 * j:6 * j + 6]
                spread = spread.at[6 * i:6 * i + 6, 6 * j:6 * j + 6].set(
                    jnp.matmul(jnp.matmul(inverse[i], block, precision=PRECISION), inverse[j].T, precision=PRECISION))
    symmetric = 0.5 * (spread + spread.T)
    return jnp.max(jnp.abs(jnp.linalg.eigvalsh(symmetric)))


def lagged_amplification(own: Array, off_diagonal: Array, masses: Array, dt: float) -> Array:
    """The lagged step's amplification of an error, the spectral radius of ``(M/dt + R_d)^-1 (M/dt - R_off)``: each
    body's step ``(M/dt + R_aa) dV = F - R_aa V - sum_b R_ab V_b`` with the others' twists a step behind takes an error
    ``e`` to ``(M/dt + R_d)^-1 (M/dt - R_off) e``. The pencil is symmetric as :func:`lagged_gain`'s is, so its eigenvalues
    are those of ``L^-1 (M/dt - R_off) L^-T`` (``L L^T = M/dt + R_d``), found by ``eigvalsh``. Overdamped (``M/dt``
    small against ``R``) it is :func:`lagged_gain`; with inertia it is below it, and the gain's check is the sufficient
    one: a gain below one makes this below one where ``R`` is definite, not conversely."""
    bodies = own.shape[0]
    factors = jnp.linalg.cholesky(masses / dt + own)
    inverse = jnp.linalg.inv(factors)
    spread = jnp.zeros_like(off_diagonal)
    for i in range(bodies):
        for j in range(bodies):
            block = masses[i] / dt if i == j else -off_diagonal[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            spread = spread.at[6 * i:6 * i + 6, 6 * j:6 * j + 6].set(
                jnp.matmul(jnp.matmul(inverse[i], block, precision=PRECISION), inverse[j].T, precision=PRECISION))
    symmetric = 0.5 * (spread + spread.T)
    return jnp.max(jnp.abs(jnp.linalg.eigvalsh(symmetric)))


def booked_work(own: Array, loads: Array, twists: Array, dt: float) -> tuple[Array, Array]:
    """The near field's books over one step, summed over the cluster as the energy ledger pairs them on each body (one
    pairing rule): each body's own coupled resistance ``R_aa`` paired at the step's solution, ``-dt V_a'^T R_aa V_a'``,
    and its cross load ``F_a`` as a load paired at the configuration's change, ``dt F_a . V_a'`` (a body no contact
    projects moves by ``dt V'`` in a step), from the blocks and loads the node delivered (``own`` ``(K, 6, 6)``, ``loads``
    ``(K, 6)``) and the twists the bodies took (``twists`` ``(K, 6)``); with the scale of its arithmetic, each product in
    magnitude. Negative where the fluid took energy from the bodies."""
    resisted = jnp.sum(twists * jnp.einsum("kij,kj->ki", own, twists, precision=PRECISION))
    loaded = jnp.sum(loads * twists)
    magnitude = jnp.abs(twists)
    scale = jnp.sum(magnitude * jnp.einsum("kij,kj->ki", jnp.abs(own), magnitude, precision=PRECISION)) \
        + jnp.sum(jnp.abs(loads) * magnitude)
    return dt * (loaded - resisted), dt * scale


#: How far behind a body's surface a point may lie, by the tangent plane at its nearest force point, in that force
#: point's cell's own size, and still be read as touching it, not inside it: the tangent plane stands off a curved
#: surface by about a quarter of the squared distance from the force point over the surface's radius, under a tenth of
#: a cell where the radius is a cell or more; a quarter keeps room.
OVERLAP_RESOLUTION = 0.25


def pair_gaps(placed: Sequence[Placed]) -> tuple[Array, Array]:
    """The gap between each ordered pair of bodies, estimated from their discretisations: ``(K, K)`` with, at ``(i,
    j)``, the least over body ``i``'s force points and its centre of each one's height above body ``j``'s surface, read
    on the tangent plane at ``j``'s nearest force point, ``(x - y) . n_y`` (the distance ``|x - y|`` where ``j`` gives no
    normals), in m; and whether one of those points lies behind ``j``'s surface by more than
    :data:`OVERLAP_RESOLUTION` of its nearest cell's size: the bodies overlap. The diagonal is ``inf`` and ``False``.
    The centre is a point of a solid body inside its surface, so a body inside another, or coincident with it, is
    read as overlapping though none of its force points lies behind the other's surface. A pure function of the
    placements; a judgement, for the caller to hold out of every derivative."""
    bodies = len(placed)
    dtype = placed[0].force.dtype
    gaps = jnp.full((bodies, bodies), jnp.inf, dtype=dtype)
    beyond = jnp.zeros((bodies, bodies), dtype=bool)
    for i, target in enumerate(placed):
        points = jnp.concatenate([target.force, target.centre[None, :]])
        for j, source in enumerate(placed):
            if i == j:
                continue
            apart = points[:, None, :] - source.force[None, :, :]
            distance = jnp.linalg.norm(apart, axis=-1)
            nearest = jnp.argmin(distance, axis=1)
            offset = points - source.force[nearest]
            height = jnp.linalg.norm(offset, axis=-1) if source.normal is None else \
                jnp.sum(offset * source.normal[nearest], axis=-1)
            gaps = gaps.at[i, j].set(jnp.min(height))
            beyond = beyond.at[i, j].set(jnp.any(height < -OVERLAP_RESOLUTION * jnp.sqrt(source.area[nearest])))
    return gaps, beyond


def quadratic_form(resistance: Array, twists: Array) -> tuple[Array, Array]:
    """The cluster's dissipation rate ``V^T R V`` at the twists ``(K, 6)`` and its rounding's scale ``|V|^T |R| |V|``:
    each body's own dissipation and the cross term together, at one velocity."""
    flat = twists.reshape(-1)
    form = jnp.dot(flat, jnp.matmul(resistance, flat, precision=PRECISION), precision=PRECISION)
    magnitude = jnp.abs(flat)
    scale = jnp.dot(magnitude, jnp.matmul(jnp.abs(resistance), magnitude, precision=PRECISION), precision=PRECISION)
    return form, scale


__all__ = ["BODY_FRAME_CONSTANT", "CROSS_KERNEL", "OVERLAP_RESOLUTION", "PRECONDITIONER", "BodyOperator", "Joint",
           "Placed", "blocks_of", "booked_work", "cross_image", "cross_loads", "grand_resistance",
           "iterative_resistance", "joint_matrix", "joint_motions", "lagged_amplification", "lagged_gain", "offsets",
           "pair_gaps", "quadratic_form"]
