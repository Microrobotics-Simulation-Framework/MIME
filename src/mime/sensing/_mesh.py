"""Static meshes in the in-loop camera: a bounding volume hierarchy built once on the host and traced for every hit.

**The tree.** A complete binary tree in heap order (node 1 the root, its leaves ``2^D`` to ``2^(D+1) - 1``), built on the
host by balanced splits of the triangles along each group's longest centroid axis, :data:`LEAF` triangle slots a leaf,
its boxes fixed (the mesh stands). Its arrays are the camera node's parameter leaves (:data:`KEYS`), not constants of
the program: the step's compile is unchanged in time by a mesh in view and its program holds no copy of it.

**The reduction in one bin** (the design's all-hits table). A ray is traced to every crossing, with no pruning: each
crossing adds its ``t`` signed by its facing (leaving the solid ``+t``, entering it ``-t``, the triangle's outward normal
being ``e1 x e2``), so the sum is the chord inside a closed, consistently wound mesh, wherever the camera is; and each
adds one to the crossing count. A sub-ray's factor from a mesh is ``exp(-mu L)`` where it attenuates, ``(1 - alpha)^n``
where it shows a display opacity, zero where it is opaque and crossed, one where missed; a pixel's is the mean over
``S x S`` sub-rays (:data:`MESH_SAMPLES`) of the product over the meshes, so overlapping meshes are composed per sub-ray.

**Derivatives.** The rays and the tree are taken under ``stop_gradient``: no derivative reaches the traversal's while
loop, which has none in reverse mode, and a derivative in the camera's nine numbers would drop the mesh's part, so it is
refused by name where a mesh is traced (``camera.mesh-derivative``)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import jax
import numpy as np

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.core._node import jnp
from mime.sensing._coverage import sub_offsets
from mime.sensing._detector import CameraPose, Pinhole, Rays, ray_table
from mime.sensing._subjects import Appearance, Look

lax: Any = jax.lax

#: Sub-rays a pixel each way, ``S``: a pixel's coverage at a mesh's silhouette is the mean of ``S x S`` sub-rays, within
#: about ``1 / (2 S)`` of a pixel on a straight edge. A declared default, set by the mesh swap gate
#: (``tests/sensing/test_mesh_swap.py``); not a setting, so a saved experiment reproduces at its own MIME version.
MESH_SAMPLES = 4
#: Rays a tile of the traversal: a tile's state stays in cache, and a tile of rays that miss leaves at its first step.
TILE = 64
#: Triangle slots a leaf of the tree.
LEAF = 4
#: The tree's arrays, the camera node's parameter leaves in this order: the nodes' boxes (lowest and highest corners,
#: heap order, index 0 unused), each leaf slot's three corners (the triangle's own, so that two triangles sharing an
#: edge see the same numbers: the watertight test), and the mesh each slot's triangle belongs to (``-1`` for an empty
#: slot), in the run's dtype.
KEYS = ("mesh_lo", "mesh_hi", "mesh_v0", "mesh_v1", "mesh_v2", "mesh_owner")
#: Where an empty node's box lies: a point box far from any scene, met by no ray in practice, and finite in float32
#: under the slab test's reciprocal of a zero direction component.
EMPTY_BOX = 1.0e6


def readonly(array: Any) -> np.ndarray:
    """``array`` as a copy that cannot be written, and cannot be made writable: NumPy keeps an array over a ``bytes``
    object read-only, whatever is done to its flags, so a write through it (``a[...] = x``) raises ``ValueError``."""
    held = np.asarray(array)
    return cast(np.ndarray, np.frombuffer(held.tobytes(), dtype=held.dtype).reshape(held.shape))


@dataclass(frozen=True, eq=False)
class Frozen:
    """An array a node holds as a constant, which MADDENING does not take for one that can change in place. It keeps a
    NumPy ``ndarray`` (the host's float64 triangles, the owners) as a registered pytree of one leaf, so that the graph's
    check for written parameters, which compares an ndarray's bytes at every chunk of a run, does not read the megabytes
    of a mesh's triangles (the derived graph still records the array's dtype, shape and sha256). The array is a copy that
    cannot be written (:func:`readonly`): a mesh is held as built, and a write to it would be an image that no longer
    follows the mesh."""

    array: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "array", readonly(self.array))

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self.array, dtype=dtype)


jax.tree_util.register_dataclass(Frozen, data_fields=["array"], meta_fields=[])


@dataclass(frozen=True)
class Tree:
    """The tree's arrays by :data:`KEYS` (float64 on the host, read-only) and each leaf slot's triangle index (``-1``
    empty)."""

    arrays: Mapping[str, np.ndarray]
    slots: np.ndarray

    @property
    def depth(self) -> int:
        return int(round(math.log2(self.slots.shape[0])))


def tree(triangles: np.ndarray, owners: np.ndarray, leaf: int = LEAF) -> Tree:
    """The tree of ``triangles`` (``(T, 3, 3)``, metres, wound counter-clockwise seen from outside) whose meshes are
    ``owners`` (``(T,)``): balanced splits along each group's longest centroid axis to ``2^D`` leaves of at most
    ``leaf`` triangles, ``D`` the least depth that holds them."""
    count = len(triangles)
    depth = max(0, math.ceil(math.log2(max(1, math.ceil(count / leaf)))))
    centroid = triangles.mean(axis=1)
    order = np.arange(count)
    segments = [(0, count)]
    for _ in range(depth):
        split: list[tuple[int, int]] = []
        for a, b in segments:
            if b - a > 1:
                held = order[a:b]
                spread = centroid[held]
                axis = int(np.argmax(spread.max(axis=0) - spread.min(axis=0)))
                order[a:b] = held[np.argsort(spread[:, axis], kind="stable")]
            middle = a + (b - a + 1) // 2
            split += [(a, middle), (middle, b)]
        segments = split
    slots = -np.ones((1 << depth, leaf), dtype=np.int64)
    for k, (a, b) in enumerate(segments):
        slots[k, :b - a] = order[a:b]
    valid = slots >= 0
    corners = triangles[np.maximum(slots, 0)]
    filled = valid.any(axis=1)
    low = np.where(valid[..., None, None], corners, np.inf).min(axis=(1, 2))
    high = np.where(valid[..., None, None], corners, -np.inf).max(axis=(1, 2))
    levels_low, levels_high, levels_filled = [low], [high], [filled]
    while levels_low[0].shape[0] > 1:
        levels_low.insert(0, levels_low[0].reshape(-1, 2, 3).min(axis=1))
        levels_high.insert(0, levels_high[0].reshape(-1, 2, 3).max(axis=1))
        levels_filled.insert(0, levels_filled[0].reshape(-1, 2).any(axis=1))
    held = np.concatenate([[False]] + levels_filled)
    node_low = np.where(held[:, None], np.concatenate([np.zeros((1, 3))] + levels_low), EMPTY_BOX)
    node_high = np.where(held[:, None], np.concatenate([np.zeros((1, 3))] + levels_high), EMPTY_BOX)
    v0, v1, v2 = (np.where(valid[..., None], corners[:, :, k], 0.0) for k in range(3))
    owner = np.where(valid, np.asarray(owners, dtype=np.float64)[np.maximum(slots, 0)], -1.0)
    held_arrays = (node_low, node_high, v0, v1, v2, owner)
    return Tree({key: readonly(each) for key, each in zip(KEYS, held_arrays, strict=True)}, readonly(slots))


def triangles_of(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The corners of each triangle, ``(T, 3, 3)``."""
    return np.asarray(points, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def welded(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``faces`` over ``points`` with exactly repeated points made one (a triangle soup's shared corners joined)."""
    unique, inverse = np.unique(np.asarray(points, dtype=np.float64), axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)[np.asarray(faces, dtype=np.int64)]


def signed_volume(triangles: np.ndarray) -> float:
    """The volume a closed mesh encloses by the divergence theorem: positive where its triangles face outward."""
    return float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0)


def closed_problem(points: np.ndarray, faces: np.ndarray) -> str | None:
    """Why a mesh is not a closed, consistently wound solid facing outward (what attenuation along its chord needs), or
    ``None``: each directed edge once, each with its reverse, and a positive enclosed volume."""
    joined, held = welded(points, faces)
    edges = np.concatenate([held[:, [0, 1]], held[:, [1, 2]], held[:, [2, 0]]])
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    repeated = int(np.sum(counts > 1))
    found = {(int(a), int(b)) for a, b in unique}
    unmatched = sum((b, a) not in found for a, b in found)
    if repeated or unmatched:
        return (f"it is not closed and consistently wound: {repeated} edges run twice one way and {unmatched} have no "
                f"face the other way")
    if not signed_volume(triangles_of(joined, held)) > 0.0:
        return "its triangles face inward (it encloses a negative volume): wind them counter-clockwise seen from outside"
    return None


def changed_mesh(key: str, built: Any, now: Any) -> None:
    """Refuse (``camera.cache-stale``) a camera whose record of a mesh (its constructor argument ``key``) is no longer the
    one its tree and its held structure were made from: a different array in it. The same numbers in another object are
    taken for the same mesh."""
    if isinstance(now, Frozen) and np.array_equal(now.array, built.array):
        return
    raise ArgumentRefused([Refusal(
        "camera.cache-stale", RefusalClass.USER,
        f"the camera's record of its static meshes ({key}) was changed after it was built: its bounding volume hierarchy "
        f"and the static structure it holds were made from the meshes it was built with, and would be read as the new ones",
        "change the mesh in the scene and call `exp.build()` again: a mesh is held as built, and a study cannot vary it",
        why="a static mesh is held as built: its tree and the image of it are made once, from the build's meshes")])


def inside(triangles: np.ndarray, point: Sequence[float] | np.ndarray) -> bool:
    """Whether ``point`` lies inside a closed mesh: the parity of its crossings along a ray in a direction no edge of a
    regular mesh lies along."""
    direction = np.array([1.0, math.pi * 1e-3, math.e * 1e-3])
    direction /= np.linalg.norm(direction)
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    usable = np.abs(det) > 0.0
    inv = 1.0 / np.where(usable, det, 1.0)
    s = np.asarray(point, dtype=np.float64) - v0
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    crossed: np.ndarray = usable & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    return int(np.count_nonzero(crossed)) % 2 == 1


def signature(arrays: Sequence[Any], dtype: Any) -> Any:
    """What the tree's arrays are, as ``4 * len(KEYS)`` numbers in ``dtype``: for each array, the sum of its elements'
    bits and the sum of them weighted by position (the golden-ratio multiplier, so a swap changes it), both modulo
    ``2^32`` in unsigned integers, each split into two 16-bit halves that a float32 holds exactly. Integer sums do not
    depend on the order they are summed in, so the host's call and the one inside a compiled step give the same numbers;
    a write to any one element changes the first sum (and the second), and a change of several that cancels both would
    have to be chosen to: a checksum, not a hash. It is what the held static structure is compared with (the camera's
    ``holds_at``), so that a tree written after the structure was made never reads as the structure's own."""
    parts: list[Any] = []
    for each in arrays:
        bits = jnp.reshape(lax.bitcast_convert_type(jnp.asarray(each), jnp.uint32), (-1,))
        weights = jnp.arange(bits.shape[0], dtype=jnp.uint32) * jnp.uint32(2654435761) + jnp.uint32(1)
        for total in (jnp.sum(bits, dtype=jnp.uint32), jnp.sum(bits * weights, dtype=jnp.uint32)):
            parts += [(total & jnp.uint32(0xFFFF)).astype(dtype), (total >> jnp.uint32(16)).astype(dtype)]
    return jnp.stack(parts)


class Hits(NamedTuple):
    """Every crossing of each ray, by mesh: the signed chord (``(R, M)``, metres) and the crossing count (``(R, M)``);
    and the nearest crossing's ``t`` and its leaf slot (``leaf * LEAF + k``; ``-1`` for none)."""

    chord: Any
    count: Any
    nearest: Any
    slot: Any


def slab_pad(dtype: Any) -> float:
    """The factor the slab test's exit distance is widened by, ``1 + 2 gamma_3``, with ``gamma_n = n u / (1 - n u)`` and
    ``u = eps / 2`` the unit roundoff of ``dtype`` (Ize, 2013, "Robust BVH ray traversal"): a plane's distance is
    ``(plane - o) * (1 / d)``, three roundings (the reciprocal, the difference, the product), so each of the entry and
    exit distances is within ``gamma_3`` of its exact value, and the nearest exit exceeds the farthest entry wherever the
    ray crosses the box, by at most the two errors (``2 gamma_3``), which the widening takes up. Without it a ray through
    an edge of a box (a silhouette, where the entry and the exit are one distance) is dropped at the root or a node by
    the rounding of one side, and the crossing at the edge is lost; the node's boxes hold the triangles' corners rounded
    to the run's dtype, so no other slack is needed."""
    u = float(jnp.finfo(dtype).eps) / 2.0
    gamma = 3.0 * u / (1.0 - 3.0 * u)
    return 1.0 + 2.0 * gamma


def _slab(o: Any, inverse: Any, low: Any, high: Any) -> tuple[Any, Any]:
    ta = (low - o) * inverse
    tb = (high - o) * inverse
    return jnp.max(jnp.minimum(ta, tb), axis=-1), jnp.min(jnp.maximum(ta, tb), axis=-1) * slab_pad(ta.dtype)


def _component(a: Any, k: Any) -> Any:
    """``a``'s component ``k`` (one axis a ray, ``(R,)``) of each of its rows (``(R, ...,3)``)."""
    return jnp.take_along_axis(a, jnp.reshape(k, (-1,) + (1,) * (a.ndim - 1)), axis=-1)[..., 0]


def _sheared(d: Any) -> tuple[Any, ...]:
    """Each ray's frame for the watertight test: the axis its direction is largest along (``z``) and the two after it in
    turn (swapped where that component is negative, so a triangle's winding is kept), and the shear that takes the
    direction onto ``z``."""
    kz = jnp.argmax(jnp.abs(d), axis=-1)
    kx, ky = (kz + 1) % 3, (kz + 2) % 3
    dz = _component(d, kz)
    kx, ky = jnp.where(dz < 0.0, ky, kx), jnp.where(dz < 0.0, kx, ky)
    return kx, ky, kz, _component(d, kx) / dz, _component(d, ky) / dz, 1.0 / dz


def _crossings(o: Any, axes: tuple[Any, ...], a: Any, b: Any, c: Any) -> tuple[Any, Any, Any]:
    """Whether each ray crosses each triangle ``(a, b, c)`` (``(R, slots, 3)``), at what ``t``, and whether it leaves
    the solid there, by the watertight test (Woop, Benthin and Wald, 2013): the corners moved to the ray's origin and
    sheared so the ray runs along ``z``, each edge's function computed from its two corners alone, so the two triangles
    sharing an edge compute exactly opposite values, and a ray through an edge is claimed by exactly one of them (an edge
    function of exactly zero counts by the edge's direction, reversed for a triangle seen from behind). Möller-Trumbore's
    barycentric test is not watertight: at a shared edge in float32 a ray can cross both triangles or neither, and one
    crossing more or less moves the signed chord by its whole ``t``."""
    kx, ky, kz, sx, sy, sz = axes
    shear = (kx[:, None], ky[:, None], kz[:, None], sx[:, None], sy[:, None], sz[:, None])

    def moved(p: Any) -> tuple[Any, Any, Any]:
        p = p - o[:, None, :]
        z = _component(p, shear[2])
        return _component(p, shear[0]) - shear[3] * z, _component(p, shear[1]) - shear[4] * z, shear[5] * z

    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = moved(a), moved(b), moved(c)
    # Each product rounded on its own before the difference: XLA's CPU backend contracts ``p q - r s`` into a fused
    # multiply-add, which rounds one product and not the other, so the two triangles of an edge would no longer compute
    # exactly opposite values (and an empty slot's ``x y - y x`` would not be zero). The barrier keeps them apart.
    products = lax.optimization_barrier((cx * by, cy * bx, ax * cy, ay * cx, bx * ay, by * ax))
    u = products[0] - products[1]
    v = products[2] - products[3]
    w = products[4] - products[5]
    det = u + v + w
    behind = det < 0.0

    def claims(e: Any, dx: Any, dy: Any) -> Any:
        leading = (dy > 0.0) | ((dy == 0.0) & (dx < 0.0))
        return jnp.where(e == 0.0, leading ^ behind, jnp.where(behind, e < 0.0, e > 0.0))

    inside = claims(u, cx - bx, cy - by) & claims(v, ax - cx, ay - cy) & claims(w, bx - ax, by - ay)
    reach = u * az + v * bz + w * cz
    crossed = inside & (det != 0.0) & jnp.where(behind, reach < 0.0, reach > 0.0)
    t = reach / jnp.where(det != 0.0, det, 1.0)
    return crossed, t, behind


def _tile(arrays: Sequence[Any], meshes: int, o: Any, d: Any, live: Any) -> Hits:
    """Every crossing of a tile of rays from ``o`` along ``d`` (``(R, 3)``, unit), each ray's own stack of the nodes it
    has still to visit (depth plus one deep), in one while loop that ends when every ray's stack is empty."""
    low, high, v0, v1, v2, owner = arrays
    first_leaf = v0.shape[0]
    depth = int(round(math.log2(first_leaf)))
    rays = d.shape[0]
    dtype = d.dtype
    inverse = 1.0 / jnp.where(jnp.abs(d) > 0.0, d, jnp.finfo(dtype).tiny)
    tn, tf = _slab(o, inverse, low[1], high[1])
    start = jnp.where(live & (tf >= jnp.maximum(tn, 0.0)), 1, 0).astype(jnp.int32)
    axes = _sheared(d)
    lanes = jnp.arange(depth + 1)[None, :]
    which = jnp.arange(meshes, dtype=dtype)
    state = (start, jnp.zeros(rays, dtype=jnp.int32), jnp.zeros((rays, depth + 1), dtype=jnp.int32),
             jnp.zeros((rays, meshes), dtype=dtype), jnp.zeros((rays, meshes), dtype=jnp.int32),
             jnp.full(rays, jnp.inf, dtype=dtype), jnp.full(rays, -1, dtype=jnp.int32))

    def going(held: tuple[Any, ...]) -> Any:
        return jnp.any(held[0] > 0)

    def step(held: tuple[Any, ...]) -> tuple[Any, ...]:
        node, top, stack, chord, count, nearest, slot = held
        active = node > 0
        at_leaf = node >= first_leaf
        left = jnp.clip(2 * node, 1, max(2 * first_leaf - 2, 1))
        right = left + 1
        inner = active & ~at_leaf
        tnl, tfl = _slab(o, inverse, low[left], high[left])
        tnr, tfr = _slab(o, inverse, low[jnp.minimum(right, 2 * first_leaf - 1)],
                         high[jnp.minimum(right, 2 * first_leaf - 1)])
        hit_left = inner & (tfl >= jnp.maximum(tnl, 0.0))
        hit_right = inner & (tfr >= jnp.maximum(tnr, 0.0))
        index = jnp.clip(node - first_leaf, 0, first_leaf - 1)
        crossed, t, leaving = _crossings(o, axes, v0[index], v1[index], v2[index])
        crossed = crossed & (at_leaf & active)[:, None] & (owner[index] >= 0.0)
        signed = jnp.where(crossed, jnp.where(leaving, t, -t), 0.0)
        mine = owner[index][:, :, None] == which
        chord = chord + jnp.sum(jnp.where(mine, signed[:, :, None], 0.0), axis=1)
        count = count + jnp.sum(mine & crossed[:, :, None], axis=1).astype(jnp.int32)
        reach = jnp.where(crossed, t, jnp.inf)
        k = jnp.argmin(reach, axis=1)
        closest = jnp.take_along_axis(reach, k[:, None], axis=1)[:, 0]
        nearer = closest < nearest
        nearest = jnp.where(nearer, closest, nearest)
        slot = jnp.where(nearer, index.astype(jnp.int32) * v0.shape[1] + k.astype(jnp.int32), slot)
        both = hit_left & hit_right
        go = jnp.where(hit_left, left, jnp.where(hit_right, right, -1))
        stack = jnp.where((lanes == top[:, None]) & both[:, None], right[:, None], stack)
        top = top + both.astype(jnp.int32)
        pop = active & (go < 0)
        last = jnp.take_along_axis(stack, jnp.maximum(top - 1, 0)[:, None], axis=1)[:, 0]
        following = jnp.where(pop, jnp.where(top > 0, last, 0), go)
        top = jnp.where(pop & (top > 0), top - 1, top)
        return jnp.where(active, following, 0), top, stack, chord, count, nearest, slot

    held = lax.while_loop(going, step, state)
    return Hits(held[3], held[4], held[5], held[6])


def traced(arrays: Sequence[Any], meshes: int, origin: Any, directions: Any, live: Any) -> Hits:
    """Every crossing of each of the rays from ``origin`` along ``directions`` (``(N, 3)``) that is ``live``, in tiles of
    :data:`TILE` rays, the rays and the tree under ``stop_gradient``."""
    arrays = tuple(lax.stop_gradient(a) for a in arrays)
    directions = lax.stop_gradient(directions)
    origin = lax.stop_gradient(origin)
    n = directions.shape[0]
    tiles = -(-n // TILE)
    pad = tiles * TILE - n
    d = jnp.concatenate([directions, jnp.ones((pad, 3), dtype=directions.dtype)]).reshape(tiles, TILE, 3)
    alive = jnp.concatenate([live, jnp.zeros(pad, dtype=bool)]).reshape(tiles, TILE)
    o = jnp.broadcast_to(origin, (TILE, 3))
    def one(tile: tuple[Any, Any]) -> Hits:
        return _tile(arrays, meshes, o, tile[0], tile[1])

    found = lax.map(one, (d, alive))
    return Hits(*(jnp.reshape(a, (tiles * TILE, *a.shape[2:]))[:n] for a in found))


def _factor(looks: Sequence[Appearance], hits: Hits) -> Any:
    """Each ray's factor, the product over the meshes of what each passes where it is crossed."""
    total = jnp.ones(hits.chord.shape[:1], dtype=hits.chord.dtype)
    for m, appearance in enumerate(looks):
        crossings = hits.count[:, m]
        if appearance.look is Look.ATTENUATING:
            part = jnp.exp(-jnp.asarray(appearance.attenuation, dtype=total.dtype) * jnp.maximum(hits.chord[:, m], 0.0))
        elif appearance.look is Look.FLAT:
            part = (1.0 - jnp.asarray(appearance.blocks, dtype=total.dtype)) ** crossings.astype(total.dtype)
        else:
            part = jnp.where(crossings > 0, 0.0, 1.0).astype(total.dtype)
        total = total * part
    return total


def mesh_part(looks: Sequence[Appearance], arrays: Sequence[Any], rays: Rays, samples: int = MESH_SAMPLES) -> Any:
    """Each pixel's factor from the meshes (``(rows, columns)``): the mean over its ``samples x samples`` sub-rays of
    their factors, each sub-ray the pixel's ray moved along the ray table's derivatives; a pixel beyond the distortion
    passes everything."""
    rays = cast(Rays, jax.tree_util.tree_map(lax.stop_gradient, rays))
    d = jnp.stack(rays.direction, axis=-1)
    du = jnp.stack(rays.along_u, axis=-1)
    dv = jnp.stack(rays.along_v, axis=-1)
    shape = d.shape[:2]
    origin = jnp.stack([jnp.asarray(c, dtype=d.dtype) for c in rays.origin])
    live = ~jnp.reshape(rays.beyond, (-1,))
    offsets = jnp.asarray(sub_offsets(samples), dtype=d.dtype)

    def one(total: Any, offset: Any) -> tuple[Any, None]:
        moved = d + offset[0] * du + offset[1] * dv
        moved = moved / jnp.linalg.norm(moved, axis=-1, keepdims=True)
        hits = traced(arrays, len(looks), origin, jnp.reshape(moved, (-1, 3)), live)
        return total + jnp.reshape(_factor(looks, hits), shape), None

    total, _ = lax.scan(one, jnp.zeros(shape, dtype=d.dtype), offsets)
    return total / samples ** 2


@dataclass(frozen=True)
class MeshChords:
    """Each mesh's signed chord along each sub-ray of each pixel and its crossings: ``chords`` and ``crossings`` of shape
    ``(meshes, samples * samples, rows, columns)``, the meshes in ``names``' order, the sub-rays row by row across the
    pixel (``samples`` of 1: its centre), chords in metres. The path lengths a transmission model reads (an X-ray
    reduction: ``exp(-sum_m mu_m(E) L_m)`` on each sub-ray, averaged); no derivative passes through the rays."""

    names: tuple[str, ...]
    chords: Any
    crossings: Any


def _chords(arrays: tuple[Any, ...], rays: Rays, meshes: int, samples: int) -> tuple[Any, Any]:
    d = jnp.stack(rays.direction, axis=-1)
    du = jnp.stack(rays.along_u, axis=-1)
    dv = jnp.stack(rays.along_v, axis=-1)
    shape = d.shape[:2]
    origin = jnp.stack([jnp.asarray(c, dtype=d.dtype) for c in rays.origin])
    live = ~jnp.reshape(rays.beyond, (-1,))
    chords: list[Any] = []
    counts: list[Any] = []
    for su, sv in sub_offsets(samples):
        moved = d + su * du + sv * dv
        moved = moved / jnp.linalg.norm(moved, axis=-1, keepdims=True)
        hits = traced(arrays, meshes, origin, jnp.reshape(moved, (-1, 3)), live)
        chords.append(jnp.reshape(hits.chord.T, (meshes, *shape)))
        counts.append(jnp.reshape(hits.count.T, (meshes, *shape)))
    return jnp.stack(chords, axis=1), jnp.stack(counts, axis=1)


_chords_jit: Any = cast(Any, jax).jit(_chords, static_argnames=("meshes", "samples"))


def mesh_chords(meshes: Mapping[str, tuple[Any, Any]], detector: Pinhole, pose: CameraPose, *,
                samples: int = 1) -> MeshChords:
    """The signed chord of each mesh along each of ``samples x samples`` sub-rays of every pixel of ``detector`` at
    ``pose``, and its crossings (:class:`MeshChords`): ``meshes`` maps a name to its points (``(n, 3)``, metres, the
    world) and triangles (``(m, 3)`` indices, counter-clockwise seen from outside), the chord being the length inside a
    closed, consistently wound mesh. Traced by the in-loop camera's tree, offline, in the dtype of the detector's
    numbers; the rays are taken under ``stop_gradient``, so no derivative passes through them."""
    if samples < 1 or not meshes:
        raise ArgumentRefused([Refusal(
            "camera.subject", RefusalClass.USER,
            f"mesh_chords takes at least one mesh and samples from 1, not {len(meshes)} meshes and samples={samples!r}",
            "write mesh_chords({'vessel': (points, triangles)}, detector, pose, samples=1)")])
    names = tuple(meshes)
    triangles = [triangles_of(np.reshape(np.asarray(points, dtype=np.float64), (-1, 3)), np.asarray(faces))
                 for points, faces in (meshes[name] for name in names)]
    owners = np.concatenate([np.full(len(held), i, dtype=np.int64) for i, held in enumerate(triangles)])
    built = tree(np.concatenate(triangles), owners)
    rays = ray_table(detector, pose)
    dtype = rays.direction[0].dtype
    arrays = tuple(jnp.asarray(built.arrays[key], dtype=dtype) for key in KEYS)
    chords, crossings = _chords_jit(arrays, cast(Rays, jax.tree_util.tree_map(lax.stop_gradient, rays)),
                                    meshes=len(names), samples=samples)
    return MeshChords(names, chords, crossings)


__all__ = ["EMPTY_BOX", "Frozen", "KEYS", "LEAF", "MESH_SAMPLES", "TILE", "Hits", "MeshChords", "Tree", "changed_mesh", "closed_problem", "inside",
           "mesh_chords", "mesh_part", "readonly", "signature", "signed_volume", "slab_pad", "traced", "tree", "triangles_of",
           "welded"]
