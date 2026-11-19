"""Gather and scatter, the two primitives of a face-graph finite-volume method, each in three implementations.

``scatter``: a plain gather whose reverse mode is XLA's scatter-add, and ``segment_sum`` (the published graph solvers' form).
``padded``: every accumulation is a gather through a padded table of the positions that feed each target, summed along the
pad; the reverse of a gather is the same kind of gather through the transposed table (a custom VJP). No scatter either way.
``sorted``: accumulations are ``segment_sum`` over entries permuted into target order (``indices_are_sorted``).
Tables are static numpy arrays, passed to the compiled function as arguments (never closed over as constants)."""
import jax
import jax.numpy as jnp
import numpy as np

IMPLS = ("scatter", "padded", "sorted")


def transposed_table(idx_flat, n):
    """For each of ``n`` targets, the positions of ``idx_flat`` that name it, padded with ``len(idx_flat)``."""
    idx_flat = np.asarray(idx_flat)
    order = np.argsort(idx_flat, kind="stable")
    sidx = idx_flat[order]
    counts = np.bincount(idx_flat, minlength=n)
    width = int(counts.max())
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.arange(idx_flat.size) - start[sidx]
    table = np.full((n, width), idx_flat.size, dtype=np.int32)
    table[sidx, slot] = order
    return table, bool(counts.min() != width), order.astype(np.int32), sidx.astype(np.int32)


def _float0(tab):
    return jax.tree_util.tree_map(lambda t: np.zeros(np.shape(t), jax.dtypes.float0), tab)


def _pad(flat, ragged):
    if not ragged:
        return flat
    return jnp.concatenate([flat, jnp.zeros((1,) + flat.shape[1:], flat.dtype)])


def make_gather(idx, n, impl, ghost=False):
    """``(tab, f)`` with ``f(tab, x) = x[idx]`` for ``x`` of leading size ``n``; ``idx`` of any shape. With ``ghost`` the
    last row of ``x`` is a constant (a boundary value): what is gathered from it sends nothing back."""
    idx = np.asarray(idx, np.int32)
    tab = {"idx": idx}
    if impl == "scatter":
        return tab, lambda tab, x: x[tab["idx"]]
    size, rank = idx.size, idx.ndim
    real = n - 1 if ghost else n
    flat_idx = idx.ravel()
    kept = np.flatnonzero(flat_idx < real)
    table, ragged, order, sidx = transposed_table(flat_idx[kept], real)
    ragged = ragged or ghost
    table = np.where(table < kept.size, kept[np.minimum(table, kept.size - 1)], size).astype(np.int32)

    def with_ghost(out):
        return jnp.concatenate([out, jnp.zeros((1,) + out.shape[1:], out.dtype)]) if ghost else out
    if impl == "padded":
        tab["T"] = table

        def back(tab, ct):
            flat = _pad(ct.reshape((size,) + ct.shape[rank:]), ragged)
            return with_ghost(flat[tab["T"]].sum(axis=1))
    else:
        tab["perm"], tab["sidx"] = kept[order].astype(np.int32), sidx

        def back(tab, ct):
            flat = ct.reshape((size,) + ct.shape[rank:])
            return with_ghost(jax.ops.segment_sum(flat[tab["perm"]], tab["sidx"], real, indices_are_sorted=True))

    @jax.custom_vjp
    def gather(tab, x):
        return x[tab["idx"]]

    gather.defvjp(lambda tab, x: (x[tab["idx"]], tab), lambda tab, ct: (_float0(tab), back(tab, ct)))
    return tab, gather


def make_net(own, nei, n, impl):
    """``(tab, f)`` with ``f(tab, v)[i] = sum of v over edges owned by i - sum of v over edges whose neighbour is i``:
    the net outflow of an edge flux oriented owner to neighbour. Edges whose ``nei`` is ``>= n`` (a ghost) only leave."""
    own, nei = np.asarray(own, np.int32), np.asarray(nei, np.int32)
    edges = own.size
    inner = nei < n
    tab = {"own": own, "nei": np.where(inner, nei, own).astype(np.int32)}
    if impl == "scatter":
        tab["nei_seg"] = np.where(inner, nei, n).astype(np.int32)

        def net(tab, v):
            return jax.ops.segment_sum(v, tab["own"], n) - jax.ops.segment_sum(v, tab["nei_seg"], n + 1)[:n]
        return tab, net
    both = np.concatenate([own, nei[inner]])
    edge = np.concatenate([np.arange(edges), np.arange(edges)[inner]]).astype(np.int32)
    sign = np.concatenate([np.ones(edges), -np.ones(int(inner.sum()))])
    table, ragged, order, sidx = transposed_table(both, n)
    mask = inner.astype(np.float64)
    all_inner = bool(inner.all())
    if impl == "padded":
        valid = table < both.size
        safe = np.where(valid, table, 0)
        tab["E"] = np.where(valid, edge[safe], edges if ragged else 0).astype(np.int32)
        tab["sg"] = np.where(valid, sign[safe], 0.0)

        def forward(tab, v):
            picked = _pad(v, ragged)[tab["E"]]
            sg = tab["sg"].astype(v.dtype).reshape(tab["sg"].shape + (1,) * (v.ndim - 1))
            return (picked * sg).sum(axis=1)
    else:
        tab["he_edge"], tab["he_cell"], tab["he_sign"] = edge[order], sidx, sign[order]

        def forward(tab, v):
            sg = tab["he_sign"].astype(v.dtype).reshape((-1,) + (1,) * (v.ndim - 1))
            return jax.ops.segment_sum(v[tab["he_edge"]] * sg, tab["he_cell"], n, indices_are_sorted=True)
    if not all_inner:
        tab["inner"] = mask

    def back(tab, ct):
        out = ct[tab["own"]]
        back_nei = ct[tab["nei"]]
        if not all_inner:
            back_nei = back_nei * tab["inner"].astype(ct.dtype).reshape((-1,) + (1,) * (ct.ndim - 1))
        return out - back_nei

    @jax.custom_vjp
    def net(tab, v):
        return forward(tab, v)

    net.defvjp(lambda tab, v: (forward(tab, v), tab), lambda tab, ct: (_float0(tab), back(tab, ct)))
    return tab, net


def as_device(tab, dtype):
    """The tables as device arrays: integer tables as int32, floating ones in the run's dtype."""
    return jax.tree_util.tree_map(
        lambda t: jnp.asarray(t, dtype=dtype if np.issubdtype(np.asarray(t).dtype, np.floating) else np.int32), tab)


def morton(points, bits=10):
    """A space-filling-curve order (Morton) of points in their bounding box."""
    p = np.asarray(points, np.float64)
    lo, hi = p.min(0), p.max(0)
    q = np.minimum(((p - lo) / np.maximum(hi - lo, 1e-300) * (1 << bits)).astype(np.uint64), (1 << bits) - 1)
    key = np.zeros(len(p), np.uint64)
    for b in range(bits):
        for a in range(3):
            key |= ((q[:, a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(3 * b + a)
    return np.argsort(key, kind="stable")


def node_order(kind, points, own, nei, n, seed=0):
    """A permutation ``new_position -> old_index`` of ``n`` nodes: ``native``, ``random``, ``morton`` or ``rcm``."""
    if kind == "native":
        return np.arange(n)
    if kind == "random":
        return np.random.default_rng(seed).permutation(n)
    if kind == "morton":
        return morton(points)
    if kind == "rcm":
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        keep = nei < n
        a = coo_matrix((np.ones(int(keep.sum())), (own[keep], nei[keep])), shape=(n, n)).tocsr()
        return np.asarray(reverse_cuthill_mckee((a + a.T).tocsr(), symmetric_mode=True))
    raise ValueError(kind)
