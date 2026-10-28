"""The table evaluator's rounding against a long-double evaluation of the same compiled table: the measurement its
declared rounding rests on. Not collected by pytest.

    JAX_PLATFORMS=cpu uv run --no-sync python tests/differential/table_accuracy.py OUT.jsonl FIRST LAST [SETTING=N ...]

Over drawn tables (seeds ``FIRST`` to ``LAST``; one to four axes of uniform, log-uniform, graded, random, zero-holding
and far-scaled knots, an edge and a source range sometimes, a scalar, 3-vector or 2 x 2 value, or a 2 x 2 positive
definite one under ``cholesky``; ``linear`` and ``cubic``, either side, extrapolated or not), each read compiled in
float64 at 240 queries (inside, outside, at every knot, edge and range end and the numbers beside them, and the
numbers a search orders specially), against the same compiled coefficients evaluated in ``numpy.longdouble`` (each
axis's weights and their derivatives by powers of ``t``, the corners summed, then the exponential or ``L L^T``), where
the read is accepted and in range. A long double holds 64 bits of mantissa on x86-64 Linux, where it was measured.
Each ``SETTING=N`` is a table setting the reads take (``gathered_at_once=0``).

One JSON line per table: the largest value difference in float64 roundings of the read's largest value, and per axis
the largest forward-derivative difference in roundings of that value over the axis's smallest cell; a ``cholesky``
table's line also holds the largest magnitude of its stored factor form, whose product ``L L^T`` multiplies the
factor's derivative by the factor. ``loglog`` tables and one-point ranges are left out (the exponential of a
logarithm's rounding, and a derivative that is NaN by design). The summary prints each kind's tail."""

from __future__ import annotations

import collections
import dataclasses
import itertools
import json
import sys
from typing import Any

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mime.core.tables import Axis, ConstantExtrapolation, Table, TableInvalid, evaluate  # noqa: E402

LONG = np.longdouble
EPS = float(np.finfo(np.float64).eps)


def drawn_axis(rng: Any, name: str, kind: str, positive: bool) -> Axis | None:
    count = int(rng.integers(2, 10))
    edges: tuple[float, ...] = ()
    source = None
    if kind == "uniform":
        x = (rng.uniform(0.1, 3) if positive else rng.uniform(-3, 3)) + rng.uniform(0.05, 1.0) * np.arange(count)
    elif kind == "log-uniform":
        x = rng.uniform(0.01, 2.0) * rng.uniform(1.2, 3.0) ** np.arange(count)
    elif kind == "graded":
        start = rng.uniform(0.05, 2) if positive else rng.uniform(-2, 2)
        x = start + np.concatenate([[0], np.cumsum(0.1 * rng.uniform(1.0, 1.8) ** np.arange(count - 1))])
    elif kind == "random":
        start = rng.uniform(0.05, 2) if positive else rng.uniform(-2, 2)
        x = start + np.concatenate([[0], np.cumsum(rng.uniform(0.02, 1.0, count - 1))])
    elif kind == "zero":
        x = np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 1.0, count - 1))])
        if rng.random() < 0.5:
            x = x - x[min(len(x) - 1, 1)]
    else:  # far-scaled
        x = 10.0 ** rng.choice([-12, -6, 6, 12]) * (1 + np.cumsum(rng.uniform(0.05, 1.0, count)))
    knots = [float(v) for v in x]
    if len(set(knots)) < 2:
        return None
    if rng.random() < 0.25 and len(knots) >= 4:  # an edge: an interior knot given twice
        k = int(rng.integers(1, len(knots) - 1))
        knots = knots[:k + 1] + [knots[k]] + knots[k + 1:]
        edges = (knots[k],)
    if rng.random() < 0.25:
        low, high = knots[0], knots[-1]
        ends = sorted(rng.uniform(low, high, 2))
        if rng.random() < 0.3:
            ends = [ends[0], ends[0]]
            if ends[0] in edges:
                ends = [ends[0] + 1e-3 * (high - low)] * 2 if ends[0] + 1e-3 * (high - low) < high else ends
        source = (float(ends[0]), float(ends[1]))
    try:
        return Axis(name, "u", tuple(knots), edges=edges, source_range=source)
    except TableInvalid:
        return None


def drawn(seed: int) -> Table | None:
    """The table of ``seed``, as the measurement drew it (``loglog`` among them, left out by the caller)."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        count = int(rng.integers(1, 5))
        scheme = str(rng.choice(["linear", "cubic", "loglog"], p=[0.35, 0.45, 0.2]))
        structure = "cholesky" if scheme != "loglog" and rng.random() < 0.2 else "plain"
        kinds = ["uniform", "log-uniform", "graded", "random", "far"] + ([] if scheme == "loglog" else ["zero"])
        axes = []
        for a in range(count):
            axis = drawn_axis(rng, f"a{a}", str(rng.choice(kinds)), scheme == "loglog")
            if axis is None:
                break
            axes.append(axis)
        if len(axes) != count or np.prod([len(a.x) for a in axes]) > 60000:
            continue
        knots = tuple(len(a.x) for a in axes)
        if structure == "cholesky":
            factor = rng.normal(size=(*knots, 2, 2))
            y = factor @ np.swapaxes(factor, -1, -2) + 0.5 * np.eye(2) * rng.uniform(0.5, 3)
        else:
            shape = [(), (3,), (2, 2)][int(rng.integers(0, 3))]
            y = rng.normal(size=(*knots, *shape)) * 10.0 ** rng.uniform(-2, 2)
            if scheme == "loglog":
                y = np.exp(y / max(1e-9, np.abs(y).max()) * 2)
            elif rng.random() < 0.3:
                y = y + 1000.0 * rng.choice([-1, 1])  # a large offset, a small variation
        extrapolation = ConstantExtrapolation("the end value") if rng.random() < 0.5 else None
        side = str(rng.choice(["left", "right"]))
        try:
            return Table(tuple(axes), y, scheme=scheme, structure=structure, side=side,  # type: ignore[arg-type]
                         extrapolation=extrapolation)
        except TableInvalid:
            continue
    return None


def queries(table: Table, rng: Any, count: int = 240) -> list[np.ndarray]:
    columns = []
    for axis in table.axes:
        x = np.array(axis.x)
        low, high = axis.readable_range
        span = high - low if high > low else x[-1] - x[0]
        special = np.array([*x, *axis.edges, low, high])
        beside = np.concatenate([special, np.nextafter(special, np.inf), np.nextafter(special, -np.inf),
                                 np.nextafter(np.nextafter(special, np.inf), np.inf)])
        pool = [rng.uniform(low, high, count // 3), rng.uniform(x[0] - 0.3 * span, x[-1] + 0.3 * span, count // 6),
                rng.choice(beside, count // 3),
                np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, np.finfo(float).tiny, -np.finfo(float).tiny,
                          np.finfo(float).max, -np.finfo(float).max, np.finfo(float).eps])]
        column = np.concatenate([np.asarray(p, dtype=np.float64) for p in pool])
        columns.append(column[rng.permutation(len(column))])
    shortest = min(len(c) for c in columns)
    return [c[:shortest] for c in columns]


def truth(table: Table, columns: list[np.ndarray], along: int | None = None) -> np.ndarray:
    """The compiled table at the queries in long double, or its derivative along axis ``along``."""
    arrays = table.arrays(np.float64)
    count, axes = len(columns[0]), len(columns)
    values = np.asarray(arrays.values).astype(LONG)
    values = values.reshape(*values.shape[:axes], -1)
    starts, weights, slopes = [], [], []
    for axis, q in zip(arrays.axes, columns, strict=True):
        index = np.searchsorted(np.asarray(axis.breaks), q, side=arrays.side)
        index = np.where(np.isnan(q), len(np.asarray(axis.breaks)), index)
        coefficients = np.asarray(axis.coefficients).astype(LONG)[index]
        width = np.asarray(axis.width).astype(LONG)[index]
        with np.errstate(invalid="ignore"):  # an infinite width outside the range: no weight is read there
            t = (q.astype(LONG) - np.asarray(axis.origin).astype(LONG)[index]) / width
        weight, slope = np.zeros((count, coefficients.shape[2]), LONG), np.zeros((count, coefficients.shape[2]), LONG)
        for power in range(coefficients.shape[1]):
            weight += coefficients[:, power, :] * (t ** power)[:, None]
            if power >= 1:
                slope += coefficients[:, power, :] * (power * t ** (power - 1))[:, None]
        starts.append(np.asarray(axis.start)[index])
        weights.append(weight)
        slopes.append(slope / width[:, None])
    held, held_slope = np.zeros((count, values.shape[-1]), LONG), np.zeros((count, values.shape[-1]), LONG)
    for corner in itertools.product(*(range(w.shape[1]) for w in weights)):
        value = values[tuple(starts[a] + corner[a] for a in range(axes))]
        held += np.prod([weights[a][:, corner[a]] for a in range(axes)], axis=0)[:, None] * value
        if along is not None:
            held_slope += np.prod([(slopes if a == along else weights)[a][:, corner[a]] for a in range(axes)],
                                  axis=0)[:, None] * value
    if arrays.structure == "cholesky":
        size = arrays.value_shape[0]
        factor, factor_slope = np.zeros((count, size, size), LONG), np.zeros((count, size, size), LONG)
        for i in range(size):
            factor[:, i, i] = np.exp(held[:, i])
            factor_slope[:, i, i] = factor[:, i, i] * held_slope[:, i]
        rows, cols = np.tril_indices(size, -1)
        factor[:, rows, cols], factor_slope[:, rows, cols] = held[:, size:], held_slope[:, size:]
        held = np.einsum("nik,njk->nij", factor, factor).reshape(count, -1)
        held_slope = (np.einsum("nik,njk->nij", factor_slope, factor)
                      + np.einsum("nik,njk->nij", factor, factor_slope)).reshape(count, -1)
    return held_slope if along is not None else held


@jax.jit
def read(arrays: Any, columns: list[Any]) -> tuple[Any, list[Any]]:
    def value(*cs: Any) -> Any:
        return evaluate(arrays, *cs, owner="toy.table").value

    found = evaluate(arrays, *columns, owner="toy.table")
    derivatives = [jax.jvp(value, tuple(columns), tuple(jax.numpy.ones_like(c) if b == a else jax.numpy.zeros_like(c)
                                                          for b, c in enumerate(columns)))[1]
                   for a in range(len(columns))]
    return found, derivatives


def measured(seed: int, settings: dict[str, int]) -> dict[str, Any] | None:
    table = drawn(seed)
    if table is None or table.scheme == "loglog" or any(a.readable_range[0] == a.readable_range[1]
                                                          for a in table.axes):
        return None
    table = dataclasses.replace(table, **settings)
    columns = queries(table, np.random.default_rng(10_000 + seed))
    found, derivatives = read(table.arrays(np.float64), [jax.numpy.asarray(c) for c in columns])
    count = len(columns[0])
    accepted = np.asarray(found.accepted) & np.asarray(found.in_range)
    held = [np.where(np.abs(c) < np.finfo(float).tiny, 0.0, c) for c in columns]  # a compiled read flushes them
    expected = truth(table, held)
    ok = accepted & np.isfinite(expected).all(axis=1)
    largest = float(np.abs(expected[ok]).max()) if ok.any() else 0.0
    if largest == 0.0:
        return None
    value = np.asarray(found.value).reshape(count, -1)
    row: dict[str, Any] = {"seed": seed, "scheme": table.scheme, "structure": table.structure,
                           "axes": len(table.axes), "value": float(np.abs(value[ok] - expected[ok]).max() / EPS
                                                                    / largest), "derivatives": []}
    if table.structure == "cholesky":
        row["factor"] = float(np.abs(np.asarray(table.arrays(np.float64).values)).max())
    for a, axis in enumerate(table.axes):
        slope = truth(table, held, along=a)
        good = ok & np.isfinite(slope).all(axis=1)
        cell = float(np.min(np.diff(np.unique(axis.x))))
        difference = np.abs(np.asarray(derivatives[a]).reshape(count, -1)[good] - slope[good])
        row["derivatives"].append(float(difference.max() / EPS / (largest / cell)) if good.any() else None)
    return row


def summary(rows: list[dict[str, Any]]) -> None:
    kinds: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    scaled: list[float] = []
    for row in rows:
        derivatives = [d for d in row["derivatives"] if d is not None]
        kinds[row["scheme"], row["structure"]] += derivatives
        if row["structure"] == "cholesky":
            scaled += [d / (1 + 2 * row["factor"]) for d in derivatives]
    print(f"{len(rows)} tables; values within {max(r['value'] for r in rows):.1f} roundings of the read's largest")
    for (scheme, structure), found in sorted(kinds.items()):
        found.sort()
        print(f"  {scheme} {structure}: {len(found)} derivatives, median {found[len(found) // 2]:.2f}, 99th "
              f"percentile {found[int(0.99 * len(found))]:.1f}, largest {found[-1]:.1f} roundings of the read's "
              f"largest value over the axis's smallest cell")
    if scaled:
        print(f"  cholesky, over (1 + 2 x the factor form's largest magnitude) times that: largest {max(scaled):.1f}")


def main() -> None:
    out, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    settings = {name: int(value) for name, value in (item.split("=") for item in sys.argv[4:])}
    rows = []
    with open(out, "a", encoding="utf-8") as held:
        for seed in range(first, last):
            row = measured(seed, settings)
            if row is not None:
                rows.append(row)
                held.write(json.dumps(row) + "\n")
                held.flush()
    summary(rows)


if __name__ == "__main__":
    main()
