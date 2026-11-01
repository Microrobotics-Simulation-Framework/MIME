"""The table evaluator's profile by operations: each configuration lowered and compiled, never run, so that a shared
machine's load does not reach it. Not collected by pytest.

    JAX_PLATFORMS=cpu uv run --no-sync python tests/differential/table_profile.py OUT.jsonl [--only wall|pairs|generic]
    JAX_PLATFORMS=cpu MIME-port041/.venv/bin/python -I tests/differential/table_profile.py OUT.jsonl --port

The read before this one is profiled by the same script, run against that evaluator's own source: its tables are
built through each tree's ``Table.arrays``, and a setting the evaluator does not have is left out of its
configurations, so that both columns of a comparison come from one harness:

    git archive origin/v0.3 src | tar -x -C BEFORE
    JAX_PLATFORMS=cpu PYTHONPATH=BEFORE/src uv run --no-sync python tests/differential/table_profile.py OUT.jsonl \
        --label before

Per configuration (scheme and structure, one to four axes, float32 and float64, batches of 1, 1,000 and 1,000,000
queries unless ``--batches`` says otherwise; the forward read, ``jvp``, ``vjp``, and per-query ``jacfwd`` and
``jacrev`` by ``vmap``), one JSON line:

- the jaxpr's equations by primitive, counted through every sub-jaxpr, a loop's body once;
- the optimised HLO's kernels, gathers and gathered elements, dynamic slices, scatters, while loops and elementwise
  operations, each while body times its known trip count (``hlo``);
- XLA's ``cost_analysis()``, flops and bytes accessed, which counts a while body once whatever its trip count, and
  counts a gather's whole operand among its bytes;
- ``memory_analysis()``: argument, output and temporary bytes; for ``vjp``, the bytes reverse mode keeps between its
  passes (the leaves of the pullback, by shape).

The tables are synthetic: 16 knots per axis (uniform or graded) and a scalar value, or a 3 x 3 positive definite one
for ``cholesky``; the wall table, 40 x 40 x 64 x 128 knots on the 0.2 grid's own knots (``references/port_table.npz``)
with a 3 x 3 value, its values a ``ShapeDtypeStruct``, never allocated; and the MRS's form, every pair of 64 or 3,100
points, the radii a column and a row. Stored polynomials are profiled on one to three axes, and on four linear ones.
``--port`` profiles the 0.2 lookup the same way on the wall table, in the port's environment
(``table_budget.baseline_lookup``), importing nothing of MIME's: its float32 lookup without x64, as it ran."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from table_budget import baseline_lookup  # noqa: E402

REFERENCE = Path(__file__).parent / "references" / "port_table.npz"
OWNER = "toy.table"
BATCHES = (1, 1000, 1_000_000)
DTYPES = (np.float32, np.float64)
_SHAPE = re.compile(r"^(?P<dtype>[a-z0-9]+)\[(?P<dims>[0-9,]*)\]")
_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s*=\s*"
                          r"(?P<shape>\([^()]*\)|[a-z0-9]+\[[0-9,]*\](?:\{[^}]*\})?)\s+(?P<op>[a-z\-]+)\((?P<rest>.*)$")
_HEADER = re.compile(r"^(?:ENTRY\s+)?%(?P<name>[\w.\-]+)\s*\(")
_ARITHMETIC = {"add", "subtract", "multiply", "divide", "maximum", "minimum", "compare", "select", "negate", "abs",
               "floor", "ceil", "convert", "and", "or", "xor", "not", "clamp", "shift-left", "shift-right-logical",
               "shift-right-arithmetic", "remainder", "round-nearest-even", "round-nearest-afz", "sign", "is-finite",
               "atan2"}
_TRANSCENDENTAL = {"exponential", "log", "power", "sqrt", "rsqrt", "tanh", "exponential-minus-one", "log-plus-one",
                   "sine", "cosine", "logistic", "cbrt"}
_NOT_KERNELS = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast", "after-all", "partition-id",
                "replica-id"}


def _elements(shape: str) -> int:
    found = _SHAPE.match(shape)
    dims = [int(d) for d in found.group("dims").split(",") if d] if found else []
    return int(np.prod(dims, dtype=np.int64)) if found else 0


def _computations(text: str) -> tuple[dict[str, list[dict[str, Any]]], str]:
    held: dict[str, list[dict[str, Any]]] = {}
    entry, current = "", None
    for line in text.splitlines():
        header = _HEADER.match(line)
        if header and line.rstrip().endswith("{"):
            current = header.group("name")
            held[current] = []
            entry = current if line.startswith("ENTRY") else entry
        elif line.startswith("}"):
            current = None
        elif current is not None and (found := _INSTRUCTION.match(line)):
            rest = found.group("rest")
            trip = re.search(r'"known_trip_count":\{"n":"(\d+)"\}', rest)
            held[current].append({"name": found.group("name"), "shape": found.group("shape"), "op": found.group("op"),
                                  "operands": re.findall(r"%([\w.\-]+)", rest.split("),")[0]),
                                  "called": dict(re.findall(r"\b(calls|body|condition|to_apply)=%([\w.\-]+)", rest)),
                                  "trip": int(trip.group(1)) if trip else 1})
    return held, entry


def hlo_counts(text: str) -> dict[str, int]:
    """The optimised HLO's counts, each while body (and condition) times its known trip count: kernels (a fusion,
    or an operation outside one), gathers, dynamic slices and scatters with their elements, elementwise and
    transcendental operations by output element, and the elements reductions read."""
    held, entry = _computations(text)
    sizes = {(name, i["name"]): _elements(i["shape"]) for name, items in held.items() for i in items}
    counts: Counter[str] = Counter()

    def walk(name: str, times: int, fused: bool) -> None:
        for instruction in held.get(name, []):
            op, called, shape = instruction["op"], instruction["called"], instruction["shape"]
            if op == "while":
                counts["while_loops"] += times
                counts["loop_iterations"] += times * instruction["trip"]
                walk(called["body"], times * instruction["trip"], fused)
                walk(called["condition"], times * (instruction["trip"] + 1), fused)
                continue
            if op == "fusion":
                counts["fusions"] += times
                counts["kernels"] += times
                walk(called["calls"], times, True)
                continue
            if op == "call" and "to_apply" in called:
                walk(called["to_apply"], times, fused)
            if not fused and op not in _NOT_KERNELS:
                counts["kernels"] += times
            if op in _ARITHMETIC:
                counts["elementwise"] += times * _elements(shape)
            elif op in _TRANSCENDENTAL:
                counts["transcendental"] += times * _elements(shape)
            elif op == "reduce":
                counts["reduced_elements"] += times * sizes.get((name, instruction["operands"][0]), 0)
            for kind, key, elements in (("gather", "gathers", "gathered_elements"),
                                        ("dynamic-slice", "dynamic_slices", "dynamic_sliced_elements"),
                                        ("scatter", "scatters", "scattered_elements")):
                if op == kind:
                    counts[key] += times
                    counts[elements] += times * _elements(shape)

    walk(entry, 1, False)
    return dict(counts)


def _jaxpr_counts(closed: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    import jax.extend.core as core

    def walk(jaxpr: Any) -> None:
        for equation in jaxpr.eqns:
            counts[equation.primitive.name] += 1
            for value in equation.params.values():
                for item in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(item, (core.ClosedJaxpr, core.Jaxpr)):
                        walk(item.jaxpr if isinstance(item, core.ClosedJaxpr) else item)

    walk(closed.jaxpr)
    return counts


def profile(function: Any, arguments: tuple, queries: int) -> dict[str, Any]:
    equations = _jaxpr_counts(jax.make_jaxpr(function)(*arguments))
    compiled = jax.jit(function).lower(*arguments).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    memory = compiled.memory_analysis()
    return {"queries": queries, "jaxpr_equations": int(sum(equations.values())), "jaxpr": dict(equations),
            "hlo": hlo_counts(compiled.as_text()), "xla_flops": float(cost.get("flops", 0.0)),
            "xla_bytes": float(cost.get("bytes accessed", 0.0)), "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes, "temporary_bytes": memory.temp_size_in_bytes}


def residual_bytes(value: Any, arguments: tuple) -> int:
    """The bytes reverse mode keeps between its passes, the table's own arrays among them where they are leaves."""
    shapes = jax.eval_shape(lambda *a: jax.vjp(lambda q: value(*a[:-1], q), a[-1])[1], *arguments)
    return int(sum(np.prod(leaf.shape, dtype=np.int64) * np.dtype(leaf.dtype).itemsize
                   for leaf in jax.tree_util.tree_leaves(shapes)))


def modes(value: Any, read: Any) -> dict[str, tuple[Any, Any]]:
    """Each mode's function of (table, queries, extra), and the extra argument it takes for a batch and dtype: the
    forward read whole (``read``), every derivative of its value."""
    def single(table, point):
        return value(table, point[None])[0]

    return {
        "forward": (read, None),
        "jvp": (lambda table, q, dq: jax.jvp(lambda x: value(table, x), (q,), (dq,)), "queries"),
        "vjp": (lambda table, q, ct: jax.vjp(lambda x: value(table, x), q)[1](ct)[0], "cotangent"),
        "jacfwd": (lambda table, q: jax.vmap(jax.jacfwd(lambda p: single(table, p)))(q), None),
        "jacrev": (lambda table, q: jax.vmap(jax.jacrev(lambda p: single(table, p)))(q), None),
    }


def run(value: Any, table: Any, axes: int, value_shape: tuple[int, ...], label: dict, out: Any,
        x64_float32: bool = True, read: Any = None, batches: tuple[int, ...] = BATCHES) -> None:
    for dtype in DTYPES if label.get("dtype") is None else (label["dtype"],):
        jax.config.update("jax_enable_x64", x64_float32 or dtype == np.float64)
        held = table(dtype)
        for batch in batches:
            q = jax.ShapeDtypeStruct((batch, axes), np.dtype(dtype))
            for mode, (function, extra) in modes(value, read or value).items():
                arguments = [held, q]
                if extra == "queries":
                    arguments.append(q)
                elif extra == "cotangent":
                    arguments.append(jax.ShapeDtypeStruct((batch, *value_shape), np.dtype(dtype)))
                started = time.perf_counter()
                found = profile(function, tuple(arguments), batch)
                if mode == "vjp":
                    found["residual_bytes"] = residual_bytes(value, (held, q))
                found.update({**label, "dtype": np.dtype(dtype).name}, batch=batch, mode=mode,
                             seconds=round(time.perf_counter() - started, 2))
                out.write(json.dumps(found) + "\n")
                out.flush()


def port(out: Any) -> None:
    reference = np.load(REFERENCE)
    knots = (reference["full_rho"], reference["full_angle"], reference["full_offset"])

    def value(table, q):
        return baseline_lookup(*table, q[:, 0], q[:, 1], q[:, 2], q[:, 3])

    def table(dtype):
        return (*(jax.ShapeDtypeStruct((len(k),), np.dtype(dtype)) for k in knots),
                jax.ShapeDtypeStruct((40, 40, 64, 128, 3, 3), np.dtype(dtype)))

    # the 0.2 float32 twin ran without x64: under it, its float64 zeros would promote the lookup to float64
    run(value, table, 4, (3, 3), {"label": "0.2", "table": "wall", "scheme": "linear", "axes": 4,
                                  "jax": jax.__version__}, out, x64_float32=False)


def mime(out: Any, only: str | None, label: str, batches: tuple[int, ...]) -> None:
    import dataclasses

    from mime.core.tables import Axis, Table, evaluate

    settings_held = {field.name for field in dataclasses.fields(Table)}

    def read(table, q):
        return tuple(evaluate(table, *(q[:, a] for a in range(q.shape[-1])), owner=OWNER))

    def value(table, q):
        return read(table, q)[0]

    def structs(arrays):
        return jax.tree_util.tree_map(lambda leaf: leaf if isinstance(leaf, jax.ShapeDtypeStruct)
                                      else jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype), arrays)

    def held(settings: dict) -> bool:
        """Whether this tree's evaluator has every one of ``settings``."""
        return set(settings) <= settings_held

    reference = np.load(REFERENCE)
    grid = (reference["full_rho"], reference["full_rho"], reference["full_angle"], reference["full_offset"])

    def wall(scheme, **settings):
        """The wall table as this tree compiles it: each axis compiled as a table of its own, the four put together,
        the values a ``ShapeDtypeStruct``, with ``settings``."""
        def table(dtype):
            schemes = scheme if isinstance(scheme, tuple) else (scheme,) * 4
            axes = tuple(Table((Axis(n, "1", tuple(float(v) for v in k)),), np.zeros(len(k)), scheme=s)
                         .arrays(dtype).axes[0] for n, k, s in zip(("rho", "rho0", "angle", "offset"), grid, schemes,
                                                                  strict=True))
            shell = Table((Axis("a", "1", (0.0, 1.0)),), np.zeros((2, 9)), scheme=schemes[0],
                          **{k: v for k, v in settings.items() if k != "float32_values"})
            stored = np.float32 if settings.get("float32_values") else dtype
            return structs(dataclasses.replace(
                shell.arrays(dtype), axes=axes, scheme=scheme, value_shape=(3, 3),
                values=jax.ShapeDtypeStruct((*(len(k) for k in grid), 9), np.dtype(stored))))
        return table

    def per_axis() -> bool:
        """Whether this tree's evaluator takes a scheme per axis."""
        try:
            Table((Axis("a", "1", (0.0, 1.0)), Axis("b", "1", (0.0, 1.0))), np.zeros((2, 2)), scheme=("cubic", "linear"))
        except ValueError:
            return False
        return True

    def label_of(**fields) -> dict:
        return {"label": label, **fields}

    if only in (None, "wall"):
        for scheme in ("linear", "cubic", ("cubic", "cubic", "linear", "linear")):
            name = scheme if isinstance(scheme, str) else "cubic radii"
            if not isinstance(scheme, str) and not per_axis():
                continue
            run(value, wall(scheme), 4, (3, 3), label_of(table="wall", scheme=name, axes=4), out, read=read,
                batches=batches)
            if not isinstance(scheme, str):
                continue
            for setting, flag, suffix in (("recomputed_in_reverse", True, "recomputed"),
                                          ("recomputed_in_reverse", False, "windows kept"),
                                          ("gathered_at_once", 0, "one combination a gather")):
                if held({setting: flag}):
                    run(value, wall(scheme, **{setting: flag}), 4, (3, 3),
                        label_of(table="wall", scheme=f"{scheme}, {suffix}", axes=4), out, read=read, batches=batches)
            if held({"float32_values": True}):
                run(value, wall(scheme, float32_values=True), 4, (3, 3),
                    label_of(table="wall", scheme=f"{scheme}, float32 values", axes=4, dtype=np.float64), out,
                    read=read, batches=batches)
    if only in (None, "pairs"):
        for scheme in ("linear", "cubic"):
            for dtype in DTYPES:
                for points in (64, 3100):
                    for shared in (True, False):
                        shapes = [(points, 1), (1, points), (points, points), (points, points)] if shared else \
                            [(points, points)] * 4
                        found = profile(lambda table, *cs: tuple(evaluate(table, *cs, owner=OWNER)),
                                        (wall(scheme)(dtype), *(jax.ShapeDtypeStruct(s, np.dtype(dtype))
                                                                for s in shapes)), points * points)
                        found.update(label_of(table="pairs", scheme=scheme, axes=4, dtype=np.dtype(dtype).name,
                                              batch=points * points, mode="forward", points=points,
                                              form="radii shared" if shared else "broadcast"))
                        out.write(json.dumps(found) + "\n")
                        out.flush()
    if only in (None, "generic"):
        for scheme, structure in (("linear", "plain"), ("loglog", "plain"), ("cubic", "plain"),
                                  ("linear", "cholesky"), ("cubic", "cholesky")):
            for count in (1, 2, 3, 4):
                for kind in ("uniform", "graded"):
                    positive = scheme == "loglog"
                    knots = (1.0 if positive else -2.0) + 0.25 * np.arange(16) if kind == "uniform" else \
                        (0.5 if positive else -1.0) + np.concatenate([[0.0], np.cumsum(0.05 * 1.15 ** np.arange(15))])
                    axes = tuple(Axis(f"a{i}", "m", tuple(float(v) for v in knots)) for i in range(count))
                    rng = np.random.default_rng(count)
                    if structure == "cholesky":
                        factor = rng.uniform(-0.3, 0.3, (*(16,) * count, 3, 3)) + 2.0 * np.eye(3)
                        y = factor @ np.swapaxes(factor, -1, -2)
                    else:
                        y = rng.uniform(0.5, 2.0, (16,) * count) if positive else rng.standard_normal((16,) * count)
                    # stored polynomials on four axes would hold 15**4 * 4**4 coefficients per component: not profiled
                    variants = [({}, "")]
                    if count < 4 or scheme == "linear":
                        variants.append(({"stored_polynomials": True}, ", stored"))
                    if count > 1:
                        variants.append(({"gathered_at_once": 0}, ", one combination a gather"))
                    for settings, suffix in variants:
                        if not held(settings):
                            continue
                        built = Table(axes, y, scheme=scheme, structure=structure, **settings)
                        run(value, lambda dtype, t=built: structs(t.arrays(dtype)), count, built.value_shape,
                            label_of(table="generic", scheme=scheme + suffix, structure=structure, axes=count,
                                     grid=kind), out, read=read, batches=batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--port", action="store_true", help="the 0.2 lookup, in the port's environment")
    parser.add_argument("--only", choices=["wall", "pairs", "generic"])
    parser.add_argument("--label", default="after", help="each line's label: the tree the profile reads")
    parser.add_argument("--batches", default=",".join(str(b) for b in BATCHES), help="the batches, comma-separated")
    options = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    with open(options.out, "a", encoding="utf-8") as out:
        if options.port:
            port(out)
        else:
            mime(out, options.only, options.label, tuple(int(b) for b in options.batches.split(",")))


if __name__ == "__main__":
    main()
