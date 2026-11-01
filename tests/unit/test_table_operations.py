"""The table evaluator's operation counts, read from the lowered and compiled read, never run: guards that hold a read
at the counts its profile measured (``tests/differential/table_profile.py``), each failed by the 7f read it replaced,
kept here as the reference (``reference_read``: a search per axis, every corner gathered in one gather and held, the
stencil's polynomial combined before it is evaluated).

- **No search** where an axis has a bucket index: the wall table's four axes and a material table's one.
- **Gathered elements per query**: the corners' values and a few numbers per axis, no more.
- **Elementwise operations per query** in the optimised program, each loop's body times its trip count, within a
  fiftieth of the measured count: one more operation where a window is summed fails it.
- **Temporary memory per query**: no corner's value held between its gather and its sum.
- **Reverse mode**: the same counts of the reverse derivative, and the bytes it keeps between its passes per query;
  recomputing the windows (the default) keeps under a quarter of what keeping them does, which either way of
  ignoring the setting fails.
- **Kernels at a step's batch**: a read of 1, 10 and 100 queries of a 4-axis cubic table launches no more kernels than
  measured, in each mode; the 7f read and this read gathering one combination at a time launch more.
- **Locating once per distinct coordinate**: a read over the pairs of points, its source and target radii given as
  a column and a row, gathers per pair only what each pair's own angle and offset select; the radii broadcast to
  every pair first fail it.

The wall table is its own shape, 40 x 40 x 64 x 128 knots on the 0.2 grid's knots (the committed slice's ``full_*``
arrays) with a 3 x 3 value, its values a ``ShapeDtypeStruct``: lowering allocates none of its 900 MB. At its 4,096
queries a read gathers one combination of the leading axes' knots at a time, the loop the guards hold."""

import re
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.core.tables import Axis, Table, TableArrays, evaluate
from mime.core.tables._compile import _compiled_axis

OWNER = "toy.wall_table"
GRID = np.load(Path(__file__).parents[1] / "differential" / "references" / "port_table.npz")
WALL = (GRID["full_rho"], GRID["full_rho"], GRID["full_angle"], GRID["full_offset"])
QUERIES = 4096


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def wall(scheme: str, dtype=np.float32, **settings) -> TableArrays:
    """The wall table compiled as ``Table.arrays`` compiles it, its values a ``ShapeDtypeStruct``, with the table's
    default settings or ``settings``."""
    description = SimpleNamespace(scheme=scheme, side="right", extrapolation=None, stored_polynomials=False)
    axes = tuple(_compiled_axis(Axis(name, "1", tuple(float(v) for v in knots)), scheme, description, np.dtype(dtype))
                 for name, knots in zip(("rho", "rho0", "angle", "offset"), WALL, strict=True))
    arrays = TableArrays(axes, jax.ShapeDtypeStruct((*(len(k) for k in WALL), 9), np.dtype(dtype)), scheme, "plain",
                         "right", (3, 3), False, False, **settings)
    return jax.tree_util.tree_map(lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), leaf.dtype)
                                  if not isinstance(leaf, jax.ShapeDtypeStruct) else leaf, arrays)


def material() -> TableArrays:
    """A material table as OpenMaterica's records give one: twelve knots of temperature, a scalar value, float64."""
    table = Table((Axis("temperature", "K", tuple(263.0 + 10.0 * k for k in range(12))),), np.linspace(1.0, 2.0, 12))
    return table.arrays(np.float64)


def columns(count: int, dtype, shapes=None) -> list:
    return [jax.ShapeDtypeStruct(shape, np.dtype(dtype)) for shape in (shapes or [(QUERIES,)] * count)]


# -- counting ------------------------------------------------------------------------------------------------------

_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*(?P<shape>\([^()]*\)|[a-z0-9]+\[[0-9,]*\]\S*)\s+"
                          r"(?P<op>[a-z\-]+)\((?P<rest>.*)$")
_HEADER = re.compile(r"^(?:ENTRY\s+)?%(?P<name>[\w.\-]+)\s*\(.*\{\s*$")
#: The optimised program's elementwise operations: each output element one operation.
ARITHMETIC = frozenset({"add", "subtract", "multiply", "divide", "maximum", "minimum", "compare", "select", "negate",
                        "abs", "floor", "convert", "and", "or", "not", "exponential", "log", "is-finite", "clamp"})
#: The optimised program's instructions that are no kernel: a fusion is one kernel, and so is each other instruction
#: outside one but these.
NOT_KERNELS = frozenset({"parameter", "constant", "tuple", "get-tuple-element", "bitcast", "after-all", "partition-id",
                         "replica-id"})


def _elements(shape: str) -> int:
    found = re.match(r"^[a-z0-9]+\[([0-9,]*)\]", shape)
    return int(np.prod([int(d) for d in found.group(1).split(",") if d], dtype=np.int64)) if found else 0


def counts(function, arguments, queries: int) -> dict:
    """The compiled program's while loops (their trip counts), its kernels (each loop's body times its trip count,
    its condition once more), and per query its gathered elements, elementwise operations (a loop's body times its
    trip count) and temporary bytes."""
    compiled = jax.jit(function).lower(*arguments).compile()
    held: dict[str, list[tuple[str, str, str]]] = {}
    entry, current = "", None
    for line in compiled.as_text().splitlines():
        if header := _HEADER.match(line):
            current = header.group("name")
            held[current] = []
            entry = current if line.startswith("ENTRY") else entry
        elif current is not None and (instruction := _INSTRUCTION.match(line)):
            held[current].append((instruction.group("shape"), instruction.group("op"), instruction.group("rest")))
    found = {"loops": [], "kernels": 0, "gathered": 0.0, "elementwise": 0.0}

    def walk(name: str, times: int, fused: bool = False) -> None:
        for shape, op, rest in held.get(name, []):
            called = dict(re.findall(r"\b(calls|body|condition|to_apply)=%([\w.\-]+)", rest))
            if op == "call":
                found["kernels"] += 0 if fused else times
                walk(called["to_apply"], times, fused)
            elif op == "while":
                trip = int(re.search(r'"known_trip_count":\{"n":"(\d+)"\}', rest).group(1))
                found["loops"].append(trip)
                walk(called["body"], times * trip, fused)
                walk(called["condition"], times * (trip + 1), fused)
            elif "calls" in called:
                found["kernels"] += 0 if fused else times
                walk(called["calls"], times, True)
            else:
                found["kernels"] += 0 if fused or op in NOT_KERNELS else times
                if op == "gather":
                    found["gathered"] += times * _elements(shape) / queries
                elif op in ARITHMETIC:
                    found["elementwise"] += times * _elements(shape) / queries

    walk(entry, 1)
    found["temporary"] = compiled.memory_analysis().temp_size_in_bytes / queries
    return found


def kept(function, arguments, queries: int) -> float:
    """The bytes per query reverse mode keeps between its passes: the leaves of ``function``'s pullback that hold a
    dimension per query, which leaves out the table's own arrays."""
    leaves = jax.tree_util.tree_leaves(jax.eval_shape(function, *arguments))
    return sum(int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize for leaf in leaves
               if queries in leaf.shape) / queries


# -- the reference: the 7f read ------------------------------------------------------------------------------------

def _along_last(held, coefficients, t):
    """7f's interpolation along the last stencil dimension of ``held``: the polynomial's coefficients from that
    stencil's values, then the polynomial at ``t`` by Horner's rule."""
    def across(column):
        return column.reshape(column.shape[0], *([1] * (held.ndim - 2)))

    powers = []
    for power in range(coefficients.shape[1]):
        term = across(coefficients[:, power, 0]) * held[..., 0, :]
        for knot in range(1, coefficients.shape[2]):
            term = term + across(coefficients[:, power, knot]) * held[..., knot, :]
        powers.append(term)
    result = powers[-1]
    for power in range(len(powers) - 2, -1, -1):
        result = powers[power] + across(t) * result
    return result


def reference_read(arrays: TableArrays, *columns):
    """The 7f read, the guards' control, on the same compiled table: each query located by a search on every axis,
    after the queries are broadcast to every pair; every corner's index computed and every corner gathered in one
    gather; each axis's stencil combined with its polynomial's coefficients before the polynomial is evaluated; the
    flags gathered per axis."""
    count = len(arrays.axes)
    batch = jnp.broadcast_shapes(*(c.shape for c in columns))
    flat = [jnp.broadcast_to(c, batch).reshape(-1) for c in columns]
    located = []
    for axis, q in zip(arrays.axes, flat, strict=True):
        index = jnp.searchsorted(axis.breaks, q, side=arrays.side, method="scan")
        stencil = axis.start[index][:, None] + jnp.arange(axis.coefficients.shape[-1], dtype=axis.start.dtype)
        located.append((index, stencil, axis.coefficients[index], (q - axis.origin[index]) / axis.width[index]))
    values = arrays.values
    knots = values.shape[:count]
    strides = [int(np.prod(knots[a + 1:], dtype=np.int64)) for a in range(count)]
    corners = sum((stencil[(slice(None), *(slice(None) if b == a else None for b in range(count)))] * stride
                   for a, ((_, stencil, _, _), stride) in enumerate(zip(located, strides, strict=True))),
                  start=jnp.zeros((), dtype=jnp.int32))
    held = values.reshape(-1, values.shape[-1])[corners]
    for _, _, coefficients, t in reversed(located):
        held = _along_last(held, coefficients, t)
    in_range = jnp.all(jnp.stack([axis.inside[index] for axis, (index, *_) in zip(arrays.axes, located,
                                                                                 strict=True)]), axis=0)
    stopped = jnp.any(jnp.stack([axis.refused[index] for axis, (index, *_) in zip(arrays.axes, located,
                                                                                 strict=True)]), axis=0)
    accepted = ~stopped & (in_range | arrays.extrapolated) & jnp.all(jnp.stack([jnp.isfinite(q) for q in flat]), axis=0)
    return held.reshape(*batch, -1), in_range.reshape(batch), stopped.reshape(batch), accepted.reshape(batch)


def read(arrays: TableArrays, *columns):
    return tuple(evaluate(arrays, *columns, owner=OWNER))


def forward_derivative(reader):
    """The read's forward derivative along every axis at once, beside its value."""
    def derivative(arrays, *columns):
        return jax.jvp(lambda *cs: reader(arrays, *cs)[0], columns, tuple(jnp.ones_like(c) for c in columns))
    return derivative


def reverse_derivative(reader):
    """The read's reverse derivative along every axis, of the values weighted by a cotangent."""
    def derivative(arrays, cotangent, *columns):
        return jax.vjp(lambda *cs: reader(arrays, *cs)[0], *columns)[1](cotangent)
    return derivative


def pullback(reader):
    """What the read's reverse mode keeps between its passes: its pullback, whose leaves are its residuals."""
    def held(arrays, *columns):
        return jax.vjp(lambda *cs: reader(arrays, *cs)[0], *columns)[1]
    return held


# -- the guards ----------------------------------------------------------------------------------------------------

#: Each configuration's counts as this read's profile measured them (CPU, the locked JAX): the loops (the one over the
#: leading axes' window knots, never a search), the gathered elements per query, exactly, and bounds a fiftieth above
#: the measured elementwise operations and a quarter above the temporary bytes and the bytes kept for reverse mode per
#: query. A fiftieth: a change of a few operations in a hundred, one more multiply where a window is summed, fails it.
GUARDED = {
    "wall linear": {"loops": [8], "gathered": 197.0, "elementwise": 763.1, "temporary": 128.1},
    "wall cubic": {"loops": [64], "gathered": 2405.0, "elementwise": 7659.5, "temporary": 148.1},
    "wall cubic, forward derivative": {"loops": [64], "gathered": 2401.0, "elementwise": 19661.8, "temporary": 392.1},
    "wall linear, reverse derivative": {"loops": [8, 8], "gathered": 193.0, "elementwise": 956.2, "temporary": 316.2,
                                        "kept": 104.0},
    "wall cubic, reverse derivative": {"loops": [64, 64], "gathered": 2401.0, "elementwise": 11373.2,
                                       "temporary": 780.3, "kept": 328.0},
    "material": {"loops": [], "gathered": 13.0, "elementwise": 68.0, "temporary": 28.0},
}
CONFIGURATIONS = {
    "wall linear": (lambda **settings: wall("linear", **settings), 4, np.float32, None),
    "wall cubic": (lambda **settings: wall("cubic", **settings), 4, np.float32, None),
    "wall cubic, forward derivative": (lambda **settings: wall("cubic", **settings), 4, np.float32, "forward"),
    "wall linear, reverse derivative": (lambda **settings: wall("linear", **settings), 4, np.float32, "reverse"),
    "wall cubic, reverse derivative": (lambda **settings: wall("cubic", **settings), 4, np.float32, "reverse"),
    "material": (lambda: material(), 1, np.float64, None),
}
#: The share above a measured count a guard allows, by count.
ALLOWED = {"elementwise": 1.02, "temporary": 1.25, "kept": 1.25}


def exceeded(found: dict, guard: dict) -> list[str]:
    """Each count of ``found`` beyond its guard."""
    problems = [f"loops {found['loops']}, guarded {guard['loops']}"] if found["loops"] != guard["loops"] else []
    if found["gathered"] != guard["gathered"]:
        problems.append(f"{found['gathered']} elements gathered per query, guarded {guard['gathered']}")
    problems += [f"{found[key]:.1f} {key} per query, guarded at most {guard[key] * share:.1f}"
                 for key, share in ALLOWED.items() if key in guard and found[key] > guard[key] * share]
    return problems


def measured(name: str, reader, **settings) -> dict:
    """The counts of configuration ``name`` read by ``reader``, the wall table's settings ``settings``."""
    make, count, dtype, derivative = CONFIGURATIONS[name]
    arrays = make(**settings)
    queries = columns(count, dtype)
    if derivative == "forward":
        return counts(forward_derivative(reader), (arrays, *queries), QUERIES)
    if derivative == "reverse":
        cotangent = jax.eval_shape(lambda a, *cs: reader(a, *cs)[0], arrays, *queries)
        found = counts(reverse_derivative(reader), (arrays, cotangent, *queries), QUERIES)
        return {**found, "kept": kept(pullback(reader), (arrays, *queries), QUERIES)}
    return counts(reader, (arrays, *queries), QUERIES)


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_a_read_holds_its_measured_operation_counts(name):
    assert exceeded(measured(name, read), GUARDED[name]) == []


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_control_the_7f_read_fails_each_guard(name):
    found = exceeded(measured(name, reference_read), GUARDED[name])
    print(f"\n{name}, the 7f read: {found}")
    assert found


@pytest.mark.parametrize("scheme", ["linear", "cubic"])
def test_recomputed_in_reverse_keeps_a_few_numbers_per_axis_where_keeping_the_windows_keeps_every_corner(scheme):
    """Reverse mode on the wall table: recomputed (the default), it keeps under a quarter of the bytes per query that
    keeping the windows does (measured: 104 against 800 linear, 328 against 10,752 cubic) and fewer temporaries, so
    either way the setting were ignored fails it."""
    name = f"wall {scheme}, reverse derivative"
    recomputed, windows = (measured(name, read, recomputed_in_reverse=flag) for flag in (True, False))
    assert recomputed["kept"] * 4 <= windows["kept"] and recomputed["temporary"] < windows["temporary"], \
        (recomputed, windows)


#: The kernels of a read of few queries, a step's own: the 4-axis cubic table, float64, of 16 knots per axis, read at
#: 1, 10 and 100 queries, forward, forward derivative and reverse derivative, as measured (each loop's body times its
#: trip count, its condition once more). The 7f read launches 157, 163 and 210 at one query (191, 193 and 231 at 100),
#: and this read gathering one combination of the leading axes' knots at a time (``gathered_at_once=0``) 275, 345 and
#: 1,063 (358, 428 and 1,459).
KERNELS = {("forward", 1): 12, ("forward", 10): 36, ("forward", 100): 43, ("jvp", 1): 20, ("jvp", 10): 36,
           ("jvp", 100): 45, ("vjp", 1): 40, ("vjp", 10): 53, ("vjp", 100): 57}


def small_batch(mode: str, batch: int, reader, **settings) -> int:
    """The kernels of ``reader`` on the 4-axis cubic table at ``batch`` queries, in ``mode``."""
    knots = tuple(float(v) for v in -1.0 + np.concatenate([[0.0], np.cumsum(0.05 * 1.15 ** np.arange(15))]))
    table = Table(tuple(Axis(f"a{i}", "m", knots) for i in range(4)),
                  np.random.default_rng(4).standard_normal((16,) * 4), scheme="cubic", **settings)
    arrays, queries = table.arrays(np.float64), columns(4, np.float64, [(batch,)] * 4)
    if mode == "forward":
        return counts(reader, (arrays, *queries), batch)["kernels"]
    if mode == "jvp":
        return counts(forward_derivative(reader), (arrays, *queries), batch)["kernels"]
    cotangent = jax.eval_shape(lambda a, *cs: reader(a, *cs)[0], arrays, *queries)
    return counts(reverse_derivative(reader), (arrays, cotangent, *queries), batch)["kernels"]


@pytest.mark.parametrize(("mode", "batch"), list(KERNELS))
def test_a_read_of_few_queries_launches_its_measured_kernels(mode, batch):
    assert small_batch(mode, batch, read) <= KERNELS[mode, batch]


@pytest.mark.parametrize(("mode", "batch"), list(KERNELS))
def test_control_the_7f_read_and_one_combination_at_a_time_launch_more_kernels(mode, batch):
    assert small_batch(mode, batch, reference_read) > KERNELS[mode, batch]
    assert small_batch(mode, batch, read, gathered_at_once=0) > KERNELS[mode, batch]


def test_the_reference_is_the_7f_read_its_values_within_the_reads_rounding():
    """The control is a faithful copy, and the read it replaced is reproduced within the rounding its reordering
    allows: on a small table of three kinds of axis, the flags bit for bit, the values within 64 roundings of the
    read's largest, the derivative within 64 roundings of that value over the smallest cell (the scale of every
    derivative a cell's polynomial has)."""
    axes = (Axis("a", "m", (0.0, 0.3, 1.0, 1.6, 2.0)), Axis("b", "m", tuple(0.5 * k for k in range(6))),
            Axis("c", "Hz", tuple(0.1 * 2.0 ** k for k in range(5))))
    mesh = np.stack(np.meshgrid(*(np.asarray(axis.x) for axis in axes), indexing="ij"), axis=-1)
    table = Table(axes, np.sin(mesh.sum(axis=-1))[..., None] * np.asarray([1.0, 2.0, -3.0]), scheme="cubic")
    rng = np.random.default_rng(3)
    at = [jnp.asarray(rng.uniform(axis.x[0], axis.x[-1], 64)) for axis in axes]
    arrays = table.arrays(np.float64)
    ours, theirs = jax.jit(read)(arrays, *at), jax.jit(reference_read)(arrays, *at)
    assert all(np.array_equal(a, b) for a, b in zip(ours[1:], theirs[1:], strict=True))
    largest = float(np.max(np.abs(np.asarray(ours[0]))))
    assert np.max(np.abs(np.asarray(ours[0]) - np.asarray(theirs[0]))) <= 64 * np.finfo(float).eps * largest
    # the derivative along every axis at once, on its inputs' scale: the largest value over the smallest cell
    slope, reference_slope = (jax.jit(forward_derivative(reader))(arrays, *at)[1] for reader in (read, reference_read))
    scale = largest / min(float(np.min(np.diff(axis.x))) for axis in axes)
    assert np.max(np.abs(np.asarray(slope) - np.asarray(reference_slope))) <= 64 * np.finfo(float).eps * scale


#: The MRS's form at 64 points: per pair, the gathered elements (exactly) and elementwise operations (a fiftieth above)
#: measured with each point's radius given as a column and a row, which the same read with the radii broadcast to
#: every pair first exceeds.
PAIRS = 64
PAIRWISE = {"gathered": 2359.71875, "elementwise": 7360.1}


def pairwise(shared: bool) -> dict:
    shapes = [(PAIRS, 1), (1, PAIRS), (PAIRS, PAIRS), (PAIRS, PAIRS)] if shared else [(PAIRS, PAIRS)] * 4
    return counts(read, (wall("cubic"), *columns(4, np.float32, shapes)), PAIRS * PAIRS)


def test_a_read_over_pairs_locates_each_points_radius_once():
    found = pairwise(shared=True)
    assert found["gathered"] == PAIRWISE["gathered"], found
    assert found["elementwise"] <= PAIRWISE["elementwise"] * ALLOWED["elementwise"], found


def test_control_the_radii_broadcast_to_every_pair_first_fail_the_pairwise_guard():
    found = pairwise(shared=False)
    assert found["gathered"] > PAIRWISE["gathered"], found
    assert found["elementwise"] > PAIRWISE["elementwise"] * ALLOWED["elementwise"], found
