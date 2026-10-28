"""The table evaluator (``mime.core.tables``) against analytic functions, finite differences and its own contract; its
switch, declared by a test entry; and its code, traced and scanned as a node's step would be.

- **The form.** Every rule of OpenMaterica's table form, and each of MIME's own conditions, refuses a table that
  breaks it when the table is made.
- **Exact where its scheme is.** Each scheme interpolates (a read at a knot is that knot's value) and reproduces the
  polynomials it is built on, at random queries in random tables (one to three axes, unequal knot spacings): linear,
  every function linear in each axis; cubic, every function at most quadratic in each axis; loglog, every power law.
  Within the rounding of their inputs: the polynomial's terms and its shift, whose values can cancel to nothing. These
  cannot see a wrong cell, where every cell's polynomial is the same one: the cases (``test_table_cases.py``) and the
  differential (``test_port_table.py``) do.
- **An axis at the dtype's extremes**, wider than its range or with a denormal scale for its buckets, reads as the
  search locates; a control plants each half of the bucket index's check alone.
- **Derivatives**, in float64, against central finite differences at random queries inside cells of random smooth
  data; a cubic's one-sided derivatives agree at every interior knot (it is continuously differentiable there), and a
  linear table's do not (the control: the check can fail).
- **Positive definite by construction**, the Cholesky variant, over draws of resistance-like tables at physical
  scales (graded grids, smooth in the offset), in float64 and float32, and over rough draws (independent matrices at
  every knot of an ungraded grid) in float64; and where a table's factor is too ill-conditioned for float32, in
  float64. The smooth draws are a regression check: interpolating their values entry by entry stays definite, so they
  do not tell the construction from a plain interpolation. The rough draws do (entry by entry loses definiteness on
  most of them), and so do the controls: interpolating the same values entry by entry with the cubic scheme loses
  definiteness on a constructed table, every run, and on a share of the draws, measured; a cubic whose slopes are the
  secants (exact only on linear functions) fails the quadratic oracle on every draw, measured. A table definite in
  float64 but not once rounded to float32 is refused when compiled, naming its axes.
- **The switch.** A test entry owning a linear table declares its location switch a kink, one owning a cubic table
  continuously differentiable, each status derived from it. The harness checks that neither is a jump at a grid node
  (it tells a jump from anything continuous: a declared order of 1 or more it checks as at least 0, so the kink and the
  smooth table both pass); the cubic's order 1 is carried by its one-sided derivative test, and the status each test
  entry declares by the status test, from ``Table.value_order``. Outside
  the readable range with no extrapolation the forced branch gives no number, a refusal rather than a branch.
- **The scan.** A step reading every scheme and structure is traced and ``mime``'s package scanned over it: no raw
  non-smooth operation, no switch inside a transform, no owner unread or passed by position; its one domain guard (a
  loglog query's logarithm) changes nothing where its condition is false."""

import ast
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mime
import mime.core.tables._compile as compile_module
import mime.core.tables._evaluate as evaluate_module
from mime.build._maddening import make_jaxpr
from mime.core import switch
from mime.core.tables import (
    Axis,
    ConstantExtrapolation,
    Table,
    TableInvalid,
    TableRefused,
    evaluate,
    query_range_problems,
    refuse,
)
from mime.docs._switches import TracedStep, scan_switches
from mime.gradients import GradientStatus, status_problem
from mime.registry import GradientRecord, declared_switches
from mime.testing import KINK_LIMIT, SwitchRow, classify, fallback_problems
from mime.testing.toys import ToySpring

PACKAGE = Path(mime.__file__).parent
OWNER = "toy.table"
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def line(x, y, **settings) -> Table:
    return Table((Axis("temperature", "K", tuple(x), **settings),), np.asarray(y, dtype=float))


#: A read and a derivative compiled once per shape of table and query, the table an argument (its compiled form, a
#: pytree), so that draws of one shape share a compilation.
_READ = jax.jit(lambda arrays, q: evaluate(arrays, *q.T, owner=OWNER))


@jax.jit
def _probe(arrays, point, step):
    """The derivative at ``point``, and the central differences of the reads ``step`` either side of it on each
    axis."""
    gradient = jax.grad(lambda q: evaluate(arrays, *q, owner=OWNER).value)(point)
    shifts = jnp.diag(step)
    plus, minus = (evaluate(arrays, *(point + sign * shifts).T, owner=OWNER).value for sign in (1.0, -1.0))
    return gradient, (plus - minus) / (2 * step)


def evaluated(table: Table, points, dtype=np.float64):
    return _READ(table.arrays(dtype), jnp.asarray(np.asarray(points, dtype=dtype).reshape(-1, len(table.axes))))


def read(table: Table, *points) -> np.ndarray:
    return np.asarray(evaluated(table, points).value)


# -- the form ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("axis", "complaint"), [
    ({"x": (1.0, 1.0)}, "fewer than two distinct"),
    ({"x": (1.0, 3.0, 2.0)}, "decreases"),
    ({"x": (1.0, 2.0, 2.0, 3.0), "edges": (2.0, 2.0)}, "given twice"),
    ({"x": (1.0, 2.0, 3.0), "edges": (3.0,)}, "strictly between"),
    ({"x": (1.0, 2.0, 3.0), "edges": (2.0,)}, "exactly twice"),
    ({"x": (1.0, 2.0, 2.0, 2.0, 3.0), "edges": (2.0,)}, "exactly twice"),
    ({"x": (1.0, 2.0, 2.0, 3.0)}, "no declared edges"),
    ({"x": (1.0, 2.0, 3.0), "source_range": (2.5, 2.0)}, "first not above"),
    ({"x": (1.0, 2.0, 3.0), "source_range": (4.0, 5.0)}, "shares no value"),
    ({"x": (1.0, 2.0, 2.0, 3.0), "edges": (2.0,), "source_range": (2.0, 2.0)}, "shares no value"),
    ({"x": (1.0, math.nan)}, "finite numbers"),
    ({"x": (1.0, 2.0), "unit": ""}, "unit is a string"),
])
def test_a_table_that_breaks_its_axis_form_is_refused_when_made(axis, complaint):
    settings = {"name": "temperature", "unit": "K", **axis}
    with pytest.raises(TableInvalid, match=complaint):
        Axis(**settings)


@pytest.mark.parametrize(("table", "complaint"), [
    ({"y": [1.0, 2.0, 3.0]}, "leading dimensions"),
    ({"y": [1.0, math.inf]}, "not finite"),
    ({"scheme": "spline"}, "the schemes are"),
    ({"scheme": "loglog", "y": [1.0, -2.0]}, "only positive"),
    ({"structure": "cholesky"}, "square matrix"),
    ({"structure": "cholesky", "y": [[[1.0, 0.5], [0.4, 1.0]]] * 2}, "symmetric"),
    ({"structure": "cholesky", "y": [[[1.0, 2.0], [2.0, 1.0]]] * 2}, "positive definite"),
    ({"scheme": "loglog", "structure": "cholesky", "y": [[[1.0]]] * 2}, "no structure"),
    ({"axes": (Axis("t", "K", (1.0, 2.0)), Axis("t", "K", (1.0, 2.0))), "y": [[1.0, 1.0]] * 2}, "named once"),
])
def test_a_table_that_breaks_its_form_is_refused_when_made(table, complaint):
    settings = {"axes": (Axis("t", "K", (1.0, 2.0)),), "y": [1.0, 2.0], **table}
    with pytest.raises(TableInvalid, match=complaint):
        Table(**settings)


def test_the_readable_range_is_the_knots_narrowed_by_the_source_range_never_widened():
    assert Axis("t", "K", (1.0, 2.0, 3.0), source_range=(1.5, 9.0)).readable_range == (1.5, 3.0)
    assert Axis("t", "K", (1.0, 2.0, 3.0), source_range=(0.0, 9.0)).readable_range == (1.0, 3.0)


def test_a_table_whose_knots_an_int32_index_cannot_reach_is_refused_when_compiled():
    """Refused before its values are read, so its axes alone are enough here: such a table's values would not fit in
    memory to make one."""
    axis = Axis("a", "m", tuple(float(v) for v in range(1300)))
    with pytest.raises(TableInvalid, match="int32 index"):
        compile_module.compiled(SimpleNamespace(axes=(axis,) * 3), np.dtype(np.float64))


@pytest.mark.parametrize(("nodes", "refused"), [(2 ** 31 - 3, False), (2 ** 31 - 2, False), (2 ** 31 - 1, True),
                                                (2 ** 31, True), (2 ** 33, True)])
def test_the_int32_guard_refuses_from_the_count_a_read_reaches_and_not_before(nodes, refused):
    """The boundary itself, so that a guard loosened or tightened by one knot, or by a share, is found."""
    if refused:
        with pytest.raises(TableInvalid, match="int32 index"):
            compile_module.reachable(nodes)
    else:
        compile_module.reachable(nodes)


def test_a_table_that_does_not_survive_float32_is_refused_when_compiled_in_it():
    close = line((1.0, 1.0 + 1e-9, 2.0), (1.0, 2.0, 3.0))
    assert np.isfinite(read(close, 1.5)).all()
    with pytest.raises(TableInvalid, match="does not survive"):
        close.arrays(np.float32)


# -- zero, the commonest coordinate: no denormal break --------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("side", ["right", "left"])
def test_a_declared_edge_at_zero_refuses_a_read_at_zero_from_either_side(dtype, side):
    """A break one unit off zero is the smallest denormal, which a compiled read flushes to zero: so the break is the
    smallest normal number, and a read at zero is at the edge."""
    table = dataclasses.replace(line((-1.0, 0.0, 0.0, 1.0), (1.0, 2.0, 5.0, 6.0), edges=(0.0,)), side=side)
    for zero in (0.0, -0.0):
        found = evaluated(table, [[zero]], dtype)
        assert bool(found.refused[0]) and not bool(found.accepted[0]) and np.isnan(np.asarray(found.value)[0])
    below, above = evaluated(table, [[-0.5], [0.5]], dtype).value
    assert float(below) == pytest.approx(1.5) and float(above) == pytest.approx(5.5)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(("x", "side"), [
    ((-2.0, -1.0, 0.0), "right"),  # the last readable knot is zero
    ((0.0, 1.0, 2.0), "left"),  # the first readable knot is zero, read from the left
    ((-1.0, 0.0, 1.0), "right"),  # controls: zero an interior knot, or an end read from the side that has a cell
    ((-1.0, 0.0, 1.0), "left"),
    ((0.0, 1.0, 2.0), "right"),
    ((-2.0, -1.0, 0.0), "left"),
], ids=["last-right", "first-left", "interior-right", "interior-left", "first-right", "last-left"])
def test_a_table_whose_readable_end_or_knot_is_zero_reads_a_finite_value_there(dtype, x, side):
    values = (10.0, 20.0, 30.0)
    table = dataclasses.replace(line(x, values), side=side)
    found = evaluated(table, [[0.0]], dtype)
    assert bool(found.in_range[0]) and bool(found.accepted[0]) and not bool(found.refused[0])
    assert float(found.value[0]) == pytest.approx(values[x.index(0.0)])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_range_ending_at_zero_by_the_source_range_reads_zero_in_range(dtype):
    table = line((-2.0, -1.0, 1.0), (10.0, 20.0, 30.0), source_range=(-5.0, 0.0))
    found = evaluated(table, [[0.0], [0.5]], dtype)
    assert bool(found.in_range[0]) and not bool(found.in_range[1])
    assert float(found.value[0]) == pytest.approx(25.0)


def test_a_knot_the_dtype_holds_as_a_denormal_is_refused_when_compiled_in_it():
    """1e-40 is a float32 denormal, which a compiled read flushes to zero (the knots 0 and 1e-40 would meet)."""
    table = line((-1.0, 1e-40, 1.0), (1.0, 2.0, 3.0))
    assert np.isfinite(read(table, 0.5)).all()
    with pytest.raises(TableInvalid, match="denormal"):
        table.arrays(np.float32)


# -- an axis at the dtype's extremes: no bucket a read maps wrongly -------------------------------------------------

#: Float32 axes whose buckets a read would map a query into wrongly: one wider than float32's range, so a query's
#: offset from the first knot overflows (a read at 4.29e37 took the cell above it, every flag the same), and two knots
#: whose buckets' scale is a float32 denormal, which a compiled read flushes to zero.
EXTREME_AXES = {"wider than the range": (-3e38, -1e38, 0.0, 1e38, 3e38), "a denormal scale": (-1.4753196e38,
                                                                                              1.4589544e38)}
#: What each axis's case needs of the buckets' check: the half that keeps it right, the other half planted alone.
ONE_HALF = {"wider than the range": lambda last, origin, scale, kind: float(np.finfo(kind).tiny) <= scale < math.inf,
            "a denormal scale": lambda last, origin, scale, kind: bool(np.isfinite(kind(last) - kind(origin)))}


def extreme_misreads(x: tuple[float, ...]) -> np.ndarray:
    """The queries at which a float32 read of a linear table on the axis ``x``, extrapolated, differs from the same
    read located by the search, in value or flags: every knot and the numbers beside it, points across each cell,
    4.29e37, and the numbers a search orders specially."""
    table = Table((Axis("a", "u", x),), np.arange(len(x), dtype=float) ** 2,
                  extrapolation=ConstantExtrapolation("the end value"))
    arrays = table.arrays(np.float32)
    searched = dataclasses.replace(arrays, axes=tuple(dataclasses.replace(axis, buckets=None) for axis in arrays.axes))
    knots = np.asarray(x)
    across = (knots[:-1, None] * (1.0 - np.linspace(0.0, 1.0, 9)) + knots[1:, None] * np.linspace(0.0, 1.0, 9))
    held = np.asarray(x, dtype=np.float32)
    points = np.concatenate([held, np.nextafter(held, np.float32(np.inf)), np.nextafter(held, np.float32(-np.inf)),
                             across.ravel().astype(np.float32),
                             np.asarray([4.29e37, 0.0, -0.0, np.nan, np.inf, -np.inf, 3.4e38, -3.4e38],
                                        dtype=np.float32)])[:, None]
    found, expected = (_READ(a, jnp.asarray(points)) for a in (arrays, searched))
    differs = np.zeros(len(points), dtype=bool)
    for a, b in zip(found, expected, strict=True):
        differs |= ~(np.asarray(a) == np.asarray(b)) & ~(np.isnan(np.asarray(a)) & np.isnan(np.asarray(b)))
    return points[differs, 0]


@pytest.mark.parametrize("name", EXTREME_AXES)
def test_an_axis_at_float32s_extremes_reads_as_the_search_locates(name):
    assert extreme_misreads(EXTREME_AXES[name]).size == 0
    if name == "wider than the range":  # the line through the cell holding 4.29e37, from 4 at 0 to 9 at 1e38
        value = np.asarray(evaluated(Table((Axis("a", "u", EXTREME_AXES[name]),), np.arange(5.0) ** 2), [[4.29e37]],
                                     np.float32).value)
        assert float(value[0]) == pytest.approx(4.0 + 5.0 * 0.429, rel=1e-6)


@pytest.mark.parametrize("name", EXTREME_AXES)
def test_control_buckets_made_without_one_half_of_the_dtypes_check_misread_the_axis_that_needs_it(name, monkeypatch):
    """Each half of ``mapped_as_made`` planted alone: without the overflow half, the axis wider than float32's range
    reads 4.29e37 in the wrong cell; without the scale half, the two knots' denormal scale sends every query to the
    first bucket."""
    monkeypatch.setattr(compile_module, "mapped_as_made", ONE_HALF[name])
    assert extreme_misreads(EXTREME_AXES[name]).size > 0


# -- a readable range of one point -----------------------------------------------------------------------------------

POINT = line((280.0, 300.0, 320.0, 340.0), (1000.0, 996.0, 990.0, 986.0), source_range=(310.0, 310.0))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("side", ["right", "left"])
def test_a_readable_range_of_one_point_reads_its_value_with_no_derivative(dtype, side):
    table = dataclasses.replace(POINT, side=side, extrapolation=ConstantExtrapolation("the value there"))
    found = evaluated(table, [[310.0], [305.0], [315.0]], dtype)
    assert [bool(v) for v in found.in_range] == [True, False, False] and not bool(np.asarray(found.refused).any())
    assert np.allclose(np.asarray(found.value), 993.0, rtol=1e-6)
    query = jnp.asarray([310.0, 305.0], dtype=dtype)
    forward = jax.jvp(lambda q: evaluate(table, q, owner=OWNER).value, (query,), (jnp.ones_like(query),))[1]
    backward = jax.grad(lambda q: evaluate(table, q, owner=OWNER).value.sum())(query)
    assert np.isnan(np.asarray(forward)[0]) and float(forward[1]) == 0.0
    assert np.isnan(np.asarray(backward)[0]) and float(backward[1]) == 0.0


@pytest.mark.parametrize("point", [280.0, 340.0, 300.0])
@pytest.mark.parametrize("side", ["right", "left"])
def test_a_readable_range_of_one_point_at_a_knot_or_an_end_reads_that_knots_value(point, side):
    table = dataclasses.replace(line((280.0, 300.0, 320.0, 340.0), (1000.0, 996.0, 990.0, 986.0),
                                     source_range=(point, point)), side=side)
    value = {280.0: 1000.0, 340.0: 986.0, 300.0: 996.0}[point]
    found = evaluated(table, [[point]])
    assert bool(found.in_range[0]) and float(found.value[0]) == value
    assert np.isnan(float(jax.grad(lambda q: evaluate(table, q, owner=OWNER).value.sum())(jnp.asarray([point]))[0]))


def test_a_range_that_is_one_point_only_in_float32_reads_as_one_point_there():
    table = line((280.0, 300.0, 320.0), (1000.0, 996.0, 990.0), source_range=(310.0, 310.0 + 1e-9))
    found = evaluated(table, [[310.0], [312.0]], np.float32)
    assert [bool(v) for v in found.in_range] == [True, False]


def test_the_derivative_of_another_axis_is_unaffected_by_a_one_point_axis():
    table = Table((Axis("a", "m", (0.0, 1.0, 2.0), source_range=(1.0, 1.0)), Axis("b", "m", (0.0, 1.0, 2.0))),
                  np.arange(9.0).reshape(3, 3) ** 2)
    slopes = np.asarray(jax.grad(lambda q: evaluate(table, q[0], q[1], owner=OWNER).value)(jnp.asarray([1.0, 0.5])))
    assert np.isnan(slopes[0]) and np.isfinite(slopes[1]) and slopes[1] != 0.0


ONE_POINT_AXIS = Table((Axis("a", "m", (0.0, 1.0, 2.0), source_range=(1.0, 1.0)), Axis("b", "m", (0.0, 1.0, 2.0))),
                       np.arange(9.0).reshape(3, 3) ** 2)


def stacked(table, *queries, owner):
    """The evaluator as it was before it took one argument per axis: the queries stacked into one array, each axis
    read from a slice of it. The planted fault of the control below."""
    stack = jnp.stack(jnp.broadcast_arrays(*queries), axis=-1)
    return evaluate(table, *(stack[..., a] for a in range(len(queries))), owner=owner)


def finite_difference_in_b(read_at, dtype):
    point, step = jnp.asarray(0.5, dtype=dtype), jnp.asarray(0.05, dtype=dtype)
    return float((read_at(point + step) - read_at(point - step)) / (2 * step))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_constant_query_on_a_one_point_axis_leaves_the_other_axis_derivative_finite_in_both_modes(dtype):
    """The one-point axis's query is a constant, no tracer: it carries no tangent, so nothing is NaN along the other
    axis, forward or reverse, and the derivative there is that of a finite difference (the table is linear in ``b``
    there: 16 - 9)."""
    arrays = ONE_POINT_AXIS.arrays(dtype)

    def read_at(b):
        return evaluate(arrays, jnp.asarray(1.0, dtype=dtype), b, owner=OWNER).value

    expected = finite_difference_in_b(read_at, dtype)
    assert expected == pytest.approx(7.0, rel=1e-3)
    point = jnp.asarray(0.5, dtype=dtype)
    assert float(jax.jacrev(read_at)(point)) == pytest.approx(expected, rel=1e-3)
    assert float(jax.jacfwd(read_at)(point)) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_control_the_stacked_read_makes_a_constant_one_point_query_nan_in_forward_mode(dtype):
    """The planted fault: the same constant-query read through the old stacking has a NaN forward derivative along
    ``b``, which the test above would fail on (reverse mode, which drops the zero cotangent, is unaffected)."""
    arrays = ONE_POINT_AXIS.arrays(dtype)

    def read_at(b):
        return stacked(arrays, jnp.asarray(1.0, dtype=dtype), b, owner=OWNER).value

    assert np.isnan(float(jax.jacfwd(read_at)(jnp.asarray(0.5, dtype=dtype))))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_known_limit_a_traced_query_on_a_one_point_axis_is_nan_in_reverse_mode_along_it_and_in_every_direction_forward(
        dtype):
    """Pins a limit that is known and kept (``MIME-ANO-115``), not a property to keep: with the one-point axis's query
    traced, the specification's NaN (no derivative along a point) holds along it in reverse mode, and in forward mode
    in every direction, since a linear tangent rule cannot give NaN along one direction and a number along another
    (``0 * NaN`` is NaN). Reverse mode keeps the other axis's derivative, matching a finite difference."""
    arrays = ONE_POINT_AXIS.arrays(dtype)
    point = jnp.asarray([1.0, 0.5], dtype=dtype)

    def read_at(q):
        return evaluate(arrays, q[0], q[1], owner=OWNER).value

    expected = finite_difference_in_b(lambda b: read_at(jnp.stack([point[0], b])), dtype)
    assert expected == pytest.approx(7.0, rel=1e-3)
    reverse = np.asarray(jax.jacrev(read_at)(point))
    forward = np.asarray(jax.jacfwd(read_at)(point))
    assert np.isnan(reverse[0]) and reverse[1] == pytest.approx(expected, rel=1e-3)
    assert np.isnan(forward).all()


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_boundary_is_whether_the_one_point_query_carries_a_tangent_not_whether_it_is_traced(dtype):
    """The workaround's boundary (``MIME-ANO-115``): a query on the one-point axis that is a ``jit`` argument, traced
    but not differentiated, carries no tangent, so the forward derivative along the other axis is finite and matches
    a finite difference; the same query given to ``jvp`` with a zero tangent carries one, and makes it NaN."""
    arrays = ONE_POINT_AXIS.arrays(dtype)
    a, b = jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.5, dtype=dtype)
    expected = finite_difference_in_b(lambda y: evaluate(arrays, a, y, owner=OWNER).value, dtype)
    traced = jax.jit(lambda x, y: jax.jacfwd(lambda z: evaluate(arrays, x, z, owner=OWNER).value)(y))(a, b)
    assert float(traced) == pytest.approx(expected, rel=1e-3)
    tangent = jax.jvp(lambda x, y: evaluate(arrays, x, y, owner=OWNER).value, (a, b),
                      (jnp.zeros_like(a), jnp.ones_like(b)))[1]
    assert np.isnan(float(tangent))


# -- exact where its scheme is -------------------------------------------------------------------------------------

@st.composite
def grids(draw, positive: bool = False) -> tuple[tuple[float, ...], ...]:
    """One to three axes of two to six knots, at spacings up to forty to one, a knot of any one of them sometimes at
    exactly zero (a signed offset, an angle). The start is a number of a physical scale, never a denormal in float32 (the narrowest dtype a test reads in)."""
    axes = []
    for _ in range(draw(st.integers(1, 3))):
        count = draw(st.integers(2, 6))
        if positive:
            start = draw(st.floats(0.01, 10.0))
            ratios = draw(st.lists(st.floats(1.2, 8.0), min_size=count - 1, max_size=count - 1))
            axes.append(tuple(float(v) for v in start * np.cumprod([1.0, *ratios])))
        else:
            start = draw(st.floats(-10.0, 10.0, width=32, allow_subnormal=False))
            gaps = draw(st.lists(st.floats(0.05, 2.0), min_size=count - 1, max_size=count - 1))
            knots = np.cumsum([start, *gaps])
            if draw(st.booleans()):
                knots = knots - knots[draw(st.integers(0, count - 1))]
            axes.append(tuple(float(v) for v in knots))
    return tuple(axes)


def queries(draw, grid, count: int = 8) -> np.ndarray:
    """Points inside the grid, each coordinate anywhere in its axis's range, knots included."""
    points = [[draw(st.one_of(st.floats(x[0], x[-1]), st.sampled_from(x))) for x in grid] for _ in range(count)]
    return np.asarray(points)


def tensor_polynomial(degrees: tuple[int, ...], coefficients: np.ndarray):
    """``sum_p c_p prod_a x_a ** p_a`` over every power ``p_a`` from 0 to ``degrees[a]`` on each axis ``a``."""
    powers = np.stack(np.meshgrid(*[np.arange(d + 1) for d in degrees], indexing="ij"), axis=-1).reshape(-1,
                                                                                                       len(degrees))

    def value(points: np.ndarray) -> np.ndarray:
        return np.prod(points[..., None, :] ** powers, axis=-1) @ coefficients[:len(powers)]

    return value


#: A polynomial's coefficient: zero, or of a magnitude from a thousandth to a thousand.
COEFFICIENTS = st.one_of(st.just(0.0), st.floats(1e-3, 1e3), st.floats(-1e3, -1e-3))


def on_grid(function, grid) -> np.ndarray:
    mesh = np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1)
    return function(mesh)


def worst_relative_miss(grid, degrees: tuple[int, ...], coefficients: np.ndarray, points: np.ndarray,
                        scheme: str = "linear") -> float:
    """The largest miss at ``points`` of a ``scheme`` table of the polynomial plus one, relative to its inputs'
    scale: the sum of its terms' magnitudes and the shift at the grid's corner farthest from zero. That bounds the
    rounding of the table's values and of the oracle, where the values themselves can cancel to nothing (a constant
    of -1 against the shift makes every value zero)."""
    function = tensor_polynomial(degrees, coefficients)
    table = Table(tuple(Axis(f"a{i}", "m", x) for i, x in enumerate(grid)), on_grid(function, grid) + 1.0,
                  scheme=scheme)
    farthest = np.asarray([max(abs(v) for v in x) for x in grid])
    scale = float(tensor_polynomial(degrees, np.abs(coefficients))(farthest)) + 1.0
    return float(np.max(np.abs(read(table, *points) - (function(points) + 1.0)))) / scale


#: The rounding floor of a read, relative to its inputs' scale: a cubic read of three axes combines 64 knot values,
#: each weighted by up to a few units, in a few operations each.
ROUNDING = 2048 * EPS


@settings(deadline=None)
@given(st.data())
def test_linear_is_exact_on_every_function_linear_in_each_axis(data):
    grid = data.draw(grids())
    coefficients = np.asarray(data.draw(st.lists(COEFFICIENTS, min_size=8, max_size=8)))
    assert worst_relative_miss(grid, (1,) * len(grid), coefficients, queries(data.draw, grid)) <= ROUNDING


@settings(deadline=None)
@given(st.data())
def test_cubic_is_exact_on_every_function_at_most_quadratic_in_each_axis(data):
    """Quadratic on an axis of three knots or more; linear on one of two, where a cubic has only the secant."""
    grid = data.draw(grids())
    degrees = tuple(min(2, len(x) - 1) for x in grid)
    coefficients = np.asarray(data.draw(st.lists(COEFFICIENTS, min_size=27, max_size=27)))
    assert worst_relative_miss(grid, degrees, coefficients, queries(data.draw, grid), "cubic") <= ROUNDING


@pytest.mark.parametrize("scheme", ["linear", "cubic"])
def test_a_table_whose_constant_cancels_its_shift_is_exact_on_its_inputs_scale(scheme):
    """The draw that once divided by zero, judged on the table's largest value: on one axis of two knots, a constant
    of exactly -1 cancels the shift of +1, so every value is zero, read at zero."""
    coefficients = np.asarray([-1.0] + [0.0] * 26)
    assert worst_relative_miss(((0.0, 1.0),), (1,), coefficients, np.zeros((8, 1)), scheme) <= ROUNDING


@settings(deadline=None)
@given(st.data())
def test_loglog_is_exact_on_every_power_law(data):
    grid = data.draw(grids(positive=True))
    scale = data.draw(st.floats(1e-6, 1e6))
    exponents = np.asarray(data.draw(st.lists(st.floats(-3.0, 3.0), min_size=len(grid), max_size=len(grid))))
    power_law = lambda points: scale * np.prod(points ** exponents, axis=-1)  # noqa: E731
    table = Table(tuple(Axis(f"a{i}", "Hz", x) for i, x in enumerate(grid)), on_grid(power_law, grid),
                  scheme="loglog")
    points = queries(data.draw, grid)
    found = read(table, *points)
    # a power law's rounding is relative to its own value, through a logarithm of up to a few hundred
    assert np.max(np.abs(found / power_law(points) - 1.0)) <= 1e3 * ROUNDING


@pytest.mark.parametrize("scheme", ["linear", "loglog", "cubic"])
def test_a_read_at_a_knot_is_that_knots_value(scheme):
    x = (0.5, 1.0, 2.5, 4.0, 4.0, 7.0)
    y = np.asarray([3.0, 1.0, 2.0, 5.0, 0.5, 4.0])
    table = line(x, y, edges=(4.0,))
    table = dataclasses.replace(table, scheme=scheme)
    found = read(table, *(v for v in x if v != 4.0))
    assert np.allclose(found, [v for v, at in zip(y, x, strict=True) if at != 4.0], rtol=64 * EPS, atol=0.0)


# -- derivatives ---------------------------------------------------------------------------------------------------

def smooth(points: np.ndarray) -> np.ndarray:
    return np.sin(1.3 * points[..., 0]) * np.cos(0.7 * points[..., -1]) + np.exp(0.2 * points.sum(axis=-1))


@pytest.mark.parametrize("scheme", ["linear", "cubic"])
@settings(deadline=None)
@given(data=st.data())
def test_the_derivative_is_the_finite_difference_inside_a_cell(scheme, data):
    """Central differences at a step far below the query's distance to any knot, on the cell's own polynomial: in
    float64, within its truncation and rounding (the rounding of the values over the step)."""
    grid = data.draw(grids())
    table = Table(tuple(Axis(f"a{i}", "m", x) for i, x in enumerate(grid)), on_grid(smooth, grid), scheme=scheme)
    cells = [data.draw(st.integers(0, len(x) - 2)) for x in grid]
    fractions = [data.draw(st.floats(0.1, 0.9)) for _ in grid]
    point = np.asarray([x[k] + f * (x[k + 1] - x[k]) for x, k, f in zip(grid, cells, fractions, strict=True)])
    widths = np.asarray([x[k + 1] - x[k] for x, k in zip(grid, cells, strict=True)])
    step = 1e-4 * widths
    gradient, central = (np.asarray(v) for v in _probe(table.arrays(np.float64), jnp.asarray(point),
                                                       jnp.asarray(step)))
    scale = float(np.max(np.abs(table.y))) / widths
    assert np.all(np.abs(gradient - central) <= 1e-6 * scale), (gradient, central)


@pytest.mark.parametrize(("scheme", "continuous"), [("cubic", True), ("linear", False)])
def test_a_cubic_tables_one_sided_derivatives_agree_at_every_interior_knot_and_a_linear_tables_do_not(scheme,
                                                                                                      continuous):
    x = (0.0, 0.4, 1.5, 1.9, 3.0, 4.2)
    y = smooth(np.asarray(x)[:, None])
    sides = [dataclasses.replace(line(x, y), scheme=scheme, side=side) for side in ("left", "right")]
    slopes = np.asarray([[float(jax.grad(lambda q, t=t: evaluate(t, q, owner=OWNER).value)(
        jnp.asarray(knot))) for knot in x[1:-1]] for t in sides])
    agree = np.abs(slopes[0] - slopes[1]) <= 64 * EPS * np.max(np.abs(y)) / min(np.diff(x))
    assert bool(agree.all()) is continuous, slopes


# -- positive definite by construction -----------------------------------------------------------------------------

#: A resistance's scales in SI: a millimetre body in water, mu R (force per velocity), mu R^2 (coupling) and mu R^3
#: (torque per rate of turn), as the Cholesky factor's rows carry them (the square roots of the diagonal blocks').
VISCOSITY, RADIUS = 1e-3, 1e-3
SCALES = np.sqrt(np.asarray([VISCOSITY * RADIUS] * 3 + [VISCOSITY * RADIUS ** 3] * 3))


@st.composite
def resistance_tables(draw) -> tuple[tuple[float, ...], np.ndarray]:
    """A resistance-like table over the offset, as a body's resistance varies across a tube: two to six knots on a
    graded grid (each cell within a factor of two of its neighbour's width), and at each knot ``S L L^T S``, ``S`` the
    scales above and ``L`` a factor varying smoothly with the offset: its diagonal ``exp(a + b sin(w s + p))`` and the
    entries below it ``c sin(w s + q)``, ``s`` the offset's fraction of the range, ``a`` from ``log(1/2)`` to
    ``log(2)``, ``b`` and ``c`` up to a half in magnitude, ``w`` up to three radians over the range."""
    count = draw(st.integers(2, 6))
    start, first = draw(st.floats(-10.0, 10.0, width=32, allow_subnormal=False)), draw(st.floats(0.05, 2.0))
    ratios = draw(st.lists(st.floats(0.5, 2.0), min_size=count - 2, max_size=count - 2))
    x = tuple(float(v) for v in start + np.concatenate([[0.0], np.cumsum(first * np.cumprod([1.0, *ratios]))]))
    fraction = (np.asarray(x) - x[0]) / (x[-1] - x[0])

    def floats(low: float, high: float, size: int) -> np.ndarray:
        return np.asarray(draw(st.lists(st.floats(low, high), min_size=size, max_size=size)))

    level, swing, below = floats(math.log(0.5), math.log(2.0), 6), floats(-0.5, 0.5, 6), floats(-0.5, 0.5, 15)
    rate, phases = draw(st.floats(0.0, 3.0)), floats(0.0, 2 * math.pi, 21)
    factors = np.zeros((count, 6, 6))
    factors[:, np.arange(6), np.arange(6)] = np.exp(level + swing * np.sin(rate * fraction[:, None] + phases[:6]))
    rows, columns = np.tril_indices(6, -1)
    factors[:, rows, columns] = below * np.sin(rate * fraction[:, None] + phases[6:])
    scaled = SCALES[None, :, None] * factors
    return x, scaled @ np.swapaxes(scaled, -1, -2)


def definite(matrices: np.ndarray, dtype) -> bool:
    """Every matrix symmetric to its rounding and with a Cholesky factor in ``dtype``, to its rounding (below)."""
    held = np.asarray(matrices, dtype=dtype)
    skew = np.abs(held - np.swapaxes(held, -1, -2)).max(axis=(-1, -2))
    if np.any(skew > 64 * np.finfo(dtype).eps * np.abs(held).max(axis=(-1, -2))):
        return False
    diagonal = np.diagonal(held, axis1=-2, axis2=-1)
    if np.any(diagonal <= 0):
        return False
    # Judged on the scale it came from. A resistance mixes force and torque scales, and a product of factors one of
    # which is nearly singular rounds each entry relative to its row's scale: scaled to a unit diagonal, the smallest
    # eigenvalue of a definite product is only known to a few units in the last place (-4.2 of them over 3000 rough
    # draws, which the unscaled factorisation cannot see past at a condition number of 1e18). So "definite" here is
    # definite to 64 units in the last place of that scale; entry-by-entry interpolation misses it by far more (three
    # draws in four).
    root = np.sqrt(diagonal)
    scaled = held / (root[..., :, None] * root[..., None, :])
    try:
        np.linalg.cholesky(scaled + 64 * np.finfo(dtype).eps * np.eye(held.shape[-1]))
    except np.linalg.LinAlgError:
        return False
    return True


def planted_loss(seed: int, ulps: float) -> np.ndarray:
    """A resistance-like matrix at physical scales whose diagonal is positive and which, scaled to a unit diagonal, has
    one eigenvalue of ``-ulps`` units in the last place: a rank-deficient matrix of random eigenvectors, scaled to a
    unit diagonal, pushed ``ulps`` below singular along its null vector, then given the resistance's scales."""
    rng = np.random.default_rng(seed)
    vectors, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    singular = vectors @ np.diag([0.0, *rng.uniform(0.5, 2.0, 5)]) @ vectors.T
    root = np.sqrt(np.diagonal(singular))
    unit = singular / np.outer(root, root)
    eigenvalues, eigenvectors = np.linalg.eigh(unit)
    pushed = unit - (ulps * EPS + eigenvalues[0]) * np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
    return SCALES[:, None] * (pushed + pushed.T) / 2 * SCALES[None, :]


@settings(deadline=None)
@given(seed=st.integers(0, 2 ** 32 - 1))
def test_control_a_loss_past_the_rounding_allowance_with_positive_diagonals_is_seen(seed):
    """The check's resolution, which the plain-interpolation controls do not reach (their losses are gross, a
    non-positive diagonal): a loss of definiteness of 96 units in the last place of the unit-diagonal scale, half as
    much again as the allowance, with every diagonal positive, is refused on every draw. A loss within the allowance
    is not seen: measured over 2000 draws, one of 32 units passes every one, one of 64 about half, one of 80 or more
    none."""
    planted = planted_loss(seed, 96.0)
    assert np.all(np.diagonal(planted) > 0)
    assert not definite(planted, np.float64)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("scheme", ["cubic", "linear"])
@settings(deadline=None)
@given(data=st.data())
def test_the_cholesky_variant_is_positive_definite_at_every_query(scheme, dtype, data):
    """A regression check on smooth, graded draws, which entry-by-entry interpolation keeps definite too: it does not
    tell the construction from a plain interpolation (the rough draws below do, and so do the controls)."""
    x, knots = data.draw(resistance_tables())
    table = Table((Axis("offset", "m", x),), knots, scheme=scheme, structure="cholesky")
    points = np.asarray([[data.draw(st.floats(x[0], x[-1]))] for _ in range(16)], dtype=dtype)
    found = evaluated(table, points, dtype)
    assert found.value.dtype == dtype and bool(found.accepted.all())
    assert definite(found.value, dtype)


def test_a_factor_too_ill_conditioned_for_float32_is_positive_definite_in_float64():
    """Construction holds in exact arithmetic: the factor's diagonal is an exponential, so positive at every query.
    The product is representably positive definite only where the factor's condition number squared is well below the
    dtype's reciprocal epsilon. On an ungraded grid (cells of two and a twentieth) whose factor changes over the short
    cell, the cubic's slopes overshoot across the long one: at its middle the value's condition number is about 1e15,
    positive definite in float64 and beyond what float32 can represent, which is a property of the table, not of the
    read. (A draw of an earlier, ungraded strategy, which no designed grid has: the property draws graded ones.)"""
    knots = np.asarray([np.diag([1e-6, 2.5e-7, 1e-6, 1e-12, 1e-12, 1e-12]),
                        np.diag([1e-6, 2.5e-7, 5e-7, 1e-12, 1e-12, 1e-12]),
                        np.diag([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])])
    knots[1, 1, 2] = knots[1, 2, 1] = 2.5e-7
    table = Table((Axis("offset", "m", (0.0, 2.0, 2.05078125)),), knots, scheme="cubic", structure="cholesky")
    value = np.asarray(evaluated(table, [[1.0]]).value)[0]
    eigenvalues = np.linalg.eigvalsh(value)
    assert definite(value, np.float64) and eigenvalues.min() > 0
    assert eigenvalues.max() / eigenvalues.min() > 1 / np.finfo(np.float32).eps


def test_control_entrywise_cubic_interpolation_of_definite_values_loses_definiteness_and_cholesky_does_not():
    """The (0, 0) entry runs small, small, one: the cubic's slopes overshoot it below zero in the first cell."""
    small = 1e-3
    knots = np.asarray([np.diag([small, 1.0]), np.diag([small, 1.0]), np.diag([1.0, 1.0])])
    x = (0.0, 1.0, 2.0)
    points = jnp.asarray([[0.5]])
    entrywise = evaluate(Table((Axis("offset", "m", x),), knots, scheme="cubic"), *points.T, owner=OWNER).value
    factored = evaluate(Table((Axis("offset", "m", x),), knots, scheme="cubic", structure="cholesky"),
                        *points.T, owner=OWNER).value
    assert not definite(entrywise, np.float64) and definite(factored, np.float64)


def rough_draw(seed: int) -> tuple[tuple[float, ...], np.ndarray, np.ndarray]:
    """Independent resistance-like matrices at physical scales at the five knots of an ungraded grid, and 16 queries:
    harder than the smooth draws, which entry-by-entry interpolation also keeps definite."""
    rng = np.random.default_rng(seed)
    x = tuple(float(v) for v in np.cumsum(np.concatenate([[rng.uniform(-10, 10)], rng.uniform(0.05, 2.0, 4)])))
    factors = np.zeros((len(x), 6, 6))
    factors[:, np.arange(6), np.arange(6)] = rng.uniform(0.5, 2.0, (len(x), 6))
    rows, columns = np.tril_indices(6, -1)
    factors[:, rows, columns] = rng.uniform(-0.5, 0.5, (len(x), 15))
    scaled = SCALES[:, None] * factors
    return x, scaled @ np.swapaxes(scaled, -1, -2), rng.uniform(x[0], x[-1], (16, 1))


@settings(deadline=None)
@given(seed=st.integers(0, 2 ** 32 - 1))
def test_the_cholesky_variant_is_positive_definite_on_rough_draws_and_a_plain_interpolation_is_not(seed):
    """Unlike the smooth draws above, these tell the construction from interpolating the entries: entry by entry
    loses definiteness on most of them, so a ``cholesky`` table that interpolated plainly fails this property on the
    first example or the second."""
    x, knots, points = rough_draw(seed)
    factored = evaluated(Table((Axis("offset", "m", x),), knots, scheme="cubic", structure="cholesky"), points).value
    assert definite(factored, np.float64)


def test_control_how_often_entrywise_cubic_loses_definiteness_where_cholesky_never_does():
    """Measured over two hundred seeded rough draws on one axis of five knots: the share where entry-by-entry cubic
    interpolation of the same knots gives a matrix with no Cholesky factor, printed; the Cholesky variant is definite
    in float64 on every one."""
    losses = 0
    for seed in range(200):
        x, knots, points = rough_draw(seed)
        entrywise = evaluated(Table((Axis("offset", "m", x),), knots, scheme="cubic"), points).value
        factored = evaluated(Table((Axis("offset", "m", x),), knots, scheme="cubic", structure="cholesky"),
                             points).value
        losses += not definite(entrywise, np.float64)
        assert definite(factored, np.float64), seed
    print(f"\nentrywise cubic lost definiteness on {losses} of 200 draws")
    assert losses > 100


def test_control_cubic_slopes_from_secants_fail_the_quadratic_oracle_on_every_draw(monkeypatch):
    """A planted slope rule, each knot's slope the secant of the cell below it (exact only on linear functions):
    over a hundred random quadratic tables of two axes of four knots, the share failing the oracle, which must be
    all."""
    def secant(x, j, first, last):
        lower = max(j - 1, first)
        return {lower: -1.0 / (x[lower + 1] - x[lower]), lower + 1: 1.0 / (x[lower + 1] - x[lower])}

    monkeypatch.setattr(compile_module, "_slope", secant)
    misses = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        grid = tuple(tuple(float(v) for v in np.cumsum(np.concatenate([[rng.uniform(-10, 10)],
                                                                       rng.uniform(0.05, 2.0, 3)])))
                     for _ in range(2))
        coefficients = rng.uniform(-1e3, 1e3, 27)
        points = np.stack([rng.uniform(x[0], x[-1], 8) for x in grid], axis=-1)
        misses.append(worst_relative_miss(grid, (2,) * len(grid), coefficients, points, "cubic") > ROUNDING)
    print(f"\nsecant slopes fail the quadratic oracle on {sum(misses)} of {len(misses)} draws")
    assert all(misses)


# -- the switch, declared by a test entry --------------------------------------------------------------------------

GRID = ((0.0, 0.5, 1.25, 2.0), (-1.0, 0.0, 0.75, 3.0))
NODE = (1.25, 0.0)
LINEAR = Table(tuple(Axis(f"a{i}", "m", x) for i, x in enumerate(GRID)),
               on_grid(lambda p: np.sin(2.0 * p[..., 0]) + p[..., 1] ** 3, GRID))
CUBIC = dataclasses.replace(LINEAR, scheme="cubic")


def knot_switch(owner: str, table: Table) -> switch.Declaration:
    smooth_across = table.value_order >= 1
    return switch.Declaration(
        owner=owner, indicator="a query's offset from the nearest break of its axis (a knot, an edge or an end of "
                               "the readable range), in that axis's unit", orders={"value": table.value_order},
        route="exact across every knot already; the readable range's ends stop the run" if smooth_across else
              "the cubic scheme, continuously differentiable across every knot")


@dataclass(frozen=True, kw_only=True)
class TabulatedLinear(ToySpring):
    """A test entry owning a linear table, never registered: its switch is a kink."""

    name = "toy.tabulated_linear"
    differentiability = GradientStatus.PIECEWISE_KINK
    gradient = GradientRecord(switches=(knot_switch("toy.linear_table", LINEAR),))


@dataclass(frozen=True, kw_only=True)
class TabulatedCubic(ToySpring):
    """A test entry owning a cubic table, never registered: continuously differentiable across its knots, so exact."""

    name = "toy.tabulated_cubic"
    differentiability = GradientStatus.EXACT
    gradient = GradientRecord(switches=(knot_switch("toy.cubic_table", CUBIC),))


def test_each_test_entrys_status_is_the_one_its_tables_switch_gives_and_a_misdeclared_one_fails():
    for entry in (TabulatedLinear, TabulatedCubic):
        assert status_problem(entry.differentiability, declared_switches(entry)) is None
    assert status_problem(GradientStatus.EXACT, declared_switches(TabulatedLinear)) is not None
    assert status_problem(GradientStatus.PIECEWISE_KINK, declared_switches(TabulatedCubic)) is not None
    extrapolated = dataclasses.replace(CUBIC, extrapolation=ConstantExtrapolation("a declared constant"))
    assert extrapolated.value_order == 0


def node_row(entry, table: Table, node=NODE, delta: float = 1e-3) -> SwitchRow:
    (declared,) = declared_switches(entry)

    def at(offset: float) -> dict:
        query = [jnp.asarray([v + offset]) for v in node]
        return {"value": evaluate(table, *query, owner=declared.owner).value}

    return SwitchRow(declared, at, delta)


@pytest.mark.parametrize(("entry", "table"), [(TabulatedLinear, LINEAR), (TabulatedCubic, CUBIC)],
                         ids=["linear", "cubic"])
def test_the_switch_harness_measures_each_tables_switch_within_its_declared_order_at_a_grid_node(entry, table):
    found = classify(node_row(entry, table), KINK_LIMIT)
    print(f"\n{found.line()}")
    assert found.problems == [] and not found.unmeasured
    assert found.measures["value"].continuous


def test_at_a_readable_range_end_under_an_extrapolation_the_switch_is_a_kink():
    extrapolated = dataclasses.replace(CUBIC, extrapolation=ConstantExtrapolation("a declared constant"))
    declared = knot_switch("toy.cubic_table", extrapolated)
    entry = type("Extrapolated", (TabulatedCubic,), {"gradient": GradientRecord(switches=(declared,))})
    found = classify(node_row(entry, extrapolated, node=(2.0, 3.0)), KINK_LIMIT)
    assert found.problems == [] and found.measures["value"].continuous, found.line()


def test_outside_with_no_extrapolation_the_forced_branch_is_a_refusal_not_a_branch():
    query = (jnp.asarray([2.0 - 1e-4]), jnp.asarray([0.5]))
    with switch.forcing({"toy.cubic_table": True}):
        outside = evaluate(CUBIC, *query, owner="toy.cubic_table")
    assert np.isnan(np.asarray(outside.value)).all()
    assert not bool(outside.in_range[0]) and not bool(outside.accepted[0])
    assert bool(evaluate(CUBIC, *query, owner="toy.cubic_table").accepted[0])


# -- the scan ------------------------------------------------------------------------------------------------------

LOGLOG = Table((Axis("frequency", "Hz", (1.0, 10.0, 100.0)),), [0.5, 0.125, 0.0625], scheme="loglog",
               extrapolation=ConstantExtrapolation("the end value"))
CHOLESKY = Table((Axis("offset", "m", (0.0, 1.0, 2.0)),), np.asarray([np.eye(2), 2 * np.eye(2), np.eye(2)]),
                 scheme="cubic", structure="cholesky")
EDGED = Table((Axis("t", "K", (263.0, 273.0, 273.0, 283.0), edges=(273.0,)),), [918.0, 916.0, 1000.0, 999.5])


def every_read(q):
    """A step reading every scheme and structure, one, two and three axes."""
    three = Table(tuple(Axis(f"a{i}", "m", (0.0, 1.0, 2.0)) for i in range(3)), np.arange(27.0).reshape(3, 3, 3),
                  scheme="cubic")
    return (evaluate(LINEAR, q[:, 0], q[:, 1], owner=OWNER).value.sum()
            + evaluate(CUBIC, q[:, 0], q[:, 1], owner=OWNER).value.sum()
            + evaluate(LOGLOG, q[:, 0], owner=OWNER).value.sum() + evaluate(CHOLESKY, q[:, 0], owner=OWNER).value.sum()
            + evaluate(EDGED, q[:, 0] + 270.0, owner=OWNER).value.sum()
            + evaluate(three, q[:, 0], q[:, 1], q[:, 2], owner=OWNER).value.sum())


def test_a_step_reading_tables_passes_the_switch_scan_over_mimes_package():
    jaxpr = make_jaxpr(every_read, jnp.full((4, 3), 0.5))
    found = scan_switches([TracedStep("tables", "tables", object, jaxpr)], PACKAGE)
    scope = {(function.path, function.name) for function in found.scope}
    assert {("core/tables/_evaluate.py", "evaluate"), ("core/tables/_evaluate.py", "_located"),
            ("core/tables/_evaluate.py", "_interpolated"), ("core/tables/_evaluate.py", "_restored")} <= scope
    assert (found.hits, found.unmatched, found.unreadable_owners, found.positional_owners) == ({}, [], [], [])
    assert [(guard.path, guard.kind) for guard in found.guards] == [("core/tables/_evaluate.py", "domain_guard")]


def test_control_a_raw_cell_lookup_in_the_evaluator_is_found(monkeypatch, tmp_path):
    """The scan reads the evaluator's own source: a copy of the package whose evaluator locates a query with a raw
    ``jnp.searchsorted`` is flagged."""
    source = (PACKAGE / "core/tables/_evaluate.py").read_text(encoding="utf-8")
    planted = source.replace("index = switch.searchsorted(", "index = jnp.searchsorted(", 1)
    assert planted != source
    from mime.docs._switches import scan_definition, tree_aliases
    import ast

    tree = ast.parse(planted)
    located = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_located")
    aliases = tree_aliases(tree, "mime.core.tables._evaluate")
    found = scan_definition(located, aliases, lambda node: f"line {getattr(node, 'lineno', 0)}",
                            lambda family: family)
    assert [hit.site for hit in found.hits] == ["searchsorted"]


def test_the_loglog_logarithms_domain_guard_changes_nothing_where_its_condition_is_false():
    def value():
        return {"value": evaluate(LOGLOG, jnp.asarray([-1.0, 0.0, 5.0]), owner=OWNER).value}

    def derivative():
        return {"value": jax.jacfwd(lambda q: evaluate(LOGLOG, q, owner=OWNER).value)(jnp.asarray([-1.0, 0.0]))}

    found = fallback_problems(value, derivative, "a loglog read at and below zero")
    assert found.problems == [] and any(found.guard_sites.values())


# -- reads, compiled and as params ---------------------------------------------------------------------------------

def test_a_compiled_table_is_a_pytree_a_jitted_step_takes_and_differentiates_in_its_values():
    arrays = CUBIC.arrays(np.float64)
    query = jnp.asarray([[0.3, 0.2], [1.7, 2.5]])
    compiled = jax.jit(lambda held, q: evaluate(held, *q.T, owner=OWNER).value)
    assert np.array_equal(np.asarray(compiled(arrays, query)), read(CUBIC, *np.asarray(query)))
    gradient = jax.grad(lambda values: compiled(dataclasses.replace(arrays, values=values), query).sum())(
        jnp.asarray(arrays.values))
    assert np.isfinite(np.asarray(gradient)).all() and np.abs(np.asarray(gradient)).sum() > 0


def test_a_read_runs_in_its_querys_dtype():
    found = evaluate(CHOLESKY, jnp.asarray([0.5], dtype=jnp.float32), owner=OWNER)
    assert found.value.dtype == jnp.float32 and found.value.shape == (1, 2, 2)


def test_float64_without_x64_is_refused():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(TypeError, match="jax_enable_x64"):
            dataclasses.replace(LINEAR).arrays(np.float64)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_a_query_of_the_wrong_number_of_axes_is_refused():
    with pytest.raises(ValueError, match="2 axes, one argument each; this read passed 1"):
        evaluate(LINEAR, jnp.zeros(3), owner=OWNER)


# -- refusing on the flags -----------------------------------------------------------------------------------------

def test_the_build_finds_a_query_range_that_would_stop_the_run():
    assert query_range_problems(EDGED, [264.0], [272.0]) == []
    assert "holds the declared edge" in query_range_problems(EDGED, [264.0], [273.0])[0]
    assert "leaves the readable range" in query_range_problems(EDGED, [260.0], [270.0])[0]
    narrowed = line((263.0, 273.0, 273.0, 283.0), [918.0, 916.0, 1000.0, 999.5], edges=(273.0,),
                    source_range=(275.0, 283.0))
    extrapolated = dataclasses.replace(narrowed, extrapolation=ConstantExtrapolation("the end value"))
    assert query_range_problems(extrapolated, [274.0], [280.0]) == []
    assert "lies across the declared edge" in query_range_problems(extrapolated, [270.0], [272.5])[0]
    assert "holds the declared edge" in query_range_problems(extrapolated, [270.0], [280.0])[0]


def test_a_host_check_refuses_reads_that_stop_the_run_and_passes_the_rest():
    refuse(evaluate(EDGED, jnp.asarray([265.0, 280.0]), owner=OWNER))
    with pytest.raises(TableRefused, match="1 reads at or across a declared edge and 1 outside"):
        refuse(evaluate(EDGED, jnp.asarray([273.0, 290.0, 280.0]), owner=OWNER))
    outside = evaluate(LOGLOG, jnp.asarray([1000.0]), owner=OWNER)
    refuse(outside)
    assert not bool(outside.in_range[0]) and float(outside.value[0]) == pytest.approx(0.0625, rel=1e-12)


def test_accepted_is_false_at_and_across_an_edge_whatever_extrapolation_is_declared():
    """``accepted`` holds a ``refused`` clause: under a declared extrapolation a read outside the range is accepted,
    and a read at or across an edge is not, though it is outside the range too."""
    narrowed = line((263.0, 273.0, 273.0, 283.0), [918.0, 916.0, 1000.0, 999.5], edges=(273.0,),
                    source_range=(275.0, 283.0))
    extrapolated = dataclasses.replace(narrowed, extrapolation=ConstantExtrapolation("the end value"))
    for table in (EDGED, dataclasses.replace(EDGED, extrapolation=ConstantExtrapolation("the end value")), extrapolated):
        at_the_edge = evaluate(table, jnp.asarray([273.0]), owner=OWNER)
        assert bool(at_the_edge.refused[0]) and not bool(at_the_edge.accepted[0])
    across = evaluate(extrapolated, jnp.asarray([270.0]), owner=OWNER)
    assert bool(across.refused[0]) and not bool(across.in_range[0]) and not bool(across.accepted[0])
    outside = evaluate(extrapolated, jnp.asarray([290.0]), owner=OWNER)
    assert not bool(outside.refused[0]) and not bool(outside.in_range[0]) and bool(outside.accepted[0])


@pytest.mark.parametrize("query", [math.nan, math.inf, -math.inf])
def test_a_query_that_is_not_a_number_is_not_accepted_under_a_declared_extrapolation(query):
    table = dataclasses.replace(LINEAR, extrapolation=ConstantExtrapolation("the end value"))
    found = evaluate(table, jnp.asarray([query]), jnp.asarray([0.5]), owner=OWNER)
    assert not bool(found.accepted[0]) and not bool(found.in_range[0])
    assert bool(evaluate(table, jnp.asarray([9.0]), jnp.asarray([0.5]), owner=OWNER).accepted[0])


def test_the_build_compares_in_the_dtype_the_run_reads():
    """A query bound that float32 rounds onto a declared edge stops a float32 run and not a float64 one."""
    assert query_range_problems(EDGED, [264.0], [272.99999999]) == []
    assert "holds the declared edge" in query_range_problems(EDGED, [264.0], [272.99999999], np.float32)[0]
    assert query_range_problems(EDGED, [264.0], [272.99999999], np.float64) == []
    assert float(evaluate(EDGED, jnp.asarray([272.99999999], dtype=jnp.float32), owner=OWNER).refused[0]) == 1.0


def test_a_cholesky_table_definite_in_float64_but_not_once_rounded_is_refused_naming_its_axes():
    nearly = 1.0 - 1e-9
    knots = np.asarray([[[1.0, nearly], [nearly, 1.0]]] * 2)
    table = Table((Axis("offset", "m", (0.0, 1.0)),), knots, scheme="linear", structure="cholesky")
    assert definite(np.asarray(evaluated(table, [[0.5]]).value), np.float64)
    with pytest.raises(TableInvalid, match=r"cholesky table over \['offset'\].*float32"):
        table.arrays(np.float32)


def raw_evaluator():
    """The evaluator's own source with its switch call replaced by the index it computes (a search, or the axis's
    bucket index: ``switch._sorted_index``), no owner, resolution or indicator: the program the switch call must leave
    unchanged when no context is open."""
    path = Path(evaluate_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))

    class Raw(ast.NodeTransformer):
        replaced = 0

        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if ast.unparse(node.func) == "switch.searchsorted":
                Raw.replaced += 1
                keywords = [keyword for keyword in node.keywords if keyword.arg in ("side", "method", "buckets")]
                return ast.copy_location(ast.Call(ast.parse("switch._sorted_index", mode="eval").body, node.args,
                                                  keywords), node)
            return node

    planted = ast.fix_missing_locations(Raw().visit(tree))
    assert Raw.replaced == 1
    module = ModuleType("raw_evaluator")
    exec(compile(planted, str(path), "exec"), module.__dict__)
    return module.evaluate


def test_with_no_context_open_the_evaluators_program_is_the_raw_operations_at_its_switch_site():
    """The whole read, traced, is the program of the same evaluator with the raw index where it calls the switch
    function, so that the resolution and the indicator, lazy, add no equation (an eager one would, even one the
    switch function ignores)."""
    def reading(evaluator):
        def read_all(q):
            return (evaluator(LINEAR, q[:, 0], q[:, 1], owner=OWNER).value.sum()
                    + evaluator(EDGED, q[:, 0] + 270.0, owner=OWNER).value.sum()
                    + evaluator(LOGLOG, q[:, 0], owner=OWNER).value.sum())

        return str(make_jaxpr(read_all, jnp.full((4, 3), 0.5)))

    assert reading(evaluate) == reading(raw_evaluator())
