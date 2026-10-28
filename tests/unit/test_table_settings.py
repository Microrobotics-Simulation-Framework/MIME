"""The table evaluator's settings beyond OpenMaterica's form, each against the read it changes: a scheme per axis, the
three settings that trade memory for time, and a read whose arguments share a coordinate, which locates it once. Each
is held to what its change can do to the arithmetic, stated before it was measured:

- **Bit for bit**: the same arithmetic. A scheme given per axis, the same on every axis, is the single scheme; a read
  whose arguments share a coordinate is the read of them broadcast first; a read recomputed in reverse mode has the
  read's value.
- **Within a stated rounding**: the same polynomial, its arithmetic reordered. Stored polynomials (computed in float64
  when the table is compiled, evaluated by powers of ``t``) read the stencil's within 64 roundings of the read's
  largest value, and its derivatives within 64 roundings of that value over the axis's smallest cell, the scale of
  every derivative a cell's polynomial has; the derivatives of a read recomputed in reverse mode, which XLA fuses
  differently around the recomputation, within 16 of the same.
- **Within a stated accuracy**: float32 values under a float64 read are each rounded once to float32, so a read is
  within ``2**-24`` of the table's largest value times the largest sum of the magnitudes of a read's corner weights,
  which a stencil bounds (twice the largest value covers every case here).

The cases are fixed tables of every scheme and structure, one to four axes, with an edge, a narrowed range and an
extrapolation among them, read compiled with the table an argument, at fixed queries: a property over drawn tables
would compile a read per draw."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core.tables import Axis, ConstantExtrapolation, Table, TableInvalid, evaluate

OWNER = "toy.table"


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


def smooth(mesh: np.ndarray) -> np.ndarray:
    return np.sin(1.3 * mesh[..., 0]) * np.cos(0.7 * mesh[..., -1]) + np.exp(0.2 * mesh.sum(axis=-1))


def case(name: str) -> Table:
    """A fixed table: its grids graded, uniform, log-uniform, edged or narrowed by a source range."""
    graded = tuple(float(v) for v in np.concatenate([[0.0], np.cumsum(0.2 * 1.4 ** np.arange(5))]))
    uniform = tuple(-1.0 + 0.5 * k for k in range(7))
    edged = Axis("e", "m", (0.0, 0.4, 1.0, 1.0, 1.5, 2.2), edges=(1.0,))
    axes = {"graded": Axis("g", "m", graded), "uniform": Axis("u", "m", uniform),
            "log": Axis("l", "Hz", tuple(float(v) for v in 0.1 * 3.0 ** np.arange(6))),
            "narrowed": Axis("n", "m", uniform, source_range=(-0.8, 1.7)), "edged": edged}
    kinds, scheme, structure, extrapolated = {
        "linear-1": (("graded",), "linear", "plain", False),
        "loglog-2": (("log", "log"), "loglog", "plain", True),
        "cubic-2-edged": (("edged", "graded"), "cubic", "plain", True),
        "mixed-3": (("uniform", "narrowed", "graded"), ("cubic", "linear", "cubic"), "plain", False),
        "cholesky-2": (("graded", "uniform"), "cubic", "cholesky", False),
        "cubic-4": (("graded", "uniform", "narrowed", "graded"), "cubic", "plain", False),
    }[name]
    held = tuple(dataclasses.replace(axes[kind], name=f"a{i}") for i, kind in enumerate(kinds))
    mesh = np.stack(np.meshgrid(*(np.asarray(axis.x) for axis in held), indexing="ij"), axis=-1)
    values = smooth(mesh)
    if scheme == "loglog":  # a power law in each axis, varied
        values = 2.0 * mesh[..., 0] ** 0.7 * mesh[..., 1] ** -0.4 * (1.1 + 0.1 * np.sin(mesh[..., 0]))
    if structure == "cholesky":
        factor = np.eye(3) * (1.0 + 0.2 * values[..., None, None]) + 0.1 * np.tril(np.ones((3, 3)), -1)
        values = factor @ np.swapaxes(factor, -1, -2)
    else:
        values = values[..., None] * np.asarray([1.0, 0.5 if scheme == "loglog" else -0.5])
    return Table(held, values, scheme=scheme, structure=structure,
                 extrapolation=ConstantExtrapolation("the end value") if extrapolated else None)


CASES = ["linear-1", "loglog-2", "cubic-2-edged", "mixed-3", "cholesky-2", "cubic-4"]


def points(table: Table) -> list[np.ndarray]:
    """Per axis, 48 queries: inside the readable range, at its knots, and, under an extrapolation, outside it."""
    rng = np.random.default_rng(len(table.axes))
    found = []
    for axis in table.axes:
        low, high = axis.readable_range
        inside = rng.uniform(low, high, 40)
        knots = rng.choice([v for v in axis.x if low <= v <= high and v not in axis.edges], 4)
        outside = [low - 0.1, high + 0.1] * 2 if table.extrapolation is not None else rng.uniform(low, high, 4)
        found.append(np.concatenate([inside, knots, outside]))
    return found


@jax.jit
def _reads(arrays, columns):
    """The value and flags, the forward derivative along every axis at once (a unit step on each) and the reverse
    derivative of the values' sum along each axis."""
    def value(*cs):
        return evaluate(arrays, *cs, owner=OWNER).value

    found = evaluate(arrays, *columns, owner=OWNER)
    forward = jax.jvp(value, tuple(columns), tuple(jnp.ones_like(c) for c in columns))[1]
    reverse = jax.grad(lambda *cs: jnp.sum(value(*cs)), argnums=tuple(range(len(columns))))(*columns)
    return found, [forward], list(reverse)


def reads(table: Table, dtype=np.float64):
    found, forward, reverse = _reads(table.arrays(dtype), [jnp.asarray(p, dtype=dtype) for p in points(table)])
    return (*(np.asarray(v) for v in found),), [np.asarray(f) for f in forward], [np.asarray(r) for r in reverse]


def scales(table: Table, value: np.ndarray) -> list[float]:
    """Each axis's derivative scale: the read's largest value over the axis's smallest cell (in logarithms on a
    loglog table, whose value's derivative is the value's times the logarithm's)."""
    largest = float(np.nanmax(np.abs(value)))
    cells = [np.diff(np.unique(np.log(axis.x) if table.scheme == "loglog" else axis.x)) for axis in table.axes]
    lowest = [float(np.min(axis.x)) if table.scheme == "loglog" else 1.0 for axis in table.axes]
    return [largest / float(np.min(cell)) / low for cell, low in zip(cells, lowest, strict=True)]


def worst(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """The largest difference where both are numbers, in roundings of ``scale``; a NaN in one and not the other is
    infinitely far."""
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    both = ~np.isnan(a)
    return float(np.max(np.abs(a[both] - b[both]), initial=0.0) / scale / np.finfo(a.dtype).eps)


# -- a scheme per axis ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("scheme", "complaint"), [(("cubic",), "each of the table's 2 axes"),
                                                   (("cubic", "loglog"), "one of linear and cubic"),
                                                   (("cubic", "spline"), "one of linear and cubic")])
def test_a_scheme_per_axis_is_linear_or_cubic_for_every_axis(scheme, complaint):
    with pytest.raises(TableInvalid, match=complaint):
        Table((Axis("a", "m", (0.0, 1.0)), Axis("b", "m", (0.0, 1.0))), np.zeros((2, 2)), scheme=scheme)


def test_a_table_cubic_on_every_axis_is_smooth_and_linear_on_any_a_kink():
    axes = (Axis("a", "m", (0.0, 1.0, 2.0)), Axis("b", "m", (0.0, 1.0, 2.0)))
    assert Table(axes, np.zeros((3, 3)), scheme=("cubic", "cubic")).value_order == 1
    assert Table(axes, np.zeros((3, 3)), scheme=("cubic", "linear")).value_order == 0


_EXACT = jax.jit(lambda arrays, columns: evaluate(arrays, *columns, owner=OWNER).value)


@pytest.mark.parametrize("schemes", [("cubic", "linear", "cubic"), ("linear", "cubic", "linear")])
@settings(deadline=None)
@given(coefficients=st.lists(st.floats(-3.0, 3.0), min_size=8, max_size=8))
def test_a_scheme_per_axis_is_exact_on_functions_quadratic_along_its_cubic_axes_and_linear_along_the_others(
        schemes, coefficients):
    """A product of a quadratic along each cubic axis and a line along each linear one, plus a constant: each axis's
    own scheme reproduces its factor."""
    grid = case("mixed-3").axes
    factors, at = [], 0
    for scheme in schemes:
        degree = 2 if scheme == "cubic" else 1
        factors.append(np.asarray(coefficients[at:at + degree + 1]))
        at += degree + 1

    def function(p):
        return np.prod([np.polyval(c, p[..., a]) for a, c in enumerate(factors)], axis=0) + 10.0

    mesh = np.stack(np.meshgrid(*(np.asarray(axis.x) for axis in grid), indexing="ij"), axis=-1)
    table = Table(grid, function(mesh), scheme=schemes)
    at_points = points(table)
    found = np.asarray(_EXACT(table.arrays(np.float64), [jnp.asarray(p) for p in at_points]))
    assert np.max(np.abs(found - function(np.stack(at_points, axis=-1)))) <= 2048 * np.finfo(float).eps * \
        np.max(np.abs(table.y))


@pytest.mark.parametrize("name", ["linear-1", "cubic-2-edged", "cholesky-2", "cubic-4"])
def test_a_scheme_per_axis_the_same_on_every_axis_is_that_scheme_bit_for_bit(name):
    table = case(name)
    single, per_axis = reads(table), reads(dataclasses.replace(table, scheme=table.schemes))
    for a, b in zip(jax.tree_util.tree_leaves(single), jax.tree_util.tree_leaves(per_axis), strict=True):
        assert np.array_equal(a, b, equal_nan=True)


# -- a read whose arguments share a coordinate ---------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_read_whose_arguments_share_a_coordinate_is_the_read_of_them_broadcast_bit_for_bit(dtype):
    """The BEM's form: every pair of 24 points, the first two axes a point's own coordinate as a column and a row, the
    others per pair. Value, flags, forward mode's derivative along every axis, and reverse mode's along each pair's
    own axes (a radius's reverse derivative sums its pairs', in an order of XLA's)."""
    table = case("cubic-4")
    rng = np.random.default_rng(7)
    ranges = [axis.readable_range for axis in table.axes]
    radius = rng.uniform(*ranges[0], 24)
    shared = [radius[:, None], rng.uniform(*ranges[1], 24)[None, :], rng.uniform(*ranges[2], (24, 24)),
              rng.uniform(*ranges[3], (24, 24))]

    @jax.jit
    def everything(arrays, held):
        found = evaluate(arrays, *held, owner=OWNER)
        forward = [jax.jvp(lambda x, a=a: evaluate(arrays, *(x if b == a else c for b, c in enumerate(held)),
                                                   owner=OWNER).value, (held[a],), (jnp.ones_like(held[a]),))[1]
                   for a in range(4)]
        reverse = jax.grad(lambda *cs: jnp.sum(evaluate(arrays, *cs, owner=OWNER).value), argnums=(2, 3))(*held)
        return [*found, *forward, *reverse]

    arrays = table.arrays(dtype)
    pairs = everything(arrays, [jnp.asarray(c, dtype=dtype) for c in shared])
    broadcast = everything(arrays, [jnp.asarray(np.broadcast_to(c, (24, 24)), dtype=dtype) for c in shared])
    for a, b in zip(pairs, broadcast, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# -- the settings that trade memory for time ------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
def test_stored_polynomials_read_the_stencils_polynomial_within_its_rounding(name):
    table = case(name)
    on_read, stored = reads(table), reads(dataclasses.replace(table, stored_polynomials=True))
    assert all(np.array_equal(a, b) for a, b in zip(on_read[0][1:], stored[0][1:], strict=True))  # the flags
    assert worst(on_read[0][0], stored[0][0], float(np.nanmax(np.abs(on_read[0][0])))) <= 64
    each = scales(table, on_read[0][0])
    assert worst(on_read[1][0], stored[1][0], sum(each)) <= 64
    for reverse, reverse_stored, scale in zip(on_read[2], stored[2], each, strict=True):
        assert worst(reverse, reverse_stored, scale) <= 64


def test_stored_polynomials_hold_each_cells_coefficients_by_power_of_each_axis():
    """Two cubic axes of five and four knots and a value of two components: four by three cells, four powers each
    way."""
    table = Table((Axis("a", "m", (0.0, 1.0, 2.0, 3.0, 4.0)), Axis("b", "m", (0.0, 0.5, 1.5, 2.0))),
                  np.zeros((5, 4, 2)), scheme="cubic", stored_polynomials=True)
    assert table.arrays(np.float64).values.shape == (4, 3, 4, 4, 2)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("name", CASES)
def test_a_read_recomputed_in_reverse_mode_is_the_read(name, dtype):
    table = case(name)
    kept, recomputed = reads(table, dtype), reads(dataclasses.replace(table, recomputed_in_reverse=True), dtype)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(kept[0], recomputed[0], strict=True))
    each = scales(table, kept[0][0])
    assert worst(kept[1][0], recomputed[1][0], sum(each)) <= 16
    for reverse, reverse_recomputed, scale in zip(kept[2], recomputed[2], each, strict=True):
        assert worst(reverse, reverse_recomputed, scale) <= 16


@pytest.mark.parametrize("name", CASES)
def test_float32_values_under_a_float64_read_are_within_their_rounding(name):
    table = case(name)
    exact, rounded = reads(table)[0][0], reads(dataclasses.replace(table, float32_values=True))[0][0]
    assert rounded.dtype == np.float64
    assert worst(exact, rounded, 1.0) * np.finfo(np.float64).eps <= 2.0 * 2.0 ** -24 * np.nanmax(np.abs(exact))


def test_float32_values_halve_a_float64_tables_values_and_keep_the_read_in_float64():
    arrays = Table((Axis("a", "m", (0.0, 1.0, 2.0)),), np.arange(3.0), float32_values=True).arrays(np.float64)
    assert arrays.values.dtype == np.float32 and arrays.dtype == np.float64
    assert evaluate(arrays, jnp.asarray([0.5]), owner=OWNER).value.dtype == np.float64
