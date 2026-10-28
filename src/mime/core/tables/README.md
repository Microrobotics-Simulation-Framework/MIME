# `mime.core.tables`

**The table evaluator**: one differentiable reader of tables, in pure JAX, for MIME's own solver-output tables (the
BEM's wall correction, a resistance, a field map) and for OpenMaterica's material tables alike. Nodes import it, so it
lives in `mime.core` and imports nothing from MIME above it; `mime.materials` converts an OpenMaterica record's table
into it (`evaluator_table`), and is the only importer of OpenMaterica.

## A table

`Table(axes, y, scheme, structure, side, extrapolation)` is OpenMaterica's table description (its record format,
§6) on as many axes as the table has:

- each `Axis` holds the condition it tabulates (`name`) and its `unit`, its knots `x`, its declared `edges` and its
  `source_range`, as OpenMaterica's `axis`, `axis_unit`, `x`, `edges` and `source_range`. Its **readable range** runs
  from its first knot to its last, narrowed (never widened) by the source range, ends included;
- `y` holds the values, its leading dimensions the axes' knots in order, the rest the shape of one value;
- `scheme` is how the table interpolates along each axis, `structure` whether it keeps a structure of its values,
  `side` the side whose derivative a read returns at an interior knot, and `extrapolation` the consumer's declared
  extrapolation past the readable range (`ConstantExtrapolation`, with its basis), or none.

Every rule of OpenMaterica's form is checked when a table is made, and a table that breaks one is refused
(`TableInvalid`, OpenMaterica's `form_invalid`). So are MIME's own conditions, when the table is compiled for a dtype:
knots the dtype cannot tell apart, a knot, edge or range end that is a denormal there (a compiled read flushes it to
zero; zero itself is an ordinary coordinate, and the break beside it is the smallest normal number), and a table too
large for a read's int32 index.

A readable range may be **one point** (OpenMaterica's tables T12 and T13): a read there is the value at the point, with
a derivative that is NaN (`mime.core.switch.without_slope`; a point has no inside to differentiate along), and a read
elsewhere is outside the range.

**A known limit of that NaN** (`MIME-ANO-115`, undecided): in forward mode, a read whose queries are traced is NaN in
every direction, the other axes' included, once a query sits in a one-point range; in reverse mode it is NaN along the
one-point axis only. A consumer differentiating such a read in forward mode drops the one-point axis from the table
first, or uses reverse mode. `tests/unit/test_tables.py` pins the limit and changes with the decision.

## Schemes and structures

| | What it is | Exact on | Across a knot |
|---|---|---|---|
| `linear` | OpenMaterica's: linear in each axis | functions linear in each axis | a kink |
| `loglog` | OpenMaterica's: linear in the logarithms of each axis and of the value, for positive ones | power laws | a kink |
| `cubic` | MIME's smooth scheme: a piecewise cubic, each knot's slope that of the parabola through it and its neighbours (a secant where a run of knots between edges has only two) | functions at most quadratic in each axis | continuously differentiable |

`structure="cholesky"` takes values that are symmetric positive definite matrices (a resistance) and interpolates
their Cholesky factor, the logarithms of its diagonal and the entries below it, with the scheme: every value it
returns is `L L^T` with a positive diagonal in `L`, so positive definite by construction. Interpolating the entries
themselves with the cubic scheme does not keep definiteness.

**The float32 limit.** The construction is exact in exact arithmetic; in a dtype the product `L L^T` is representably
definite only while the factor's condition number squared is well below the reciprocal of the dtype's epsilon. On a
graded grid (cells within a factor of two of their neighbours') that holds; on an ungraded one, where the cubic's slopes
overshoot across a long cell, a value can be definite in float64 and not float32. A table whose values are definite in
float64 but not once rounded to the dtype is refused when compiled in it (`TableInvalid`, naming the table's axes).

The potential variant, divergence- and curl-free, waits for its first consumer (a2's coil maps).

## Outside the readable range, and at an edge

A table is **never clipped silently**. A read returns, beside the value:

- `in_range`: whether the query lies in the readable range on every axis;
- `refused`: whether it lies at a declared edge on some axis, or across one from the readable range, where the table
  gives no value whatever is declared;
- `accepted`: whether the run may go on: not refused, and in range or under the declared extrapolation.

Outside the readable range the value is the value at the range's nearest end, taken from inside it, with the derivative
zero, under a declared extrapolation, and no number (NaN) without one; at or across an edge, no number. A query that is
not a number (NaN or infinite) is never `accepted`. The NaN is a
second line only: a consumer refuses on the flags. At build, where it knows the range its queries will span,
`query_range_problems(table, low, high, dtype)` says why reads over it would stop the run, comparing in the dtype the
run reads (a bound float32 rounds onto an edge stops a float32 run). At run time it publishes `accepted` (or the two
flags) in its node's state, and `refuse(evaluation)` raises `TableRefused` on the host where any read must stop the
run. A compiled run is not stopped from inside: the flags are the mechanism. **Neither check has a caller in `src`
yet**: the first consumer, the BEM's wall table, wires them, and its brief carries the control (a node whose
build omits the check, or whose step omits publishing the flags, fails a named test).

## The switch

Locating a query among an axis's knots is the one non-smooth operation a read evaluates, through
`mime.core.switch.searchsorted`, under the owner its caller passes (`evaluate(table, at, owner=...)`). The entry that
owns the table declares that owner; `Table.value_order` gives its order in the value: a kink for `linear` and `loglog`;
continuously differentiable for `cubic`, across every knot inside its readable range, whose ends stop the run unless
an extrapolation is declared, under which they are kinks. What the switch harness measures of it is that neither is a
jump: it checks any declared order above a jump only as "not a jump", so it does not tell the kink from the smooth table. The
cubic's continuous differentiability is carried by its own test of the one-sided derivatives at every interior knot,
and each test entry's declared order by the status test, from `Table.value_order`; the entry that owns the BEM's table
(the first consumer) ties its declared order to `Table.value_order` the same way. A table's switch is never evaluated inside a
transform: a read takes a whole batch of queries at a step's top level.

## Reading in a step

`evaluate(table, at, *, owner)` reads the queries `at`, of shape `(..., axes)`, in their dtype: a node's step reads in
the rig's dtype, and float64 needs `jax_enable_x64`. `Table.arrays(dtype)` compiles the table for a dtype once, into a
`TableArrays`, a pytree of device arrays that a node can hold in its `params` and a compiled step takes as an argument;
`evaluate` takes either. Differentiable in the queries, and in the values the table holds.

**How a read works.** Compiling a table places, on each axis, the breaks a query is located among: the readable range's
ends, its interior knots, and each edge in an interval of its own, each where the dtype puts it, so that the declared
side, the range's closed ends and the edges come out of one location. Every interval between two breaks holds what a
read there returns: inside the readable range, the polynomial of its knot cell, by its coefficients on the values of a
few knots; outside it, the constant end value or NaN; at or across an edge, NaN. A read gathers those knots' values and
interpolates one axis after another by Horner's rule, in products and sums only, so that no matrix unit's reduced
precision reaches it.

## Its oracles

- OpenMaterica's table-evaluation cases, every one of them, in float64 and float32, with each side declared, with and without an
  extrapolation (`tests/unit/test_table_cases.py`), with controls for the faults the cases name.
- Analytic functions each scheme reproduces, central finite differences, one-sided derivatives at the knots, and
  positive definiteness over resistance-like draws at physical scales (`tests/unit/test_tables.py`).
- The v0.2 line's wall-table lookup, the weakest oracle, on two blocks of its own grid (the near-wall corner, and the
  axis end with every angle and offset)
  (`tests/differential/test_port_table.py`), and its per-lookup cost against that lookup's
  (`tests/differential/table_budget.py`).

## Status

Since phase 7f: the description, the three schemes and the Cholesky structure, the flags, the build-time and
host-side refusals, and `switch.searchsorted`. Its first consumer is the BEM's wall table, in phase 7h; `MaterialAt`,
a2's coil maps and the table surrogate type come later.

Phases and releases are explained in the [package manifest](../../README.md#status).
