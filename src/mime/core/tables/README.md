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

**A known limit of that NaN** (`MIME-ANO-115`, narrowed): the NaN follows the one-point axis's query. A read
takes one query argument per axis, so a query there that carries no tangent in the mode used (a constant, or a `jit`
argument the derivative is not taken in) leaves the derivatives along the other axes finite in forward and reverse
mode. A query there that carries one keeps the NaN, even a zero tangent given to `jvp`, and any query `jacfwd` traces:
along that axis in reverse mode, in every direction in forward mode, which no linear tangent rule avoids. A consumer
differentiating the other axes in forward mode leaves the one-point axis's query out of what it differentiates, or
uses reverse mode; a study differentiating a whole step with respect to a state that query is derived from cannot.
`tests/unit/test_tables.py` pins each case.

## Schemes and structures

| | What it is | Exact on | Across a knot |
|---|---|---|---|
| `linear` | OpenMaterica's: linear in each axis | functions linear in each axis | a kink |
| `loglog` | OpenMaterica's: linear in the logarithms of each axis and of the value, for positive ones | power laws | a kink |
| `cubic` | MIME's smooth scheme: a piecewise cubic, each knot's slope that of the parabola through it and its neighbours (a secant where a run of knots between edges has only two) | functions at most quadratic in each axis | continuously differentiable |

For MIME's own tables, `scheme` may instead be a tuple, `linear` or `cubic` for each axis (a wall table cubic in
the two radii and linear in the angle and the offset reads a quarter of the knots an all-cubic read does); a table
linear along any axis is a kink there. OpenMaterica's single scheme is unchanged.

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
not a number (NaN or infinite) is never `accepted`. The NaN is a second line only: a consumer refuses on the flags.
At build, where it knows the range its queries will span, `query_range_problems(table, low, high, dtype)` says why
reads over it would stop the run, comparing in the dtype the run reads (a bound float32 rounds onto an edge stops a
float32 run). At run time it publishes `accepted` (or the two
flags) in its node's state, and `refuse(evaluation)` raises `TableRefused` on the host where any read must stop the
run. A compiled run is not stopped from inside: the flags are the mechanism. **Neither check has a caller in `src`
yet**: the first consumer, the BEM's wall table, wires them, and its brief carries the control (a node whose
build omits the check, or whose step omits publishing the flags, fails a named test).

## The switch

Locating a query among an axis's knots is the one non-smooth operation a read evaluates, through
`mime.core.switch.searchsorted` (by the axis's bucket index where it has one), under the owner its caller passes
(`evaluate(table, *at, owner=...)`). The entry that
owns the table declares that owner; `Table.value_order` gives its order in the value: a kink for `linear` and `loglog`;
continuously differentiable for `cubic`, across every knot inside its readable range, whose ends stop the run unless
an extrapolation is declared, under which they are kinks. What the switch harness measures of it is that neither is a
jump: it checks any declared order above a jump only as "not a jump", so it does not tell the kink from the smooth
table. The cubic's continuous differentiability is carried by its own test of the one-sided derivatives at every
interior knot, and each test entry's declared order by the status test, from `Table.value_order`; the entry that owns
the BEM's table (the first consumer) ties its declared order to `Table.value_order` the same way. A table's switch is
never evaluated inside a transform: a read takes a whole batch of queries at a step's top level.

## Reading in a step

`evaluate(table, *at, owner)` reads the queries `at`, one argument per axis in the table's order, broadcast together,
in their dtype (there is no stacked `(..., axes)` form: one array holding every axis would carry a one-point axis's
NaN into the others' forward derivatives): a node's step reads in the rig's dtype, and float64 needs `jax_enable_x64`.
Each axis is located at its own argument's shape, before the arguments are broadcast, so a coordinate that many
queries share is located once: over the pairs of a body's points, pass a radius as a column and a row, not the two
broadcast to every pair. `Table.arrays(dtype)` compiles the table for a dtype once, into a `TableArrays`, a pytree of
device arrays that a node can hold in its `params` and a compiled step takes as an argument; `evaluate` takes either.
Differentiable in the queries, and in the values the table holds.

**How a read works.** Compiling a table places, on each axis, the breaks a query is located among: the readable range's
ends, its interior knots, and each edge in an interval of its own, each where the dtype puts it, so that the declared
side, the range's closed ends and the edges come out of one location. Every interval between two breaks holds what a
read there returns: inside the readable range, the polynomial of its knot cell, by its coefficients on the values of a
window of consecutive knots; outside it, the constant end value or NaN; at or across an edge, NaN. A read evaluates
each axis's polynomial into a weight per window knot, then sums the windows' values with them (below), in products and
sums only, so that no matrix unit's reduced precision reaches it.
## What a read costs

Profiled by operations, not wall time, on the lowered and compiled read (`tests/differential/table_profile.py`), and
held there by the guards in `tests/unit/test_table_operations.py`, which the read it replaced fails.

- **Located without a search.** Compiling an axis builds a bucket index of its breaks wherever it compares a query
  with no more breaks than the search takes steps (`mime.core.switch.Buckets`): equal buckets in the coordinate, or in
  its logarithm on a positive axis. A query's bucket is `floor((q - origin) * scale)`, and comparing it with the few
  breaks within a margin of that bucket gives the search's index exactly, NaN, the infinities and either zero
  included. On a uniform or log-uniform axis, and on the wall table's graded ones, that is one comparison. An axis
  wider than its dtype's range, where a query's offset from the first break overflows, or whose buckets' scale would
  be a denormal, which a compiled read flushes to zero, keeps the search.
- **Weights first.** Each axis's stencil weights are its interval's polynomial evaluated at `t`; a window's values
  are then summed with them, as its first knot's value plus each weight times a difference from it, and the windows
  the same way, as the first corner's value plus each window's weight times its sum less that value. Every stencil's
  weights sum to one, so a derivative along any axis is a weighted sum of differences, never a difference of sums the
  size of the values, and the first corner, a value of the table, adds nothing to it.
- **Gathered together where the batch is small, looped where it is large.** The last axis's window, its knots'
  values contiguous, is gathered for each combination of the leading axes' window knots. As many of the leading axes
  (those nearest the last) are gathered together as keep one gather within the table's `gathered_at_once` elements;
  the rest are looped over, one gather an iteration, and XLA fuses each gather into its sums. A step's few queries
  are read in one gather and a few kernels, where a loop would launch a kernel or more an iteration; millions are read
  in a loop that holds one iteration's windows, never every corner at once, in a program of one iteration's size.
- **Located once per distinct coordinate.** Each axis is located at its own argument's shape, before the arguments
  are broadcast: a read over pairs of points, a radius given as a column and a row, locates each point's radius once.

Reverse mode recomputes, by default, the gathered windows in its backward pass rather than keep them between its
passes (`recomputed_in_reverse`): it keeps a few numbers per axis per query, where keeping the windows keeps every
corner's values. On the CPU that is no trade: it keeps fewer bytes, gathers fewer elements and launches fewer kernels
than keeping them, for a few more operations, and a reverse derivative takes less time. That was measured on the CPU
only; the GPU is unmeasured.

### Settings that trade memory for time

| Setting | Default | The other choice |
|---|---|---|
| `stored_polynomials` | off: the values, one per knot | each knot cell's polynomial, by power of each axis's `t`: the product over the axes of their degrees plus one, per cell and value component, so that a read gathers its cell's coefficients and evaluates them with no stencil to combine |
| `float32_values` | off: the values in the dtype a read runs in | the values in float32 under a float64 read: half the table's memory and of the bytes a read gathers; each value rounded once to float32 (below) |
| `recomputed_in_reverse` | on: reverse mode recomputes the gathered windows | reverse mode keeps them between its passes: every corner's values per query kept |
| `gathered_at_once` | `GATHERED_AT_ONCE` elements, half a megabyte in float64 | zero: one combination of the leading axes' window knots a gather, the least memory and the most kernels; or more, for fewer kernels at larger batches |

The defaults keep a table at its values' memory and a read at a step's standard derivative or less, except that a
read of few queries gathers up to `gathered_at_once` elements at once, for fewer kernels: its temporaries are then a
few times that gather's, whatever the batch, since a larger batch gathers fewer axes together.

**Float32 values.** Each value is rounded once to float32, in the form the table stores it (the logarithms of a
`loglog` table's values, the Cholesky factor of a `cholesky` table's), so a plain read lies within float32's rounding
of the table's largest value times the sum of its corners' weight magnitudes, and its derivative within the same
times the sum of their derivatives' magnitudes. Measured against the float64-valued read over drawn tables, a
derivative differs by up to 6.5 units of float32's rounding of the read's largest value over the axis's smallest cell
on a plain table, and up to 21 on a Cholesky one, whose product `L L^T` multiplies its factor's rounding: a
measurement, not a bound. A table whose stored values float32 holds only past its range (a read would return no
number) or below its normal numbers (a compiled read flushes them to zero) is refused when it is made.

**Recorded.** The settings are the table's, beside its scheme and structure. Nothing saves a `Table` yet: they are
recorded when tables enter an experiment's saved form, with their first consumers (phases 7h and 7i).

## Its oracles

- OpenMaterica's table-evaluation cases, every one of them, in float64 and float32, with each side declared, with and
  without an extrapolation (`tests/unit/test_table_cases.py`), with controls for the faults the cases name.
- Analytic functions each scheme reproduces, central finite differences, one-sided derivatives at the knots, and
  positive definiteness over resistance-like draws at physical scales (`tests/unit/test_tables.py`).
- The v0.2 line's wall-table lookup, the weakest oracle, on two blocks of its own grid (the near-wall corner, and the
  axis end with every angle and offset)
  (`tests/differential/test_port_table.py`), and its per-lookup cost against that lookup's
  (`tests/differential/table_budget.py`).
- A long-double evaluation of the same compiled coefficients over drawn tables, values and forward derivatives
  (`tests/differential/table_accuracy.py`, run by hand, never collected): the measurement the read's declared
  rounding rests on. A `cholesky` table's derivatives are judged on the read's largest value over the axis's
  smallest cell times one plus twice its stored factor form's largest magnitude, since `L L^T` multiplies the
  factor's derivative by the factor.

## Status

Since phase 7f: the description, the three schemes and the Cholesky structure, the flags, the build-time and
host-side refusals, and `switch.searchsorted`. Since the performance pass ahead of phase 7h: the bucket index, the
read by windows, a scheme per axis and the three settings that trade memory for time. Its first consumer is the BEM's
wall table, in phase 7h; `MaterialAt`, a2's coil maps and the table surrogate type come later.

Phases and releases are explained in the [package manifest](../../README.md#status).
