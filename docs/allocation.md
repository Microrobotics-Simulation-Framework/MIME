# The exact allocation

`mime.control.exact_allocation` chooses commands, within bounds you give, that come nearest the quantities you want
through an affine model: coil currents for a wanted field, gradient or force, or any command whose effect is linear. It
is JAX, it runs inside a controller's compiled step, and a derivative passes through it. `allocate()`, the coil array's
host function, runs the same law.

## What it solves

<!-- mime-fragment: a signature and its formula, with names a reader fills in -->
```python
from mime.control import exact_allocation

found = exact_allocation(rows, wanted, lower=-limit, upper=limit, scales=scales, owner="my_controller.active_set")
found.commands, found.active, found.objective

# minimises, over lower <= u <= upper:
#   sum_q ((rows @ u + offset - wanted)_q / scales_q) ** 2 + sum_k ((u - reference)_k / reference_scales_k) ** 2
```

Every array is in SI, and may be traced:

- `rows`: each quantity's row per unit command (T/A, T/(m A), N/A), one column a command;
- `wanted`, `offset` (zero unless given) and `scales`, each in its row's unit. **`scales` is required**: rows in
  different units are weighed against each other only through the scales you give, a miss of one scale counting the same
  in every row. Unweighted rows in SI favour whichever quantity has the larger numbers, a gradient in T/m over a field in
  T;
- `lower` and `upper`, one value or one per command, in the command's unit; keeping `lower <= upper` is yours;
- `reference` and `reference_scales`: with `reference_scales` the regularisation term pulls the commands towards
  `reference` (the last command, for smoothness; zero, for least effort), and the problem has one minimiser whatever the
  rows. Without it there is no such term.

Where several commands minimise it (more commands than the rows fix, as eight coils asked for a field alone), it returns
the one nearest `reference`, by the least squared distance from it: least effort for a reference of zero.

`found.active` says where each command sits in the selected active set: minus one at its lower bound, zero free, plus one
at its upper bound. `found.objective` is the minimised sum.

## How it is exact

Each command is free or at one of its bounds, so there are as many active sets as a power of three in the commands. A
set's free commands are the least squares of its free columns against what its bound commands leave (where the free
commands outnumber the rows, the solution of those rows nearest the reference), and the minimiser is the feasible
candidate of least residual, ties going to the least effort. A set's factors depend only on which commands are free, so
the law makes one QR factor per free subset and every set's candidate from its subset's by matrix products. The
selection is taken with no derivative; the selected set is then solved once more, by one QR (by its free columns'
pseudo-inverse where there are more commands than rows), and the derivative is that set's.

**Its guards.** A free subset whose triangle has a diagonal entry below `RCOND` times its largest is rank deficient: its
triangle is replaced by a stand-in before the solve and its sets are kept from the selection after it, so no singular
solve reaches a value or a derivative. A command with no effect (a column of zero length, and no regularisation) is held
at the reference clipped into its bounds, never driven to a bound by a tie. The returned commands are clipped into the
bounds, which their solve meets to rounding.

## Its bound, and why it is eight

`mime.control.EXACT_COMMANDS` is the most commands the law takes; it refuses more (`allocation.exact-size`) with its
route, the iterative path, which takes any number and arrives in a later 0.3.0 development release. **The bound is a
choice against the rule set before the cost was measured.** The rule was the largest bound at which the allocation costs
no more than three times its cost at five commands; it gives six. Eight was taken because every coil array MIME ships
has at most eight coils (Navion_2 three, CardioMag and OctoMag eight): at six, `allocate()` on CardioMag and OctoMag,
which works today, would be refused until the iterative path, or keep a second solver beside the law. Changing
`EXACT_COMMANDS` changes the bound, and nothing else depends on it. The costs measured on GitHub's runners, per base step
and against the commands, are in the CHANGELOG's entry for the exact allocation.

A slower controller pays the allocation every base step (MADDENING computes a node at a slower rate every step and
holds its result between its ticks), so the cost per base step is the one to read.

## Its derivative

Within an active set the commands are the selected set's linear solve, so the derivative is exact there. Where a command
reaches a bound or leaves it the derivative jumps (the commands stay continuous where the minimiser is unique), and it
is never NaN.

**A caller declares its own owner.** `owner` is required and has no default: the law's switches belong to the entry
that calls it, so a controller names an owner of its own (`"my_controller.active_set"`), passes that string to
`exact_allocation`, and declares the same owner in its gradient record with `allocation_switch(owner, orders)`. The
orders are a kink where the problem is regularised or its rows fix the commands, a jump where they do not (the
least-effort minimiser can move between faces). MIME's switch scan counts an owner as declared only when an entry that
MIME ships declares it; `allocate()` passes its own, `allocation.active_set`.

## Float32

Each solve is a QR on columns of unit length, so float32 loses the scaled columns' condition times its rounding, not its
square; what that costs on each array MIME ships is in the CHANGELOG's entry.

## Which path to use

- **Inside a controller's law** (a derivative wanted, or a controller in the compiled step): `exact_allocation`, with
  the rows the law makes (for a coil array, `placed_unit_fields` at the body's point).
- **Outside a run** (a script choosing currents once): `allocate(actuation, field=, gradient=, limits=, field_scale=,
  gradient_scale=)`, which runs the same law on the host in float64 and returns the currents with what they make and
  miss.
- **More commands than the bound**: neither yet; the iterative path is the route.
