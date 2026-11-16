# Allocation: the exact path and the iterative path

`mime.control.exact_allocation` chooses commands, within bounds you give, that come nearest the quantities you want
through an affine model: coil currents for a wanted field, gradient or force, or any command whose effect is linear. It
is JAX, it runs inside a controller's compiled step, and a derivative passes through it. `allocate()`, the coil array's
host function, runs the same law. `mime.control.iterative_allocation` takes what the exact path does not: more commands
than it enumerates, and models that are not affine (an arm's frame from its joint positions), by projected Newton, its
derivative the implicit function's. A controller's `ctx.allocator` chooses between them when its law is made.

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
route, the iterative path, which takes any number (`ctx.allocator` takes it by itself above the bound). **The bound is a
choice against the rule set before the cost was measured.** The rule was the largest bound at which the allocation costs
no more than three times its cost at five commands; it gives six. Eight was taken because every coil array MIME ships
has at most eight coils (Navion_2 three, CardioMag and OctoMag eight): at six, `allocate()` on CardioMag and OctoMag,
which works today, would move to the iterative path, whose answer is the minimiser to its tolerance, not by enumeration. Changing
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

## Where it stops being exact

The law is exact to rounding for the problems it is given, except in one corner it does not refuse. Where the commands
outnumber the rows and there is no regularisation, a free subset with more commands than rows is solved in the commands'
own units, because the least-effort solution it returns is a statement about the commands, not about the commands
rescaled, and its rank test reads that solve's triangle at `RCOND`. One column whose norm exceeds the others' by more than
about the reciprocal of `RCOND` makes that test call the whole free set rank deficient, the set is dropped, and the
objective returned is above the optimum with nothing said. **Measured** on random rows of four commands against three
(the test in `tests/control/test_exact_allocation.py`, against SciPy's BVLS on the columns normalised): the objective is
the optimum to rounding up to a ratio of about a billion between one column's norm and the others' in float64, and ten
thousand in float32, the edge one decade below the reciprocal of `RCOND` in each; a decade and a half past it the
objective is far above, by a large share of the wanted values' squared size. The CHANGELOG's entry has the figures. Where
commands do not outnumber the rows, or the problem is regularised (it then always has more rows than commands), every
subset is solved on columns of unit length and no such limit was found (held to a ratio of a trillion in float64 and a
million in float32). The arrays MIME ships span about two decades. Caller-given `scales` act on the rows and do not
remove it; a caller whose columns span more than the limit changes its commands' unit (the matrix's columns times a
column scale, the bounds times it, the commands divided by it afterwards) so that the columns are alike, at the price that
the least-effort tie-break then reads the least rescaled command.

## Float32

Each solve is a QR on columns of unit length, so float32 loses the scaled columns' condition times its rounding, not its
square; what that costs on each array MIME ships is in the CHANGELOG's entry.

## Through the controller's context: `ctx.allocator`

A controller's law can ask its context for the allocation instead of writing the rows itself. The model is the
controller's own, its belief of the apparatus (`ctx.believed`, the true declaration where it declares none), and the
bounds are the commands' ports':

<!-- mime-fragment: a law's lines, with names a reader fills in -->
```python
solve = ctx.allocator("emns", commands=("currents",), objective={"force": N(1e-3)}, owner="my_controller.active_set")

def step(state, reads):
    commands, report = solve({"force": wanted}, {"point": point, "moment": moment}, {"currents": state["last"]})
```

- **`objective`** names each quantity of the model to come near (a coil array's `flux_density`, `gradient`, and `force`
  and `torque` on the moment given as `at["moment"]`) with its scale, in its own unit: a miss of one scale counts the
  same in every row. For a fixed moment the force, `F_j = m_i dB_i/dx_j`, and the torque, `m x B`, are linear in the
  field and its gradient, so they stay affine in the currents and the allocation stays exact. A gradient's rows are its
  five independent entries, as `allocate()` stacks them. The model's `gradient` is declared in the base units MIME writes
  for a field gradient (the form `unit_symbol` gives it), which the scale is checked against: write a scale in
  `T_per_m` or `mT_per_m`.
- **The bounds** are the port's declared `bounds`, else the bounds its receiver states (an arm's joint limits). A command
  with neither is refused (`allocation.bounds`): the exact allocation chooses within bounds.
- **The slew.** Where the port declares a `slew`, the bounds are narrowed to `last` plus or minus the slew times the
  base step times `g` base steps: what the stage delivers before the next write arrives. For a port sampling every `R`
  base steps and a controller writing every `P`, `g = R * floor(P / R)` where `R <= P`, else `R / ceil(R / P)`: with the
  port's period at the controller's or dividing it, every write is sampled and the next is at least `R * floor(P / R)`
  steps later; with the port slower, one sample can take several writes' moves at once, with `R` steps to make them.
  Both give `P` where one period divides the other, and a latency on either side delays every write alike and changes
  neither.
  **The rule assumes the stage stood at `last`** (the commands the law wrote last), which holds by induction only if
  every earlier command was deliverable. It does not hold on the first tick from rest where the law's initial `last` is
  not the port's `rest`, nor after a `trip` latches the port to its `rest`: the stage is elsewhere, and the allocation
  asks for more than it delivers. The build report says so beside each boxed command. A trip is flagged by the run
  (`tripped_<command>`, a warning); a delivery the slew limited raises no flag, and is read from the port's recorded
  traffic (`run(record=["ports:<apparatus>.written_<command>", "ports:<apparatus>.<command>"])`).
- **`effort`** adds the regularisation towards zero or the last command (`toward="last"`); without it, ties among
  several minimisers go to the commands nearest that reference.
- **`owner`** is the switch's owner, which the controller declares, as for `exact_allocation`.

**The path is chosen when the law is made**, from the model's declaration and the commands' count alone, never from a
measured time: the exact path where the model is affine in every command asked for (`affine_in`), there are at most
`EXACT_COMMANDS` of them and each quantity's miss is the difference of its rows; the iterative path otherwise (an arm's
joint positions; nine coils; a frame's orientation, whose miss is a rotation). The build report names each allocation's
path and why.

What it refuses, at the build, each with its own code: commands that name no one part of the apparatus
(`allocation.model`), an objective naming no quantity of the model or a scale of another unit (`allocation.objective`),
an iterative allocation with no effort whose rows cannot fix its commands (`allocation.objective`: three rows of a
frame's position over six joints; give `effort=`), and a command with no bounds (`allocation.bounds`). It is one function
in both modes: traced inside the compiled step, called on the host as a callback, where no derivative passes through it.

**What it bounds, and what it estimates.** On camera spike R2's sphere the allocator writes the currents a hand-written
law writes from the same rows, bounds and scales, bit for bit in float64; its first tick in float32 agrees to rounding
(`tests/control/test_allocator.py`). That says the allocator is the law, through the controller's model; whether the
model is the hardware is the model's own claim (a software oracle's, for the coil arrays: see the calibrated
multipole's validity).

**A coil array mounted on its chain's frame** (`CalibratedMultipole(mount="motor.rotor", ...)`): its model takes the
frame's world pose as an input, `x["mount_pose"]`, and stays affine in the currents at every pose, so the allocation
is exact at each tick: `solve({"flux_density": wanted}, {"point": point, "mount_pose": frames[row]})`. At each tick of a
moving chain the currents written make the wanted field at the pose the controller read, by the host's `actuation`
there, and are the host's own exact allocation at that pose (`tests/fields/test_mounted_array.py`). The field delivered
later is the array's at the pose the chain has moved to by then. The law reads the frame's pose as a state read
(`reads={"frames": "<apparatus>.<segment>.frame_poses"}`), which a controller inside the compiled step cannot
make: its node runs first in each step and each of its read edges must carry the state at the step's start, and nothing
it writes reaches the chain, so the schedule runs the chain before it and the read would carry the pose at the step's
end. The build refuses it (`controller.unreached-read`, from `mime.build._in_graph.order_refusals`) rather than give the
law a late pose; a callback (`mode="callback"`) reads the pose at the start of its period.

## Which path to use

- **Inside a controller's law, through its model and its ports**: `ctx.allocator(...)`, above.
- **Inside a controller's law, with rows of its own** (a derivative wanted, or a controller in the compiled step):
  `exact_allocation`, with the rows the law makes (for a coil array, `placed_unit_fields` at the body's point).
- **Outside a run** (a script choosing currents once): `allocate(actuation, field=, gradient=, limits=, field_scale=,
  gradient_scale=)`, which runs the same law on the host in float64 and returns the currents with what they make and
  miss.
- **More commands than the bound, or a model that is not affine**: `ctx.allocator` takes the iterative path by itself;
  with rows of your own, `iterative_allocation`.

## The iterative path

<!-- mime-fragment: a signature and what it returns, with names a reader fills in -->
```python
from mime.control import iterative_allocation

found = iterative_allocation(rows, start, lower=lower, upper=upper, owner="my_controller.active_set")
found.commands, found.active, found.objective, found.converged, found.stationarity, found.iterations

# minimises, over lower <= u <= upper:  sum_i rows(u)_i ** 2 / 2
```

`rows` maps the commands to rows, each a miss over its scale (and an effort's rows, where you want one), in JAX; it may
close over traced values (what is wanted, a believed model's numbers), and each carries its derivative. The bounds may be
infinite. It starts from `start`, projected into the bounds: on a model that is not affine the minimiser it returns is
the local one reached from there (the last command, in `ctx.allocator`), which need not be the nearest. Unregularised,
near a singular pose of an arm, Newton's step along the direction the rows barely fix is long: on the AR4's wrist the
solve can end at a joint's limit far from its start, a constrained minimiser off the pose, where a damped solve met the
pose close by. A slew box around the start bounds how far the solve can go.

**The method** is Bertsekas's projected Newton, in commands scaled to unit columns: an epsilon-active set; Newton's step
on the free commands where the exact Hessian is positive definite there, Gauss-Newton's where it is not (far from a
minimiser); a scaled gradient step on the held ones; the bounds as a projection and Armijo's test on the projection arc,
nonmonotone (Grippo, Lampariello and Lucidi) for Newton's step on a positive definite Hessian that no bound cuts: it is
judged against the largest objective of the last four points, since in the curved valley of an arm's redundant
joints that step first raises the objective and then lowers it past where it began (the Maratos effect); judged against
the current point alone, those steps were halved at every other iteration, and a warm start near the minimiser took the
whole cap. A step the box cuts, and a Gauss-Newton step, are judged against the current point: with the allowance, a
box's corners can trade places for the whole cap. Where a step had to be cut, the next is damped (Levenberg's term,
small, raised at each further cut and lowered at each step taken whole), so that on a curved valley's floor the
steps shorten and turn towards the gradient instead of crawling by halvings to the cap. The step's path bends with the
rows (Transtrum and Sethna's geodesic acceleration): a second-order term, the least-squares fit of the rows' second
derivative along the step, keeps the step in a curved valley that the straight step leaves, where a cut step and the
damped one after it had zigzagged a warm start to the cap; on an affine model the term is zero. Its decrease is measured
as a difference of squares, and a decrease within what the objective resolves passes, judged on the scale the rows came from:
a row is a difference of terms (a frame's position and its goal) whose rounding is the terms', not the row's, so each
row's rounding is taken as the dtype's of the problem's size, which bounds its terms (exactly so on an affine model). On
an affine model Newton's and Gauss-Newton's steps are one, and the iteration is exact once its active set is the
solution's.

**Its local phase.** From the first point within the tolerance on, the solve takes full Newton steps with no line
search, while a step is longer than the commands' rounding, at most eight, and returns the point within the
tolerance whose step was the shortest. The tolerance bounds the stationarity, not the distance to the minimiser, and
the derivative is taken at the point returned. On a weakly regularised problem (an arm's position rows over more joints
than they fix, a weak effort holding the rest, whose scaled Hessian's least eigenvalues are far below its largest) the
two part far: a warm start met the tolerance a step or two in, where the derivative was measurably not the
minimiser's, over a closed loop's ticks as well, and a solve from a little further met it with its objective well
above the minimum. There the objective cannot judge a step: along a soft direction the decrease is below its rounding,
and in the curved valley the full step can raise it while it converges quadratically. With the local phase, the
loop's derivative is central differences to the differences' own truncation; the CHANGELOG's entry gives each
measurement.

**Its guards, declared**, as MIME's rule for single precision asks: a cap on its iterations (`ITERATIONS`, read at a
point already weighed, so that one within the tolerance still goes on to its local phase), a tolerance by dtype at or
above the dtype's reachable floor (`TOLERANCE`, on the stationarity over the problem's size; the CHANGELOG's entry gives
the floor measured in each), and a stagnation stop (`STALLS` steps in a row that leave the commands within their rounding,
whose decrease is within what the objective resolves and which do not halve the stationarity: a step that moves the
commands further is progress the objective cannot show, as in single precision on an unreachable target, whose
objective is too large to resolve the steps that still converge). The stationarity is the projected gradient's length in the scaled
commands over the problem's size, the rows' miss plus what the commands make, so its floor is the dtype's rounding of
that size whatever the rows cancel.

**`converged` is read at the commands returned**, never from the loop's own account: the stationarity within its
tolerance there, and the free commands' rows of full rank (the scaled normal matrix's least eigenvalue over its largest
above the floor), so that the minimiser and its derivative are defined. **Where it did not converge**, the commands are the
last iterate, inside the bounds, and `found.converged` says so; through `ctx.allocator` the controller raises its run
flag `allocation_unconverged` at that tick, its value the stationarity, which the run counts and warns of and
`Study.gradient`'s result carries. A value is never silent.

**Its derivative** is the implicit function's at the converged commands: `jax.lax.custom_root` on the projected
gradient's fixed point, its tangent system the exact Hessian's on the free commands (never the Gauss-Newton one, which
is wrong wherever the rows are not met, an unreachable target's) and, on a held command, its bound's own derivative (a
port's bound, or the slew box around the last command, so a derivative in the last command passes there). The tangent
system is one dense LU of order the commands. Forward and reverse mode both pass through it. Where the solve did not
converge it is the implicit function's at the iterate, which solves nothing, and the step's flag says so.

**Its status, per configuration**, declared as the controller's switch (`allocation_switch(owner, orders)`), evaluated
once at the solution:

| configuration | status | why |
|---|---|---|
| an affine model, regularised or its rows of full column rank | `piecewise_kink` | a convex quadratic: the minimiser is continuous, and its derivative jumps where a command reaches a bound or leaves it |
| a model that is not affine (an arm) | `piecewise_jump` | the local minimiser followed from the start can change branch, or settle at a bound, as what is wanted moves |
| a step whose solve did not converge | flagged | the derivative solves nothing there: `allocation_unconverged` marks the step |

**What it bounds, and what it estimates.** Against the exact path on the arrays MIME ships (three to eight coils, both
dtypes), against SciPy's BVLS on nine to sixteen commands, and against the rigid-body library's inverse kinematics on
reachable poses of the AR4, the iterative path's commands are the oracle's within the line its claim derives: the oracle's
own least-squares perturbation bound plus the natural residual's error bound in the scaled commands (the distance to the
minimiser at most the projected gradient's length times the scaled Hessian's largest eigenvalue, plus one, over its
least) (`tests/control/test_iterative_allocation.py`). Its derivative is central differences of converged
reference solves, and reverse mode is forward mode. These bound the solver against software oracles on the problems
drawn; they say nothing of whether a model is the hardware, which is the model's own claim. The figures measured are in
the CHANGELOG's entry.
