# `mime.control`

**The controller**: declared once, runnable four ways.

## The declaration

A controller is declared like an interaction: a registered name and version, settings, what it reads, what it
writes, a rate and a latency. The build derives every external input from that one declaration. An input therefore
cannot be declared twice, or wired and declared at once.

## The four modes

1. **In the graph**: a JAX node at its own rate. Gradients flow through the closed loop.
2. **Callback**: a Python class, called once per control period.
3. **External client**: a separate process that steps the simulation, as a robot controller steps a Webots world.
   A Python client comes first.
4. **FMU**: the graph exported for a tool such as Simulink, which then holds the controller.

Swapping modes never changes the physics. A swap test runs one controller in two modes and compares them.

The rigid-body maths a controller uses (forward and inverse kinematics, Jacobians, dynamics) is in
[`mime.core.multibody`](../core/multibody/), below the nodes, since nodes import it too.

**The exact allocation** (`_allocation.py`), a law a controller's step may call: `exact_allocation(rows, wanted, *,
lower, upper, scales, owner, offset=None, reference=None, reference_scales=None)`, the commands within explicit bounds
that minimise the caller-scaled residual of an affine model (and a regularisation towards a reference, if scaled), exact
by its active sets: each free subset's QR, every set's candidate from its subset's, the feasible one of least residual
selected with no derivative (ties to the least squared distance from the reference), then the selected set solved once more,
differentiably. Its switch's owner is the caller's, declared with `allocation_switch(owner, orders)`. More than
`EXACT_COMMANDS` commands is refused (`allocation.exact-size`), the iterative path its route. Its solves are the
registered sites `allocation_exact` and `allocation_least_norm`. `mime.fields.allocate()` runs it on the host in float64.
[The allocation note](../../../docs/allocation.md) is the user's page.

**The iterative allocation** (`_iterative.py`): `iterative_allocation(rows, start, *, lower, upper, owner)`, the
commands within bounds that minimise half the squared rows a function of them makes, for any number of commands and
any model: projected Newton (Newton's step on the exact Hessian where it is definite on the free commands,
Gauss-Newton's where not), the bounds a projection, Armijo's test on a difference of squares; its cap, tolerance and
stagnation stop declared; `converged` read at the commands returned (a first-order stationary point, isolated, the
rank judged from singular values at `RANK`; not a local minimiser). Its derivative's `resolved` and `resolution` read
the tangent system's condition. Its derivative is the implicit function's
(`jax.lax.custom_root`, the exact Hessian's tangent system, one dense LU through `mime.core.solves.dense_solve`). Its
switch is the caller's owner, evaluated once at the solution; its forward solve and its root are traced with any switch
collector hidden (`mime.core.switch.not_collected`), whose rows would keep their tracers.

**A controller's flags** (`mime.nodes.control._flags`). A controller declares the flags its step raises in its class's
`raises` (each a `mime.registry.Flag`, its `field` the flag's name); what its law calls declares its own when the law is
made (`ctx.allocator` on the iterative path declares `allocation_unconverged` and
`allocation_derivative_unresolved`). Its step raises one with
`raise_flag(name, value)`, nonzero where raised, the flag's value at the tick the largest magnitude raised under its
name; the gatherer (`raising`) is entered at the level the step is called at, and a raise below it (inside a
`jax.lax` body, or a function transformed on its own) or under a name not declared is refused with its fix. Inside the
compiled step the controller's node holds each as its state `flag_<name>`; as a callback, a node of its own
(`<controller>.flags`, fed at each tick from what the step raised; a step the law compiled with `jax.jit` is compiled
again through the function it compiles, so that the flags leave it, and one compiled with nothing underneath it is
refused, `hidden_step`) holds them. The run lays them out at one path in both modes, `<controller>.<name>` (the
layout's `shown`), and counts them as a law's: the steps raised, the first and the last, a warning for one that warns.
A controller that declares none builds nothing new. A setting read only by a flag's value is a leaf the step reads.

**Believed models and the allocator through them** (`_models.py`). `set_controller(..., believed={key: entry})` gives
the controller its own declaration of an apparatus's provider or chain segment (`<apparatus>.<field>`,
`<apparatus>.<segment>`, the apparatus's name alone where it holds one part, saved and reported in full) or of a camera
(its id): an entry of the same kind taking the same commands, its settings checked in phase 1. Where none is declared
the belief is the truth, and the build report says `believed = true`. The law reads one as `ctx.believed(key)`: a
provider's or segment's forward model (`mime.registry.ForwardModel`) or a camera's projection (`CameraModel`, the
camera's own `project`). Inside the compiled step each of its numbers the step reads is a leaf of the controller's node,
`believed_<apparatus>__<part>__<parameter>`, so a derivative reaches the belief, and a fit moves the belief, never the
truth; no node is built for a belief. `ctx.allocator(apparatus, commands=..., objective=..., owner=...)` is the
allocation through the believed model, its bounds the ports', with the slew box around `last` where a port declares a
slew, which takes the stage to have stood at `last` ([the allocation note](../../../docs/allocation.md)); its path is
chosen when the law is made, the exact one where the model is affine in at most `EXACT_COMMANDS` commands, the
iterative one otherwise. A refusal either raises inside `law(ctx)` keeps its own code;
any other error a law raises is its entry's defect.

## What exists

- **The declaration.** `Experiment.set_controller(controller, reads={...}, writes={...}, mode=..., rate=...,
  latency=...)`. The controller is a registered entry (`mime.registry.Controller`): its fields are its settings, and
  it declares the labels it reads and writes. The experiment binds each read label to a thing's state
  (`"umr.pose"`, `"rpm.arm.joint_positions"`), an observation's output (`"swim_speed:umr.speed"`) or an apparatus's
  measurement port (`"imaging.tracker"`, and its age in base steps, `"imaging.tracker.age"`), and each write label to
  an apparatus's command, which then takes no drive; where the command has a port, the write enters its stage.
- **Its law.** `law(ctx)` makes the controller ready from its settings: its state at the start, the writes it holds
  before its first acts, and its step, `(state, reads) -> (state, writes)`. The step is the same in every mode.
- **External inputs**, for a callback, come from the writes alone: one per write, fed by the run. A command has one
  writer.
- **Inside the compiled step**, the first mode. The controller's node runs its law's step at the controller's own rate,
  MADDENING's multi-rate schedule holding its writes between ticks; a delay line holds them for its latency. Its
  reads are edges from the state, its writes edges into the commands: no external input. It runs first in each step,
  and the rest as MADDENING runs the callback's graph, so every other edge reads the step it reads there. A derivative
  of the run passes through it. A step that cannot run there (Python on a traced value, a call to the host, a state
  that changes shape) is refused for this mode, and so is a read of what its writes do not reach. **A step must be
  pure** to run there: it is traced, so what changes between ticks lives in its state. A step that reads Python state
  its own calls change (a counter, a global, a random generator) is refused (`controller.impure`: it is traced twice,
  and the two programs differ). A step with a side effect on the world (`print`, a file, a list it appends to) is
  *not* seen by any trace: inside the compiled step the effect runs once, at the build, where the callback runs it at
  every tick, so the two differ with no refusal. **A run need not be whole periods** there: it accepts any duration of base steps,
  and splits that fall off a period rejoin bit for bit, since the schedule's phase persists across runs. The callback
  refuses a run that is not whole periods (`ValueError`), so a swap test runs whole periods.
- **The callback** (`_run.py`). A run is compiled chunks of one control period. Between them the controller
  is called on the host with the state as the period starts, and its writes are fed through the steps they act on.
  No derivative passes through a callback: its status is `cut` whatever the law (`Controller.callback_gradient`, the
  binding's), recorded beside the law's own in the lock, the registry export and the entry's page, and kept as a line
  of `mime docs check`'s cut summary with its route (the mode inside the compiled step).
- **Latency**, in whole base steps, the same in every mode. A tick at time t reads the state at t; its writes act from
  the step that starts at t + latency. Before the first lands, the held writes act.
- **The per-chunk hook.** `run(duration, hook=...)` and `run_to(..., hook=...)` call the hook after every chunk with
  its step, time and state.
- **Live inputs.** A controller declares the values a live session may set (`live`, each a `LiveInput`: units, shape,
  meaning). In the callback mode the session hands each one set to the step under its label beside its reads, from
  the next tick on; until one is set, the step sees its reads alone, so every other run is unchanged. `ArmTracking`
  declares `offset`, which moves the point it holds the magnet above along its axis.
- **Classed reads.** Each read is the physics state ("ground truth, not deployable": an implicit ideal port), an
  observation's output, or a measurement port ("as deployable as its declaration"). The
  build report and the lock list each with its class. `mime.build.refused_reads` is where a profile will judge them.
- **Settings as leaves.** A setting a study's axis names (`controller.<setting>`, `mime.study`) and the law reads
  through its context in its step is a leaf of the node inside the compiled step, the law made again from the leaves
  inside the traced step; one read through `self` is refused for an axis (`controller.setting-unread`). A law's
  context holds its bare numbers (a gain) beside its dimensioned settings in SI. In the callback mode the law is made
  on the host from each run's own settings, and a derivative through it is cut.
- **`ArmTracking`**, the arm-driven experiment's controller, ported from the v0.2 line: a magnet on an arm held above
  a body as it swims. It reads its numbers (smoothing, gain, damping) through its context. Its switches are written through `mime.core.switch` and declared, so inside the compiled step
  its status is `piecewise_jump`; as a callback no derivative passes through it.

## Status

The callback and the mode inside the compiled step are built, and give the same writes
at the same steps. The external client and the FMU come in a2.

Releases are explained in the [package manifest](../README.md#status).
