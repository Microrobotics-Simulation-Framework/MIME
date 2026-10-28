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

## What exists

- **The declaration.** `Experiment.set_controller(controller, reads={...}, writes={...}, mode=..., rate=...,
  latency=...)`. The controller is a registered entry (`mime.registry.Controller`): its fields are its settings, and
  it declares the labels it reads and writes. The experiment binds each read label to a thing's state
  (`"umr.pose"`, `"rpm.arm.joint_positions"`) or an observation's output (`"swim_speed:umr.speed"`), and each write
  label to an apparatus's command, which then takes no drive.
- **Its law.** `law(ctx)` makes the controller ready from its settings: its state at the start, the writes it holds
  before its first acts, and its step, `(state, reads) -> (state, writes)`. The step is the same in every mode.
- **External inputs** come from the writes alone: one per write, fed by the run. A command has one writer.
- **The callback** (`_run.py`). A run is compiled chunks of one control period. Between them the controller
  is called on the host with the state as the period starts, and its writes are fed through the steps they act on.
  No derivative passes through a callback.
- **Latency**, in whole base steps. A tick at time t reads the state at t; its writes act from the step that starts
  at t + latency. Before the first lands, the held writes act.
- **The per-chunk hook.** `run(duration, hook=...)` calls the hook after every chunk with its step, time and state.
  The viewer bridge will use it.
- **Classed reads.** Each read is the physics state ("ground truth, not deployable") or an observation's output. The
  build report and the lock list each with its class. `mime.build.refused_reads` is where a profile will judge them.
- **`ArmTracking`**, the arm-driven experiment's controller, ported from the v0.2 line: a magnet on an arm held above
  a body as it swims.

## Status

The callback is built (dev1, phase 7b1). The mode inside the compiled step comes next (phase 7b2), with the latency it
must match; the external client and the FMU come in a2.

Phases and releases are explained in the [package manifest](../README.md#status).
