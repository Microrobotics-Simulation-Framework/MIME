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

## Status

Empty. The declaration and modes 1 and 2 land in dev1's phase 7, with the arm-driven experiment. Modes 3 and 4 come
in a2.

Phases and releases are explained in the [package manifest](../README.md#status).
