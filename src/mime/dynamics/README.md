# `mime.dynamics`

**How each thing's state evolves**: the registered dynamics entries.

## What goes here

- **`body/`**:
  - `locked`, `rigid_overdamped`, `rigid_inertial`. The inertial body integrates resistance implicitly,
    `(M/dt + R)·V' = (M/dt)·V + F`.
  - Confinement and its velocity rules: remove the outward velocity, inelastic, or a friction cone.
- **`flow/`**: `quiescent`, `prescribed`, and the finite-volume flow solver. The solver is rewritten for 0.3. Coupling
  a coarse flow to a fine one near the body (multi-resolution) is a stretch goal.
- **`apparatus/`**: `fixed`, `arm`, `motor`, and a coil driver.

## How an entry works

In phase 1, an entry plans. It declares its targets (what interactions may deliver into it), its reads, its capability
record, its validity and its differentiability.

In phase 2, it builds its node or nodes once, with every registration addressed to it. For example, a flow solver is
constructed with the bodies it must include.

## Boundaries

- Built by `build`, using nodes from `nodes/`.
- MIME owns the solvers its value depends on, such as the fluid solver, and keeps each one narrow.

## Status

Since phase 4b, **`body.RigidInertial`** (`rigid_inertial` 0.2.0 since phase 4c3; 0.1.0 in 4b): a rigid body with mass and inertia (principal
moments about the body axes), its state the pose and twist of its centre of mass. It sums every applied wrench and
background load, integrates every delivered resistance implicitly, solved for the change in velocity,
`(M/dt + R) dV = F - R V` (0.2.0; 0.1.0 solved `(M/dt + R) V' = (M/dt) V + F`, which float32 rounded, MIME-ANO-113),
with the gyroscopic term linearly implicit at the midpoint (it was explicit, and gained energy: MIME-ANO-130; there is no explicit setting), turns by the exponential of `dt omega'`, holds its
confinement inside the solve while the body starts a step on its surface moving outward (0.2.0), and projects it after
the step. Its node declares first order in time, which the node walk measures. It reads its body's `mass` and
`inertia` (declared in the experiment until the part resolver lands). The rest land in dev1's phase 7 (the arm and
motor) and a1 (the rewritten flow solver).

Since phase 7a2, an apparatus's **chain segments** (`apparatus/`), each a `mime.registry.ChainSegment` that publishes
the world pose of every frame it names (`apparatus.frame_poses`), alike under every rung, and mounts on a frame of the
segment before it:
- **`apparatus.Arm`** (`arm` 0.1.0), general over its URDF file, with its `kinematic` rung: its joint
  positions commanded, held within its joints' limits, followed at once or through a first-order servo, every frame
  by forward kinematics; no mass matrix and no load, a law that would load it refused. Its `dynamic` rung is reserved.
  Its description's elements are classified for the report and the saved form; a closed chain is refused.
- **`apparatus.Motor`** (`motor` 0.1.0), with its `kinematic` rung: a prescribed angle or speed about an axis of its
  base frame, the frame it mounts on; its frames `base` and `rotor`. Its `dynamic` rung is reserved.

Each states its limits in its capability record: rigid joints and links, an ideal source of its motion, and no load
applied.

Since phase 7a3, their **dynamic rungs**:
- **`Arm.Dynamic`**: joint torques in (`<segment>.joint_torques`), against the mass matrix and the bias of
  `mime.core.multibody` (gravity, velocity products and the loads on its frames, `apparatus.link_wrench`, which it
  accepts), each joint's damping (implicit) and Coulomb friction from the description, its limits rigid, inelastic
  stops; the stops and friction act as impulses through the step's system, so that what holds one joint moves every
  joint coupled to it, and never add kinetic energy; semi-implicit Euler, first order in time; it publishes its joint
  velocities too. Its base is fixed in the world (the chain's first segment, `chain.fixed-base`), a description whose
  moving body has geometry and no inertial is refused (`chain.inertia-gap`), and so is a joint that moves no inertia
  (`chain.massless-joint`). Its two rungs declare their own gradient statuses: kinematic `piecewise_kink`, dynamic
  `piecewise_kink` (a joint reaching its stop arrives at it at the end of the step, its velocity continuous).
- **`Motor.Dynamic`**: a torque in (`<segment>.torque`) against its rotor's inertia, its bearing's damping and the
  loads on its rotor about the axis; an ideal torque source (MIME-ANO-102). It accepts loads on its frames, and its
  stator passes the drive's reaction and the rotor's loads across its axis to the frame it is mounted on, which a
  dynamic segment takes a step late (`b.reaction`; MIME-ANO-103).

Since phase 7a4 (`motor` 0.2.0): **`Motor.Dynamic`** is a spinner, carrying its rotor's absolute spin, with the parts
on its rotor adding their moment about its axis, its momentum and its inequality across the axis handed to its mount,
and `top_speed` to judge a rotor's imbalance or tilt; **`Arm.Dynamic`** carries what is mounted on its links, welded on
at build, the spinners' gyroscopic torques and turning inertia, and refuses an articulated payload and an imbalance or
a tilt beyond a residue, which the kinematic rung takes. Both rungs of both publish their frames' angular velocities
(the kinematic ones where something reads them), and the arm's commands are typed by joint kind.

Phases and releases are explained in the [package manifest](../README.md#status).
