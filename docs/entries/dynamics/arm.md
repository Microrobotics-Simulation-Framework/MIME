---
describes: "0.1.0"
---
# A robot arm, as a chain segment

**What it models.** A robot arm carrying something, a magnet or a source, as one segment of an apparatus's chain:
`Apparatus(..., chain={"arm": Arm.Kinematic(description=asset("arm.urdf"), ...)})`. It is general over its
description: the arm is whatever multibody tree its URDF file describes, so another arm is another file. Every frame
of the tree is published as a world pose and an angular velocity, every step, and a motor or a part mounts on one of
them by name: the flange, or a link part-way along the arm. A segment mounts on a frame of the segment before it
(`mount="link_3"`), or of an earlier one (`mount="arm.link_3"`), so that a mid-arm aimer and a motor on the flange
both ride on the arm.

The `kinematic` rung prescribes the arm's motion. Its joint positions are a command, written by a drive or by the
controller: `<segment>.joint_positions` for its revolute joints, in radians, and `<segment>.joint_displacements` for
its sliding ones, in metres, each in the order of its kind's joints. Each is held within its joint's limits where the
description gives them, and the joints follow it at once or, with a servo time constant, as a first-order lag. The
positions it starts at must lie within the limits. Forward kinematics gives every frame's pose, and the joints' rates
over the step its angular velocity. It has no mass matrix and takes no load, and no mass mounted on it changes its
motion: it takes every mounted mass, tilted, unbalanced or articulated, and carries none.

The `dynamic` rung drives the joints by torques, `Arm.Dynamic(description=..., initial_positions=...)`, the command
`<segment>.joint_torques` for its revolute joints (N m) and `<segment>.joint_forces` for its sliding ones (N). They
move against the arm's inertia (the mass matrix from the description's inertials and what is mounted on its links, by
the composite-rigid-body algorithm), the experiment's gravity, each joint's damping and Coulomb friction from the
description, and the loads on its frames, all in the recursive Newton-Euler algorithm. Its loads are the reactions of
what it carries: a magnet's, where a law acts on a body through the field of a magnet mounted on it, and a motor's,
whose stator passes on the drive's reaction and what its rotor takes across its axis. A joint limit is a rigid,
inelastic stop: a joint driven past it arrives at it at the end of the step, and the stop takes the rest of its
velocity on the next. The stops and the friction act as impulses through the arm's inertia, so that what holds one
joint moves every joint coupled to it: the step's velocity is the one nearest the unconstrained step's, less what
friction dissipates, that keeps every joint within its limits, which never holds more kinetic energy than the
unconstrained step. An arm whose friction can hold it under its weight stays exactly still, and a torque below a
joint's friction moves nothing. It publishes its joint velocities beside its positions and frame poses. Its step is
semi-implicit Euler, first order in time, with the damping taken at the end of the step.

What is mounted on its links with a mass is grafted onto them at build, each a rigid body welded to its link in the
arm's tree, the link's own inertial untouched: a part with a density (from its material, or declared as
`<apparatus>.parts.<part>.density`) or a declared mass (`<apparatus>.parts.<part>.mass`), a part on a motor's
stator, and a dynamic motor's rotor, of which the link carries the mass and the mean inertia across the axis while
the motor carries its inertia about the axis by its spin. A part with no mass changes nothing. The build report lists
each mass with what carries it. A controller that models the arm reads this built tree; one that reads a bare
description of its own makes that choice.

The description is read through MIME's URDF reader: every element is read and used, recognised and not used, or
refused. Each kind of element not used is warned about once, and every one is listed in the build report and the
saved form. An element refused because dropping it would change the mechanism (a mimic joint, an element URDF does not
define) is read past only when the experiment acknowledges it with its reason. The file is pinned by its sha256 hash, and
the saved form carries it.

**When not to use it.**
- The kinematic rung when what the arm feels matters: a load on it is refused at build, since a prescribed motion
  cannot be moved by it, and the reaction of a field on a magnet it carries is not applied (see MIME-ANO-103). Use the
  dynamic rung.
- The dynamic rung where the arm is not the chain's first segment: its base is fixed in the world.
- The dynamic rung for a description whose moving bodies have geometry and no inertial, or whose moving joint moves
  a body with neither geometry nor inertial: they have no mass to move, and it refuses them; MIME infers no mass from
  geometry.
- The dynamic rung for an articulated payload, a segment with joints of its own carrying mass (a pan-tilt aimer, a
  gripper, a kinematic motor with a massive rotor): its tree holds no joint beyond its own, and it refuses one. Grafting
  jointed sub-trees on a link comes later. It refuses a rotor turned off its spin axis, or with its centre of mass off
  it, beyond a residue whose rotating load at the motor's top speed is a thousandth of the rotor's weight. The kinematic
  rung takes every one of these. A spinner mounted on it hands it the rotor's momentum and its inequality across its
  axis: the arm applies the gyroscopic torque at its own velocity, with no work at any step size, and carries the
  turning inertia in its mass matrix.
- Where a joint's speed must stay within the description's limit: the dynamic rung does not apply the limit.
- The dynamic rung where the stops' and friction's impulses on many coupled joints must be exact: they are solved by
  a fixed count of sweeps, exact for one joint and wherever every joint with friction sticks. On coupled joints a
  change a converged solve would still make to the velocity remains: small on A0's arm, and largest where strongly
  coupled joints reach their stops in the same step; the record's limits state it as measured. The kinetic energy
  never rises, and no joint passes its stop, whatever the sweeps leave.
- Where a mounted motor's reaction must reach the arm in the same step: it arrives a step late (see MIME-ANO-103).
- When the arm's structure moves the result: its joints and links are rigid, with no structural mode (see
  MIME-ANO-101).
- For a closed chain, such as a delta robot or a four-bar linkage, which URDF cannot describe: it is out of scope
  until someone needs it. Ask at the issues, or register your own chain segment, which publishes the same frame poses,
  so that everything mounted on it works unchanged.
- When the commanded motion is faster than the real arm can follow: the kinematic rung follows its command whatever
  its speed; the watch on joint speeds and accelerations against the description's limits comes with validity lines.

**Versions.** `arm` 0.1.0 is its first: the kinematic rung, and since dev1's phase 7a3 the dynamic rung, added before
any release with no change to what the kinematic rung computes. Since dev1's phase 7a4, before any release, both rungs
publish their frames' angular velocities, their commands are typed by joint kind, and the dynamic rung carries what is
mounted on its links; what the kinematic rung computed is unchanged to the last bit on A0.
