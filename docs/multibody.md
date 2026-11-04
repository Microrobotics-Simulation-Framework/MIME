# The multibody library: chains, mounts and armature

A robot arm or a motor is not a body in MIME: it is a **segment of an apparatus's chain**, and what rides on it is
**mounted** on one of its frames. This note holds the definitions; the entries that use them
([`arm`](entries/dynamics/arm.md), [`motor`](entries/dynamics/motor.md)) say what each computes.

## The library under the segments

`mime.core.multibody` is the rigid-body library the segments are built on. A description file is read (`read_urdf`) into
one canonical `MultibodyTree`: bodies with optional inertials, revolute, prismatic and fixed joints (each with limits,
damping, Coulomb friction and an armature), and **named frames** (every link's own frame, and frames fixed to a link
with an offset), which are the frames a part or a segment mounts on. The tree compiles once into an `Articulation`, the arrays
the maths and the nodes read. The reader sorts every element of the file into read and used, recognised and not used, or
refused, and warns once about each kind not used. A closed chain is refused by every algorithm there.

## How a chain is declared

A chain is the apparatus's `chain` mapping, each segment's name to its dynamics, in chain order:

<!-- mime-fragment: a chain declared with the elision dots of a sketch (description=..., initial_positions=...), not a complete call -->
```python
Apparatus("rig", chain={"arm": Arm.Dynamic(description=asset("arm.urdf"), initial_positions=...),
                        "motor": Motor.Dynamic(axis=Z, inertia=..., mount="link_6")}, ...)
```

- **The first segment is placed in the world** (`at=`, `orientation=`) and takes no `mount=` (`chain.mount-first`).
- **Every later segment mounts on a frame** of the segment before it (`mount="link_6"`), or `<segment>.<frame>` of an
  earlier one (`mount="arm.link_3"`); any other name is refused with the frames that exist (`chain.mount`).
- A segment is mounted on by what it publishes: its frames' poses. A lab's own segment that publishes the same poses
  (`mime.registry.register_dynamics`) is mounted on, and mounts, like MIME's.
- A segment's **command** is written by a drive or a controller under the segment's name (`<segment>.joint_positions`,
  `<segment>.joint_torques`, `<segment>.torque`).

## How a mount is declared

A **part** is a solid fixed to a frame, `Part(geometry, material=..., at=..., mount="motor.rotor")`: `mount` is
`<segment>.<frame>`, `at` places the part's centroid in that frame, and the part's own frame is not turned in it. With no
`mount` the part is fixed in the world. A part is a magnet by role, `Apparatus(magnets={"name": PermanentMagnet(...)})`, with
its magnetisation direction in the part's frame.

A part has **mass** where its material gives a density, or where the experiment declares its density or its mass
(`exp.declare("rig.parts.<part>.density", ..., reason=...)`, or `.mass`); with none it is massless and changes nothing.
How a value reaches the part is [the materials note](materials.md). A **dynamic** segment carries the mass: at build each
part is grafted onto its frame's link as a rigid body welded to it, the link's own inertial left as the description gave
it. A **kinematic** segment prescribes its motion, so it takes every mounted mass and carries none.

## How armature is declared

A geared joint's drive reflects its motor's rotor onto the joint, and a description's links do not hold it. The dynamic
arm takes each joint's **armature** (a moment of inertia on a revolute joint, a mass on a sliding one) from one of three
places, each named in the build report:

1. **Declared per joint**: `Arm.Dynamic(armature={"joint_6": kg_m2(..., reason="...")})`. A declaration, not a measurement
   of this arm: the reason says where the number came from.
2. **Reflected through a transmission** of the description, from the rotor behind it: `actuators={"motor_6": kg_m2(...,
   reason=...)}`, or the name of a registered actuator (`mime.dynamics.register_actuator`), a datasheet's record with its
   source. The reduction of the transmission is applied squared.
3. **Neither**: a convention of none, the description's own meaning, which the report says and the research profile
   lists.

A joint given both ways, or a joint or an actuator given twice, is refused (`chain.armature`). Armature sits on the
diagonal of the mass matrix and on the tree a model-based controller reads.
