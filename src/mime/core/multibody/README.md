# `mime.core.multibody`

**The rigid-body library**: description readers into one canonical tree, and the maths of an open tree on it. Nodes,
dynamics entries and controllers import it; it imports nothing from MIME above `mime.core`.

## The tree

`MultibodyTree` is the canonical form every reader builds and every algorithm reads:

- **bodies**, each with its own frame and an optional `Inertial` (mass, centre of mass, inertia about it, in the
  body's frame); `None` means the description declares none;
- **joints**: revolute, prismatic and fixed, each placing its child's frame in its parent's at zero position, with an
  axis in the child's frame, limits (position, velocity, effort, as declared), damping, Coulomb friction and armature;
- **named frames**: every body's own (URDF links), and extra frames fixed to a body with an offset (MJCF sites, mount
  points), in one namespace for parts to mount on;
- **loop closures** (`LoopClosure`, `connect` or `weld` between two frames), held as data so that a closed-chain
  mechanism needs no new data model, and refused by every algorithm here;
- **the source**: the file's format, name and SHA-256, which pins it.

A tree is immutable and compares by value; its numbers are Python floats in SI. It compiles once into an
`Articulation`: the moving joints numbered depth first from the root (siblings in the order listed, so a parent comes
before its children), each body welded by fixed joints into the rigid body it moves with, their inertials combined
about their common centre of mass, and every frame's offset in that rigid body. The articulation's arrays are JAX
pytree leaves, so a mass or an axis can be differentiated.

## The URDF reader

`read_urdf(path, sha256=None, acknowledged=None)` returns the tree and a `DescriptionClassification` listing every
element of the file in one of three groups:

| Group | Elements |
|---|---|
| read and used | links, inertials, joints (revolute, continuous, prismatic, fixed) with origin, parent, child, axis, limit, dynamics (damping, friction) |
| recognised and not used | visual, collision, material, transmission, `gazebo`, `ros2_control`, a joint's calibration and safety controller |
| refused | a mimic joint and any element URDF does not define (both can be read past with an acknowledged reason, which is recorded); floating and planar joints and unexpanded xacro (outright) |

Each kind of element not used, or read past, is warned about once (`DescriptionWarning`), saying what MIME does
instead, where to look and how to ask for it; a refused element raises `DescriptionRefused`, which carries the whole
classification. A link with geometry and no `<inertial>` is massless, as URDF reads it, and is recorded as an
`InertiaGap`; where the rigid body it moves with declares no inertial at all, `needed_by_dynamics` marks the mass a
dynamic chain would need and the file does not give. Read, never written.

## The maths

Pure JAX: `jit`, `grad` and `vmap` traceable, finite (with finite derivatives) at zero rotation, and computed in the
dtype of the joint positions given (float32 in, float32 out). Everything is in the root's frame. Poses are
`(x, y, z, qw, qx, qy, qz)` on `mime.core`'s quaternions; Jacobian rows and spatial vectors are `[linear; angular]`.

- `forward_kinematics` (every frame), `frame_poses` (the frames named), `frame_pose`, `frame_jacobian`;
- `inverse_kinematics`: damped least squares for a fixed number of iterations; the result reports the position and
  rotation error, whether the target was reached within the tolerances asked for, and whether the joints are within
  their limits. Never clamped;
- `mass_matrix` (the composite-rigid-body algorithm), `inverse_dynamics` (the recursive Newton-Euler algorithm), both
  after Featherstone, *Rigid Body Dynamics Algorithms* (2008); `bias_forces`, `gravity_forces`, and
  `forward_dynamics`, `q_ddot = M^-1 (tau - h)`. The armature is on the mass matrix's diagonal and in inverse
  dynamics; damping and friction are joint forces a dynamic chain adds to `tau`.

## Closed chains

Out of scope until a downstream user needs one: ask at
[the issues](https://github.com/Microrobotics-Simulation-Framework/MIME/issues). Until then, a lab can register its own
dynamics entry for a chain segment (`mime.registry.register_dynamics`) that takes its own description of the mechanism
and publishes the same frame poses as MIME's chains, so that mounts and everything downstream work unchanged.

## Status

Since phase 7a1: the tree, the URDF reader and the maths. The MJCF reader, with MuJoCo as the oracle of both readers
and the maths, comes in dev2.

Since phase 7a2, the kinematic arm (`mime.dynamics.Arm`) reads its description with this reader and computes
its frames with this library's forward kinematics; its node takes the articulation's arrays as its parameters.

Since phase 7a3, **loads on the links**: `frame_forces` gathers wrenches given on the named frames (a force and a
torque about each frame's origin) into a spatial force on each rigid body, and `inverse_dynamics` and
`forward_dynamics` take them (`link_forces`) as the recursion's external forces, equal to the Jacobian transpose's
mapping (its oracle); `cholesky_solve` is public. The dynamic arm steps on them.

Phases and releases are explained in the [package manifest](../../README.md#status).
