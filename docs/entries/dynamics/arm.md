---
describes: "0.1.0"
---
# A robot arm, as a chain segment

**What it models.** A robot arm carrying something, a magnet or a source, as one segment of an apparatus's chain:
`Apparatus(..., chain={"arm": Arm.Kinematic(description=asset("arm.urdf"), ...)})`. It is general over its
description: the arm is whatever multibody tree its URDF file describes, so another arm is another file. Every frame
of the tree is published as a world pose, every step, and a motor or a part mounts on one of them by name.

The `kinematic` rung prescribes the arm's motion. Its joint positions are a command, written by a drive or, later, by
the controller; each is held within its joint's limits where the description gives them, and the joints follow it at
once or, with a servo time constant, as a first-order lag. The positions it starts at must lie within the limits.
Forward kinematics gives every frame's pose. It has no mass matrix and takes no load.

The description is read through MIME's URDF reader: every element is read and used, recognised and not used, or
refused. Each kind of element not used is warned about once, and every one is listed in the build report and the
saved form. An element refused because dropping it would change the mechanism (a mimic joint, an element URDF does not
define) is read past only when the experiment acknowledges it with its reason. The file is pinned by its sha256 hash, and
the saved form carries it.

**When not to use it.**
- When what the arm feels matters: a load on a kinematic arm is refused at build, since a prescribed motion cannot be
  moved by it, and the reaction of a field on a magnet it carries is not applied (see MIME-ANO-103). The `dynamic`
  rung, which takes joint torques and loads on its links, is reserved and not built yet.
- When the arm's structure moves the result: its joints and links are rigid, with no structural mode (see
  MIME-ANO-101).
- For a closed chain, such as a delta robot or a four-bar linkage, which URDF cannot describe: it is out of scope
  until someone needs it. Ask at the issues, or register your own chain segment, which publishes the same frame poses,
  so that everything mounted on it works unchanged.
- When the commanded motion is faster than the real arm can follow: the kinematic rung follows its command whatever
  its speed; the watch on joint speeds and accelerations against the description's limits comes with validity lines.

**Versions.** `arm` 0.1.0 is its first: the kinematic rung, with the dynamic rung reserved.
