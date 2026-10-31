---
describes: "0.1.0"
---
# A motor, as a chain segment

**What it models.** A rotor turning about an axis relative to its stator, as one segment of an apparatus's chain,
usually mounted on the flange of an arm: `chain={..., "motor": Motor.Kinematic(axis=Z, prescribed="velocity",
initial_angle=rad(start), mount="link_6")}`. Its frames are `base`, the stator, and `rotor`; a magnet on the rotor mounts
on `motor.rotor`. The axis is stated in the motor's own base frame, the frame it mounts on, never in the world, so it
turns with whatever carries it.

The `kinematic` rung prescribes its motion: an angle, or a speed integrated over each step, from its initial angle. It
is an ideal source of that motion.

The `dynamic` rung turns its rotor by a torque, `Motor.Dynamic(axis=Z, inertia=..., damping=..., initial_angle=...)`,
the command `<segment>.torque`, against the rotor's inertia about its axis, its bearing's viscous damping and what the
loads on the rotor exert about the axis: an ideal torque source. It takes loads on its frames, the reaction on a magnet
on its rotor among them. Its stator holds the rotor: it takes the drive's reaction and the bearing's, and the loads on
the rotor across its axis, and passes them, with the loads on the stator, to the frame it is mounted on. That segment
takes them a step late, what the stator passed over the step before, where it takes loads: an arm on its dynamic rung.

**When not to use it.**
- The kinematic rung when the motor's own behaviour matters: it has no rotor inertia, no cogging, imbalance or
  thermal term (see MIME-ANO-102, as it applies to a prescribed motion), and the torque that turns its rotor is not
  returned to the link it is mounted on (see MIME-ANO-103). Use the dynamic rung.
- The dynamic rung where the drive's electrical side matters: the torque commanded is the torque on the rotor, with no
  armature, saturation, cogging, imbalance or thermal terms (see MIME-ANO-102).
- The dynamic rung where the rotor's own mass or inertia, or its gyroscopic torque as the arm turns it, matters: the
  frame it is mounted on carries none of them, its inertia about its axis included, and its speed is relative to its
  stator. On A0 the rotor's inertia about its axis is several times the inertia of the link it is mounted on about
  the same axis, so the arm's last joint turns with a fraction of the inertia it would have (MIME-ANO-122 states the
  numbers).
- Where its reaction must reach the arm in the same step: it arrives a step late (see MIME-ANO-103).
- When a prescribed speed must hold its phase over a long run in single precision: the angle is a sum over steps, and
  its rounding grows with it.

**Versions.** `motor` 0.1.0 is its first: the kinematic rung, and since dev1's phase 7a3 the dynamic rung, added
before any release with no change to what the kinematic rung computes.
