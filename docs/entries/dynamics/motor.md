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

**When not to use it.**
- When the motor's own behaviour matters: there is no rotor inertia, no cogging, imbalance or thermal term (see
  MIME-ANO-102, as it applies to a prescribed motion).
- When its reaction on the arm matters: the torque that turns the rotor is not returned to the link it is mounted on
  (see MIME-ANO-103). The `dynamic` rung, which takes a torque and has the rotor's inertia, is reserved and not built
  yet.
- When a prescribed speed must hold its phase over a long run in single precision: the angle is a sum over steps, and
  its rounding grows with it.

**Versions.** `motor` 0.1.0 is its first: the kinematic rung, with the dynamic rung reserved.
