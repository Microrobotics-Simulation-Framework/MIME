---
describes: "0.2.0"
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
the command `<segment>.torque`, against the rotor's inertia about its axis, its bearing's viscous damping on its speed
relative to its stator, and what the loads on the rotor exert about the axis: an ideal torque source. It is a spinner:
it carries the rotor's absolute spin about its axis, whose rate is exactly the torques about the axis over the inertia,
with no term from how its stator turns, and it reads the angular velocity of the frame it is mounted on (a step late,
as it reads the frame's pose) for the rotor's turn on its stator and the bearing's damping. The parts mounted on
`<segment>.rotor` with a mass are its rotor: their moments about the axis add to `inertia`, and on a dynamic arm their
mass and mean inertia across the axis are welded on the link the motor is mounted on. It takes loads on its frames,
the reaction on a magnet on its rotor among them. Its stator holds the rotor: it takes the drive's reaction and the
bearing's, the loads on the rotor across its axis, and the rotor's gyroscopic torque as the arm turns its axis, and
passes them, with the loads on the stator, to the frame it is mounted on. That segment takes them a step late, what
the stator passed over the step before, where it takes loads: an arm on its dynamic rung. A rotor unequal across its
axis on a dynamic arm needs `top_speed`, the most its spin reaches, for the worst case of the term it does not carry,
which the build report gives.

**When not to use it.**
- The kinematic rung when the motor's own behaviour matters: it has no rotor inertia, no cogging, imbalance or
  thermal term (see MIME-ANO-102, as it applies to a prescribed motion), and the torque that turns its rotor is not
  returned to the link it is mounted on (see MIME-ANO-103). Use the dynamic rung.
- The dynamic rung where the drive's electrical side matters: the torque commanded is the torque on the rotor, with no
  armature, saturation, cogging, imbalance or thermal terms (see MIME-ANO-102).
- The dynamic rung on a dynamic arm for a rotor tilted off its axis, with its centre of mass off it, or so unequal
  across it that the term it does not carry could pass half the gyroscopic torque it carries: each is refused, since a
  spinner holds the rotor symmetric about its axis. Its stator's turning couples into such a rotor's spin by terms it
  does not carry. On a kinematic arm each is accepted, and its own spin carries none of those terms. A0's magnet,
  spun about a diameter, is unequal across its axis, and accepted: MIME-ANO-122 states its worst case at the AR4's
  joint speed limits, and the build report gives it for each rotor.
- The kinematic rung for mass on its rotor on a dynamic arm: its prescribed turn moves that mass by a joint the arm
  does not hold, an articulated payload, and it is refused. The dynamic rung carries it.
- Where its reaction must reach the arm in the same step: it arrives a step late (see MIME-ANO-103).
- When a prescribed speed must hold its phase over a long run in single precision: the angle is a sum over steps, and
  its rounding grows with it.

**Versions.** `motor` 0.1.0 was its first: the kinematic rung, and since dev1's phase 7a3 the dynamic rung. `motor`
0.2.0, since dev1's phase 7a4 and before any release, carries the dynamic rotor's absolute spin, its parts' inertia
and its gyroscopic torque, and publishes both rungs' frames' angular velocities; what the kinematic rung computes is
unchanged to the last bit on A0.
