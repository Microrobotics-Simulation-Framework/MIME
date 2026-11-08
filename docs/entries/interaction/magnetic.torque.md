---
describes: ["0.1.0", "0.2.0"]
---
# The torque of a field on a magnet

**What it models.** The torque of the field bound to `field` on the body's permanent magnetic moment, about the
centre of mass: `exp.attach(Magnetic.Torque(), body=umr, field=rpm.magnetic)`. The moment is the body's
`magnetic_moment`, in the body's frame, turned into the world by the body's orientation; the torque is its cross product
with the flux density. It delivers no force: the pull of a field's gradient on the moment is another law, `Magnetic.Force`,
and the two are attached together for the whole of a magnet's response. A field that varies in space is read where the
magnet is, at the body's `magnet_centre` (from its parts, or declared) turned by its orientation, so that with
`Magnetic.Force` it is the derivative of a point dipole's energy there and does no net work round a closed loop of the
body's pose; a uniform field is read as it always was, with no points.

It reads the body's pose at the start of the step, as every law reads its body's state, so the torque of a step turns
the body as it stood after the step before: a lag of one step in phase at a rotating field's rate. Its reaction, the
opposite torque on the field's source, is delivered to the frame that source is mounted on wherever that frame's
segment takes loads: a segment on its dynamic rung. With `Magnetic.Force`'s reaction beside it, that is the whole of
what the body's moment exerts on a point dipole source. A source fixed in the world, or carried by a kinematic segment,
takes none.

It is a member of the magnetic family: its sign, frame, units, closed form and gradients are checked by the family's
suite, with `Magnetic.Force` its closed-loop case (no net work round a closed loop of the pose in a point dipole's
field, nor, with their reactions, round closed loops of the source's pose and of the two together), and its reaction
against the force and torque of the body's dipole on the source's. For the energy ledger the pair is a potential law,
the derivative of `-m . B` at the magnet's centroid, which the family states once for the pair and its energy case
checks the pair's wrench against; where the field's source is outside (a uniform field, a magnet on a kinematic arm)
their work is supplied by it. It is a complementary
delivery to `Magnetic.Force`, not an alternative to it.

**When not to use it.**
- Where the field varies across the magnet: it is read at one point, the magnet's centroid.
- For a body with several magnets in a field that varies in space: one centroid does not hold them. A body with parts
  and several magnets among them, or none (its moment declared), is refused there unless its `magnet_centre` is
  declared.
- Inside a point dipole source's near field, where the source's own model fails before this law does (see MIME-ANO-100).
- When the reaction on a source carried by a kinematic segment matters: a prescribed motion takes no load, so the
  reaction is delivered only where the source's segment is on its dynamic rung (see MIME-ANO-103).
- For an induced moment, which depends on the field: the moment here is permanent.
- When the lag of one step matters: it is a phase lag of the field's rate times the timestep, so it grows with both.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `magnetic.torque` 0.2.0 reads a field that varies in space at the magnet's centroid. 0.1.0 read it at the
centre of mass, so with the magnet off the centre of mass the pair with `magnetic.force` 0.1.0 missed `m x (G l)` and
did net work round a closed loop of the pose, `m l pi G_xx` for a turn about x on a dipole's axis (MIME-ANO-137). With
the magnet at the centre of mass, or in a uniform field, the two compute the same, and A0's and script A's runs were
measured the same to the last bit on either, in float32 and float64. 0.1.0 stays registered (`mime.interactions.KEPT`),
so that a folder saved with it loads and reproduces, and a script that writes `Magnetic.Torque()` gets 0.2.0. 0.1.0 was
the first version; it delivers its reaction on a source mounted on a segment that takes loads, which no earlier
experiment had.
