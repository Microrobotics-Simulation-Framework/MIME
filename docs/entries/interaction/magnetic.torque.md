---
describes: "0.1.0"
---
# The torque of a field on a magnet

**What it models.** The torque of the field bound to `field` on the body's permanent magnetic moment, about the
centre of mass: `exp.attach(Magnetic.Torque(), body=umr, field=rpm.magnetic)`. The moment is the body's
`magnetic_moment`, in the body's frame, turned into the world by the body's orientation; the torque is its cross product
with the flux density. It delivers no force: the pull of a field's gradient on the moment is another law, `Magnetic.Force`,
and the two are attached together for the whole of a magnet's response. A field that varies in space is read at the body's
centre of mass, and a uniform field is read as it always was, with no points.

It reads the body's pose at the start of the step, as every law reads its body's state, so the torque of a step turns
the body as it stood after the step before: a lag of one step in phase at a rotating field's rate. Its reaction, the
opposite torque on the field's source, is delivered to the frame that source is mounted on wherever that frame's
segment takes loads: a segment on its dynamic rung. With `Magnetic.Force`'s reaction beside it, that is the whole of
what the body's moment exerts on a point dipole source. A source fixed in the world, or carried by a kinematic segment,
takes none.

It is a member of the magnetic family: its sign, frame, units, closed form and gradients are checked by the family's
suite, and its reaction against the force and torque of the body's dipole on the source's. It is a complementary
delivery to `Magnetic.Force`, not an alternative to it.

**When not to use it.**
- Where the field varies across the body: it is read at one point, the body's centre of mass.
- Inside a point dipole source's near field, where the source's own model fails before this law does (see MIME-ANO-100).
- When the reaction on a source carried by a kinematic segment matters: a prescribed motion takes no load, so the
  reaction is delivered only where the source's segment is on its dynamic rung (see MIME-ANO-103).
- For an induced moment, which depends on the field: the moment here is permanent.
- When the lag of one step matters: it is a phase lag of the field's rate times the timestep, so it grows with both.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `magnetic.torque` 0.1.0 is its first; since dev1's phase 7a3 it delivers its reaction on a source mounted
on a segment that takes loads, which no earlier experiment had, so what it computes elsewhere is unchanged.
