---
describes: "0.1.0"
---
# The force of a field's gradient on a magnet

**What it models.** The force on a body's permanent magnetic moment from the gradient of the field it sits in,
`exp.attach(Magnetic.Force(), body=umr, field=rpm.magnetic)`: the force on a point dipole in a current-free field, the
moment turned into the world by the body's orientation, the gradient read at the body's centre of mass. The force acts
where the body's magnet sits, its `magnet_centre`, from its parts or declared, so where that is off the centre of mass
it delivers a torque about it too. A uniform field's gradient is zero, so beside a uniform field it delivers nothing.
The field's torque on the moment is another law, `Magnetic.Torque`; attach both for the whole of a magnet's response.

Its reaction, the equal and opposite force on the magnet that makes the field, along the same line, is delivered to
the frame that magnet is mounted on wherever that frame's segment takes loads: a segment on its dynamic rung, an arm or
a motor's rotor. With `Magnetic.Torque`'s reaction beside it, that is the whole of what the body's moment exerts on a
point dipole source. A source fixed in the world, or carried by a kinematic segment, takes none.

It is a member of the magnetic family: its sign, frame, units, closed form and gradients are checked by the family's
suite, the equal and opposite force between two point dipoles by a test of its own, and its reaction on the source,
with the torque's, against the force and torque of the body's dipole on the source's.

**When not to use it.**
- Where the gradient varies across the body: it is read at one point, the body's centre of mass.
- Inside a point dipole's near field, where the source's own model fails before this law does (see MIME-ANO-100).
- When the reaction on a source carried by a kinematic segment matters: a prescribed motion takes no load, so the
  reaction is delivered only where the source's segment is on its dynamic rung (see MIME-ANO-103).
- For an induced moment, which depends on the field: the moment here is permanent.

**Versions.** `magnetic.force` 0.1.0 is its first; since dev1's phase 7a3 it delivers its reaction on a source
mounted on a segment that takes loads, which no earlier experiment had, so what it computes elsewhere is unchanged.
