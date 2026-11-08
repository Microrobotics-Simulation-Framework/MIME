---
describes: ["0.1.0", "0.2.0"]
---
# The force of a field's gradient on a magnet

**What it models.** The force on a body's permanent magnetic moment from the gradient of the field it sits in,
`exp.attach(Magnetic.Force(), body=umr, field=rpm.magnetic)`: the force on a point dipole in a current-free field, the
moment turned into the world by the body's orientation, the gradient read where the body's magnet sits, its
`magnet_centre` (from its parts or declared) turned by the body's orientation. The force acts there, so where that is
off the centre of mass it delivers a torque about it too. A uniform field's gradient is zero, so beside a uniform field
it delivers nothing. The field's torque on the moment is another law, `Magnetic.Torque`; attach both for the whole of a magnet's response.

Its reaction, the equal and opposite force on the magnet that makes the field, along the same line, is delivered to
the frame that magnet is mounted on wherever that frame's segment takes loads: a segment on its dynamic rung, an arm or
a motor's rotor. With `Magnetic.Torque`'s reaction beside it, that is the whole of what the body's moment exerts on a
point dipole source. A source fixed in the world, or carried by a kinematic segment, takes none.

It is a member of the magnetic family: its sign, frame, units, closed form and gradients are checked by the family's
suite, with `Magnetic.Torque` its closed-loop case (no net work round a closed loop of the pose in a point dipole's
field, nor, with their reactions, round closed loops of the source's pose and of the two together), the equal and
opposite force between two point dipoles by a test of its own, and its reaction on the source, with the torque's,
against the force and torque of the body's dipole on the source's. For the energy ledger the pair is a potential law,
the derivative of `-m . B` at the magnet's centroid, stated once for the pair by its family, whose energy case checks
the pair's wrench against it; where the field's source is outside, their work is supplied by it.

**When not to use it.**
- Where the gradient varies across the magnet: it is read at one point, the magnet's centroid.
- For a body with several magnets: a field's gradient pulls on each where it sits, which one centroid does not hold;
  a body with several magnets in its parts is refused unless its `magnet_centre` is declared.
- Inside a point dipole's near field, where the source's own model fails before this law does (see MIME-ANO-100).
- When the reaction on a source carried by a kinematic segment matters: a prescribed motion takes no load, so the
  reaction is delivered only where the source's segment is on its dynamic rung (see MIME-ANO-103).
- For an induced moment, which depends on the field: the moment here is permanent.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `magnetic.force` 0.2.0 reads the gradient at the magnet's centroid, where the force acts. 0.1.0 read it at
the centre of mass (MIME-ANO-137); with the magnet at the centre of mass, or in a uniform field, the two compute the
same, and A0's runs were measured the same to the last bit on either, in float32 and float64. 0.1.0 stays registered
(`mime.interactions.KEPT`), so that a folder saved with it loads and reproduces, and a script that writes
`Magnetic.Force()` gets 0.2.0. 0.1.0 was the first version; it delivers its reaction on a source mounted on a segment
that takes loads, which no earlier experiment had.
