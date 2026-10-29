---
describes: "0.1.0"
---
# The field of a permanent magnet, as a point dipole

**What it models.** The magnetic field of a magnet part, and its gradient, at each reader's point, as the field of a
point dipole at the magnet's centroid: `fields={"magnetic": PointDipole(part="rotor_magnet")}`. The part is a
part of the apparatus with the `PermanentMagnet` role; its moment is the apparatus's `magnetic_moment`, declared as a
magnitude along the direction the role gives in the part's frame. The magnet sits where its mount puts it, so it
follows the chain that carries it, with no edge written by hand. The gradient is computed from its closed form, and is
the field's derivative.

**When not to use it.**
- Near the magnet. A point dipole is faithful only far from it, measured against the magnet's radius, the radius of
  the smallest sphere about its centroid that holds it, which the provider's node holds; closer, a finite magnet's
  field differs by more (see MIME-ANO-100). No finite-magnet model is built.
- At the magnet itself, where the field is singular: it is evaluated outside the magnet only, and a body that starts
  inside the smallest sphere that holds the magnet is refused at build. The validity line that checks the distance
  against the magnet's radius is a1's.
- For a magnet whose magnetisation is not rigid and uniform along one direction.

**Versions.** `point_dipole` 0.1.0 is its first.
