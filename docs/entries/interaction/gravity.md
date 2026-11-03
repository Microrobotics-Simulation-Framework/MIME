---
describes: "0.1.0"
---
# Weight and buoyancy

**What it models.** The weight of a body less the weight of the fluid it displaces, as one wrench about the body's centre of
mass: `exp.attach(Gravity(), body=umr, medium=water)`. The weight acts at the centre of mass and the buoyancy at the body's
centre of buoyancy, so where a body declares that point off its centre of mass the law also delivers the couple the
buoyancy makes about it, and reads the body's pose to turn the offset into the world; it reads the pose only then. The
body's mass and buoyant volume, the medium's density and the experiment's gravity vector are its inputs, and they are
parameters of the graph, so a study can move them.

It is a member of the gravity family.

**When not to use it.**
- With a medium that moves and carries its own hydrostatic load: the medium here is at rest, and with a flow that
  counts the hydrostatic pressure the buoyancy would be counted twice. No such flow is built yet.
- For a body whose displaced volume changes, or that floats at a surface: the buoyant volume is a constant of the body,
  and the fluid's density is uniform.
- For anything but weight and buoyancy: it is not the drag, the contact or any hydrodynamic load on the body.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `gravity` 0.1.0 is its first.
