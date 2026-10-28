---
describes: 0.1.0
---

# `sphere`

A solid ball, given by its radius and centred on its centroid: a bead, a particle, or the body of a magnetic sphere
pulled by a field gradient. As a body's part it gives the body its volume and inertia, its extent (its radius, about
any axis) and its surface samples by patch, for a solver that integrates over the surface; at run time a node
evaluates it by the shape `sphere`.

**When not to use it.** For a body that is not round: its extent and anything read from its samples would be a
sphere's.

**Its derivative** is the shape's: at the centre the distance is a cone and the normal has no limit. Within a band
about the centre the distance is minus the radius and the normal zero, each with the derivative zero in the point.

**Versions.** The first, landed with the capsule and the de Jongh helix in phase 7d.
