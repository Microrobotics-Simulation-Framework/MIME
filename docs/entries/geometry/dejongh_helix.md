---
describes: 0.1.0
---

# `dejongh_helix`

The screw-shaped body of de Jongh et al. (the equation of their surface): a cylinder of a reference radius and a length along its own
x axis, its cross-section modulated by a number of starts and turned along the axis by the normalised wavenumber, its
sign the handedness, closed by flat caps. Its origin is its centroid: the body the earlier line ran from one end is
moved half its length along the axis, the cross-section's angle still measured at that end, so a reference point is
stated where it was assumed (MIME-ANO-112).

It gives a body its volume and inertia in closed form, its extent (the crests' radius), its family parameters (the
radius, length, wavenumber, modulation and starts, which the de Jongh drag surrogate reads in place of declared
values), the thread's direction as its tangent, and its surface samples by patch: `lateral`, the screw surface, and
`caps`. At run time its regions are evaluated by the shapes `dejongh_screw` (the screw surface, unbounded along the
axis) and `slab` (between the end planes): the body is their intersection.

**When not to use it.** For a screw of another cross-section, or a body with fins or a tail: it is this family only.
A drag surrogate trained on it covers the modulation and starts it was trained at, and refuses others.

**Its derivative.** The screw's distance comes from a root find whose gradients are stopped, then a Newton step
with them flowing; its seam, where two points of the surface are equally near (the axis among them), is a kink in the
distance and a jump in the normal and the tangent. The shape conformance suite checks the derivatives against finite
differences off, near and on it.

**Versions.** The first, landed in phase 7d.
