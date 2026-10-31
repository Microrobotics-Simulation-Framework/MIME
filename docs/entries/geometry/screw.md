---
describes: 0.2.0
---

# `screw`

The screw-shaped body of de Jongh et al. (the equation of their surface): a cylinder of a reference radius and a
length along its own x axis, its cross-section modulated by a number of starts and turned along the axis by the
normalised wavenumber, its sign the handedness, closed by flat caps. Its origin is its centroid: the body the earlier
line ran from one end is moved half its length along the axis, the cross-section's angle still measured at that end,
so a reference point is stated where it was assumed (MIME-ANO-112).

It gives a body its volume and inertia in closed form, its extent (the crests' radius), its family parameters (the
radius, length, wavenumber, modulation and starts, which the de Jongh drag surrogate reads in place of declared
values), the thread's direction as its tangent, and its surface samples by patch: `lateral`, the screw surface, and
`caps`. Each cap is sampled in rings: a sample at its centre and rings out to the rim, each a copy of the section at a
fraction of the way out, its count a multiple of the starts growing with its radius, so that no two of a cap's samples
lie closer than a declared share of the spacing (`CAP_SEPARATION`), the samples are themselves after a turn by one
start, as the section is, and the cap's area and first and second moments are exact. A cap carries as many samples as
a polar grid at the spacing would, less the few its symmetry needs, so that a boundary-element solve's blob, set by the
samples' mean spacing, barely moves; a section whose troughs and flanks are too narrow for that many carries fewer.
The caps' counts are one number, the samples on each. At run time its regions are evaluated by the shapes `screw` (the screw surface, unbounded along the
axis) and `slab` (between the end planes): the body is their intersection.

**When not to use it.** For a screw of another cross-section, or a body with fins or a tail: it is this family only.
A drag surrogate trained on it covers the modulation and starts it was trained at, and refuses others.

**Its derivative.** The screw's distance comes from a root find whose gradients are stopped, then a Newton step
with them flowing; its seam, where two points of the surface are equally near (the axis among them), is a kink in the
distance and a jump in the normal and the tangent. The seam ends at the focal set, each crest's and each trough's
centre of curvature turned along the axis, where the nearest point moves without bound as the point does: the
distance is once differentiable there and the normal and the tangent continuous with no derivative, and within a band
of rounding about it the Newton step, singular there, is not taken. The shape conformance suite checks the
derivatives against finite differences off, near and on both, and the distance on the focal set against its exact
value.

**Versions.** 0.2.0 samples each cap in rings; 0.1.0, the first (phase 7d), sampled it on a polar grid, as many
samples on its innermost ring, near the centre, as on its rim, which crowded the centre and made a boundary-element
matrix with a tube's wall correction singular in float64 (MIME-ANO-119). The lateral patch, the shapes and every
closed form are unchanged.
