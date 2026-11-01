---
describes: ["0.2.0", "0.3.0"]
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
a polar grid at the spacing would, less the few its symmetry needs, so that an MRS solve's blob, set by the
samples' mean spacing, barely moves; a section whose troughs and flanks are too narrow for that many carries fewer.
The caps' counts are one number, the samples on each. At run time its regions are evaluated by the shapes `screw` (the
screw surface, unbounded along the axis) and `slab` (between the end planes): the body is their intersection.

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

**Its sampling.** At a spacing, for a solver whose unknowns sit at the samples (the MRS rung), it samples its surface
quasi-uniformly: as many samples as its quadrature has at that spacing, chosen from its quadrature at a finer spacing
by farthest points and Lloyd's relaxation (`mime.scene.quasi_uniform`), each a quadrature sample with its cell's area as
its weight, so that no two crowd. Its hull query, its reach in a direction, is an envelope, the cylinder of its crests,
which holds the body: exact along its axis and across it, and between them beyond the body by up to the kind's
`reach_envelope` of its enclosing radius, a declaration from the supremum over every direction on de Jongh's design
(MIME-ANO-128). It is conservative, and its farthest point is the envelope's, not the body's, so a contact takes its
point from `mime.scene.body_reach`, the reach of the body's own samples, which never overstates it and whose farthest
point is a sample.

**Versions.** 0.3.0 (phase 7m-a) samples its thread and caps together quasi-uniformly: 0.2.0's sampling was its
quadrature, which gives the caps several times their share of the area and leaves about a quarter of the samples
closer than half the mean spacing to another; on de Jongh's design in free space at the frozen spacing the change moves the rung's
swim speed most of the way to a converged reference (`tests/oracles/test_helix_reference.py`). 0.2.0, the frozen model
of phase 7j, stays registered so that a folder saved with it loads and reproduces; a script that writes `Screw(...)` gets
0.3.0. 0.2.0 samples each cap in rings; 0.1.0, the first
(phase 7d), sampled it on a polar grid, as many samples on its innermost ring, near the centre, as on its rim, which
crowded the centre and made an MRS matrix with a tube's wall correction singular in float64 (MIME-ANO-119). The lateral
patch, the shapes and every closed form are unchanged.
