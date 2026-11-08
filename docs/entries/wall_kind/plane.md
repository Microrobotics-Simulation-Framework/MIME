---
describes: 0.1.0
---

# `plane`

An unbounded plane with a no-slip wall, the medium filling the half-space its normal points into. The method of
regularised Stokeslets adds the plane's image system to the free-space kernel: the flow a point force makes above the
wall, less the free Stokeslet [@Blake1971]. It is the image of the drag's own regularised Stokeslet, of the drag's own
blob [@Ainley2008], in closed form: a Stokeslet and a Stokes doublet at the source's mirror on the source's blob, a
potential dipole on another, and a rotlet of the blob's square. With the free Stokeslet of that blob it vanishes on the
wall and has no divergence, to rounding, and as the blob vanishes it is Blake's singular image. No table and no asset.

**What it reads.** The plane's normal and offset, from the half-space region the medium fills (a
[plane structure](../geometry/plane.md)'s `half_space`), and the drag's blob: the drag passes its own regularisation to
every wall kind, and this one uses it (the build report says so, with the blob's size). The offset is a leaf a study
may move; the normal is held as built.

**Where it holds.** Every sample of the body above the plane. A sample through the plane is outside the image's
domain, and nothing holds a body off a plane yet (no contact against it): the build refuses a body that starts
through it, and in a run the drag reports the body's height above the plane each step (`wall_gap`, below zero through
it) and raises `near_wall` within a force spacing.

**How accurate.** The image is exact for the blob. The drag near the plane is as accurate as its force points resolve
the gap: on a sphere, within the drag's free-space error down to about a radius, the normal entry failing first as the
gap falls below about half a force spacing; the drag's near-wall validity lines hold here as in the tube.

**When not to use it.** For two walls: two planes' images do not add up to a two-wall image (each reflects the other's
flow). For a run with a grid: there the grid carries the walls.

**The gradient.** Exact: the image is a smooth function of the samples, the offset, the normal and the blob, with no
switch.

**Versions.** The first.
