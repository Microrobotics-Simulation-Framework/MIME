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
every wall kind, and this one uses it (the build report says so, with the blob's size). The offset is the drag node's
leaf `wall_offset`, which a gradient or a caller of the node reaches, and the plane's `floor.offset` is a study's axis, a
run's override and a range: the override writes that leaf, and the resistance with it is a fresh build's at the offset,
entry for entry; an offset that puts the plane through the body is refused at its point. The drag's own names for it
(`drag:<body>.wall_offset`, `drag:<body>.offset`) are no axis (`study.unknown-path`). The normal is held as built.

**Where it holds.** Every sample of the body above the plane. A sample through the plane is outside the image's
domain, and nothing holds a body off a plane yet (no contact against it): the build refuses a body that starts
through it, and in a run the drag reports the body's height above the plane each step (`wall_gap`, below zero through
it) and raises `near_wall` within a force spacing.

**How accurate.** The image is exact for the blob. The drag near the plane is as accurate as its force points resolve
the gap, so the error is a function of the gap in force spacings, not in radii (a radius is 3.3 spacings at 0.3 radii
and 5 at 0.2). The numbers are **estimates, not bounds**: one sphere, against Brenner's exact series for the normal
entry and the method of fundamental solutions with Blake's image for the others, at force spacings of 0.3 and 0.2
radii, a blob of 0.1 and float64.

- The image at the force points (the default), the normal entry: +3.6% at one spacing from the plane, +5.4% at 0.75,
  +10.3% at 0.55 (the drag's clear line), +14% at 0.5, +23% at 0.45 and +70% at 0.4; at 0.2 radii +11.8% at 0.55 and
  +16.5% at 0.5. At 0.36 spacings (0.3 radii) one pose was indefinite, so the run stops (`drag.mrs-indefinite`), and
  at 0.33 a definite resistance read the normal entry 34% low: definite does not mean accurate here. At the default
  blob of 0.5 the same image reads +30% at 0.75 spacings and +60% at 0.55, and is indefinite from 0.33.
- The image integrated over the cells (`wall_refinement=2`), the worst entry at 0.5 spacings: 2.6% unturned and 4.6%
  turned at 0.3 radii, 14.8% with a pole of the sphere's sampling grid facing the wall and 5.0% turned at 0.2 radii.
  The error depends on the orientation, so a body that turns sees more than one of these.
- `near_wall` is raised within one spacing, where the point image's normal entry is already +3.6% and rising; the last
  spacing is what the near-contact hand-over closes, not this image.
- float32 against float64 holds to 5e-6 of an entry's diagonal scale for a plane through the origin. The height is
  `y . n - offset` in the working dtype, so a displaced plane costs precision as the offset grows over the body's
  size: 8e-7 at offset 0, 2e-5 at 0.5 m and 1.7e-4 at 5 m for a 1 mm sphere (5,000 radii, past the 1e-4 line). Nothing
  flags or refuses it: past about a thousand body sizes in float32, run in float64 or place the world's origin on the
  plane.

**When not to use it.** For two walls: two planes' images do not add up to a two-wall image (each reflects the other's
flow). For a run with a grid: there the grid carries the walls.

**The gradient.** Exact: the image is a smooth function of the samples, the offset, the normal and the blob, with no
switch.

**Versions.** The first.
