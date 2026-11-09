---
describes: 0.1.0
---

# `helix`

A helical filament: a filament of a circular section round a helix of a radius, a pitch (its sign the handedness) and
a number of turns, its axis along its frame's x, closed by hemispherical caps, so that it is every point within the
filament's radius of its centreline's arc. The geometry of slender-body theory and of many microswimmers' tails, and
the slender kind a `Drag.RFT` body takes. Its frame's origin is its centroid, which lies off the axis (by the caps,
and by a part turn). As a body's part it gives the body its volume, inertia, extent and its surface samples by patch,
`lateral` and `caps`.

**Its centreline** is one segment: the helix by arc length, its frame's normal the rotation-minimising one
[@Bishop1975], its radius the filament's and its wavelength the arc length of a turn. Every value is a JAX function of
the four settings, so a study's derivative in them flows through the centreline and the samples.

**When not to use it.** For a body that is a thread round a core (a screw-shaped body): that is `screw`, whose section
is no filament's. For a filament whose radius or pitch varies along it, or whose section is not round. Settings where
the filament folds on its inner side or meets its neighbouring turn are refused when declared.

**Its derivative** is exact: its samples, centreline, volume and centroid are smooth functions of its settings, and
it evaluates no switch. Its inertia is integrated at build, as every kind's mass properties are.

**Its sampling.** Quasi-uniform at a spacing (`mime.scene.quasi_uniform`), from a quadrature of the trapezoid rule
round the filament and Gauss-Legendre nodes along it, and on each cap Gauss-Legendre nodes in the cosine of the angle
from its pole. No registered shape evaluates its region; the conformance harness holds it to a level set written from
its definition, the distance from its centreline's arc.

**Regions, surfaces, directions and frames.** What a kind names and how a law or a stage reads it: [the scene note](../../scene.md).

**Versions.** 0.1.0 is the first.
