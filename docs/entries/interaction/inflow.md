---
describes: "0.1.0"
---
# A volume flux into a flow at an opening

**What it models.** A volume flux into a medium's flow through one of its openings, steady or with a harmonic part:
`exp.attach(Inflow(steady=mL_per_s(...)), medium=blood, surface=tube.start)`, and for a pulsatile inflow `amplitude=`
and `frequency=` (and `phase=`, in radians). The flow holds it in its own step: the opening's face is evolved like any
other, and its pressure is the unknown that pins its flux to the inflow at the step's end, solved with the flow's other
openings through the flow's affine response (a flux row). It pins the opening's whole net flux, the lumen's, the
wall's band's and the solid's faces on its plane together, never the lumen's alone: the lumen carries the inflow less
what the solid carries, which runs either way (the flow's entry gives its size: a few per cent with the inflow in
steady flow at a coarse spacing; in pulsed flow, where the drive's frequency nears the stiff interior's damping rate,
a swing with the drive of up to about a third of the inflow either way, the lumen then carrying up to about a third
more than the inflow). The profile across the opening is the flow's own, developed
by the vessel upstream of it, never imposed. With the energy ledger it is a source: it declares no class, and the flow
books its work, the pressure it returns times the flux the law gives, under the law's name, as supplied.

**When not to use it.**
- At every opening of a flow: an incompressible flow takes in what it lets out, so with no pressure law at any opening
  the inflows must cancel at every instant, or the build refuses them (`flow.openings-ill-posed`).
- To impose a measured velocity profile across a short segment: the profile is the flow's, not the law's.
- Where the lumen's own flux must be the one given, at a coarse spacing in pulsed flow fast against the stiff
  interior's damping: the solid's swing moves the lumen's flux by up to about a third (a finer spacing or a stiffer
  interior, `Band(delta_in=...)`, lowers what sets it, the drive's frequency over the interior's damping rate, as the
  square of the interior's depth).
- As a closed valve at an opening the flow leaves by: that is a `Valve`, gated against the walled-off vessel.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `inflow` 0.1.0 is its first.
