---
describes: "0.1.0"
---
# A volume flux into a flow at an opening

**What it models.** A volume flux into a medium's flow through one of its openings, steady or with a harmonic part:
`exp.attach(Inflow(steady=mL_per_s(...)), medium=blood, surface=tube.start)`, and for a pulsatile inflow `amplitude=`
and `frequency=` (and `phase=`, in radians). The flow holds it in its own step: the opening's face is evolved like any
other, and its pressure is the unknown that pins its flux to the inflow at the step's end, solved with the flow's other
openings through the flow's affine response (a flux row). It pins the whole opening's flux, the wall's band included:
the band carries its Darcy share of it (about 0.1 to 1 %), so the lumen takes the inflow less that share. The profile across the opening is the flow's own, developed
by the vessel upstream of it, never imposed. With the energy ledger it is a source: it declares no class, and the flow
books its work, the pressure it returns times the flux the law gives, under the law's name, as supplied.

**When not to use it.**
- At every opening of a flow: an incompressible flow takes in what it lets out, so with no pressure law at any opening
  the inflows must cancel at every instant, or the build refuses them (`flow.openings-ill-posed`).
- To impose a measured velocity profile across a short segment: the profile is the flow's, not the law's.
- As a closed valve at an opening the flow leaves by (`Inflow(steady=...)` of zero there): it is admitted, and not
  gated yet.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `inflow` 0.1.0 is its first.
