---
describes: "0.1.0"
---
# A closed valve at an opening

**What it models.** An opening shut: no flux through it, `exp.attach(Valve(), medium=blood, surface=tree.end_1)`. The
flow holds it in its own step as a flux row of zero, so the opening's face is evolved like any other and its pressure
is the unknown that holds its flux at zero, solved with the flow's other openings through the flow's affine response:
the pressure behind the valve. It does no work (that pressure on a flux of zero), and with the energy ledger its row is
workless. Its gate is the walled-off vessel: a closed outlet against the same vessel with that outlet capped by solid,
whose own leak through the cap (the solid's Darcy flow) is the gate's measured tolerance.

**When not to use it.**
- As a valve that opens and closes with the pressure across it: that is not built.
- In place of a cap of solid where the solid's own flow matters: the valve holds the opening's flux at zero exactly,
  the solid's flow through the opening's face included.
- At every opening of a flow but one: an incompressible flow then has no flux at all, and with an `Inflow` it is
  refused (`flow.openings-ill-posed`).

**Where its values come from.** It has none.

**Versions.** `valve` 0.1.0 is its first.
