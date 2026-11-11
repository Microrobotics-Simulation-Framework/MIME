---
describes: "0.1.0"
---
# A closed valve at an opening

**What it models.** An opening shut: no flux through it, `exp.attach(Valve(), medium=blood, surface=tree.end_1)`. The
flow holds it in its own step as a flux row of zero, so the opening's face is evolved like any other and its pressure
is the unknown that holds its flux at zero, solved with the flow's other openings through the flow's affine response:
the pressure behind the valve. It does no work (that pressure on a flux of zero), and with the energy ledger its row is
workless. It holds the opening's net flux at zero, not each face's: the lumen behind it carries out what the solid's
faces on the opening bring back in, the solid's Darcy flow, which falls as the square of the stiff interior's depth.
On two parallel tubes from one inlet, one outlet closed, that exchange is a few per cent of the inflow at the default
stiff interior and a hundred times less at a tenth of its depth, `Band(delta_in=...)` (the flow's entry gives the
figures). Its gate is the walled-off vessel, the same vessel with that outlet capped by solid, whose cap and the solid
around it leak as much: the inlet pressures agree to the two runs' solid flows carried through the open tube's
resistance, so the gate sees a valve error from about a twelfth of the inflow at the default interior and from about a
thousandth of it at the stiffer one (two thousandths in single precision).

**When not to use it.**
- As a valve that opens and closes with the pressure across it: that is not built.
- As a closure tighter than the solid's own flow: at the default stiff interior a closed outlet's lumen still exchanges
  a few per cent of the vessel's flow through the solid around it; an interior a tenth as deep lowers it a hundredfold.
- At every opening of a flow but one: an incompressible flow then has no flux at all, and with an `Inflow` it is
  refused (`flow.openings-ill-posed`).

**Where its values come from.** It has none.

**Versions.** `valve` 0.1.0 is its first.
