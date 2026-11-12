---
describes: "0.1.0"
---
# A porous structure's drag on the fluid in it (Darcy's law)

**What it models.** A porous material filling a structure's region, a clot, its solid at rest, the fluid permeating
it: `exp.attach(Darcy(), medium=blood, structure=clot)`, the structure a [`Plug`](../geometry/plug.md) and its
permeability declared, `exp.declare("clot.permeability", um2(...), reason="...")`. The medium's flow (the FVM's
monolithic schemes, IMEX Euler or its steady limit) holds it in its own step as a resistivity `mu / K` on the fluid
inside the region, added to the wall's: Darcy's law where the fluid's viscous term is small beside it, Brinkman's
(Darcy's drag with the fluid's viscous term, the effective viscosity the fluid's own) at its edges. The region is
sampled on the faces through the band the wall is lifted by: its own indicator is the band's sine ramp about its
surface. Where it meets the wall's band, the region and the wall's smeared solid lie side by side along the wall, so a
face's Darcy conductance is the fluid's part of it times `K`: the region's resistivity there is its share of the fluid
over the face's fluid fraction (held above a declared floor, `FLUID_FLOOR`), and the region reaches through the wall's
solid behind it, so the band's solid half and the stiff interior's ramp carry no flow around a plug that closes the
vessel. Its loss is booked under its own id (`darcy:clot`) with the energy ledger, and the flow reports each step the
region's mean pressure on the planes a band's half width outside its two ends (`porous_face_pressure`, Pa), their
difference the drop a Darcy reading takes. Static: the region and its permeability are the run's, sampled each step
from its leaves, so a study or a range moves them, and its derivative passes through both.

**What holds, by how resolved its layer is.** A clot's Brinkman layer (about `sqrt K`) is far thinner than any cell
the flow can afford. Its bulk Darcy flux and pressure drop still hold through the implicit solve, first order in the
spacing; its profile across the layer is resolved from about a cell; its face against free fluid (a plug with a gap
round it) acts as a wall the band smears, shifted outward by a fraction of a cell where the layer is sub-cell; a plug
driven by a body force alone, with no pressure drop across it (a porous medium filling a periodic box), reads a few
per cent low. A plug closing a straight tube is resolved down to a `sqrt(K)` of a few thousandths of a cell with the
default stiff interior (a hundredth with the stiffer one) and unmeasured below, where the first step's cold solve
stops (flagged); the wall lets no measurable flux past it (the floor, bounded far under the plug's own flux there).
These hold for the flux the run reports, `lumen_flux`, which with a porous region is the box's whole axial flux: the
flux through every section, the plug's throughflow. In a periodic box driven by a body force the velocity next to a
closing plug is not its throughflow's: the pressure that holds the drive back pushes the wall's porous solid (which the
drive does not reach) backward along the open part, and the plug closes the solid in its own span, so the open part
carries a loop through the wall's band, a few thousandths of the open tube's flux at 16 cells across with the default
interior, a tenth of that with the stiffer one, whatever `K` (`bypass_ratio` near -1 shows it). A plug that leaves a gap
to the wall (a channel past it) is measured within 2 % where `sqrt(K)` is at least a cell and reads 12 to 16 % low at a
tenth to a hundredth of one. The flow's entry gives the measured figures; the build report states the run's own: its
`sqrt(K)` in cells where the layer is under a cell, and where it lies outside what was measured; and the run raises
`porous_flagged`, with a warning, at each step a region lies outside (under the range measured, or a gap under a
cell). A periodic box reads a plug anywhere along the tube at its image in the period. The flow's other "Darcy flow",
the wall's stiff interior's bypass, is the wall's numerical porosity, not the clot's.

**When not to use it.**
- For a clot that cores, moves or breaks: the region is static (an evolving permeability is a dynamics' state, later).
- For the clot's face as a contact a robot presses on: that is a contact law's, later.
- With RK3: its stages split the penalty from the projection, which leaves a Darcy flux to the splitting
  (`flow.porous-scheme`).
- In a tilted vessel: the box must run along z (`flow.porous-axis`).
- Longer than a periodic box's period (it would overlap its own image, `flow.porous-period`), or outside an open
  vessel's span between its openings (`flow.porous-outside`).
- For the velocity field next to a closing plug in a box driven by a body force (the loop above); its throughflow and
  pressure drop hold.
- In a tube tree, or tilted against the tube, its flux is unmeasured (a plug tilted 10 degrees is bounded only near
  2e-7 of the open flux).

**Where its values come from.** The structure's `permeability` (in square metres, above zero), declared in the experiment with its
reason: no OpenMaterica record holds a permeability yet. A value of zero or below is refused where it is declared
(`value.not-positive`), and a run whose value a study or a range moved there is refused after it (`flow.permeability-not-positive`).
A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `darcy` 0.1.0 is its first.
