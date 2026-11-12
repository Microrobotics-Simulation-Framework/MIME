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
per cent low. A plug closing a straight tube was resolved down to a `sqrt(K)` of a few thousandths of a cell with the
default stiff interior (a hundredth with the stiffer one), but only with the solve's iteration cap raised far past its
default, which an experiment cannot raise. At the default cap the two-way solve converges down to the hand-over's line,
which rises with the lumen's cells across (D/h), since the solve's count does at a fixed line: a hundredth of a cell up to sixteen cells across, three hundredths
at thirty-two and a tenth at sixty-four, with either interior, in a periodic tube and (to thirty-two across) in an
open vessel. The line
is the box type's (a periodic tube's, an open vessel's), measured over a range of D/h; below the line at a D/h outside
it the run raises `porous_flagged`. The wall lets no measurable flux past a closing plug (the floor, bounded far under
the plug's own flux there). A step whose velocity goes not finite from a finite start and drive in a flow holding a
porous region is refused after the run (`flow.porous-solve-not-finite`); a NaN in the drive or a body's pose is the
run's own refusal (`run.non-finite`). These hold for the flux the run reports, `lumen_flux`, which with a porous region
is the box's whole axial flux: the flux through every section, the plug's throughflow (below the line, the plug
counted at its reading, next). In a periodic box driven by a body force the velocity next to a
closing plug is not its throughflow's: the pressure that holds the drive back pushes the wall's porous solid (which the
drive does not reach) backward along the open part, and the plug closes the solid in its own span, so the open part
carries a loop through the wall's band, a few thousandths of the open tube's flux with the default interior and a
tenth of that with the stiffer one, whatever `K` (`bypass_ratio` near minus one shows it). A plug that leaves a gap to
the wall (a channel past it) holds where `sqrt(K)` is at least a cell and reads well low below that. The flow's entry
gives the measured figures; the build report states the run's own: its
`sqrt(K)` in cells where the layer is under a cell, the route its throughflow takes and the line; and the run raises
`porous_flagged`, with a warning, at each step a region lies outside what was measured (a gap under a cell, or below
the line beside a resolved body or at a D/h outside the line's range). A periodic box reads a plug anywhere along the
tube at its image in the period. A tube tree holding a plug is refused (`flow.porous-tree`): no line was measured in one.

**Below the line: the hand-over.** A dense clot (`sqrt(K)` under the line) is a wall for the outer flow, to within the
little it lets through. The flow's solve holds it at the line's permeability, so the solve never gets stiffer than it
was measured converging at, and the outer flow carries the flux that held plug passes, some millionths of the open
tube's. Its throughflow, which lysis and drug transport need, is then read one way from the solved pressures on its two
faces: Darcy's law in Brinkman's form, `F_B K A dp / (mu L)`, the drop `dp` its face pressures' difference with the
body force on its own fluid, `A` its section within the lumen. Each step reports the throughflow by its route,
`porous_flux` (the solve's flux through the plug's mid-section at or above the line, the reading below it), and the
seepage velocity over its section, `porous_seepage`; `lumen_flux` counts the plug at its reading in place of the held
flux through it, so below the line it too is the throughflow (the velocity field next to the plug is still the held
solve's). All are differentiable in `K`, and the reading in the solved pressures. The reading's error is the held
plug's feedback on the drop, about `R_rest / R_held`: every resistance in series with the plug outside it (the open
lumen's, and in an open vessel its outlets' laws) over the held plug's, beside the face pressures' own few per cent. A
distal resistance of a small artery's beside a plug held at the line can make it several per cent (the flow's entry
gives the measured cases). **It needs the drop to be the plug's,** which holds where the flow can go elsewhere: a
branch, a gap round the plug, an opening whose law holds a pressure or a Windkessel, or a body force driving a closed
periodic box. Where a law fixes the flux (an inflow, a valve) out of a part of the lumen whose every other way out is
through plugs below the line (no opening of a pressure law in it), the solved drop is the held plugs' instead; the run
then raises `forced_flagged`, with a warning, at each step a plug's reading falls short of the solve's flux through it
(by more than a tenth: a forced plug within a tenth of the line's `K` is not told apart). Every such plug is cut at
once, so two plugs in parallel branches below one inflow are forced together, and a closed valve is no way out. That
test is structural: a held plug obeys its own Darcy relation whatever drives it, so no comparison of the solve's flux
with its reading could tell a fixed flux from a driven one, and the comparison is made only where the vessel's
topology says the flux is fixed. A body moving through the vessel can force flow too, which the test does not see, so
a plug below the line beside a resolved body raises `porous_flagged`. **Plugs in series:** a plug below the line in
one flow path with another porous region is refused after the run (`flow.porous-series`): the held plug passes the
line's flux, which every region in its path carries, so neither reading would be the plug's (two plugs in one tube read
some ninety times the closed form and nine tenths of it before the refusal). Resolve each two-way, or hold one plug a flow path. The flow's
other "Darcy flow",
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
- Tilted against the tube, its flux is unmeasured (a tilted plug's floor is bounded only far above an aligned one's:
  the flow's entry).
- Below the line, for flow forced through a plug that closes the only path (`forced_flagged`): give the flow somewhere
  else to go, or resolve the plug two-way (the line rises with D/h, so a coarser spacing lifts a plug toward it). A
  plug of another shape below the line needs a small Darcy solve inside it, which is not built.
- Below the line beside another porous region in one flow path (`flow.porous-series`), or in a tube tree
  (`flow.porous-tree`).

**Where its values come from.** The structure's `permeability` (in square metres, above zero), declared in the experiment with its
reason: no OpenMaterica record holds a permeability yet. A value of zero or below is refused where it is declared
(`value.not-positive`), and a run whose value a study or a range moved there is refused after it (`flow.permeability-not-positive`).
A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `darcy` 0.1.0 is its first.
