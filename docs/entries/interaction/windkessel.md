---
describes: "0.1.0"
---
# The three-element Windkessel at an opening

**What it models.** The downstream vessels beyond an opening as a lumped circuit: a `proximal` resistance, then the
`compliance` of a capacitor drained by a `distal` resistance to a distal pressure of zero,
`exp.attach(Windkessel(proximal=Pa_s_per_m3(...), compliance=m3_per_Pa(...), distal=Pa_s_per_m3(...)), medium=blood,
surface=tube.outlet)`. The flow holds it in its own step and steps it by backward Euler at its own instant, so its
pressure is affine in the opening's flux over the step (a pressure row) and the exchange across the opening does no work
of its own. Backward Euler's numerical dissipation, half the compliance times the capacitor's change squared, first
order in the step, is booked under its name with its capacitor's stored energy and its resistances' loss: with the
energy ledger its own books close at every step, and its loss is read net of what its capacitor gives back.

**When not to use it.**
- At an inflow, where the flux is known: that is an `Inflow`.
- Where the downstream network's compliance or inertance matters branch by branch: a lumped circuit holds one
  capacitor.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `windkessel` 0.1.0 is its first.
