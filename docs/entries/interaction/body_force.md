---
describes: "0.1.0"
---
# A body force on a medium's fluid

**What it models.** A uniform force per unit mass on the fluid of a medium, along a direction, steady or with a harmonic
part: `exp.attach(BodyForce(along=..., steady=m_per_s2(...)), medium=water)`, and for a pulsatile drive
`amplitude=` and `frequency=` (and `phase=`, in radians). The medium's flow field applies it to its fluid alone, never to
its solid, so the flux it drives does not depend on how much solid the field's domain holds. On a periodic box it is the
drive: the stand-in for the pressure drop a period of the vessel would carry. The waveform is the drive's; the field kind
is what responds to it, so a different field kind takes the same drive. It is delivered once a step, at the instant the
field's time scheme reads its drive (for IMEX Euler, the step's end), and as its waveform's coefficients, which RK3
evaluates at its own stage times and the prescribed flow (`Womersley()`) responds to analytically: one law, either rung.
A medium whose field reads the waveform takes one such law; its steady part and harmonic go in that one. With the
energy ledger it is outside the components (it declares no class and its node reads nothing of theirs), and the flow
books its work, paired with the fluid's momentum, as supplied under its name.

**When not to use it.**
- As gravity on the fluid: a uniform gravity on an incompressible fluid in a closed box is balanced by its hydrostatic
  pressure and drives nothing; this is a drive standing in for a pressure drop.
- Where the medium has openings: a drive through an opening is a port's, not a body force.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `body_force` 0.1.0 is its first.
