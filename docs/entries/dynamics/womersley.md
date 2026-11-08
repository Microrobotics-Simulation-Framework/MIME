---
describes: "0.1.0"
---
# A medium's flow, prescribed in a straight tube

**What it models.** The fully developed flow in a straight rigid circular tube driven by the medium's body force:
Poiseuille's parabola for the drive's steady part and Womersley's profile for its harmonic, the exact solution of the
Navier-Stokes equations for that flow at any Reynolds number, since convection vanishes in it:
`Medium("blood", fills=tube.lumen, fields={"flow": Womersley()})` with `exp.attach(BodyForce(...), medium=blood)`. It
provides the flow's velocity at points, with no grid and no time step of its own, so it runs on a CPU at the cost of a
function evaluation. It is the periodic state of the drive, with no start-up: the flow a run would reach after its
transients have died. It takes nothing back from what reads it.

The same drive and the same medium take `FVM(...)` in its place: the two are rungs of one experiment, compared on the
same lines.

**When not to use it.**
- Anywhere but a straight circular tube's lumen: a stenosis, a bend, a branch or a body in the flow need `FVM(...)`.
- For a start-up, or a drive that is not a steady part and one harmonic: the response is the periodic state's.
- Past its Womersley number's line, `R sqrt(omega / nu)`, which each dtype's series holds (`VALIDITY` in
  `mime.nodes.flow._womersley`, measured against SciPy): float64's is about twice float32's; a run past it is refused
  after it, with the route.
- As a drag on a body: the drag laws that read a provided flow arrive later; until then it is a flow field to read, to
  compare with `FVM(...)` and to check the FVM by.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `womersley` 0.1.0 is its first.
