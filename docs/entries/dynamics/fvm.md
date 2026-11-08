---
describes: "0.1.0"
---
# A medium's flow, solved on a grid

**What it models.** The flow of the fluid filling a medium's region, the incompressible Navier-Stokes equations (or the
Stokes equations, with `stokes=True`) on a Cartesian grid of cubic cells with the velocity on the faces (the MAC grid):
`Medium("water", fills=tube.lumen, fields={"flow": FVM(domain=PeriodicBox(length=...))})`. The wall is lifted onto
the grid as a smooth band: a solid outside the lumen whose penalty rises from nothing in the fluid to a stiff interior
beyond the band, so the wall is a first-order immersed wall, its effective position shifted from the geometry's surface by
the band's own calibration, the same in cells at every resolution. Each step solves the viscosity, the wall's penalty
and incompressibility together in one implicit solve (the projected monolithic conjugate gradient), with convection
explicit at the step's start: IMEX Euler, first order in time. `Scheme.Steady()` drops the time term and solves the
steady Stokes problem at each step's drive. `Scheme.RK3()` is the scheme where convection limits the step: three
explicit stages for convection and viscosity, the wall's penalty implicit in each stage, an exact fast projection each
stage and the pressure carried from step to step, third order in time away from the wall. It reads the drive's waveform
at each stage's own time, and a run starts from a pressure consistent with its velocity, made in the program (as does
a state handed over from another run).

**Choosing the scheme and the timestep.** With no `scheme=` the build picks one from the viscous limit: RK3 where its
explicit viscosity is stable at the run's `dt` (its viscous number, nu dt over the spacing squared, within its line), IMEX Euler otherwise. Before any step, the build
predicts the flow's CFL number from what it knows (the drive's steady part as Poiseuille's centreline, its harmonic's peak
from Womersley's profile, in the empty tube) and refuses a `dt` past the chosen scheme's line, with the timestep that
keeps it: "use dt <= X, or dt='auto'". `Experiment(..., dt="auto")` takes the largest stable step instead, dividing a
pulsatile drive's period into a whole number of steps, and the build report says which scheme it chose, its margins,
how many steps a period that is, and what the other scheme's step would have been. A saved experiment keeps the step
`dt="auto"` chose. **"Stable" is all `dt="auto"` promises**: accuracy is the timestep's own question (a scheme at its
stability line can miss a body's drag history by several per cent where many more steps a period would not), so the
report states the steps a period beside it.

The CFL lines are values of each scheme (`TimeScheme.cfl_line`, with RK3's viscous number beside it), on `max|u| dt /
h` along a grid axis, and scaled along a tilted vessel's axis `n` by `max|n_d| / sum|n_d|`: a plug flow along a diagonal
is unstable at a lower face CFL than one along an axis. RK3's line sits under a plug flow's limit at every viscous number
its rule allows, where the line first decided for it did not: a plug flow grew there. With `stokes=True` no CFL line
applies.

The flow is driven by laws bound to the medium: on a periodic box, a `BodyForce` on the fluid, which stands in for the
pressure drop a period of the vessel would carry. The fluid's viscosity and density are the medium's values, from its
material record or `exp.declare`, never settings of the field.

Each step reports its solve: the iterations, the true residual against the tolerance, and how the solve stopped (at its
cap, stagnated, at a residual that is not finite); a run whose solve stops other than converging is flagged, never
silently accepted. With convection on, each step checks the flow's CFL number against the scheme's line along the
vessel's axis, and a run past it is refused, naming the step and the timestep that keeps it. A CFL number that is not
finite counts as past the line. The fluxes of the lumen and of the solid beyond the band
are reported each step: the second is the bypass the porous solid lets through, the wall's own validity line.

**When not to use it.**
- Anywhere but a periodic box around a straight circular tube, aligned with z or tilted in the x-z plane along a lattice
  direction of the box: the region with openings, inflows and outlets, and other vessels arrive with later versions.
- Near a wall at a resolution the band does not resolve: the wall's shift is calibrated on straight tubes at the default
  resolution and finer, and about five cells across a lumen is the band's floor.
- With bodies in the flow: neither a point body's exchange nor a resolved body is coupled yet.
- Where convection matters at a timestep past the CFL line: the build refuses it; use the timestep it names, `dt="auto"`,
  or the Stokes approximation where convection is negligible.
- Where a cheaper rung answers the question: a straight tube's undisturbed flow is `Womersley()`, the same drive's
  analytic response with no grid.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `fvm` 0.1.0 is its first.
