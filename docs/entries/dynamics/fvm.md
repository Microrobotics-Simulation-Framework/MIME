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
steady Stokes problem at each step's drive.

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
- Where convection matters at a timestep past the CFL line: the run is refused; use a smaller timestep, or the Stokes
  approximation where convection is negligible.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `fvm` 0.1.0 is its first.
