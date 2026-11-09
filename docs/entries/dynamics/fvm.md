---
describes: "0.1.0"
---
# A medium's flow, solved on a grid

**What it models.** The flow of the fluid filling a medium's region, the incompressible Navier-Stokes equations (or the
Stokes equations, with `stokes=True`) on a Cartesian grid of cubic cells with the velocity on the faces (the MAC grid):
`Medium("water", fills=tube.lumen, fields={"flow": FVM(domain=PeriodicBox(length=...))})`; every other setting has
its default (the wall's band and its stiff interior, `wall=Band(...)`, grouped as the wall's). The wall is lifted onto
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
cap, stagnated, at a residual that is not finite). A step whose solve stopped other than converging raises the flag
`solve_flagged`, which the run's result reads (`result.flags`, with the count of steps that raised it), and the run
warns of it with that count after it ends; the step's report, `solve_report`, says which way it stopped, and a run that
records every step (`history=True`) keeps each step's. A solve stagnates when, over its last checks of the true
residual, none fell below a share of the best before them (the entry's validity lists the window and the share). With no
tolerance given, a float32 run's solve stops at its tolerance, or at its own rounding floor where that is higher (on a
fine grid with a stiff interior): the residual an answer reaches when each of the operator's coefficients and the
right-hand side is off by one rounding unit. A tolerance given is held as given. A float32 solve that stalls above that
floor is still flagged: with the stiffer interior (`Band`'s smaller `delta_in`) in a flow with a pressure gradient its
conjugate gradient can stall well above it, though the solve could still converge. A solve whose residual is not finite
(a NaN in the step's drive or operator) returns NaN, never a velocity it did not solve for, so the run is refused for
its state's not being finite, naming the flow (and, in a run that records every step, the step where it began).

With convection on, each step checks the flow's CFL number against the scheme's line along the vessel's axis. At the
first step past it the flow's state is held (that step and the later ones leave it unchanged), the run ends at the end of
the chunk the step lies in, and it is refused, naming the step and a timestep to try. A CFL number that is not finite
counts as past the line; the run is then refused for its state's not being finite, naming where that began.

The lumen's flux and the solid's beyond the band are reported each step: the second, as a share of the first, is the
bypass the porous solid lets through, flagged (`bypass_flagged`) above a declared share. With `stokes=True` each step
reports the lumen's Reynolds number and flags it (`reynolds_flagged`) above the declared Stokes line. The build reports a
lumen resolved by fewer cells than the band's calibration, and a float32 run whose rounding floor near a steady state is
above its tolerance. Every such line is listed in the entry's validity.

**Its derivative.** Through the viscosity, the density and the drive it is checked exact against a central difference
that rebuilds the experiment. Through the tube's radius, centre and axis it is unverified: they are not in that check
yet, and through the band a derivative in them carries a ripple within a cell of about ±20 % at 8 to 16 cells across
and ±10 % at 32 (measured on a static tube). They stay trainable. The entry declares one status, so it is
`unverified`.

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
