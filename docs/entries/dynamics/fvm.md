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
steady Stokes problem at each step's drive. `Scheme.RK3()` is the scheme where convection limits the step: three
explicit stages for convection and viscosity, the wall's penalty implicit in each stage, an exact fast projection each
stage and the pressure carried from step to step, third order in time away from the wall. It reads the drive's waveform
at each stage's own time, and a run starts from a pressure consistent with its velocity, made in the program (as does
a state handed over from another run).

**Choosing the scheme and the timestep.** With no `scheme=` the build picks one from the viscous limit: RK3 where its
explicit viscosity is stable at the run's `dt` (its viscous number, nu dt over the spacing squared, within its line), IMEX Euler otherwise. Before any step, the build
predicts the flow's CFL number from what it knows (the drive's steady part as Poiseuille's centreline and its harmonic's
peak from Womersley's profile, doubled for a start from rest, in the empty tube) and refuses a `dt` past the chosen scheme's line, with the timestep that
keeps it: "use dt <= X, or dt='auto'". All of this is judged at the step the flow itself takes: with `FVM(rate=...)`
that is the rate, a whole number of base steps, and the report states it. The prediction is arithmetic at build, in
float64 whatever the run's dtype; past the Womersley number its series is held to, it takes a bound that stays above
the flow's peak (a prediction that is not finite refuses the build). `Experiment(..., dt="auto")` takes
the largest stable step instead, dividing a pulsatile drive's period and a declared `duration` into whole numbers of
steps, and the build report says which scheme it chose, its margins, how many steps a period that is, and what the
other scheme's step would have been. Under `dt="auto"` the flow's own `rate=` is refused (the two would set one step),
and any other entry's rate must be a whole multiple of the step chosen. A saved experiment keeps the step `dt="auto"`
chose. **"Stable" is all `dt="auto"` promises**: accuracy is the timestep's own question (a scheme at its
stability line can miss a body's drag history by several per cent where many more steps a period would not), so the
report states the steps a period beside it.

The CFL lines are values of each scheme (`TimeScheme.cfl_line`, with RK3's viscous number beside it), on `max|u| dt /
h` along a grid axis, and scaled along a tilted vessel's axis `n` by `max|n_d| / sum|n_d|`: a plug flow along a diagonal
is unstable at a lower face CFL than one along an axis. RK3's line sits under a plug flow's limit at every viscous number
its rule allows, where the line first decided for it did not: a plug flow grew there. With `stokes=True` no CFL line
applies.

The build report shows the box: its cells, their spacing and the lumen's diameter in them, and the run's memory
estimate (the cells times the bytes a cell a run on a CPU was measured to hold at its peak), against the memory a run
can have: a GPU's limit, or on a CPU the smaller of the host's available memory and the limit of the control group the
process runs in (a container's, a batch job's). A box whose run will not fit is refused when the experiment is planned
(`flow.memory`), with a coarser grid or a shorter period as its route. The estimate is measured, not a bound: the peak
grows with the threads a run uses, and a GPU's bytes a cell are not measured. It is measured without the energy
ledger, whose terms make a step's program 1.15 to 1.26 times larger by XLA's analysis at D/h 32: a metered run holds
more than the estimate says. The flow provides the velocity at points
(`field.flow.velocity`, its solved-field declarations' `provides`); a drag, which assumes its medium at rest, is refused
wherever the region its medium fills carries a field.

With `EnergyLedger()` attached the build meters the flow as a receiver of the `fluid` port kind: its fluid's momentum
(`flow.momentum`, the integral of the density times the velocity over the fluid, none over the solid) against the body
force each law driving it delivers. The kinetic energy is what the flow stores; each law's drive is booked as the work
it did on the fluid, a `BodyForce` supplying its own (it reads nothing of the components, so it is outside); the
viscous loss is booked under the flow's viscosity (`water.flow.viscosity`) and the wall's band under the wall
(`tube.wall`); the scheme's own terms (backward Euler's, convection at mixed time levels and the solve's residual) are
numerical, never a loss. IMEX Euler and its steady limit close the step's identity to rounding at every step; RK3's
split penalty has no formula term, so each RK3 step is judged to have dissipated instead. The stall detector is written
with the ledger.

The flow is driven by laws bound to the medium: on a periodic box, a `BodyForce` on the fluid, which stands in for the
pressure drop a period of the vessel would carry. The fluid's viscosity and density are the medium's values, from its
material record or `exp.declare`, never settings of the field.

Each step of the implicit schemes reports its solve: the iterations, the true residual against the tolerance, and how
the solve stopped (at its cap, stagnated, at a residual that is not finite); RK3's projections are exact fast solves,
which do not stop short, so an RK3 step never raises the flag that follows. A step whose solve stopped other than
converging raises the flag `solve_flagged`, which the run's result reads (`result.flags`, with the count of steps that
raised it), and the run warns of it with that count after it ends; the step's report, `solve_report`, says which way it
stopped, and a run that records every step (`history=True`) keeps each step's. A solve stagnates when, over its last
checks of the true residual, none fell below a share of the best before them (the entry's validity lists the window and
the share). With no tolerance given, a float32 run's solve stops at its tolerance, or at its own rounding floor where
that is higher (on a fine grid with a stiff interior): the residual an answer reaches when each of the operator's
coefficients and the right-hand side is off by one rounding unit. A tolerance given is held as given. A float32 solve
that stalls above that floor is still flagged: with the stiffer interior (`Band`'s smaller `delta_in`) in a flow with a
pressure gradient its conjugate gradient can stall well above it, though the solve could still converge. A solve whose
residual is not finite (a NaN in the step's drive or operator) returns NaN, never a velocity it did not solve for, so
the run is refused for its state's not being finite, naming the flow (and, in a run that records every step, the step
where it began).

With convection on, each step checks the flow's CFL number against the scheme's line along the vessel's axis, behind
the build's prediction (for a state handed over mid-run, or a flow the prediction does not cover). At the first step
past it the flow's state is held (that step and the later ones leave it unchanged), the run ends at the end of the
chunk the step lies in, and it is refused, naming the step and a timestep to try. A CFL number that is not finite
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
- Where convection matters at a timestep past the CFL line: the build refuses it; use the timestep it names, `dt="auto"`,
  or the Stokes approximation where convection is negligible.
- Where a cheaper rung answers the question: a straight tube's undisturbed flow is `Womersley()`, the same drive's
  analytic response with no grid.

**Where its values come from.** A declaration, a record or the body's parts, and what "declared" means: [the materials note](../../materials.md).

**Versions.** `fvm` 0.1.0 is its first.
