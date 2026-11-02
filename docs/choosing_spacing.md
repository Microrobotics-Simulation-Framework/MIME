# Choosing `spacing` and `dt`

Two numbers decide what an MRS run costs and what its answer is worth, and neither has a right value in
general: **`spacing`**, how far apart the force points on the body's surface are (`Drag.MRS(spacing=...)`), and
**`dt`**, the base step. This page measures both on the two examples, [`examples/sphere_settling.py`](../examples/sphere_settling.py)
and [`examples/screw_in_tube.py`](../examples/screw_in_tube.py), so that the choice is made from numbers and not from
advice. Where `spacing` is left out, a first-look default is taken (`built.report.pending()` says so, and what was
measured of it); the numbers here are what a user needs to leave the default.

**Since drag 0.4.0 (phase 7m-b) the rung separates its quadrature from its force points**: the unknowns sit at the
force points, `spacing` apart, and the integral is taken on the kind's own quadrature `quadrature_refinement` times
finer, so that the blob (`blob` times the force points' mean spacing) can be smaller than the force spacing and
still be resolved. Every row here is at the family's defaults for the other two settings: for the sphere the blob a
tenth of the force spacing and the quadrature twenty times finer, for the screw a quarter and eight times finer. The
two other settings are numerical too; the rule for each is in the drag's entry page, and a ladder of them is as cheap
as one in free space (below).

## Which numbers are measured, and on what

- **Run times** are wall-clock seconds of `build()` then `run()`, compile included, in float32 (a script's default), on
  the CPU alone, on a laptop: a 12th Gen Intel(R) Core(TM) i7-12850HX with 24 threads, under Linux, Python 3.14. **Each is one run.** The rows of a table were
  run one after another in one process, so the first row of each table also carries the compile. The laptop was shared
  with other work while the runs were taken (its one-minute load average was between 4 and 15; every row of `docs/data/spacing_tables.json` records it),
  so a time on it is neither a floor nor a bound for an unloaded laptop, and two rows close in `spacing` can come out
  of order. A run on your machine is another. The point-force runs that phase 7l-c took on CI's runner are not retaken
  here, so the page no longer carries CI's times.
- **Accuracy columns marked float64** are computed from the rung's 6 x 6 resistance in float64, so that the number is
  the discretisation's and not float32's rounding (`scripts/default_spacing_table.py`). The sphere's reference is
  Stokes's law, a closed form, so its columns are errors. **The screw has no closed form**: in the tube its reference
  is the finest spacing measured, which is not a converged value, so its "against the finest" column is an estimate,
  not a bound; in free space it has one, the shipped helix reference (`tests/oracles/references/helix_fl9_free_space.json`, the
  model's converged limit by two routes, an estimate with its own preregistered band, the file's), which the last
  column is measured against.
- **The run's own number** is also in the tables, from the example run at that spacing. For both examples it equals the
  number the resistance gives to the digits printed, which is what lets the cheap float64 resistance stand in for a run
  where a run is too slow to make.
- What the tables **estimate**: the discretisation error of the rung on these two bodies. What they **bound**: nothing
  about another body, another fluid or another wall. The sphere's error is a property of the method on a sphere; the
  screw's is a property of the method on a thin helix, and a different helix is a different table.

Regenerate them with `scripts/example_tables.py` (its docstring gives the commands). `tests/docs/test_spacing_tables.py`
holds every table below, and every number the prose states of them, to `docs/data/spacing_tables.json`, so none is
typed by hand (the few numbers in the prose that are not in the tables are arithmetic on the examples' own values, which
the page says).

## What `spacing` costs

A body's surface is sampled so that neighbouring force points lie about `spacing` apart. **In free space** the part of
the matrix the body makes with itself is the same at every pose, so the rung assembles it once per build, over the fine
quadrature in chunks, and holds the whole resistance: each step only turns it, and a run costs its build, which grows
with the force points times the quadrature's points. **With a wall** each step adds the wall's correction at the force
points and solves a system of three unknowns for each, so the cost of a step grows as the cube of the force points'
count. Halving `spacing` multiplies the force points by about four: from 512 to 2016 force points (3.9 times) the sphere's run goes from 4.7 s to 36.1 s on the laptop (7.7 times) (free space: the build's
cost); the screw's, from 283 to 1714 force points (6.1 times, from 0.7 mm to 0.3 mm), goes from 9.6 s to 160.8 s on the laptop (16.8 times) (in a tube: the steps'). Another body's cost is its own scale.

## A sphere settling in free space

A steel sphere of ten micrometres radius released in still water, the example's starting size: `spacing` a quarter of the
radius, which gives 338 force points, and `dt` 0.1 ms, thirty steps. Stokes's law gives its settling speed; the rung
should reproduce it.

| spacing / radius | force points | run time, laptop (s) | settled speed against Stokes's law | drag against Stokes's law (float64) | rotational drag against Stokes's law (float64) |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 128 | 2.5 | -0.81% | +0.82% | +2.33% |
| 0.3 | 231 | 2.5 | -0.60% | +0.60% | +1.75% |
| 0.25 | 338 | 3.0 | -0.50% | +0.50% | +1.45% |
| 0.2 | 512 | 4.7 | -0.40% | +0.40% | +1.18% |
| 0.133 | 1152 | 13.9 | -0.27% | +0.27% | +0.79% |
| 0.1 | 2016 | 36.1 | -0.20% | +0.20% | +0.59% |

Read it as follows. The drag converges from above, and the speed from below, by the same amount: the error falls about
as the spacing does (from 0.4 to 0.1 of the radius, a factor of 4 in spacing, the drag's error falls from +0.82% to +0.20%, a factor of 4.1). The drag is within 0.50% at a quarter of the radius (338 force points) and 0.40% at a fifth (512). What remains is the regularisation's: at
its blob the sphere's drag meets the exact regularised drag at that blob within the oracle's tolerance
(`tests/oracles/test_mrs_separated.py`), so a smaller blob, with the quadrature refined to match, moves it closer at
the cost of the build. The rotational drag of the same sphere converges more slowly
(+1.45% against the translation's +0.50% at the starting size), so a run that depends on rotation should be judged on that, not on the translation.

**The Reynolds number is not a discretisation error.** The example's sphere settles at a Reynolds number of 0.03, inside the range the rung is
declared for, and the post-run watch says so. The Reynolds number goes as the cube of the radius (the settling speed as its square), so a sphere of fifty micrometres radius, five times as large, settles at about 3.7: the same comparison with Stokes's law then
comes out as close, and means nothing, since Stokes's law is out of its range there too. `python examples/sphere_settling.py --radius-um 50`
shows the flag, with the size of the inertia the rung leaves out for a sphere (Oseen's first correction).

### `dt` for the sphere

| dt (ms) | steps | run time, laptop (s) | settled speed against Stokes's law |
|---:|---:|---:|---:|
| 0.05 | 60 | 4.9 | -0.49% |
| 0.1 | 30 | 2.1 | -0.50% |
| 0.2 | 15 | 2.1 | -0.50% |
| 0.5 | 6 | 1.9 | -0.56% |
| 1 | 3 | 1.8 | -0.82% |

The settled speed is -0.49% of Stokes's law at 0.05 ms and -0.50% at 0.2 ms, then -0.56% at 0.5 ms and -0.82% at 1 ms, as the step passes the sphere's relaxation time to its settling speed, 2 rho a^2 / (9 mu) = 0.17 ms for this sphere. **Take `dt` no larger than the relaxation time** for a body released from rest; the table's first
rows cost more and give nothing for it. (The first row is slower than the one after it only because it is the first run
in its process and compiles.)

## A screw swimming along a tube

An FL-9 screw on the axis of a quarter-inch tube of glycerol, turned at 1 Hz by a rotating field: the example's starting
size is `spacing` 0.7 mm, `dt` 100 ms, a run of three turns. The tube's wall kind is the unit tube table (the asset
`unit_tube_R0.997`, fetched once with `mime assets fetch`).

| spacing (mm) | force points | run time, laptop (s) | speed per turn, from the run (um) | speed per turn, from the resistance (um, float64) | against the finest, in the tube | in free space, against the reference |
|---:|---:|---:|---:|---:|---:|---:|
| 0.83 | 206 | 8.3 | 278.6 | 278.6 | -13.0% | -6.1% |
| 0.829 (the default) | 222 | not run | not run | 283.8 | -11.3% | -2.6% |
| 0.7 | 283 | 9.6 | 285.8 | 285.9 | -10.7% | -4.0% |
| 0.6 | 416 | 14.9 | 291.6 | 291.6 | -8.9% | -3.5% |
| 0.5 | 567 | 36.5 | 301.1 | 301.1 | -6.0% | -1.3% |
| 0.4 | 957 | 43.4 | 308.9 | 308.9 | -3.5% | -0.5% |
| 0.3 | 1714 | 160.8 | 314.3 | 314.3 | -1.8% | +0.1% |
| 0.16 | 5984 | not run | not run | 320.1 | the reference | +0.7% (at 0.15 mm) |

**This is the number to take away**: at the example's starting size the speed is -10.7% from the finest in the tube, and the same screw in free space -4.0% from the shipped reference, and the default (222 force points) is -2.6% from the shipped reference in free space. Before drag 0.4.0 the same
rows, on point forces at the samples, lay far further below (this page's version of phase 7m-a): the separated
quadrature resolves the near field the point forces did not, at the same count of unknowns. The speed still
rises as the spacing falls: in the tube the speed at 0.3 mm (1714 force points) lies -1.8% from the speed at 0.16 mm (5984), and in free space the rows approach the reference from below and
the finest passes it. **In the tube the rows lie further from the finest than the same rows in free space lie from the
reference**: the wall's image is taken at the force points, which resolve it less well than the fine quadrature
resolves the body's own near field, so the tube's own effect on the speed converges more slowly (MIME-ANO-130). A
value to put in a paper needs a spacing at which the speed has stopped changing to the accuracy wanted; run the
resistance in float64 at two or three spacings, as the script does, before a run, since a resistance costs seconds
where the run in the tube costs minutes.

### The default, and a coarse ladder

The default (`Drag.MRS()` with no `spacing`) is in the table: the coarsest spacing that keeps the screw's shape, its
two caps sampled alike, as before drag 0.4.0 (a step in the tube costs as the cube of the force points, so the default
keeps the earlier count and takes the accuracy the separated quadrature gives at it). The screw has as many force
points as its quadrature has at the spacing, a ring of 16 round the turn at each of `ceil(length / spacing)` stations
along the axis and two caps; **the count of force points, not the spacing, sets the answer**, and at this coarseness it
moves in steps. By force points, the same screw in the same tube:

| spacing (mm) | force points | speed per turn, from the resistance (um, float64) | against the finest |
|---:|---:|---:|---:|
| 1.2 | 119 | 219.5 | -31.4% |
| 1 | 138 | 235.4 | -26.5% |
| 0.9 | 193 | 269.0 | -16.0% |
| 0.83 | 206 | 278.6 | -13.0% |
| 0.829 (the default) | 222 | 283.8 | -11.3% |
| 0.7 | 283 | 285.9 | -10.7% |

**What to do:** refine until three successive rows change by a shrinking amount, and trust none of the coarse ones. A
shrinking change is necessary, not sufficient.

### `dt` for the screw

| dt (ms) | steps | run time, laptop (s) | speed per turn (um) |
|---:|---:|---:|---:|
| 10 | 300 | 55.3 | 285.8 |
| 20 | 150 | 25.3 | 285.8 |
| 40 | 75 | 13.6 | 285.8 |
| 100 | 30 | 7.6 | 285.8 |
| 250 | 12 | 3.9 | 285.8 |

The screw's speed per turn is 285.8 um at 10 ms, 285.8 um at 100 ms and 285.8 um at 250 ms. In this regime (a viscous fluid, a turning Reynolds number of a few hundredths) the body follows the
field at once, so the step only has to resolve the field's turn: **ten steps to a turn is enough for the speed**, and a
run costs what its steps do. The example's test holds that the speed does not depend on the inertia declared either.

The same is not true where the body's inertia and the fluid's matter, and a step is chosen for the regime: the post-run
watch says which regime a run was in. A screw turned at 5 Hz in water, as the first-hour walk's was, is in the hundreds
of the turning Reynolds number, far past the rung's range, and the watch flags it. It is also not a body the surrogate
drag describes: with its axial moment the smallest, as a real screw's is, it turns broadside about half a second into
the rotating field, and in a tube narrower than it is long that is through the wall (`tilt_fired` is raised as the tilt
passes the range the surrogate is declared for, and `bore_fired` as the body stops fitting the bore). What the step
changes there is the timing of that turn, not an error that the step can be refined away.

## A checklist

1. Start at the example's sizes for a body like it. Run it, and read the watch: if it says the Reynolds number is past
   the range, no choice of `spacing` or `dt` repairs that; it is the model.
2. Run again at a `spacing` about 0.7 of the first (with a wall the work in a step doubles or more; in free space the
   build's does), and compare the number you care about. If it moves by more than you can accept, the first was not
   converged; if it moves by less, it may be. Two values agreeing do not prove convergence: refine until three
   successive rows change by a shrinking amount.
3. Check the cheap quantity first. For a body whose answer is a function of its resistance (a synchronous swimmer's
   speed, a sphere's drag) the resistance at a ladder of spacings costs seconds each; run it before the run.
4. Choose `dt` for the regime: no more than the body's relaxation time when it is released from rest, and enough steps
   to a turn of a drive to resolve it. Then halve it once and see that the number does not move.
