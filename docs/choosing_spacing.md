# Choosing `spacing` and `dt`

Two numbers decide what a boundary-element run costs and what its answer is worth, and neither has a right value in
general: **`spacing`**, how far apart the samples on the body's surface are (`Drag.BEM(spacing=...)`), and **`dt`**, the
base step. This page measures both on the two examples, [`examples/sphere_settling.py`](../examples/sphere_settling.py)
and [`examples/screw_in_tube.py`](../examples/screw_in_tube.py), so that the choice is made from numbers and not from
advice. Where `spacing` is left out, a first-look default is taken (`built.report.pending()` says so, and what was
measured of it); the numbers here are what a user needs to leave the default.

## Which numbers are measured, and on what

- **Run times** are wall-clock seconds of `build()` then `run()` in a fresh process, compile included, as a script's
  first run takes them, in float32 (a script's default), on the CPU alone. There are two columns, each from one machine:
  - **laptop**: a 12th Gen Intel(R) Core(TM) i7-12850HX with 24 threads, under Linux, Python 3.14. It was shared with
    other work while the runs were taken (its one-minute load average was between 5 and 15; every row of
    `docs/data/spacing_tables.json` records it), so these times are upper bounds and two rows close in `spacing` can come
    out of order.
  - **CI**: a GitHub-hosted `ubuntu-latest` runner, an AMD EPYC 7763 64-Core Processor with 4 threads, Python 3.12, one
    run of `scripts/example_tables.py` on it (the data file records the run's address). It was not shared, and it is
    **CI's runner class**, which is where the examples' tests run. It ran the spacings up to 0.5 mm for the screw,
    where the laptop ran to 0.3 mm; a dash is a row it did not run.
  A run on your machine is neither: the two columns came out within about a factor of two of each other, the runner no
  slower than the loaded laptop for these dense solves.
- **Accuracy columns marked float64** are computed from the rung's 6 x 6 resistance in float64, so that the number is
  the discretisation's and not float32's rounding (`scripts/default_spacing_table.py`). The sphere's reference is
  Stokes's law, a closed form, so its column is an error. **The screw has no closed form**: its reference is the finest
  spacing measured, and **that spacing is not converged** (below), so the screw's "against the finest" column
  *understates* the error: it is a lower bound.
- **The run's own number** is also in the tables, from the example run at that spacing. For both examples it equals the
  number the resistance gives (the screw's, to the digits printed), which is what lets the cheap float64 resistance
  stand in for a run where a run is too slow to make.
- What the tables **estimate**: the discretisation error of the rung on these two bodies. What they **bound**: nothing
  about another body, another fluid or another wall. The sphere's error is a property of the method on a sphere; the
  screw's is a property of the method on a thin helix, and a different helix is a different table.

Regenerate them with `scripts/example_tables.py` (its docstring gives the commands). `tests/docs/test_spacing_tables.py`
holds every table below to `docs/data/spacing_tables.json`, so no number in a table on this page is typed by hand. The speeds in the two machines'
columns are the same to the digits printed, which is the one check across machines these runs give.

## What `spacing` costs

A body's surface is sampled so that neighbouring samples lie about `spacing` apart, and the rung solves a system in all
the samples, three unknowns for each, at the pose of every step. Halving `spacing` of a surface multiplies its samples by
about four, and the run time rises by more than the samples do (the tables: from 512 to 2016 samples the sphere's run
goes from 3.5 s to 19 s; the screw's from 283 to 1714 samples goes from 5 s to 101 s). That is what the first-hour walk
met: halving a sphere's spacing turned a few seconds into minutes. The tables give the cost for these two bodies in
seconds; another body's is its own scale.

## A sphere settling in free space

A steel sphere of ten micrometres radius released in still water, the example's starting size: `spacing` a quarter of the
radius, which gives 338 samples, and `dt` 0.1 ms, thirty steps. Stokes's law gives its settling speed; the rung should
reproduce it.

| spacing / radius | samples | run time, laptop (s) | run time, CI (s) | settled speed against Stokes's law | drag against Stokes's law (float64) |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 128 | 0.6 | 1.3 | -2.30% | +2.44% |
| 0.3 | 231 | 0.4 | 0.9 | -1.59% | +1.70% |
| 0.25 | 338 | 2.3 | 1.1 | -1.30% | +1.30% |
| 0.2 | 512 | 3.5 | 2.2 | -1.02% | +1.00% |
| 0.133 | 1152 | 9.0 | 8.0 | -0.64% | +0.60% |
| 0.1 | 2016 | 19.0 | 30.6 | -0.47% | +0.43% |

Read it as follows. The drag converges from above, and the speed from below, by the same amount: the error falls about
as the spacing does (a factor of four in spacing, a factor of nearly six in error from 0.4 to 0.1). A user who wants
the sphere's drag to a per cent starts at a quarter of the radius; to a half per cent, at a tenth, which costs an order
of magnitude more here. The rotational drag of the same sphere converges more slowly (about four times the translation's
error at the starting size: `scripts/default_spacing_table.py` prints both), so a run that depends on rotation should
be judged on that, not on this table.

**The Reynolds number is not a discretisation error.** The example's sphere settles at a Reynolds number of 0.03, inside
the range the rung is declared for, and the post-run watch says so. A sphere of fifty micrometres radius settles at 3.7;
the same comparison with Stokes's law then comes out as close, and means nothing, since Stokes's law is out of its range
there too. `python examples/sphere_settling.py --radius-um 50` shows the flag, with the size of the inertia the rung
leaves out for a sphere (Oseen's first correction).

### `dt` for the sphere

| dt (ms) | steps | run time, laptop (s) | run time, CI (s) | settled speed against Stokes's law |
|---:|---:|---:|---:|---:|
| 0.05 | 60 | 2.4 | 1.8 | -1.29% |
| 0.1 | 30 | 1.5 | 1.1 | -1.30% |
| 0.2 | 15 | 1.4 | 0.8 | -1.30% |
| 0.5 | 6 | 0.7 | 0.7 | -1.36% |
| 1 | 3 | 0.3 | 0.7 | -1.61% |

The settled speed does not move to the digits printed from 0.05 ms to 0.2 ms, then changes as the step passes the
sphere's relaxation time to its settling speed (0.17 ms for this sphere). **Take `dt` no larger than the relaxation time**
for a body released from rest; the table's first rows cost more and give nothing for it. (The first row is slower than
the one after it only because it is the first run in its process and compiles.)

## A screw swimming along a tube

An FL-9 screw on the axis of a quarter-inch tube of glycerol, turned at 1 Hz by a rotating field: the example's starting
size is `spacing` 0.7 mm, `dt` 100 ms, a run of three turns. The tube's wall kind is the unit tube table (the asset
`unit_tube_R0.997`, fetched once with `mime assets fetch`).

| spacing (mm) | samples | run time, laptop (s) | run time, CI (s) | speed per turn, from the run (um) | speed per turn, from the resistance (um, float64) | against the finest |
|---:|---:|---:|---:|---:|---:|---:|
| 0.83 | 206 | 4.8 | 5.3 | 70.2 | 70.2 | -75% |
| 0.7 | 283 | 4.9 | 6.3 | 124.1 | 124.1 | -56% |
| 0.6 | 416 | 10.2 | 10.3 | 159.1 | 159.1 | -44% |
| 0.5 | 567 | 14.7 | 17.0 | 184.0 | 184.0 | -35% |
| 0.4 | 957 | 33.2 | - | 212.1 | 212.1 | -26% |
| 0.3 | 1714 | 100.5 | - | 243.2 | 243.2 | -15% |
| 0.16 | 5984 | not run | not run | not run | 284.8 | the reference |

**This is the number to take away: at any spacing that runs in seconds, the screw's speed is far from its converged
value, and the converged value is not known.** The speed per turn rises at every row as the spacing falls. Between the
last two rows (0.3 mm, 1714 samples, and 0.16 mm, 5984 samples) it still rises by a sixth. The rise per halving of the
spacing is shrinking (about 47%, 35% and 19% over the last three intervals), so the speed is converging, but the finest
row has not arrived. So the finest row is not the converged value, "against the finest" is a lower bound on each row's
error if the speed keeps rising, and at the example's starting size the speed is **at least** 56% too low. Why the
coarse rows are low is not measured here; the tables show that they are, and by how much.

What the example's number is worth, then: **its sign, once the spacing is fine enough to have the right one, and its order
of magnitude, and the dependence on everything else (the field, the fluid, the tube)**. At a spacing of 1.2 mm (coarser
than the table's rows, run once while the example's test was written) the speed came out *negative*, a screw swimming
the wrong way: a coarse sample set does not only underestimate, it can reverse. It is a first look. A value to put in a paper needs a spacing at which the speed has
stopped changing to the accuracy wanted, which the tables show is finer than these rows for this helix; run the
resistance in float64 at two or three spacings, as the script does, before a run, since a resistance costs seconds
where the run costs minutes.

### `dt` for the screw

| dt (ms) | steps | run time, laptop (s) | run time, CI (s) | speed per turn (um) |
|---:|---:|---:|---:|---:|
| 10 | 300 | 43.8 | 46.3 | 124.1 |
| 20 | 150 | 21.2 | 23.2 | 124.1 |
| 40 | 75 | 11.4 | 12.5 | 124.1 |
| 100 | 30 | 4.9 | 6.1 | 124.1 |
| 250 | 12 | 2.4 | 3.4 | 124.2 |

The screw's speed does not depend on `dt` to the digits printed from 10 ms to 100 ms, and moves by five parts in ten
thousand at 250 ms. In this regime (a viscous fluid, a turning Reynolds number of a few hundredths) the body follows the
field at once, so the step only has to resolve the field's turn: **ten steps to a turn is enough for the speed**, and a
run costs what its steps do. The example's test holds that the speed does not depend on the inertia declared either.

Where inertia matters this is not so. The first-hour walk's screw (the de Jongh surrogate drag, 5 Hz, in water, a
turning Reynolds number in the hundreds) gives a mean speed over the last second of a three-second run of 1.969, 1.961
and 1.957 mm/s at steps of 0.5, 0.25 and 0.125 ms (`scripts/inertia_arms.py`): a first-order dependence on `dt`, about
half a per cent at each halving. **A step is chosen for the regime, and the post-run watch says which regime a run was
in.**

## A checklist

1. Start at the example's sizes for a body like it. Run it, and read the watch: if it says the Reynolds number is past
   the range, no choice of `spacing` or `dt` repairs that; it is the model.
2. Run again at a `spacing` about 0.7 of the first (the work in a step doubles or more), and compare the number you care
   about. If it moves by more than you can accept, the first was not converged; if it moves by less, it may be. Two
   values agreeing do not prove convergence, and three with a shrinking change come close.
3. Check the cheap quantity first. For a body whose answer is a function of its resistance (a synchronous swimmer's
   speed, a sphere's drag) the resistance at a ladder of spacings costs seconds each; run it before the run.
4. Choose `dt` for the regime: no more than the body's relaxation time when it is released from rest, and enough steps
   to a turn of a drive to resolve it. Then halve it once and see that the number does not move.

## What a test costs on CI

The examples run as tests in CI's default lane at their starting sizes: the sphere in about a second, the screw in
about six seconds on the runner, from the tables above. A run at the next finer rows would be ten to thirty seconds
for the sphere and ten to seventeen for the screw's 0.6 and 0.5 mm, the CI column says. The tube's wall table (the
unit tube table, 209 MB) is fetched once for every job, not by the run.
