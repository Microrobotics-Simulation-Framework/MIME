# Tutorial A: a screw in a tube of water

Script A is de Jongh's FL-9 screw resting on the floor of a quarter-inch tube of water. A uniform field turns about the tube's
axis, the screw turns with it, and it swims along the tube. You build it twice: first as the **reproduction arm**, which
declares every value the earlier line's run of A used, each with where it came from; then as the **record arm**, which takes
water's density and viscosity, and the magnets' moment, from records instead.

Both arms are in one file, `examples/script_a.py`, and the tutorial's own run of them is `examples/tutorial_a.py`. The definitions of what the code uses are not repeated here:
[the scene](../scene.md) (frames, regions, the stage), [materials](../materials.md) (how a value reaches an entry, declared
against fitted), and [the profiles](../profiles.md).

## What this tutorial claims

It claims that the experiment builds and runs as written, and that its output is the output shown. It claims nothing about
the swim speed it prints, because the run is a short one: the earlier line's reference reads the second half of a longer run,
and this run is shorter on purpose, to fit a test's time.

What the test suite holds about A is separate, and narrow. `tests/differential/test_script_a.py` holds the reproduction arm
to the frozen outputs of the earlier code with every difference declared before the comparison ran: it bounds *how far
MIME 0.3's A is from the earlier line's A* at this operating point. The earlier code is the weakest oracle there is, so the
agreement says nothing about the flow or about hardware.

## Before you run it

**Assets.** Tutorial A fetches nothing: the drag surrogate's weights and the two material records ship with MIME. A tube
experiment whose drag is the MRS rung is different. It reads the table of the tube's wall, which is not shipped, and the
build is refused by name until it is in the cache: fetch it once with `mime assets fetch unit_tube_R0.997`
([install and environment](install.md#assets-a-run-fetches) says more).

**The time step.** A's base step is the one the earlier line's reference ran with, half a millisecond; it is kept so that the
comparison to that reference is of like with like, and nothing in this guide shows it to be the largest step that serves.
`docs/choosing_spacing.md` measures what the step does in a different experiment (the MRS screw in a tube), and its numbers
are not A's. What MIME states about a step, it states in the build report's `limits` and `validity` lines
([reading it](build_report.md#limits)). For the rigid body there is one: the gyroscopic term is resolved only while the
rotation rate times the step, `|omega| dt`, is below the number the line prints (MIME-ANO-129). A's field turns at ten hertz,
so its `|omega| dt` is well below it; and A declares the same inertia about every axis, for which the gyroscopic term is zero.

Those lines are declared, and nothing checks a step against them: the build does not refuse a step past a limit, and a step
inside the limits that are stated is not thereby shown to run. A run that ends with a state that is not finite is
`run.non-finite`. It names the first step, thing and field only where the run kept every step's state (`history=True`);
with only the series it recorded it names which of them went non-finite, and with neither it says the first step is not
known. This tutorial's runs record one series and keep no history ([reading an error](../errors.md)). If your own step
does that, shorten it, and read the `limits` lines of the report before suspecting anything else.

## The world, the body and the field

The reproduction arm starts from the world and the tube:

<!-- mime-include: examples/script_a.py#world -->

The tube is `Tube(...)` of the scene, its `lumen` is the region water fills, and the two `declare` lines give water's density
and viscosity as numbers with a reason ([materials](../materials.md) says what a declared value is). The body is the screw
and its declared mass, inertia, buoyant volume and magnetic moment, each with its reason:

<!-- mime-include: examples/script_a.py#body -->

The field is an apparatus with a field provider, turning about the tube's axis, and its two commands are the frequency and the
amplitude:

<!-- mime-include: examples/script_a.py#field -->

## The laws, and building

What acts on the body is attached, each law to the things it reads. The drag is the surrogate with lubrication, the torque is
the field's on the body's moment, and gravity and the contact keep the body in the tube:

<!-- mime-include: examples/script_a.py#laws -->

`.build()` turns the declarations into a graph and a **build report** ([reading it](build_report.md)). The tutorial's file
builds the reproduction arm and then the record arm, prints the values the report holds for water, and runs each arm for a
short time. The length of the run is a constant, `show_water` prints the report's values with their sources, and `swim` runs
the arm and reads the second half of its swim speed:

<!-- mime-include: examples/tutorial_a.py#short -->

<!-- mime-include: examples/tutorial_a.py#values -->

<!-- mime-include: examples/tutorial_a.py#run -->

The last region builds the two arms from `examples/script_a.py` and calls those functions on each, which is the output
below; the record arm it builds is described next.

<!-- mime-include: examples/tutorial_a.py#arms -->

## The record arm

The record arm replaces the declared water and magnet values with records, read at one temperature, and says what it accepts
of each record that the record does not state:

<!-- mime-include: examples/script_a.py#records -->

Run it from a source checkout of MIME, in an environment where MIME is installed (`examples/` is not part of the installed
package): `python examples/tutorial_a.py`. It prints:

<!-- mime-output: examples/tutorial_a.py -->
```text
reproduction arm: every value declared, with its reason
  water.density = 1000.0 kg/m^3, declared in the experiment
  water.viscosity = 0.001 Pa s, declared in the experiment
  swim speed over the second half of a 1 s run: 3.17 mm/s
  post-run watch on the drag: above
record arm: water's values from OpenMaterica's record
  water.density = 998.207 kg/m^3, record of material 'water'
  water.viscosity = 0.0010016 Pa s, record of material 'water'
  swim speed over the second half of a 1 s run: 3.18 mm/s
  post-run watch on the drag: above
profile exploratory: nothing pending.
```

Read it from the top:

- **Where each value came from.** In the reproduction arm water's values are `declared in the experiment`; in the record arm
  they are read from the record of the material `water`. The build report lists the same for every value an entry reads.
- **The swim speed** is read from the observation's series, over the second half of the run. It is a short run's, so it is
  not the reference's window. The record arm changes three inputs at once (water's density, its viscosity, and the
  magnets' moment), so whatever the difference between the two printed speeds is, this run does not say which input moves it;
  `tests/script_a.py` switches the records on one at a time.
- **`post-run watch on the drag: above`** is the watch of [the results note](../results.md): after a run, MIME compares
  the run's Reynolds number with the bound the drag entry declares itself valid up to, and here it finds it above. That is a
  flag on the result and never a refusal. The run is outside the range the entry declares itself for: the reproduction holds
  MIME 0.3 to the earlier line at the reference's operating point, and does not move that point into the range.
- **The warning before the run** (on standard error, so not in the block above) is the same finding, said before
  anything runs: from the field's declared frequency, the hull's diameter and water's density and viscosity, the build
  computes the turning Reynolds number at the drive's rate and finds it far above the bound the surrogate rung declares.
  It is a warning in every profile, never a refusal. The rung's range is declared, not fitted: outside it the result is
  unvalidated, which is not the same as measured to be wrong.
- **`profile exploratory: nothing pending`** is the stamp every output carries ([profiles](../profiles.md)). The record arm
  states its acceptances, so nothing is left pending. `python examples/script_a.py --explore` builds the same records
  without them and prints what sealing it takes.

## Where next

The tutorial for experiment G waits for a1. For an arm and a controller, A0 is next: [tutorial A0](tutorial_a0.md).
