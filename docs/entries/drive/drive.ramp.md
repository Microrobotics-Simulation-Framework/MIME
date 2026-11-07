---
describes: "0.1.0"
---
# A command ramped from one value to another

**What it models.** A command held at `start` for `hold`, then moved at `rate` until it reaches `to`, then held there:
`drives.Ramp(Hz(100.0), to=Hz(5.0), rate=Hz(0.25) / s(1.0), hold=s(3.0))` as a field's frequency holds it for the
hold, then ramps it down at the rate. The rate is a magnitude in the command's units per second, and the ramp moves
from `start` towards `to`, up or down. It writes one number per command.

Its node, `RampNode`, writes each step the ramp's value at the step's middle, its mean over the step wherever the ramp
neither starts nor ends inside it, so a field that integrates its frequency (`uniform_rotating` 0.2.0) turns by the
ramp's exact integral, `2 pi (f0 t + k t^2 / 2)` for a ramp at `k` from `f0`. Its numbers are parameters of the graph,
so a study can sweep a ramp's start, rate or end without rebuilding the experiment; a study's protocol `Ramp` declares
one on each point, starting at the point's value of the command.

**When not to use it.**
- On the frequency of `uniform_rotating` 0.1.0, kept for saved folders: its phase is the present frequency times the
  time, so a ramp would turn its field at the wrong rate, and the build refuses it.
- For a command that must follow what the run computes: a controller does that.
- For a step, a pulse or a schedule of several ramps: this is one ramp between two holds.
- Where the value at the step's end is wanted rather than its mean over the step: the two differ by the rate times half
  a step, for a reader that takes the value at the step's end (an amplitude, say).

**Its derivative** is piecewise: the ramp's start and its end are switches, kinks in the value (it is continuous, and
its slope jumps), declared in its gradient record with the route to a smooth derivative (the ramp's exact mean over
the step in which each falls).

**Versions.** `drive.ramp` 0.1.0 is its first.
