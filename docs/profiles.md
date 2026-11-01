# Profiles: explore first, seal later

Every experiment is built under a **profile**, which says how much the build asks of it. You pass it to the experiment,
`Experiment(..., profile="research")`; without it, the profile is **exploratory**.

- **exploratory**, the default, is for getting something running. What your experiment leaves unstated (a reason for a
  number, a condition a record does not state, a solver setting you did not choose) is waived with a *pending* reason.
  Each waiver is recorded in the saved form and stamped on every output, so nothing is silent.
- **research** asks for each of those reasons, and refuses a pending one by name. It also lists, and never refuses,
  what a reviewer would ask about: a physical value with no source, a numerical value with no convergence evidence.
- **strict** holds a higher evidence standard, and confers nothing by itself. It is not built yet: MIME refuses it by
  name, since a strict build now could apply only research's rules and would claim a standard it does not apply.

No profile relaxes a validity limit or an input error: a value in the wrong unit, a read a record does not cover, a
body outside a model's domain are refused in every profile, each refusal saying what to change.

## Explore

A steel ball settling in water, written with no paperwork but one line a record's read needs:

```python
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.materials import Record, packaged_record
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import degC, kg_per_m3, m_per_s, mm, ms, rad_per_s, s

exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=ms(1.0),
                 conditions={"temperature": degC(20.0, plus_minus=0.5)})
water = exp.add_medium(Medium("water", material=Record(
    "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)})))
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Sphere(radius=mm(0.5)))}))
exp.declare("ball.parts.shell.density", kg_per_m3(7850.0))
exp.attach(Drag.MRS(), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
exp.observe(SwimSpeed(ball, along=-Z, after=s(0.004)))

built = exp.build()
print(built.report.pending())
```

It builds. Gravity is standard gravity by convention, since the experiment states none, and the drag's wall kind is
the one that matches a medium filling no region; the build report says both. Four things are pending, and
`built.report.pending()` names each in plain words, with the line to paste and the imports those lines need:

- the ball's density was declared with no reason;
- water's viscosity record states no shear rate, which exploratory waives as an assumption;
- the MRS drag's `spacing` and `blob` took their numerical defaults, each a stated rule with its basis,
  never a converged value. What was measured of the default is printed beside it, in numbers: how far a sphere's drag
  at the default spacing is from Stokes's law, and how far a screw's speed per turn is from the finest spacing
  measured, with how long a task of a first experiment's size takes at it.

The one line of paperwork is the record's `require=`. "No paperwork" holds for what the experiment leaves unstated,
not for what a read asks of its consumer: water's record holds viscosities measured by more than one method, and only
the experiment knows which it may use, so it states it (`require={"test_method": ("reference_formulation",)}`).
Without it the build is refused in every profile, `record.unknown` on `condition_unchecked:test_method`, with the line
to write: a consumer's omission is never waived, since a waiver would accept a value the experiment never checked.

A run's printout, its `outputs.json`, its `.mat` file and its saved form all carry the same stamp: the profile and
the items pending.

## See what is pending

`built.report.pending()` prints, for each item, what it is about, why it is pending, a line of Python to paste and a
fragment of the saved document (`experiment.mime.yaml`) to merge in. Each line holds `pending()` (in the document,
`{pending: true}`) where its reason goes, so it builds as it is in exploratory.

For a saved experiment, `mime pending <folder>` prints the same thing. It exits with zero when nothing is pending,
with one when something is (a sealing check for a script or CI), and with two when the folder does not load.
`mime pending <folder>/experiment.mime.yaml` does the same for a document as it is, with no lock to check.

## Seal

Sealing is a reviewable change to the script or the document: paste each line (and the imports printed above them),
put the reason in place of `pending()`, and build with `profile="research"`. Research refuses a pending reason by name
and lists what it would still ask about, each with its line.

- **In the script**: paste the Python lines, and build as before.
- **In the saved document**: merge the fragments into `experiment.mime.yaml`, each `{pending: true}` replaced by the
  reason, and `profile: research`. A saved folder's lock is the one the build that wrote it wrote, so
  `Experiment.load(folder)` refuses an edited document (`load.input-hash`, "the document was changed after its lock
  was written"). Build the edited document with `Experiment.from_document("<folder>/experiment.mime.yaml")`, check
  it with `mime pending <folder>/experiment.mime.yaml`, and save the built experiment again for a new lock.

What research refuses is the one placeholder MIME writes, `pending()`: it cannot tell a reason from a text that only
looks like one, so `"TODO"` passes it. Judging what a reason says is a1's: named bases (`basis("name")`, a library
pinned by its hash) for the reasons themselves, and strict's evidence records for what a numerical value rests on.

## Where a value's source goes

- On the quantity itself: `mm(3.175, reason="the tube's datasheet")`. Any quantity takes one, and the saved form keeps
  it beside the value; equality and arithmetic ignore it.
- Apart from the value, by its path: `exp.source("tube.inner_radius", reason="the tube's datasheet")`, for a value
  that is not a quantity, or a source added when sealing. A path names a setting: `<structure>.<setting>`,
  `<body>.parts.<part>.<setting>`, `<apparatus>.<field>.<setting>`, `<apparatus>.<command>` for a command's value,
  `<entry id>.<setting>` (`drag:ball.spacing`), `controller.<setting>`, `dt`, `gravity`, `conditions.<name>`.
- A value you chose yourself, whatever its setting's role: `reason=chosen("the radius this study sweeps")`. The report
  records it as a declared change of the setting's role.
- A value declared in the experiment takes its reason in `exp.declare(..., reason=...)`; with none, it is pending.
- A part's placement, `Part(..., at=mm(..., reason=...))` (`<thing>.parts.<part>.at`); the controller's rate and
  latency, `exp.set_controller(..., rate=ms(2, reason=...))` (`controller.rate`, `controller.latency`); an operating
  condition, `degC(20.0, plus_minus=0.5, reason=...)` (`conditions.<name>`).
- A record's acceptance and nominal value take theirs in `Acceptance(..., reason=...)` and `Nominal(..., reason=...)`,
  `pending()` included.

A reason is a text, `chosen(...)` or `pending()`; a source given twice, on the quantity and by `exp.source`, is refused,
and so is a path that names no setting. A reason written anywhere else (on a condition's tolerance, on a nominal's
value) is judged by nothing and saved nowhere, so it is refused as an input error naming where it was written.
`gravity` takes a source by `exp.source("gravity", reason=...)` whether it is written or left to its convention.

## Roles and defaults

Every setting of every entry declares its role: **design** (the experimenter's choice), **physical** (a fact about the
setup), **numerical** (a solver's choice) or **standard** (a constant MIME builds in). A setting that declares no
default is required. A default is a claim, of one of these kinds:

| Kind | What it is | Example | In every profile? |
|---|---|---|---|
| determined | the one value other inputs allow | the drag's `wall_kind`, from the region the medium fills | yes, and reported |
| convention | what anyone assumes untold | standard gravity; a window from the run's start | yes, and reported; research lists a physical one with no source |
| numerical | a stated rule with its own basis | the drag's `spacing`, from the body's sample count | exploratory only, stamped pending |
| undeclared | a written default whose setting declares no role (a lab's entry) | `smoothing: float = 3.0` with no `setting_roles` line | exploratory only, stamped pending |

A convention on a physical setting (a tube centred on the world's origin, a screw's two starts, a segment at its
frame's origin) is what anyone assumes untold, and may not hold for the setup: research lists each one taken, with
the `exp.source(...)` line that states it.

Each entry page lists its settings with their roles, and `docs/ratchets/required_settings.txt` lists every required
setting with its reason.

## The refusals on this path

### `profile.unknown`

The profile is none of MIME's. Pass `profile="exploratory"` or `profile="research"`.

### `profile.reserved`

`strict` is not built yet. Build with `profile="research"` until it is.

### `value.reason-pending`

Research found a value declared with no reason, or with `reason=pending()`. Give it its reason:
`exp.declare("ball.parts.shell.density", kg_per_m3(7850.0), reason="...")`.

### `material.accept-pending`

Research found an acceptance whose reason is pending. Give it its reason: `Acceptance(covers=(...), reason="...")`.

### `source.pending`

Research found a setting whose source is pending, on its quantity or by `exp.source`. Give it its reason.

### `setting.numerical-default`

Research found a numerical setting left at its default. Write its value, with its convergence evidence as its reason:
the refusal prints the line, such as `spacing=m(..., reason="...")`.

### `source.unknown`

`exp.source` names a path that is no setting of the experiment. The refusal lists the nearest paths.

### `source.duplicate`

A setting, or a declared value, is given its reason twice. Give it one: on the quantity, or by `exp.source` (by
`reason=` for a declared value).

### `source.reason`

A source's reason is not one: an empty text, or something that is not a text, `chosen(...)` or `pending()`.

### `profile.not-a-waiver`

A profile rule generated something other than a waiver. Profiles act on waivers alone, so this is a defect of MIME's
tables: report it.

### `profile.tables`

The profile tables hold a row no site can apply: a rule that generates or accepts something a waiver does not cover,
one that lists what only a declaration or a fix covers, or one for a class with no code. This is a defect of the
tables a build reads (MIME's, or those `mime.validity.using_tables` holds): report it.

### `material.nominal-pending`

Research found a nominal value whose reason is pending. Give it its reason: `Nominal(T(1.345), reason="...")`.

### `source.no-site`

A quantity carries a reason where nothing judges or saves one: on a condition's tolerance, on a nominal's value, on a
quantity inside a setting's list. Take it off, and give the reason where it belongs: on the setting's own quantity or
by `exp.source(path, reason=...)`, in `degC(..., plus_minus=..., reason=...)` for a condition, in
`Nominal(..., reason=...)` for a nominal.

### `setting.undeclared-default`

Research found a setting left at a written default that its entry declares no role or default kind for (a lab's
entry), so nothing says what the default claims. State the value with its source,
`exp.source("<entry id>.<setting>", reason="...")`, or declare the setting's role and kind in the entry's
`setting_roles`.

### `setting.source-required`

A profile whose rule refuses a physical value with no source found one (no dev1 profile does: research lists it).
Give it its source.

### `setting.evidence-required`

A profile whose rule refuses a numerical value with no evidence found one (no dev1 profile does: research lists it).
Give it its reason, with its evidence.
