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

A steel ball settling in water, written with no paperwork at all:

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
exp.attach(Drag.BEM(), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
exp.observe(SwimSpeed(ball, along=-Z, after=s(0.004)))

built = exp.build()
print(built.report.pending())
```

It builds. Gravity is standard gravity by convention, since the experiment states none, and the drag's wall kind is
the one that matches a medium filling no region; the build report says both. Four things are pending, and
`built.report.pending()` names each in plain words, with the line to paste:

- the ball's density was declared with no reason;
- water's viscosity record states no shear rate, which exploratory waives as an assumption;
- the boundary-element drag's `spacing` and `blob` took their numerical defaults, each a stated rule with its basis,
  never a converged value.

A run's printout, its `outputs.json`, its `.mat` file and its saved form all carry the same stamp: the profile and
the items pending.

## See what is pending

`built.report.pending()` prints, for each item, what it is about, why it is pending, a line of Python to paste and a
fragment of the saved document (`experiment.mime.yaml`) to merge in. Each line holds `pending()` (in the document,
`{pending: true}`) where its reason goes, so it builds as it is in exploratory.

For a saved experiment, `mime pending <folder>` prints the same thing. It exits with zero when nothing is pending,
with one when something is (a sealing check for a script or CI), and with two when the folder does not load.

## Seal

Sealing is a reviewable change to the script or the document: paste each line, put the reason in place of `pending()`,
and build with `profile="research"`. Research refuses a pending reason by name, so a placeholder cannot pass for a
reason, and lists what it would still ask about, each with its line.

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

A reason is a text, `chosen(...)` or `pending()`; a source given twice, on the quantity and by `exp.source`, is refused,
and so is a path that names no setting.

## Roles and defaults

Every setting of every entry declares its role: **design** (the experimenter's choice), **physical** (a fact about the
setup), **numerical** (a solver's choice) or **standard** (a constant MIME builds in). A setting that declares no
default is required. A default is a claim, of one of these kinds:

| Kind | What it is | Example | In every profile? |
|---|---|---|---|
| determined | the one value other inputs allow | the drag's `wall_kind`, from the region the medium fills | yes, and reported |
| convention | what anyone assumes untold | standard gravity; a window from the run's start | yes, and reported |
| numerical | a stated rule with its own basis | the drag's `spacing`, from the body's size | exploratory only, stamped pending |

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
