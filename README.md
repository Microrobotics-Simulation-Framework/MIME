# MIME

MIcrorobotics Multiphysics Engine, on [MADDENING](https://microrobotica.org/maddening/).

**MIME 0.3 is being rebuilt on the `v0.3` branch.** So far it holds the core that turns a declared experiment into a
MADDENING graph, the saved form, the scene, the profiles, and the physics of its first experiments: a sphere settling
in free space, a magnetic screw swimming in a tube (script A, held to frozen outputs of the 0.2 code with every
difference declared before the comparison ran). More physics arrives phase by phase.

The 0.2 line (the last release on MADDENING 0.3.x, with its nodes, experiments, docs and verification
scripts) is preserved at the tag [`pre-0.3`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/pre-0.3).

## Install

MIME is `mime-engine` on PyPI (Python 3.12 or later). **0.3 is not released yet**: PyPI holds the 0.2 line, whose API
this README does not describe. Until 0.3 is, install it from the `v0.3` branch:

```sh
pip install "mime-engine[verify] @ git+https://github.com/Microrobotics-Simulation-Framework/MIME@v0.3"
```

Once 0.3 is released, this is `pip install "mime-engine[verify]"`. The `verify` extra adds Hypothesis and MADDENING's
verification harnesses, which `mime docs build` and MIME's own tests need; a script that only builds and runs
experiments needs neither. For a GPU on Linux add `cuda12` (`mime-engine[cuda12]`); on macOS and Windows a large job
belongs on the CPU or in the cloud ([`docs/platforms.md`](docs/platforms.md) says what each platform supports and what
is tested). To work on MIME itself, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quickstart

A steel ball 10 micrometres in radius settling in still water, written with no paperwork. It runs in a few seconds on a
laptop's CPU, and nothing in it is a record or a reason:

```python
from mime import Body, Experiment, Medium, State
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity
from mime.observations import SwimSpeed
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m, m_per_s, mm_per_s, mPa_s, ms, rad_per_s, um

exp = Experiment("settling_ball", scene=Scene.new(up=Z, units="m"), dt=ms(0.1))
water = exp.add_medium(Medium("water"))
exp.declare("water.density", kg_per_m3(998.0))
exp.declare("water.viscosity", mPa_s(1.0))
ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
    position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
    angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"steel": Part(Sphere(radius=um(10.0)))}))
exp.declare("ball.parts.steel.density", kg_per_m3(7850.0))
exp.attach(Drag.MRS(), body=ball, medium=water)
exp.attach(Gravity(), body=ball, medium=water)
settling = exp.observe(SwimSpeed(ball, along=-Z, after=ms(2.0)))

built = exp.build()
result = built.run(ms(3.0), history=True)
print(result[settling].mean_speed.to(mm_per_s))  # the speed over the last millisecond, in mm/s
print(result.text())                             # every output, its flags, the post-run watches, the profile
built.save("settling_ball")
```

The build took standard gravity by convention, the drag's wall kind from the medium (it fills no region, so free
space), and the MRS drag's `spacing` and `blob` from their numerical defaults, each a stated rule and
never a converged value. Stokes's law gives 1.493 mm/s for this ball at standard gravity (9.80665 m/s^2, the convention
the build took); the run gives a speed within two per cent of it, and its result says at what Reynolds number it
settled and that it is inside the range the model is declared for (the watch is in
[`docs/results.md`](docs/results.md)). That the default `spacing` is a first look is the subject of
[`docs/choosing_spacing.md`](docs/choosing_spacing.md), with the measured cost and accuracy of finer ones.

To see what sealing it takes: the exploratory build left three reasons and two numerical defaults **pending**, and
says so on every output. Read them from the shell (the `mime` command is on the path where `pip` installed it; with
`uv`, it is `uv run mime`):

```sh
mime pending settling_ball
```

prints each item in plain words with the Python line to paste, and exits with one while anything is pending
([`docs/profiles.md`](docs/profiles.md)).

## Where to go next

- [`examples/sphere_settling.py`](examples/sphere_settling.py) is this ball with its reasons and a comparison with
  Stokes's law; `--radius-um 50` makes the ball one whose Reynolds number is past the model's range, to see the flag
  raised.
- [`examples/screw_in_tube.py`](examples/screw_in_tube.py) is a magnetic screw swimming along a tube of glycerol, every
  value declared; [`docs/choosing_spacing.md`](docs/choosing_spacing.md) says what its number is worth.
- [`docs/results.md`](docs/results.md) reads a run: outputs by name with units, windows, flags, the exports.
- [`docs/errors.md`](docs/errors.md) reads an error, and [`docs/profiles.md`](docs/profiles.md) the path from exploring
  to sealing.
- [`examples/script_a.py`](examples/script_a.py) is script A, a screw in a tube of water, with records.

## Where things are

Every folder has a README saying what it is for.

- [`src/mime/`](src/mime/): the package. Its [README](src/mime/README.md) is the manifest: what MIME 0.3 is, how a
  declared experiment becomes a MADDENING graph, the dependency direction, each subpackage's role, and the status of
  the rewrite.
- [`tests/`](tests/): MIME's own tests, and the planned test layout.
- [`examples/`](examples/): scripts as a user writes them, each held by a test; script A's reproduction arm first.
- [`lab_fixture/`](lab_fixture/): a lab's package in miniature, which registers an entry from outside MIME.
- [`docs/`](docs/): the bibliography and the known-anomalies registry. Other docs return with the code they describe.
- [`scripts/`](scripts/): the compliance checks CI runs.

MIME is research software. It is not a medical device and is not intended for clinical use.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Releases of the 0.2 line (the tag `pre-0.3` and earlier)
keep the licence they were distributed under, which [NOTICE](NOTICE) names.
