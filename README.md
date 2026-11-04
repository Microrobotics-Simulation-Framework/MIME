# MIME

MIcrorobotics Multiphysics Engine, on [MADDENING](https://microrobotica.org/maddening/).

**MIME 0.3.0.dev1 is the first release of MIME 0.3**, a rewrite on MADDENING 0.4, and a development pre-release: its
API, its saved form and its results may change before 0.3.0. It holds the core that turns a declared experiment into a
MADDENING graph, the saved form, the scene, the profiles, and the physics of its first experiments: a sphere settling
in free space, a magnetic screw swimming in a tube (script A, held to frozen outputs of the 0.2 code with every
difference declared before the comparison ran), and the same screw driven by a magnet on a robot arm. More physics
arrives with each release; [the changelog](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/CHANGELOG.md) says what this one holds, and
[scope and limits](https://microrobotica.org/mime/0.3.0/pages/guide/scope) what it claims. The documentation is at
[microrobotica.org/mime/0.3.0](https://microrobotica.org/mime/0.3.0/).

The 0.2 line (the last release on MADDENING 0.3.x, with its nodes, experiments, docs and verification
scripts) is preserved at the tag [`pre-0.3`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/pre-0.3), and is 0.2.0 on PyPI.

## Install

MIME is `mime-engine` on PyPI (Python 3.12 or later). 0.3 is a pre-release, so pip installs it only when asked to:

```sh
pip install --pre mime-engine
```

A plain `pip install mime-engine` installs 0.2.0, whose API this README does not describe. `mime --version` says which
you have. The `verify` extra (`pip install --pre "mime-engine[verify]"`) adds Hypothesis and MADDENING's verification
harnesses, which `mime docs build` and MIME's own tests need; a script that only builds and runs experiments needs
neither. For a GPU on Linux add `cuda12` (`mime-engine[cuda12]`); on macOS and Windows a large job belongs on the CPU or
in the cloud ([`docs/platforms.md`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/platforms.md) says what each platform supports and what is tested).
A tube experiment on the MRS drag rung reads a table of 209 MB that a build never downloads; fetch it once with
`mime assets fetch unit_tube_R0.997`. To work on MIME itself, see [`CONTRIBUTING.md`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/CONTRIBUTING.md).

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
space), and the MRS drag's `spacing`, `blob` and `quadrature_refinement` from their numerical defaults, each a stated
rule and never a converged value. Stokes's law gives 1.493 mm/s for this ball at standard gravity (9.80665 m/s^2, the convention
the build took); the run gives a speed within two per cent of it, and its result says at what Reynolds number it
settled and that it is inside the range the model is declared for (the watch is in
[reading a run's results](https://microrobotica.org/mime/0.3.0/pages/results)). That the default `spacing` is a first look is the subject of
[`docs/choosing_spacing.md`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/choosing_spacing.md), with the measured cost and accuracy of finer ones.

To see what sealing it takes: the exploratory build left three reasons and three numerical defaults **pending**, and
says so on every output. Read them from the shell (the `mime` command is on the path where `pip` installed it; with
`uv`, it is `uv run mime`):

```sh
mime pending settling_ball
```

prints each item in plain words with the Python line to paste, and exits with one while anything is pending
([the profiles](https://microrobotica.org/mime/0.3.0/pages/profiles)).

## Where to go next

The examples and the tutorials are scripts in the repository, not in the package: run them from a source checkout of
this release, `git clone --branch v0.3.0.dev1 https://github.com/Microrobotics-Simulation-Framework/MIME.git`.

- [`examples/sphere_settling.py`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/examples/sphere_settling.py) is this ball with its reasons and a comparison
  with Stokes's law; `--radius-um 50` makes the ball one whose Reynolds number is past the model's range, to see the
  flag raised.
- [`examples/screw_in_tube.py`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/examples/screw_in_tube.py) is a magnetic screw swimming along a tube of
  glycerol, every value declared (it needs the tube's table, above);
  [`docs/choosing_spacing.md`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/docs/choosing_spacing.md) says what its number is worth.
- [The user guide](https://microrobotica.org/mime/0.3.0/pages/guide/index): install and environment, tutorials A and A0, reading a build report,
  the command line, and scope and limits.
- [Reading a run's results](https://microrobotica.org/mime/0.3.0/pages/results): outputs by name with units, windows, flags, the exports.
- [Reading an error](https://microrobotica.org/mime/0.3.0/pages/errors), and [the profiles](https://microrobotica.org/mime/0.3.0/pages/profiles), the path from exploring to
  sealing.
- [`examples/script_a.py`](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/examples/script_a.py) is script A, a screw in a tube of water, with records.

## Where things are

In the repository, every folder has a README saying what it is for.

- [`src/mime/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/src/mime): the package. Its [README](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/src/mime/README.md) is the manifest: what MIME 0.3
  is, how a declared experiment becomes a MADDENING graph, the dependency direction, each subpackage's role, and the
  status of the rewrite.
- [`tests/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/tests): MIME's own tests, and the planned test layout.
- [`examples/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/examples): scripts as a user writes them, each held by a test; script A's reproduction arm first.
- [`lab_fixture/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/lab_fixture): a lab's package in miniature, which registers an entry from outside MIME.
- [`docs/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/docs): the sources of the documentation, the bibliography and the known-anomalies registry.
- [`scripts/`](https://github.com/Microrobotics-Simulation-Framework/MIME/tree/v0.3.0.dev1/scripts): the compliance checks CI runs.

MIME is research software. It is not a medical device and is not intended for clinical use.

## License

Apache-2.0. See [LICENSE](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/LICENSE) and [NOTICE](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/NOTICE). Releases of the 0.2 line (the tag `pre-0.3`
and earlier) keep the licence they were distributed under, which [NOTICE](https://github.com/Microrobotics-Simulation-Framework/MIME/blob/v0.3.0.dev1/NOTICE) names.
