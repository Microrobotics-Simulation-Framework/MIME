"""Write the frozen folder of saved form 14 that form 15's step is shown on (``tests/saved/test_form15.py``).

Form 15 moves a kinematic arm's servo onto its apparatus's ports: a form-14 document's arm ``servo`` is read as a
first-order lag on each of the arm's position commands' ports, and the rebuilt document and resolved record are written
back with it for the lock's hashes. So the folder holds an arm with a servo, as form 14's own code writes it: the
two-joint arm of ``tests/descriptions/c_arm.urdf`` (a slide and two revolute joints, each bounded), kinematic, its servo
20 ms, commanded past its limits so that the clip before the lag acts.

It runs only under a MIME whose saved form is 14, the code that wrote form 14, never this branch's: a worktree at the
last commit of form 14, its ``src`` first on the path, in float64 with four usable CPUs and one BLAS thread::

    git worktree add <dir> 82219938
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=<dir>/src taskset -c 0-3 \\
        python tests/saved/write_form14.py tests/saved/form14

It writes ``<out>/servo_arm`` (refusing an existing one) and prints the sha256 of each file, the table the test pins."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

#: The arm's description, its start and what it is commanded to: the first revolute joint past its upper limit of 2 rad,
#: the slide past its 0.4 m.
DESCRIPTION = Path(__file__).parents[1] / "descriptions" / "c_arm.urdf"
START = (0.1, 0.0, 0.0)
COMMANDED_TURNS, COMMANDED_SLIDE = (2.5, -2.5), (0.6,)
SERVO_MS = 20.0


def main(out: Path) -> None:
    from mime.saved import _versions

    if _versions.CURRENT != 14:
        raise SystemExit(f"this MIME writes saved form {_versions.CURRENT}: run this under the last commit of form 14")
    target = out / "servo_arm"
    if target.exists():
        raise SystemExit(f"{target} exists: remove it first")

    from mime import Apparatus, Experiment
    from mime.drives import Constant
    from mime.dynamics import Arm
    from mime.scene import Scene, Z, asset
    from mime.units import m, m_per_s2, ms, rad, s

    experiment = Experiment("servo_arm", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1),
                            duration=s(0.05))
    experiment.add_apparatus(Apparatus("carm", chain={"arm": Arm.Kinematic(
        description=asset(DESCRIPTION), initial_positions=rad(START), servo=ms(SERVO_MS))},
        commands={"arm.joint_positions": Constant(rad(COMMANDED_TURNS)),
                  "arm.joint_displacements": Constant(m(COMMANDED_SLIDE))}))
    experiment.build().save(target)
    lock = json.loads((target / "mime.lock").read_text(encoding="utf-8"))
    assert lock["schema"] == "mime.lock/14", lock["schema"]
    for path in sorted(target.rglob("*")):
        if path.is_file():
            print(f'"{path.relative_to(target).as_posix()}": "{hashlib.sha256(path.read_bytes()).hexdigest()}",')


if __name__ == "__main__":
    main(Path(sys.argv[1]))
