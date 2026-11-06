"""One program builds to the same held matrix every time in a process, in float32 (MIME-ANO-138).

The MRS node makes the body's resistance (or, with a wall, its free part's matrix) once per build, in float64 whatever
the run's precision, and holds it. The screw's and the cylinder's cap samples come from a layout
(``mime.scene._caps.cap_rings``) cached by the process. Measured at MIME's level: ``jnp.asarray`` with x64 on, applied
to the very arrays a step had been traced with under x64 off, returned float32 for them, and float64 for a copy of
them; the cached arrays themselves were unchanged (bytes, dtype, read-only flag). So the first build of the screw in a
process held the matrix of the cap samples in float64 and every later one the matrix of samples rounded to float32
(1e-7 of themselves), which a float32 solve carries to 1e-5 of the resistance: the same program, two answers, the first
build's apart from the rest. In float64 the two were the same. It was not reduced to a case without MIME, so it is not
attributed to JAX in general.

The screw, a cylinder (whose caps take the same layout) and a ball (the tube demo's body, which has no caps) are each
built, at a coarse spacing so that a build takes seconds, three times in a fresh process (float32 is the default, and the first build of a process is the one in
question; this process has built before), and the held matrix of each is read. The path a user meets has a process of
its own: the screw's first build is run, then saved, and that run is compared with the reference the save wrote. The
controls hand the layout out as the one cached array, as it was: the first build then differs from the second and
third, and its run is reported moved against the saved reference, on every run (nothing in them is drawn).

The builds are compared with each other; that the held matrix is the float64 one is not asserted here. The layout's
samples are pinned against an extended-precision reference by ``tests/scene/test_screw_caps.py``, which kills a layout
rounded to float32 for everyone."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tube_asset

ROOT = Path(__file__).resolve().parents[2]
SPACING_MM = 2.0
PROGRAMS = ("screw", "cylinder")
#: The programs whose three builds are compared: the control needs the cap layout, which a ball has none of.
ALIKE = (*PROGRAMS, "ball")
#: What the saved reference of the screw's first-build run moves on when the first build differs from the rest.
MOVED = "drag:umr.resistance"

PRELUDE = """
import hashlib, json, sys, types
from pathlib import Path
import numpy as np
from mime.scene import Cylinder, Sphere, _caps, _screw, _solids
layout, program, example, spacing, folder = sys.argv[1:6]
if layout == "as_cached":
    _screw.cap_rings = _solids.cap_rings = _caps._cap_rings
module = types.ModuleType("example")
sys.modules["example"] = module
exec(compile(Path(example).read_text(), example, "exec"), module.__dict__)
if program == "cylinder":
    module.Screw = lambda radius, length, **_: Cylinder(radius=radius, length=length, axis=(1.0, 0.0, 0.0))
if program == "ball":
    module.Screw = lambda radius, length, **_: Sphere(radius=radius)


def build():
    return module.screw_in_tube(float(spacing)).build()


def held(built):
    return hashlib.sha256(np.ascontiguousarray(built.graph.get_node("drag:umr").held_in_body_frame).tobytes()).hexdigest()
"""

#: Three builds, the held matrix of each.
THREE_BUILDS = PRELUDE + """
print(json.dumps({"held": [held(build()) for _ in range(3)]}))
"""

#: The first build of the process, run, saved, and that run compared with the reference the save wrote; then ``verify``.
FIRST_BUILD_RUN = PRELUDE + """
from mime.saved import verify_history
from mime.testing._saved import first_steps
first = build()
run = first_steps(first, 8)
first.save(folder, steps=8)
print(json.dumps({"moved": list(verify_history(first._saved, run).moved),
                  "verify_moved": list(first.verify().moved)}))
"""


def fresh_float32_process(script: str, layout: str, program: str, folder: Path | None = None) -> dict:
    """What ``script`` prints, run in a fresh float32 process on ``program`` (``"screw"`` or ``"cylinder"``) with the
    layout handed out as MIME does (``"as_handed_out"``) or as the one cached array (``"as_cached"``)."""
    tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    done = subprocess.run([sys.executable, "-c", script, layout, program, str(ROOT / "examples" / "screw_in_tube.py"),
                           str(SPACING_MM), "-" if folder is None else str(folder)],
                          capture_output=True, text=True, check=False,
                          env={key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def held_matrices(layout: str, program: str) -> list[str]:
    return fresh_float32_process(THREE_BUILDS, layout, program)["held"]


def first_build_run(layout: str, folder: Path) -> dict:
    return fresh_float32_process(FIRST_BUILD_RUN, layout, "screw", folder / layout)


@pytest.mark.parametrize("program", ALIKE)
def test_three_builds_hold_the_same_matrix_in_float32(program):
    first, second, third = held_matrices("as_handed_out", program)
    assert first == second == third


@pytest.mark.parametrize("program", PROGRAMS)
def test_control_the_layout_as_one_cached_array_makes_the_first_build_differ_from_the_rest(program):
    first, second, third = held_matrices("as_cached", program)
    assert second == third and first != second


def test_the_first_build_of_a_process_run_agrees_with_the_reference_it_saves(tmp_path):
    """The path a user meets: in a script the process's first build is run and saved, and that run is compared with
    the saved reference (a later build's); ``verify`` reruns the reference."""
    done = first_build_run("as_handed_out", tmp_path)
    assert done["moved"] == [] and done["verify_moved"] == []


def test_control_the_first_build_run_with_the_layout_as_one_cached_array_moves_the_resistance_from_step_0(tmp_path):
    done = first_build_run("as_cached", tmp_path)
    moved = [line for line in done["moved"] if line.startswith(MOVED)]
    assert len(moved) == 1 and "first at step 0" in moved[0], done["moved"]
    assert done["verify_moved"] == [], done["verify_moved"]  # verify compares two later builds, either way
