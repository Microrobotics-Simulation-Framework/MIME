"""One program builds to the same held matrix every time in a process, in float32 (MIME-ANO-138).

The MRS node makes the body's resistance (or, with a wall, its free part's matrix) once per build, in float64 whatever
the run's precision, and holds it. The screw's cap samples come from a layout (``mime.scene._caps.cap_rings``) cached by
the process; JAX could hand back, to a conversion made with x64 on after a step had been traced with x64 off, the
float32 conversion of the same array. So the first build of the screw in a process held the matrix of the cap samples in
float64 and every later one the matrix of samples rounded to float32 (1e-7 of themselves), which a float32 solve
carries to 2e-5 of the resistance's largest entry: the same program, two answers, the first build's apart from the rest.
In float64 the two were the same.

Each test builds the screw in the tube, at a coarse spacing so that a build takes seconds, three times in a process of
its own (float32 is the default, and the first build of a process is the one in question; this process has built
before), and reads the held matrix of each. The control is the layout handed out as the one cached array, as it was:
the first build then differs from the second and third, on every run (nothing in it is drawn)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import tube_asset

ROOT = Path(__file__).resolve().parents[2]
SPACING_MM = 2.0

CHILD = """
import hashlib, importlib.util, json, sys
import numpy as np
from mime.scene import _caps, _screw
if sys.argv[1] == "as_cached":
    _screw.cap_rings = _caps._cap_rings
spec = importlib.util.spec_from_file_location("screw_in_tube", sys.argv[2])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
held = [hashlib.sha256(np.ascontiguousarray(module.screw_in_tube(float(sys.argv[3])).build().graph.get_node("drag:umr")
                                           .held_in_body_frame).tobytes()).hexdigest() for _ in range(3)]
print(json.dumps(held))
"""


def held_matrices(layout: str) -> list[str]:
    """The sha256 of the held matrix of three builds of the screw in the tube, made in a fresh float32 process, with the
    layout handed out as MIME does (``"as_handed_out"``) or as the one cached array (``"as_cached"``)."""
    tube_asset.path()  # stored by CI; a missing table fails here by name, with the command that stores it
    done = subprocess.run([sys.executable, "-c", CHILD, layout, str(ROOT / "examples" / "screw_in_tube.py"),
                           str(SPACING_MM)], capture_output=True, text=True, check=False,
                          env={key: value for key, value in os.environ.items() if key != "JAX_ENABLE_X64"})
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_three_builds_of_the_screw_in_the_tube_hold_the_same_matrix_in_float32():
    first, second, third = held_matrices("as_handed_out")
    assert first == second == third


def test_control_the_layout_as_one_cached_array_makes_the_first_build_differ_from_the_rest():
    first, second, third = held_matrices("as_cached")
    assert second == third and first != second
