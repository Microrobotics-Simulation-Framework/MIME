"""Saved form 8's table: case by case, the reference a save takes from the run's own graph, held to what ``verify``
reruns and to what a save wrote before form 8 (``reference_programs.py`` makes each row).

Each row builds the case, runs it chunked by one step keeping its steps' state (its control periods where a controller
sets the chunks), and saves: the save takes the one-step program the run compiled. Asserted in every row, the contract:
the save's reference equals a fresh graph run by the same program (what ``verify`` reruns), equals the run's own first
steps (the reference is those steps, run again from the initial state), and the history scan on the run's graph equals
a fresh graph's (what a run not chunked by one step saves). Asserted where it was measured equal: the save's reference
equals what a save wrote before form 8, a fresh graph's history scan. It differs elsewhere in the last bits of a scan of
one step against a longer one (PR #104's MRS sphere in float64; on the laptop, 15 of 28 cases in float32 and 12 of 28 in
float64, at most 4.2e-7 and 4.4e-16), and is recorded there (``record_property``), not asserted.

**Where each row runs.** In float32 each row is a fresh process, so the build is the process's first, the condition in
which MIME-ANO-138 made the first build differ from the later ones (float32 only): the cases of ``tests/saved/cases.py``
that build, and PR #104's MRS sphere (12 steps). In float64 the rows run in this process: the cases whose reference
differs from the save before in float64, #104's case, and four that are equal, so that the equality is asserted too.

**Measured on a laptop's CPU and not run here,** for the shard's time (a row of each took 15 to 85 s there, and the
table's first form, every row, took a CI shard past its 20 minutes): the tube demo, both precisions (its first 8 steps
never reach the contact; 27 of 27 arrays equal to the save before); and bound_robot_with_parts, dynamic_rpm,
massive_rpm, mrs_free_space and mrs_solids, both precisions, whose entries the rows here build in lighter cases
(bound_robot; kinematic_rpm, tracked_rpm and swing_magnet_kept; mrs_kept, mrs_point_forces, mrs_tube and lab_bead).
Every row of those, and the float64 rows not run here, held the contract (the verify and history columns empty; the
own-steps column was added after, and is asserted here). The hull contact runs here in float32 alone, on each version
(0.4.0, its loops unrolled; 0.5.0, looped), as the cases whose contact acts from the first step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from reference_programs import table_row

HERE = Path(__file__).parent
TESTS = HERE.parent
STEPS = 8

#: The cases each in a fresh float32 process: those of ``tests/saved/cases.py`` that build, less the five measured on the
#: laptop alone (the module's docstring), and PR #104's MRS sphere.
FLOAT32 = ("script_a_body", "script_a", "script_a_kept", "script_a_records", "hull_contact", "hull_contact_looped",
           "coupled_toy", "full_toy", "walled_toy", "anchored_toy", "nudged_toy", "bound_robot", "probed_shapes",
           "kinematic_rpm", "gantry_magnet", "tracked_rpm", "swing_magnet_kept", "mrs_point_forces", "mrs_kept",
           "mrs_tube", "rigid_explicit", "lab_bead", "sphere_settling")
#: The float64 rows run in this process: the cases that differ from the save before in float64 (script A, the nudged
#: toy, the MRS rows, #104's sphere) and four that do not (the coupled and full toys, the probed shapes, A0's arm).
FLOAT64 = ("script_a", "script_a_records", "nudged_toy", "mrs_point_forces", "mrs_kept", "lab_bead", "sphere_settling",
           "coupled_toy", "full_toy", "probed_shapes", "kinematic_rpm")
#: Where the save's reference equals what a save wrote before form 8, measured on the laptop (24 CPUs, a fresh process
#: per case), in each precision.
EQUAL_BEFORE = {
    False: {"script_a_body", "full_toy", "anchored_toy", "bound_robot", "bound_robot_with_parts", "gantry_magnet",
            "tracked_rpm", "dynamic_rpm", "swing_magnet_kept", "massive_rpm", "rigid_explicit", "sphere_settling"},
    True: {"script_a_body", "coupled_toy", "full_toy", "walled_toy", "anchored_toy", "bound_robot", "probed_shapes",
           "kinematic_rpm", "gantry_magnet", "swing_magnet_kept", "rigid_explicit"},
}


def steps_of(case: str) -> int:
    return 12 if case == "sphere_settling" else STEPS


def in_a_fresh_float32_process(case: str) -> dict[str, Any]:
    environment = {**os.environ, "JAX_ENABLE_X64": "0",
                   "PYTHONPATH": os.pathsep.join([str(TESTS), str(HERE), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, str(HERE / "reference_programs.py"), case, str(steps_of(case))],
                          capture_output=True, text=True, check=False, env=environment)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.splitlines()[-1])


def holds_the_contract(row: dict[str, Any], x64: bool, record_property: Any) -> None:
    record_property("table_row", json.dumps(row, sort_keys=True))
    assert row["x64"] is x64 and row["compared"] > 0
    assert row["path"] == "one-step", "a run chunked by one step compiled the one-step history program"
    assert row["verify_moved"] == {}, "verify, a fresh graph run by the lock's program, reproduces the saved reference"
    assert row["own_moved"] == {}, "the saved reference is the run's own first steps"
    assert row["history_moved"] == {}, "the history scan on the run's graph is a fresh graph's"
    if row["case"] in EQUAL_BEFORE[x64]:
        assert row["moved"] == {}, "measured equal to what a save wrote before form 8"


@pytest.mark.parametrize("case", FLOAT32)
def test_in_float32_on_a_first_build_the_saved_reference_is_what_a_fresh_graph_reruns(case: str, record_property):
    holds_the_contract(in_a_fresh_float32_process(case), False, record_property)


@pytest.mark.parametrize("case", FLOAT64)
def test_in_float64_the_saved_reference_is_what_a_fresh_graph_reruns(case: str, record_property):
    holds_the_contract(table_row(case, steps_of(case)), True, record_property)
