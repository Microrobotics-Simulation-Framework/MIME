"""The switch scan's answer does not depend on what traced before it (REVIEW of #266, M2).

The scan reads each equation's recorded frames. JAX keeps small jaxprs it builds again and again (a scatter's combiner) in
caches that outlive a trace, and an equation of such a jaxpr carries the frames of the trace that first built it. Run
``tests/scene/test_tube_tree.py`` first and the arm's step, which runs no union, had 164 equations in ``ExactUnion.evaluate``'s
frames (27 steps in all), so the scan put that function in scope and found two raw sites in it: both of its control tests failed,
in one process, on any shard that put the file before a late scan test. ``trace_fresh`` empties JAX's caches before each trace.

Each case is a process of its own (``switch_order_probe.py``): the order is the thing under test, and a cache cannot be
emptied in the process that has to show it was not."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE = Path(__file__).with_name("switch_order_probe.py")


def union_frames(order: str, stale: bool = False) -> int:
    env = {key: value for key, value in os.environ.items() if key != "MIME_PLANT_STALE_FRAMES"}
    if stale or os.environ.get("MIME_PLANT_STALE_FRAMES"):
        env["MIME_PLANT_STALE_FRAMES"] = "1"
    proc = subprocess.run([sys.executable, str(PROBE), order], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return int(json.loads(proc.stdout.strip().splitlines()[-1])["union_frames"])


def test_the_arms_step_has_no_union_frame_after_a_union_was_traced_in_the_process():
    assert union_frames("polluted") == 0


def test_control_traced_as_before_the_fix_the_same_order_puts_the_union_in_the_arms_step():
    """The order does what the review measured (164 equations at the fix's base): without it the test above says nothing."""
    assert union_frames("polluted", stale=True) > 0


def test_control_the_arms_step_alone_has_no_union_frame_even_traced_as_before_the_fix():
    """Nothing but the order puts one there."""
    assert union_frames("clean", stale=True) == 0
