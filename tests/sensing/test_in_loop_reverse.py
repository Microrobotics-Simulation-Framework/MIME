"""Reverse-mode derivatives through the in-loop camera over a run, through MIME's differentiated run (what
``Study.gradient`` compiles), checked in four arms and both dtypes: from the build's state, from a continued state (its
structure the tree's, or a leaf written over after the build), and in the chunks a CALLBACK controller's run is split
into. The case, the checks and each tolerance's mechanism are in ``reverse_gates.py``'s docstring.

Each arm runs float64 and float32 in one test (float32 is judged against float64 where central differences cannot
discriminate). The operating point is camera spike R1's fixed point; ``DETECTION_SEED`` draws another (the controls'
rates run over drawn points), and ``CAMERA_PLANT`` switches a planted error on: ``edge-stop``, ``held-stop`` and
``structure-stop``, each in every arm."""

from __future__ import annotations

import _cache_ceiling
import pytest
import reverse_gates as gates
from in_loop_cases import SEED

gates.plant()


#: How far above where a worker stands after giving its compiled programs back this file lets its resident memory
#: grow before it gives them back again (tests/_cache_ceiling.py). Each arm's test compiles the run's reverse
#: derivative afresh, and a worker that ran one kept 1 to 2 GB of its programs beside the next test; three workers at
#: 3 to 4 GB beside a memory-heavy test lost CI's shard 16 its runner (a probe on a runner, 2026-11-16).
HEADROOM_MIB = 1024
LIMIT_MIB: list[float] = []  # one worker's limit: where it stood after the last give-back, plus the headroom


@pytest.fixture(autouse=True)
def programs_given_back():
    """As ``test_lubrication.py``'s: give JAX's compiled programs back once a worker grows past its headroom."""
    yield
    now = _cache_ceiling.resident_mib()
    if now is None:
        return
    if not LIMIT_MIB or now > LIMIT_MIB[0]:
        _cache_ceiling.give_back_programs()
        LIMIT_MIB[:] = [(_cache_ceiling.resident_mib() or now) + HEADROOM_MIB]


@pytest.mark.parametrize("arm", list(gates.ARMS))
def test_the_reverse_derivative_through_the_camera_agrees_with_forward_mode_and_central_differences(arm):
    point = gates.point_of(SEED)
    f64 = gates.measured(point, gates.ARMS[arm], "float64")
    f32 = gates.measured(point, gates.ARMS[arm], "float32")
    failed = {f"{run['dtype']} {name}": check for run in (f64, f32)
              for name, check in gates.judged(run, f64, f32, gates.DETECTOR.width).items() if not check["pass"]}
    assert not failed, (arm, point, failed)
