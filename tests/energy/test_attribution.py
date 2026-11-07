"""Attribution, read from per-law records kept in the receiver (F0-1 M6.3):
with each law delivering into its own input on the rigid body's node, the unattributed inputs (the ones every law sums
into without the ledger) paired no work, exactly zero, on every draw and in both dtypes: a zero, not a difference of
sums, so no summation floor.

**The plants**, on every draw: a share of the law's load, 1e-3 and 1e-6 (F0's spike FU1's), delivered into the summed input, an
edge no law owns: the unattributed work is not zero. The load is drawn pushing along the body's motion (its force and
torque 0.1 to 1 of their length along the velocity and the angular velocity), so that it does work in the step; a
load that does none is invisible to any energy check, which is not attribution's to see."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from ledger_cases import free_states, in_dtype, metered_body, one_step

BODY = metered_body()
DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])


@DTYPES
@settings(deadline=None)
@given(drawn=free_states())
def test_with_every_law_in_its_own_input_nothing_arrives_unattributed(name, drawn):
    with in_dtype(name) as dtype:
        found = one_step(BODY, drawn, dtype)
    assert float(found["ledger_unattributed"]) == 0.0
    assert float(found["ledger_unattributed_worst"]) == 0.0


@DTYPES
@pytest.mark.parametrize("share", [1e-3, 1e-6])
@settings(deadline=None)
@given(drawn=free_states(pushing=True))
def test_a_hidden_share_of_a_load_is_caught_on_every_draw(name, share, drawn):
    with in_dtype(name) as dtype:
        found = one_step(BODY, drawn, dtype, unattributed_share=share)
    assert float(found["ledger_unattributed_worst"]) > 0.0, found["ledger_unattributed"]
    assert float(found["ledger_unattributed"]) != 0.0
