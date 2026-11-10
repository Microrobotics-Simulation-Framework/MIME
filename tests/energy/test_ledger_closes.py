"""The ledger harness's first rung (``mime.testing.ledger_closes``): one check that a metered run's books close at
every step, which any gate of any dynamics or field kind uses: a receiver's records or a run's result.

**Its resolution is the ledger's floor, in both dtypes, on every receiver kind**: a gift of 1.5 floors of each step's
own scale (planted in the books every receiver's step writes, ``mime.core.ledger.recorded``, so its size is exact
whatever the draw) is caught on every draw of a rigid body, a dynamic arm and a flow (``step_identity``'s draws,
which judge with it), and one of 0.5 floors on none: the check is no looser and no tighter than its floor, within the
step's own rounding, which reaches 0.27 floors on a rigid body's draws (0.10 on a flow's, 0.03 on an arm's; 500 to
2000 draws a kind and dtype, the PR's table). So a gift between about 0.73 and 1.27 floors is caught on some draws and
not others, and a floor loosened by 30 % passes these drawn gifts: **the floor's value is pinned on records** (no
rounding: a worst step at exactly the floor passes, the next number above fails). The rate over independent runs is
measured on Actions with this file run as a script (:func:`caught_in_one_run`), never collected as a test: CI's lanes
allow no skipped test.

**Each of its checks has a control on records** that fails it alone: the identity, unattributed work (and none where
the gate fed some), a dissipative law's gift, an indefinite operator, a step that gave energy on route 1, a jump
between steps no producer owns, the balance, a record that is not finite; and a run's result, its findings and each
metered receiver judged."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from ledger_cases import in_dtype

from mime.core import ledger as books
from mime.testing import ledger_closes, step_identity

def gifted(floors: float):
    """``mime.core.ledger.recorded`` with each step's end storage raised by ``floors`` of the identity's floor on that
    step's scale, unbooked: an energy gift of exactly that size, on any receiver whose step the ledger records."""
    real = books.recorded

    def recorded(state, step, held=()):
        eps = float(np.finfo(step.storage_end.dtype).eps)
        raised = step.storage_end + floors * books.IDENTITY_ULPS * eps * step.scale
        return real(state, dataclasses.replace(step, storage_end=raised), held)

    return recorded


def _dynamics(kind: str):
    from mime.dynamics import FVM, Arm, RigidInertial

    return {"rigid_body": RigidInertial(), "joint": Arm.__new__(Arm), "fluid": FVM.__new__(FVM)}[kind]


@pytest.mark.parametrize("floors", [1.5, 0.5])
@pytest.mark.parametrize("kind", ["rigid_body", "joint", "fluid"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_gift_over_the_floor_is_caught_on_every_draw_and_one_under_it_on_none(dtype, kind, floors, monkeypatch):
    monkeypatch.setattr(books, "recorded", gifted(floors))
    with in_dtype(dtype):
        result = step_identity(_dynamics(kind), max_examples=10)
    if floors > 1.0:
        assert result.failed and "of its floor" in result.detail, result
        assert result.n_examples <= 2, "the first draw fails (Hypothesis replays it once)"
    else:
        assert result.passed, result


def caught_in_one_run(kind: str, dtype: str, floors: float) -> bool:
    """One run of the gift's detection rate: a gift of ``floors`` on 20 draws of ``kind`` in ``dtype``; True when the
    check catches it. ``python tests/energy/test_ledger_closes.py --kind fluid --dtype float32`` is the target of
    ``.github/probe/detection_rate.py --script`` (``--plant-var MIME_LEDGER_GIFT``: the planted arm's gift in floors,
    zero on the clean arm); it exits 1 when caught."""
    real = books.recorded
    books.recorded = gifted(floors)
    try:
        with in_dtype(dtype):
            result = step_identity(_dynamics(kind), max_examples=20)
    finally:
        books.recorded = real
    return result.failed


# -- each check, on records -----------------------------------------------------------------------------------------

def _closed(dtype: str = "float64", route_one: bool = False) -> dict[str, np.ndarray]:
    """A receiver's records after a run whose books close: one law, three terms, a balance of exactly zero."""
    z = np.zeros((), dtype=dtype)
    records = {f"{books.RECORDS}{name}": np.asarray(value, dtype=dtype) for name, value in {
        "work": [2.0], "work_low": [0.0], "step_work": [0.1], "hosted": [[0.0, 0.0]], "hosted_low": [[0.0, 0.0]],
        "terms": [0.5, 0.25, 0.25], "terms_low": [0.0, 0.0, 0.0], "step_terms": [0.0, 0.0, 0.0],
        "unattributed": 0.0, "unattributed_low": 0.0, "unattributed_worst": 0.0, "between": 0.0, "between_low": 0.0,
        "between_own": 0.0, "between_own_low": 0.0, "between_worst": 0.0, "storage_own": 2.0, "storage_start": 1.0,
        "storage": 2.0, "step_identity": 0.0, "identity_worst": 1e-17 if dtype == "float64" else 1e-8,
        "scale": 10.0, "indefinite_steps": [0.0], "gift_steps": [0.0]}.items()}
    if route_one:
        records.update({f"{books.RECORDS}{name}": z for name in books.ROUTE_ONE_FIELDS})
        records[f"{books.RECORDS}route_one_worst"] = np.asarray(0.1, dtype=dtype)
    return records


PLANTS = {
    "identity": ("identity_worst", 2.0 * books.IDENTITY_ULPS, "of its floor"),
    "unattributed": ("unattributed_worst", 1e-30, "unattributed"),
    "gift": ("gift_steps", [1.0], "gave energy"),
    "indefinite": ("indefinite_steps", [3.0], "definite"),
    "between": ("between_worst", 2.0 * books.IDENTITY_ULPS, "jumped between steps"),
    "balance": ("storage", 2.5, "balance"),
    "not_finite": ("terms", [np.nan, 0.25, 0.25], "not finite"),
}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_closed_books_pass(dtype):
    assert ledger_closes(_closed(dtype)).passed
    assert ledger_closes(_closed(dtype, route_one=True)).passed


@pytest.mark.parametrize("plant", sorted(PLANTS))
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_control_each_check_fails_its_plant_alone(dtype, plant):
    """Each plant fails the check, naming what failed; the identity's and the jump's are planted at two floors."""
    field, value, words = PLANTS[plant]
    records = _closed(dtype)
    eps = float(np.finfo(dtype).eps)
    scaled = np.asarray(value, dtype=np.float64) * (eps if plant in ("identity", "between") else 1.0)
    records[f"{books.RECORDS}{field}"] = np.asarray(scaled, dtype=dtype)
    result = ledger_closes(records)
    assert result.failed and words in result.detail, result


@pytest.mark.parametrize("route_one", [False, True])
@pytest.mark.parametrize("field", ["identity_worst", "between_worst"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_floor_is_pinned_at_its_exact_value(dtype, field, route_one):
    """The drawn gifts above resolve the floor only to the step's rounding (0.73 to 1.27 floors), so a floor loosened
    by 30 % passed them. Here on records, with no rounding: a worst step at exactly 16 units of the dtype's rounding
    passes, and at the next number above it fails."""
    assert books.IDENTITY_ULPS == 16.0
    floor = np.asarray(16.0 * np.finfo(dtype).eps, dtype=dtype)
    records = _closed(dtype, route_one=route_one)
    records[f"{books.RECORDS}{field}"] = floor
    assert ledger_closes(records).passed
    records[f"{books.RECORDS}{field}"] = np.nextafter(floor, np.asarray(np.inf, dtype=dtype))
    assert ledger_closes(records).failed


def test_control_route_one_s_step_that_gave_energy_fails_and_its_balance_is_not_judged():
    records = _closed(route_one=True)
    records[f"{books.RECORDS}storage"] = np.asarray(2.5)  # route 1 leaves its scheme's dissipation in the balance
    assert ledger_closes(records).passed
    records[f"{books.RECORDS}route_one_negative_steps"] = np.asarray(1.0)
    result = ledger_closes(records)
    assert result.failed and "gave energy" in result.detail


def test_work_fed_unattributed_must_come_out_unattributed():
    records = _closed()
    assert ledger_closes(records, fed_unattributed=True).failed
    records[f"{books.RECORDS}unattributed_worst"] = np.asarray(1e-9)
    assert ledger_closes(records, fed_unattributed=True).passed


def test_a_group_member_s_jump_is_its_group_s():
    records = _closed()
    records[f"{books.RECORDS}between_worst"] = np.asarray(1e-10)
    assert ledger_closes(records).failed and ledger_closes(records, grouped=True).passed


def test_a_receiver_that_holds_no_records_is_not_metered():
    result = ledger_closes({"velocity": np.zeros(3)})
    assert result.failed and "not metered" in result.detail


def test_a_run_s_result_judges_each_metered_receiver_and_its_findings():
    class Result:
        final = {"a": _closed(), "b": {"velocity": np.zeros(3)}}
        energy = None

    assert ledger_closes(Result()).passed
    planted = _closed()
    planted[f"{books.RECORDS}gift_steps"] = np.asarray([1.0])

    class Gifted(Result):
        final = {"a": planted}

    result = ledger_closes(Gifted())
    assert result.failed and result.detail.startswith("a: ")

    class Unmetered(Result):
        final = {"b": {"velocity": np.zeros(3)}}

    assert ledger_closes(Unmetered()).failed


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["rigid_body", "joint", "fluid"], required=True)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    arguments = parser.parse_args()
    gift = float(os.environ.get("MIME_LEDGER_GIFT", "0"))
    sys.exit(1 if caught_in_one_run(arguments.kind, arguments.dtype, gift) else 0)
