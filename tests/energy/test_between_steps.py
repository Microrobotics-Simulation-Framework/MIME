"""The storage's jump between steps is judged unless a producer is declared for it (review M1 of P1b), and a run says so
when work arrived that no law or writer owns (review M2).

**M1.** A receiver's storage at a step's start less the storage the step before ended at is the jump between steps. It
is added back to the books only where a producer is declared for it: the delivered inertia's exchange (the part of the
storage that reads a delivery) and a coupling group. What no declared producer explains, the storage of the receiver's
own state at a step's start less the one the step before ended at, is judged as unattributed work is: its worst over
the step's scale is kept (``ledger_between_worst``), over 16 eps it is a failure of the step identity, and the balance
leaves it in. Without that a state edited between steps (a hand-edited state, a restart that resets a field, a node
that writes another's state) closed the books by construction.

- **The bookkeeping, alone** (drawn storages and scales, both dtypes): a jump declared as the exchange's closes the
  books and is not judged; the same jump in the receiver's own state fails them, with the jump read in floors.
- **A0's dynamic loop, the motor's spin bumped**: the review's 50 rad/s gift (1.25e-2 J, four times A0's whole 50 ms
  supply) fails on every run, the books, the step identity and the account's findings; and gifts at a fraction of a
  step's floor up to ten of them, after a warm-up, to say where detection stops: **it reads the gift in floors and
  fails above one**, so a gift of half a floor passes and one of two fails. The floor is 16 eps of the step's scale,
  about fifty times the loop's own rounding, which is the identity's resolution and not a statement about the gift's
  size. Both dtypes, one test each.
- **A declared jump** (the swinging rotor, whose exchange moves the arm's storage between steps): booked under the
  exchange, not judged, the books closing.
- **M2.** A callback-mode controller writing into a metered receiver arrives unattributed: the run's text, the ledger's
  own summary and the command line's attention lines say so, with the amount; the same run with the controller in the
  graph says nothing, and its build report no longer reads ``outside: nothing``."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from ledger_cases import in_dtype, swinging_rotor
from script_a0_dynamic import a0_dynamic

from mime.build._maddening import run_scan
from mime.cli._summary import attention_lines
from mime.core import ledger
from mime.observations import EnergyLedger
from mime.results import energy_account
from mime.units import s

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
PROPERTY = settings(max_examples=100, deadline=None, database=None)
#: The motor's spin gift of the review, rad/s, and the joules it is (the rotor's inertia and the spin's square).
REVIEW_BUMP = 50.0
WARM_UP = 50


@st.composite
def _jumps(draw: Any) -> tuple[Any, float, float]:
    """A dtype's name, a step's scale (a storage of 1e-9 to 1e3 J, times 1 to 100) and a multiple of the identity's
    floor (0.25 to 20) the jump is."""
    dtype = draw(st.sampled_from(["float32", "float64"]))
    storage = 10.0 ** draw(st.floats(-9.0, 3.0))
    return dtype, storage * draw(st.floats(1.0, 100.0)), draw(st.floats(0.25, 20.0))


def _planted(dtype: Any, scale: float, floors: float, *, declared: bool) -> tuple[Any, float]:
    dtype = jnp.dtype(dtype)
    eps = float(jnp.finfo(dtype).eps)
    storage = jnp.asarray(scale / 3.0, dtype=dtype)
    jump = jnp.asarray(floors * ledger.IDENTITY_ULPS * eps * scale, dtype=dtype)
    read = float(jump) / (ledger.IDENTITY_ULPS * eps * scale)
    own = storage if declared else storage + jump
    # The first step's start is the jumped one; the storage at its end does not change inside it.
    records = ledger.initial_records((), 0, storage)
    step = ledger.StepAccount(
        work=jnp.zeros(0, dtype=dtype), hosted=jnp.zeros((0, 2), dtype=dtype), unattributed=jnp.zeros((), dtype),
        terms=jnp.zeros(0, dtype=dtype), storage_start=storage + jump, storage_end=storage + jump,
        scale=jnp.asarray(scale, dtype=dtype), operators={}, own_start=own, own_end=own)
    return ledger.recorded(records, step, ()), read


@PROPERTY
@given(_jumps())
def test_a_jump_a_producer_declares_closes_the_books_and_is_not_judged(drawn):
    dtype, scale, floors = drawn
    with in_dtype(dtype):
        records, read = _planted(dtype, scale, floors, declared=True)
        assert float(ledger.between_floors(records)) == 0.0 and float(records["ledger_between"]) != 0.0
        floor = float(ledger.balance_floor(records))
        assert abs(float(ledger.balance(records))) <= floor, (float(ledger.balance(records)), floor)


@PROPERTY
@given(_jumps())
def test_a_jump_no_producer_declares_is_read_in_floors_and_fails_above_one_on_every_draw(drawn):
    """The control: the same jump in the receiver's own state. A jump of at least 1.5 floors fails every draw (the
    state's rounding of the jump itself is a few percent in float32); a quarter floor passes; the reading is the
    jump's, to that rounding."""
    dtype, scale, floors = drawn
    with in_dtype(dtype):
        records, read = _planted(dtype, scale, floors, declared=False)
        seen = float(ledger.between_floors(records))
        assert seen == pytest.approx(read, rel=0.15), (seen, read)
        if read >= 1.5:
            assert seen > 1.0, (seen, read)
        if read <= 0.5:
            assert seen < 1.0, (seen, read)
        floor = float(ledger.balance_floor(records))
        if read >= 1.5:
            assert abs(float(ledger.balance(records))) > floor, "the balance leaves an unexplained jump in"
        assert abs(float(ledger.balance(records, grouped=True))) <= floor, "a group owns its members' jumps whole"


def _metered(experiment: Any) -> Any:
    experiment.observe(EnergyLedger())
    return experiment.build()


def _motor_gift(graph: Any, states: Any, bump: float) -> Any:
    """One step of the loop from ``states`` with the motor's spin raised by ``bump`` (rad/s), as a state edited
    between steps."""
    for name in graph.node_names:
        graph.set_node_state(name, states[name])
    edited = dict(states["rpm.motor"])
    edited["spin"] = edited["spin"] + jnp.asarray(bump, dtype=edited["spin"].dtype)
    graph.set_node_state("rpm.motor", edited)
    return run_scan(graph, 1)


@DTYPES
def test_a_state_edited_between_steps_on_a0_s_dynamic_loop_fails_when_it_gives_the_review_s_gift_or_over_one_floor(name):
    """A0's dynamic loop with the ledger, the motor's spin raised between two steps.

    **The review's gift**: 50 rad/s from the start, 1.25e-2 J, four times A0's whole 50 ms supply. It fails the
    balance, the step identity and the account's findings, on every run (one rig, no draws).

    **Where detection stops**: after a warm-up, the gift is raised to a multiple of the step's own floor (16 eps of its
    scale, read from the same step run without the gift), and the reading is that multiple. **A gift of half a floor
    passes and one of two fails**, in both dtypes: the resolution is the identity's, one floor, about fifty times the
    loop's own rounding, which is the floor and not a statement about the gift's size.

    **The unplanted loop** reads far under its floor between steps, on every receiver."""
    with in_dtype(name):
        graph = _metered(a0_dynamic()).graph
        initial = {node: dict(graph.get_node_state(node)) for node in graph.node_names}
        warm = run_scan(graph, WARM_UP)
        for receiver in ("umr", "rpm.arm", "rpm.motor"):
            assert float(ledger.between_floors(warm[receiver])) <= 0.1, receiver

        final = _motor_gift(graph, initial, REVIEW_BUMP)
        records = final["rpm.motor"]
        assert float(ledger.between_floors(records)) > 1e3
        assert abs(float(ledger.balance(records))) > 1e3 * float(ledger.balance_floor(records))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            account = energy_account(graph, final, steps=1)
        assert account is not None
        assert account.identity_worst["rpm.motor"] > 1e3 and account.between_worst["rpm.motor"] > 1e3
        assert any("rpm.motor" in finding and "between steps" in finding for finding in account.findings), \
            account.findings
        assert account.identity_worst["rpm.arm"] <= 1.0 and account.identity_worst["umr"] <= 1.0
        assert "FINDING" in account.text()

        spin = float(warm["rpm.motor"]["spin"])
        inertia = float(graph.get_node("rpm.motor").constants()["inertia"])
        assert spin != 0.0
        trial = _motor_gift(graph, warm, 0.0)["rpm.motor"]
        eps = float(jnp.finfo(trial["ledger_scale"].dtype).eps)
        floor = ledger.IDENTITY_ULPS * eps * float(trial["ledger_scale"] - warm["rpm.motor"]["ledger_scale"])
        seen = {}
        for multiple in (0.25, 0.5, 2.0, 4.0, 10.0):
            wanted = multiple * floor
            bump = 2.0 * wanted / (inertia * (spin + np.sqrt(spin * spin + 2.0 * wanted / inertia)))
            seen[multiple] = float(ledger.between_floors(_motor_gift(graph, warm, bump)["rpm.motor"]))
        print(f"\n{name}: a gift of 0.25, 0.5, 2, 4 and 10 floors reads {seen}")
        for multiple, read in seen.items():
            assert read == pytest.approx(multiple, rel=0.3), seen
        assert seen[0.25] < 1.0 and seen[0.5] < 1.0 and seen[2.0] > 1.0 and seen[4.0] > 1.0 and seen[10.0] > 1.0


def test_a_jump_the_exchange_declares_is_booked_under_it_and_closes_the_books():
    """The swinging rotor's arm carries the rotor's inertia, predicted a step ahead: its storage jumps between steps by
    the exchange's own discretisation error. It is booked under the exchange, the receiver's own state's jump is
    rounding, and the books close."""
    with in_dtype("float64"):
        built = swinging_rotor(1e-3).build()
        final = run_scan(built.graph, 300)
        arm = final["swing.arm"]
        jumped, own = float(arm["ledger_between"]), float(arm["ledger_between_own"])
        assert abs(jumped) > 1e3 * abs(own) and abs(jumped) > 1e-9, (jumped, own)
        assert float(ledger.between_floors(arm)) <= 1.0
        assert abs(float(ledger.balance(arm))) <= float(ledger.balance_floor(arm))
        account = energy_account(built.graph, final, steps=300)
        assert account is not None and not account.findings, account.findings
        assert account.exchanges["swing.motor.inertia"] != 0.0


def test_work_a_callback_controller_writes_is_said_to_be_unattributed_where_the_run_is_read_and_a_graph_one_says_nothing():
    """A0's dynamic loop for 50 ms, the controller in its callback mode and in the graph, each with the ledger.

    **Callback**: the controller is no node, so its torques arrive unattributed on the arm and the motor. The run's
    text, the ledger's own summary and the command line's attention lines say so, with the amount; the identity
    books the work and closes, so attribution is what fails; the build report no longer reads ``outside: nothing``.
    **Graph**: nothing is said, the unattributed work is exactly zero, and the build report names the controller."""
    with in_dtype("float64"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built, result = {}, {}
        for mode in ("graph", "callback"):
            built[mode] = _metered(a0_dynamic(mode=mode))
            result[mode] = built[mode].run(s(0.05))

    account = result["callback"].energy
    assert account is not None and account.findings
    amount = account.unattributed["rpm.motor"]
    assert amount != 0.0
    attention = "\n".join(attention_lines(built["callback"], _written(result["callback"]), None, Path(".")))
    for text in (result["callback"].text(), account.text(), attention):
        assert "no law or writer owns" in text and "rpm.motor" in text, text
        assert f"{amount:+.4g} J" in text
    assert "mode 'callback'" in account.findings[0]
    report = built["callback"].report.energy
    assert any(line.startswith("outside: ") and "nothing" not in line and "controller (called on the host)" in line
               for line in report), report
    assert any("is no node of the graph" in line for line in report)
    assert account.identity_worst["rpm.motor"] <= 1.0, "the identity books the work, and closes: only attribution fails"

    clean = result["graph"].energy
    assert clean is not None and not clean.findings
    assert result["graph"].energy_findings == () and "FINDING" not in clean.text()
    assert "energy ledger:" not in result["graph"].text()
    assert all(value == 0.0 for value in clean.unattributed.values())
    assert built["graph"].report.energy[-1] == "outside: controller (receives no delivery from the components)"


class _written:
    """What ``attention_lines`` reads of a run written to a folder, for a run that was not: its result and nothing
    stopped."""

    def __init__(self, result: Any) -> None:
        self.result, self.stopped, self.steps = result, None, result.steps
