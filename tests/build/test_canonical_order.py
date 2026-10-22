"""The canonical order makes the order of a script's calls irrelevant, bit for bit, on the coupled toy.

Every order of the two ``add_*`` calls and the four ``attach`` calls (48 scripts, binding by handle) must build the
same graph and give a history equal, in every field at every step, to the toy assembled by hand in the canonical
order from independently written node classes. Two controls show the test can fail: the hand-assembled graph is
sensitive to node order and to the order loads are summed in, and a build that skips the canonical order gives more
than one history over the same 48 scripts.
"""

import itertools

import pytest
from hand_built import LOADS_A, LOADS_B, ORDER_A, ORDER_B, assemble, history

from mime.build import _testing
from mime.experiment import Experiment
from mime.testing import history_differences, run_history
from mime.testing.toys import (
    PROBE_START,
    ToyBody,
    ToyCouple,
    ToyDrag,
    ToyLift,
    ToyRelaxation,
    ToySpring,
    toy_experiment,
)
from mime.things import Body, Medium
from mime.units import s

STEPS = 50
DURATION = s(STEPS * 0.02)


def _laws(probe, bath):
    return {
        "couple": lambda e: e.attach(ToyCouple(), body=probe, medium=bath),
        "drag": lambda e: e.attach(ToyDrag.Quadratic(), body=probe),
        "lift": lambda e: e.attach(ToyLift(), body=probe, medium=bath),
        "spring": lambda e: e.attach(ToySpring(), body=probe),
    }


def script(adds: tuple[str, ...], attaches: tuple[str, ...]) -> Experiment:
    experiment = toy_experiment("coupled_toy")
    handles = {}
    for name in adds:
        if name == "bath":
            handles[name] = experiment.add_medium(Medium("bath", flow=ToyRelaxation()))
        else:
            handles[name] = experiment.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START))
    laws = _laws(handles["probe"], handles["bath"])
    for name in attaches:
        laws[name](experiment)
    return experiment


SCRIPTS = [(adds, attaches) for adds in itertools.permutations(("bath", "probe"))
           for attaches in itertools.permutations(("couple", "drag", "lift", "spring"))]


@pytest.fixture(scope="module")
def reference():
    return history(assemble(ORDER_A, LOADS_A), STEPS)


@pytest.fixture(scope="module")
def reference_snapshot():
    return script(*SCRIPTS[0]).build().snapshot()


def test_the_hand_assembled_toy_is_sensitive_to_node_and_summation_order(reference):
    """The control that the comparison can discriminate: node order moves which edge of each loop lags, and the
    order loads are summed in changes the rounding."""
    both = history_differences(reference, history(assemble(ORDER_B, LOADS_B), STEPS))
    nodes_only = history_differences(reference, history(assemble(ORDER_B, LOADS_A), STEPS))
    loads_only = history_differences(reference, history(assemble(ORDER_A, LOADS_B), STEPS))
    assert both and nodes_only and loads_only
    assert any("first at step 0" in line for line in nodes_only)


@pytest.mark.parametrize(("adds", "attaches"), SCRIPTS, ids=lambda order: "-".join(order))
def test_every_order_of_the_calls_reproduces_the_hand_ordered_reference(adds, attaches, reference,
                                                                      reference_snapshot):
    built = script(adds, attaches).build()
    assert built.snapshot() == reference_snapshot
    assert history_differences(reference, run_history(built, DURATION)) == []


def test_without_the_canonical_order_the_same_scripts_give_different_histories():
    """The control that order invariance is the canonical order's doing: built in the order the script declares, the
    48 scripts give more than one history."""
    distinct: list[dict] = []
    with _testing.declaration_order():
        for adds, attaches in SCRIPTS:
            found = run_history(script(adds, attaches).build(), DURATION)
            if not any(history_differences(held, found) == [] for held in distinct):
                distinct.append(found)
    assert len(distinct) > 1
