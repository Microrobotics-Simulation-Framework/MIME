"""The lag structure of script A's two graphs, compared as declared before the run
(``script_a_expected_differences.json``, ``lag_structure``): the frozen reference's sixteen edges, in its builder's
order from its provenance (field, magnet, gravity, mlp_drag, body, lub), classified by the rule MIME's lag accounting
applies to its own graphs, and 0.3's ten, as the build report gives them; each 0.2 edge mapped to its 0.3 counterpart
or to none, with how the two relate.

The inference this settles: the 0.2 magnet node runs before the body, so its torque reads the orientation at the
start of the step, as 0.3's does. The control: built without the canonical order, 0.3's list changes, and the
comparison names the torque's edge among what no longer maps.
"""

import json
from pathlib import Path

from script_a import reproduction_arm

from mime.build import _testing
from mime.build._builders import DELIVERY, FIELD, STATE, EdgeDecl
from mime.build._lags import EdgeLag, classify, unplanned_refusals

HERE = Path(__file__).parent
DECLARED = json.loads((HERE / "script_a_expected_differences.json").read_text())["lag_structure"]
PROVENANCE = json.loads((HERE / "references" / "script_a" / "f64_provenance.json").read_text())
TORQUE_0_2, TORQUE_0_3 = "body.orientation -> magnet.orientation", "umr.pose -> magnetic.torque:umr.pose"


def reference_lags() -> tuple[EdgeLag, ...]:
    """The reference graph's edges, classified by MIME's rule: a body's output carries its state, the field's a
    field, every other a delivery."""
    edges = []
    for i, spec in enumerate(PROVENANCE["edges"]):
        source, target = spec.removeprefix("EdgeSpec(").removesuffix(")").split(" -> ")
        a, b = source.split(".")
        edges.append(EdgeDecl("reference", (a, b), tuple(target.split(".")), False, "", i,
                              {"body": STATE, "field": FIELD}.get(a, DELIVERY)))
    return classify(list(PROVENANCE["nodes"]), edges, [])


def mismatches(old: tuple[EdgeLag, ...], new: tuple[EdgeLag, ...]) -> list[str]:
    """Every way the two graphs' classified edges depart from the declared mapping."""
    found: list[str] = []
    old_lags, new_lags = {e.text: e.lag.value for e in old}, {e.text: e.lag.value for e in new}
    if old_lags != DECLARED["edges_0_2"]:
        found.append(f"0.2's edges classify as {old_lags}, not as declared")
    if new_lags != DECLARED["edges_0_3"]:
        found += [f"0.3's {text} reads {new_lags.get(text)}, declared {lag}"
                  for text, lag in DECLARED["edges_0_3"].items() if new_lags.get(text) != lag]
        found += [f"0.3's {text} is not declared" for text in new_lags.keys() - DECLARED["edges_0_3"].keys()]
    mapped = set()
    for text, row in DECLARED["mapping"].items():
        if old_lags.get(text) != row["lag_0_2"]:
            found.append(f"0.2's {text} reads {old_lags.get(text)}, mapped as {row['lag_0_2']}")
        counterpart = row["counterpart_0_3"]
        if counterpart is not None:
            mapped.add(counterpart)
            if new_lags.get(counterpart) != row["lag_0_3"]:
                found.append(f"{text} maps to 0.3's {counterpart}, which reads {new_lags.get(counterpart)}, mapped "
                             f"as {row['lag_0_3']}")
    if old_lags.keys() != DECLARED["mapping"].keys():
        found.append("the mapping does not cover 0.2's edges exactly")
    unmapped = new_lags.keys() - mapped - DECLARED["new_in_0_3"].keys()
    found += [f"0.3's {text} has no 0.2 counterpart and is not declared new" for text in sorted(unmapped)]
    return found


def test_the_two_graphs_lag_structures_map_as_declared():
    assert mismatches(reference_lags(), reproduction_arm().build().report.lags) == []


def test_the_torque_reads_the_orientation_at_the_start_of_the_step_in_both():
    """The inference: in both graphs the torque's read of the orientation is a previous-step read, so both measured
    lags contain -omega dt and the two agree (the float64 comparison then holds the lag to its prediction)."""
    old = {e.text: e for e in reference_lags()}
    new = {e.text: e for e in reproduction_arm().build().report.lags}
    assert old[TORQUE_0_2].lag.value == new[TORQUE_0_3].lag.value == "previous step"
    assert old[TORQUE_0_2].planned and new[TORQUE_0_3].planned
    assert DECLARED["mapping"][TORQUE_0_2]["counterpart_0_3"] == TORQUE_0_3


def test_0_3s_rule_refuses_the_reference_graphs_late_drag():
    """0.2's drag reached its body a step late, a delivered load read from the previous step, which 0.3's lag
    accounting refuses unless a coupling group solves the loop; 0.3 delivers a resistance read this step instead."""
    refused = unplanned_refusals(reference_lags(), {})
    assert [r.message.split(", built by")[0] for r in refused] == [
        "the edge lub.drag_force -> body.drag_force", "the edge lub.drag_torque -> body.drag_torque"]
    assert DECLARED["refused_by_0_3s_rule_in_0_2"] == ["lub.drag_force -> body.drag_force",
                                                       "lub.drag_torque -> body.drag_torque"]


def test_without_the_canonical_order_the_comparison_notices():
    """The control: declared in the script's order the body is added before the laws, so its loops close on the
    deliveries, read a step late; the comparison must report it, the torque's edge among what no longer maps."""
    with _testing.declaration_order():
        shuffled = reproduction_arm().build().report.lags
    found = mismatches(reference_lags(), shuffled)
    print("\n".join(found))
    assert found
    assert any(TORQUE_0_3 in line for line in found)
    assert any("magnetic.torque:umr.wrench -> umr.applied_wrench" in line for line in found)
