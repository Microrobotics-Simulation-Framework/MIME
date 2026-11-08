"""The ledger through the build (F0-1 M3, M6.3; F0-2's one line), end to end on script A's reproduction arm.

Without ``EnergyLedger`` the build is the build it was: no receiver holds a ledger, no input of its own, no energy line,
and the run's result has no account. With it, each law delivers into its own input on the body, in the order its
summed input took them; the field is outside (it receives no delivery from the components), so the torque's work is
supplied by it; and over a run, in both dtypes, the body's step identity closes within its floor at every step, nothing
arrives unattributed (exactly zero), the books balance within 16 eps of the run's summed step scale, gravity's paired
work is its energy's change from the run's two ends, with its own parameters, within rounding (script A's body has no
couple), and no dissipative law gave energy.

**Attribution's controls, through the build**, each failing every run: an edge carrying a thousandth and a millionth of
the field's torque into the body's summed input, added to the built graph, is unattributed work; and the graph as built
is compared edge for edge with the deliveries the wiring made (``post-build.ledger-inputs``), so that each of these,
planted in the wiring, is refused at build by name: a law's delivery left in the summed input or sent to another
field's, a delivery dropped, a delivery made twice, two laws' own inputs swapped (the same field, or different fields
whose shapes agree, which no other check sees), and one law's two inputs swapped. With that check taken out, a dropped
delivery and one sent to another field's summed input reach an uncoded ``KeyError`` at run, and a duplicate and a swap
across fields run to a wrong account: the control that the refusals above are the check's alone. And a swap of
gravity's and the torque's inputs shows in gravity's residual, far above its floor (a witness for swaps that involve
gravity and a moving load only: gravity against drag's load is 0.008 of its floor in float32), which the identity and
attribution cannot see.

**The gradient through the ledger**: the account itself (``energy_account``, formed inside the function being
differentiated, parameters traced) gives a propulsive efficiency, the drag's axial resistance times the mean speed
squared times the time over ``supplied_total`` (lost plus the change in what is stored, each potential's included), its
derivative in the field's amplitude by reverse mode through the compiled step agrees with central differences as
``h^2`` at three step sizes (float64), and so do the potentials' change and ``supplied_total`` themselves."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ledger_cases import in_dtype
from script_a import reproduction_arm

from mime.build import BuildRefused
from mime.build import _ledger as wiring
from mime.build._maddening import full_state, graph_edges, graph_params, run_scan, step_function
from mime.core import ledger
from mime.observations import EnergyLedger
from mime.results import energy_account
from mime.units import s

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
#: Script A's laws on its body, and the run the books are judged over: 0.25 s, 500 steps.
LAWS = {"gravity:umr": "potential", "magnetic.torque:umr": "potential", "drag:umr": "dissipative",
        "contact:umr": "dissipative"}
RUN = s(0.25)


def metered(experiment=None):
    experiment = experiment if experiment is not None else reproduction_arm()
    experiment.observe(EnergyLedger())
    return experiment.build()


def floors(records: dict[str, Any]) -> float:
    return float(records["ledger_identity_worst"]) / (ledger.IDENTITY_ULPS * float(
        jnp.finfo(jnp.asarray(records["ledger_identity_worst"]).dtype).eps))


def test_without_the_ledger_the_build_is_the_build_it_was():
    plain = reproduction_arm().build()
    assert ledger.LEDGER not in plain.graph.get_node("umr").constants()
    assert not any(ledger.own_input(field) for *_, field in graph_edges(plain.graph))
    assert plain.report.energy == () and "energy ledger" not in plain.report.text(details=True)
    assert plain.run(s(0.005)).energy is None
    again = reproduction_arm().build()
    assert graph_edges(again.graph) == graph_edges(plain.graph)
    assert again.report.derived_hash == plain.report.derived_hash


def test_with_the_ledger_each_law_delivers_into_its_own_input_and_the_field_is_outside():
    built = metered()
    held = ledger.deliveries(built.graph.get_node("umr").constants())
    assert {d.law: d.energy_class for d in held} == LAWS
    assert [(d.field, d.law) for d in held] == [
        ("applied_wrench", "gravity:umr"), ("applied_wrench", "magnetic.torque:umr"),
        ("background_load", "drag:umr"), ("resistance", "drag:umr"), ("confinement", "contact:umr")]
    assert [d.kind for d in held] == ["load", "load", "load", "solved", "hosted"]
    assert {d.law: d.supplied_by for d in held}["magnetic.torque:umr"] == "field"
    into = [(source, field) for source, _, target, field in graph_edges(built.graph) if target == "umr"]
    assert into == [(d.law, ledger.law_input(d.field, i)) for i, d in enumerate(held) if d.kind != ledger.HOSTED]
    assert built.report.energy[-1] == "outside: field (receives no delivery from the components)"
    assert any(line.startswith("umr: metered (rigid_body port)") for line in built.report.energy)


@DTYPES
def test_script_a_s_books_close_through_the_build(name):
    with in_dtype(name):
        result = metered().run(RUN)
        records = result.final["umr"]
        account = result.energy
    assert account is not None
    assert floors(records) <= 1.0, floors(records)
    assert float(records["ledger_unattributed_worst"]) == 0.0 and account.unattributed == {"umr": 0.0}
    assert abs(account.balance) <= account.balance_floor, (account.balance, account.balance_floor)
    (gravity,) = [line for line in account.laws if line.law == "gravity:umr"]
    assert gravity.residual is not None and gravity.residual_floor is not None
    assert abs(gravity.residual) <= gravity.residual_floor, account.text(details=True)
    assert all(line.gift_steps == 0 and line.indefinite_steps == 0 for line in account.laws)
    assert set(account.lost) == {"drag:umr", "contact:umr"} and all(v > 0.0 for v in account.lost.values())
    assert list(account.supplied) == ["field"] and account.supplied["field"] > 0.0
    assert "supplied: field" in account.text()


def _with_a_hidden_edge(built, share: float):
    """``built``'s graph with an edge carrying ``share`` of the field's torque into the body's summed input: a
    delivery no law owns."""
    graph = built.graph
    graph.add_edge("magnetic.torque:umr", "umr", "wrench", "applied_wrench", transform=lambda value: share * value,
                   additive=True, source_units="N; N m", target_units="N; N m")
    graph.compile()
    return graph


@DTYPES
@pytest.mark.parametrize("share", [1e-3, 1e-6])
def test_a_hidden_edge_into_the_summed_input_is_unattributed_work(name, share):
    with in_dtype(name):
        clean = run_scan(metered().graph, 200)["umr"]
        planted = run_scan(_with_a_hidden_edge(metered(), share), 200)["umr"]
    assert float(clean["ledger_unattributed_worst"]) == 0.0
    assert float(planted["ledger_unattributed_worst"]) > 0.0
    assert floors(planted) <= 1.0, "the identity books the hidden edge's work as unattributed, and closes"


@contextmanager
def rewired(change):
    """The build's wiring with ``change`` applied to the edges it returns: a defect planted in the wiring."""
    real = wiring.attach

    def planted(plans, built, edges):
        built, rewired_edges, boundary, refusals = real(plans, built, edges)
        return built, change(rewired_edges), boundary, refusals

    wiring.attach = planted
    try:
        yield
    finally:
        wiring.attach = real


def swapped(edges):
    """Gravity's and the torque's edges into the body, each into the other's own input."""
    into = {e.by: e.target for e in edges if e.target[0] == "umr" and e.by in ("gravity:umr", "magnetic.torque:umr")}
    other = {"gravity:umr": "magnetic.torque:umr", "magnetic.torque:umr": "gravity:umr"}
    return [dataclasses.replace(e, target=into[other[e.by]]) if e.by in other and e.target[0] == "umr" else e
            for e in edges]


def left_in_the_summed_input(edges):
    """Gravity's edge into the body left in the input it sums into, where its work is unattributed."""
    return [dataclasses.replace(e, target=("umr", "applied_wrench")) if e.by == "gravity:umr" and e.target[0] == "umr"
            else e for e in edges]


def sent_to_another_summed_input(edges):
    """Gravity's edge into the body sent to the summed input of another field (the drag's load)."""
    return [dataclasses.replace(e, target=("umr", "background_load")) if e.by == "gravity:umr" and e.target[0] == "umr"
            else e for e in edges]


def dropped(edges):
    """Gravity's edge into the body gone."""
    return [e for e in edges if not (e.by == "gravity:umr" and e.target[0] == "umr")]


def made_twice(edges):
    """Gravity's edge into the body, twice."""
    return [copy for e in edges for copy in ((e, e) if e.by == "gravity:umr" and e.target[0] == "umr" else (e,))]


def _exchanged(edges, first, second):
    """The targets of the edges ``first`` and ``second`` (each ``(law, input's field)`` into the body) exchanged."""
    def which(e):
        return next((key for key in (first, second) if e.by == key[0] and e.target[0] == "umr"
                     and e.target[1].startswith(key[1])), None)

    targets = {which(e): e.target for e in edges if which(e) is not None}
    assert set(targets) == {first, second}, targets
    other = {first: second, second: first}
    return [dataclasses.replace(e, target=targets[other[which(e)]]) if which(e) is not None else e for e in edges]


def swapped_across_fields(edges):
    """Gravity's own input and the drag's load's exchanged: both take six components, so no check of shapes sees it."""
    return _exchanged(edges, ("gravity:umr", "applied_wrench"), ("drag:umr", "background_load"))


def one_law_s_inputs_swapped(edges):
    """The drag's load and its resistance each delivered into the other's own input."""
    return _exchanged(edges, ("drag:umr", "background_load"), ("drag:umr", "resistance"))


#: Each defect planted in the wiring, the codes the build refuses it with besides the check's (the drag's two inputs
#: differ in shape, which MADDENING's own checks see too), and what its refusal says.
WIRING_DEFECTS = {
    "swapped": (swapped, set(), "is not one the wiring made"),
    "unattributed": (left_in_the_summed_input, set(), "arrives in applied_wrench"),
    "another_summed_input": (sent_to_another_summed_input, set(), "arrives in background_load"),
    "dropped": (dropped, set(), "is missing from the graph"),
    "twice": (made_twice, set(), "is in the graph 2 times, not 1"),
    "swapped_across_fields": (swapped_across_fields, set(), "is not one the wiring made"),
    "one_laws_inputs": (one_law_s_inputs_swapped, {"maddening.shape", "maddening.units", "post-build.units"},
                        "is not one the wiring made"),
}


@pytest.mark.parametrize("name", WIRING_DEFECTS)
def test_the_wiring_s_defects_are_refused_at_build_by_the_post_build_check(name):
    change, besides, says = WIRING_DEFECTS[name]
    with rewired(change), pytest.raises(BuildRefused) as refused:
        metered()
    assert set(refused.value.codes()) == {"post-build.ledger-inputs"} | besides, refused.value
    messages = [r.message for r in refused.value.refusals if r.code == "post-build.ledger-inputs"]
    assert messages and any(says in message and "umr" in message for message in messages), messages


@pytest.mark.parametrize("name", ["dropped", "another_summed_input"])
def test_with_the_check_taken_out_a_missing_delivery_reaches_an_uncoded_key_error_at_run(name, monkeypatch):
    """The control for the refusals above: the run can reach an uncoded ``KeyError`` (a body's own input with no edge
    into it) only where the check is absent, so that its absence from the build is the check's."""
    import mime.build._checks as checks

    monkeypatch.setattr(checks, "misattributed", lambda *_: [])
    with rewired(WIRING_DEFECTS[name][0]):
        built = metered()
    with pytest.raises(Exception, match="error.uncoded|KeyError") as error:
        built.run(s(0.005))
    assert "KeyError" in str(error.value), error.value


@pytest.mark.parametrize("name", ["twice", "swapped_across_fields"])
def test_with_the_check_taken_out_a_duplicate_and_a_swap_across_fields_build_and_run(name, monkeypatch):
    """Nothing else sees them: they build and run, to an account that is wrong (a delivery counted twice, a load paired
    as another law's)."""
    import mime.build._checks as checks

    monkeypatch.setattr(checks, "misattributed", lambda *_: [])
    with rewired(WIRING_DEFECTS[name][0]):
        built = metered()
    assert built.run(s(0.005)).energy is not None


def test_every_own_input_of_a_receiver_takes_exactly_one_edge_so_the_run_has_no_key_error_to_reach():
    built = metered()
    edges = graph_edges(built.graph)
    held = ledger.deliveries(built.graph.get_node("umr").constants())
    into = [field for _, _, target, field in edges if target == "umr" and ledger.own_input(field) is not None]
    assert sorted(into) == sorted(ledger.law_input(d.field, i) for i, d in enumerate(held) if d.kind != ledger.HOSTED)
    assert len(into) == len(set(into))


def test_two_inputs_of_one_law_that_agree_in_shape_are_told_apart_by_the_wiring_they_were_made_for():
    """No registered law delivers two inputs of one shape, so the check is exercised on an edge table of its own: a law
    ``w`` delivering its force and its torque (both six components) into inputs 0 and 1 of a receiver, and the table
    with those two exchanged. A check of shapes sees nothing; this one names both."""
    delivered = [ledger.LedgerDelivery("load", "w", ledger.UNMETERED, ledger.LOAD),
                 ledger.LedgerDelivery("load_too", "w", ledger.UNMETERED, ledger.LOAD)]
    made = (("w:node", "force", "body", ledger.law_input("load", 0)), ("w:node", "torque", "body",
                                                                      ledger.law_input("load_too", 1)))
    receivers = {"body": tuple(delivered)}

    def owner_of(node):
        return "w"

    assert wiring.misattributed(list(made), receivers, owner_of, made) == []
    exchanged = [("w:node", "force", "body", ledger.law_input("load_too", 1)),
                 ("w:node", "torque", "body", ledger.law_input("load", 0))]
    found = wiring.misattributed(exchanged, receivers, owner_of, made)
    assert len(found) == 4, found
    assert sum("is not one the wiring made" in line for line in found) == 2
    assert sum("is missing from the graph" in line for line in found) == 2


@DTYPES
def test_with_the_check_taken_out_a_swap_shows_in_gravity_s_residual_and_nowhere_else(name, monkeypatch):
    import mime.build._checks as checks

    monkeypatch.setattr(checks, "misattributed", lambda *_: [])
    with rewired(swapped), in_dtype(name):
        result = metered().run(RUN)
        records, account = result.final["umr"], result.energy
    assert account is not None
    assert floors(records) <= 1.0 and float(records["ledger_unattributed_worst"]) == 0.0
    (gravity,) = [line for line in account.laws if line.law == "gravity:umr"]
    assert gravity.residual is not None and gravity.residual_floor is not None
    assert abs(gravity.residual) > 1e3 * gravity.residual_floor, account.text(details=True)


def _central(function, point, h):
    return (float(function(point * (1 + h))) - float(function(point * (1 - h)))) / (2 * h * float(point))


def test_the_account_s_own_sums_are_differentiable_and_agree_with_central_differences():
    """The derivative of the account itself, potentials included, through the compiled step by reverse mode: the
    account formed inside the function being differentiated from the final state and the parameters, as a study
    would take it (``energy_account``; no hand-assembled expression stands in for it). The propulsive efficiency is
    the drag's axial resistance times the mean speed squared times the time over ``supplied_total``; the potentials'
    change (gravity's: the body's height) and ``supplied_total`` are differentiated on their own as well, so that a
    sum of the account cut from the derivative (a potential held constant, a conversion to NumPy) fails against
    central differences at the smaller step sizes, on every run."""
    steps, dt = 400, 5e-4
    with in_dtype("float64"):
        graph = metered().graph
        tree = graph_params(graph)
        start = jax.tree_util.tree_map(jnp.asarray, full_state(graph))
        step = step_function(graph)

        def account_of(amplitude):
            params = {**tree, "nodes": {**tree["nodes"], "drive:field.amplitude": {"value": amplitude}}}
            final, _ = jax.lax.scan(lambda state, _: (step(state, {}, params), None), start, None, length=steps)
            return energy_account(graph, final, params["nodes"], steps=steps), final

        def efficiency(amplitude):
            account, final = account_of(amplitude)
            speed = (final["umr"]["pose"][0] - start["umr"]["pose"][0]) / (steps * dt)
            return account.efficiency(final["drag:umr"]["resistance"][0, 0] * speed ** 2 * steps * dt)

        def potentials(amplitude):
            return sum(account_of(amplitude)[0].potentials.values())

        def supplied(amplitude):
            return account_of(amplitude)[0].supplied_total

        amplitude = jnp.asarray(1.2e-3)
        errors = {}
        for name, function in (("efficiency", efficiency), ("potentials", potentials), ("supplied", supplied)):
            derivative = float(jax.grad(function)(amplitude))
            compiled = jax.jit(function)
            assert derivative != 0.0 and np.isfinite(derivative), name
            errors[name] = [abs(_central(compiled, amplitude, h) - derivative) / abs(derivative)
                            for h in (1e-3, 1e-4, 1e-5)]
        account, _ = account_of(amplitude)
        concrete = energy_account(graph, full_state_after(step, start, tree, steps), tree["nodes"], steps=steps)
    # The potentials are in the account being differentiated: a nonzero change, with a derivative of its own.
    assert account.potentials and float(sum(account.potentials.values())) != 0.0
    assert isinstance(concrete.supplied_total, float) and isinstance(concrete.efficiency(1e-12), float)
    # The efficiency's and supplied total's errors fall as h squared to the rounding floor; the potentials' change
    # (an energy difference of about 1e-11 J) to the larger rounding it has.
    for name in ("efficiency", "supplied"):
        coarse, middle, fine = errors[name]
        assert middle <= 1e-6 and middle < 0.03 * coarse and fine < 0.03 * middle, (name, errors[name])
    coarse, middle, _ = errors["potentials"]
    assert middle <= 1e-5 and middle < 0.03 * coarse, errors["potentials"]


def full_state_after(step, start, tree, steps):
    final, _ = jax.lax.scan(lambda state, _: (step(state, {}, tree), None), start, None, length=steps)
    return final
