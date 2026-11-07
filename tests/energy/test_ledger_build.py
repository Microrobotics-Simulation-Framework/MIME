"""The ledger through the build (F0-1 M3, M6.3; F0-2's one line), end to end on script A's reproduction arm.

Without ``EnergyLedger`` the build is the build it was: no receiver holds a ledger, no input of its own, no energy line,
and the run's result has no account. With it, each law delivers into its own input on the body, in the order its
summed input took them; the field is outside (it receives no delivery from the components), so the torque's work is
supplied by it; and over a run, in both dtypes, the body's step identity closes within its floor at every step, nothing
arrives unattributed (exactly zero), the books balance within 16 eps of the run's summed step scale, gravity's paired
work is its energy's change from the run's two ends, with its own parameters, within rounding (script A's body has no
couple), and no dissipative law gave energy.

**Attribution's controls, through the build**, each failing every run: an edge carrying a thousandth and a millionth of
the field's torque into the body's summed input, added to the built graph, is unattributed work; the build's wiring
leaving a law's delivery in the summed input, or swapping two laws' own inputs, is refused by the post-build check
(``post-build.ledger-inputs``); and, that check taken out, a swap of gravity's and the torque's inputs shows in
gravity's residual, far above its floor, which the identity and attribution cannot see.

**The gradient through the ledger**: a propulsive efficiency, the drag's axial resistance times the mean speed squared
times the time over what was supplied (lost plus the change in what is stored, as the account states it), over 0.2 s,
its derivative in the field's amplitude by reverse mode through the compiled step agrees with central differences as
``h^2`` (float64)."""

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
from mime.units import s

DTYPES = pytest.mark.parametrize("name", ["float32", "float64"])
#: Script A's laws on its body, and the run the books are judged over: 0.25 s, 500 steps.
LAWS = {"gravity:umr": "potential", "magnetic.torque:umr": "unmetered", "drag:umr": "dissipative",
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


@pytest.mark.parametrize("change", [swapped, left_in_the_summed_input], ids=["swapped", "unattributed"])
def test_the_wiring_s_defects_are_refused_by_the_post_build_check(change):
    with rewired(change), pytest.raises(BuildRefused) as refused:
        metered()
    assert set(refused.value.codes()) == {"post-build.ledger-inputs"}, refused.value


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


def test_a_propulsive_efficiency_s_derivative_through_the_ledger_agrees_with_central_differences():
    steps, dt = 400, 5e-4
    with in_dtype("float64"):
        graph = metered().graph
        tree = graph_params(graph)
        start = jax.tree_util.tree_map(jnp.asarray, full_state(graph))
        step = step_function(graph)
        held = ledger.deliveries(graph.get_node("umr").constants())

        @jax.jit
        def efficiency(amplitude):
            params = {**tree, "nodes": {**tree["nodes"], "drive:field.amplitude": {"value": amplitude}}}
            final, _ = jax.lax.scan(lambda state, _: (step(state, {}, params), None), start, None, length=steps)
            body = final["umr"]
            supplied = jnp.sum(ledger.losses(body, held)) + ledger.stored(body)
            speed = (body["pose"][0] - start["umr"]["pose"][0]) / (steps * dt)
            return final["drag:umr"]["resistance"][0, 0] * speed ** 2 * steps * dt / supplied

        amplitude = jnp.asarray(1.2e-3)
        derivative = float(jax.grad(efficiency)(amplitude))
        apart = []
        for h in (1e-3, 1e-4):
            central = (float(efficiency(amplitude * (1 + h))) - float(efficiency(amplitude * (1 - h)))) / (
                2 * h * float(amplitude))
            apart.append(abs(central - derivative) / abs(derivative))
    assert derivative != 0.0 and np.isfinite(derivative)
    assert apart[1] <= 1e-6 and apart[1] < 0.03 * apart[0], apart
