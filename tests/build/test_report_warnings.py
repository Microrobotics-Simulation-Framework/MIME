"""MADDENING's warnings reach the build report: a units warning is a refusal, and every other warning is reported
with where it came from, rather than left on stderr."""

import jax.numpy as jnp
import pytest
from maddening.core.node import BoundaryInputSpec, SimulationNode

from mime.build import BuildRefused, _construct, _testing
from mime.testing.toys import coupled_toy, full_toy

SPRING_LOAD = ("toy.spring:probe", "probe", "load", "load")


class _Lonely(SimulationNode):
    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"x": jnp.zeros(3)}

    def boundary_input_spec(self):
        return {"q": BoundaryInputSpec(shape=(3,), default=jnp.zeros(3), expected_units="m")}

    def update(self, state, boundary_inputs, dt):
        return state


def test_a_compile_warning_is_reported_with_its_source():
    report = full_toy().build().report
    (warning,) = [w for w in report.warnings if "larger feedback loop" in w.message]
    assert warning.source == "MADDENING GraphManager.compile(): UserWarning"
    assert f"[{warning.source}] {warning.message}" in report.text()


def test_a_validate_warning_is_reported_with_its_source():
    with _testing.after_construction(lambda graph, plans: graph.add_node(_Lonely("lonely", plans.dt))):
        report = coupled_toy().build().report
    assert [w.source for w in report.warnings if "'lonely' is disconnected" in w.message] == [
        "MADDENING GraphManager.validate()"]


def test_a_unit_mismatch_warning_from_compile_is_a_refusal(monkeypatch):
    """With MIME's own units check and MADDENING's validate() both out of the way, the warning compile() raises is
    still refused: the build turns a ``UnitMismatchWarning`` into a refusal, wherever it comes from."""

    def relabel(graph, plans):
        graph.remove_edge(*SPRING_LOAD)
        graph.add_edge(*SPRING_LOAD, additive=True, source_units="mN", target_units="mN")

    monkeypatch.setattr(_construct, "post_build_checks", lambda graph, plans: [])
    monkeypatch.setattr(_construct, "_validate", lambda graph: ([], []))
    with _testing.after_construction(relabel), pytest.raises(BuildRefused) as refused:
        coupled_toy().build()
    print(f"\n{refused.value}")
    assert set(refused.value.codes()) == {"maddening.units"}
    assert all(r.message.startswith("MADDENING's compile(): WARNING[units]") for r in refused.value.refusals)


def test_without_a_mismatch_the_same_bypass_builds(monkeypatch):
    """The control: the bypass alone refuses nothing, so the refusal above is the warning's."""
    monkeypatch.setattr(_construct, "post_build_checks", lambda graph, plans: [])
    monkeypatch.setattr(_construct, "_validate", lambda graph: ([], []))
    assert coupled_toy().build().report.refusals == ()
