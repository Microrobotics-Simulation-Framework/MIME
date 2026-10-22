"""A run is MADDENING's one jit-compiled step, scanned: whatever MIME adds above its nodes runs at build time, nothing
runs on the host inside a step, and every value a study may change is a ``params`` leaf the step reads. Three
checks, each with a control:

- the build report states the run's structure, and a run of a toy is one compiled scan; the control: MADDENING's own
  scan counter sees a run made of two compiled programs;
- the build traces the step once and refuses a host callback inside it, naming the node; the control: a node whose
  update calls ``jax.pure_callback`` (or ``jax.debug.print``) is refused;
- every parameter leaf reaches the compiled step without a recompilation; the control: a node that keeps its leaf in
  ``params`` but computes with a constant it took at construction is caught, because the write changes nothing.
"""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.build import BuildRefused, RefusalClass
from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing import param_reach_failures
from mime.testing.toys import LOAD, POSITION, ToySpringNode, coupled_toy, full_toy
from mime.units import s
from mime.vocabulary import register_namespace

register_namespace("checking", "0.0.1")


class _CallsBackSpringNode(ToySpringNode):
    def law(self, p, reading):
        doubled = jax.pure_callback(lambda x: np.asarray(x) * 2.0, jax.ShapeDtypeStruct(reading.shape, reading.dtype),
                                    reading)
        return super().law(p, doubled)


class _PrintingSpringNode(ToySpringNode):
    def law(self, p, reading):
        jax.debug.print("x = {x}", x=reading)
        return super().law(p, reading)


class _BakedSpringNode(ToySpringNode):
    """Keeps ``k`` in ``params``, so it saves whole and passes every post-build check, but computes with the value it
    took at construction: a write of the leaf reaches nothing."""

    def __init__(self, name, timestep, k=1.7, x0=(0.05, -0.02, 0.01)):
        super().__init__(name, timestep, k=k, x0=x0)
        self._k = float(k)

    def law(self, p, reading):
        return super().law({**p, "k": self._k}, reading)


@dataclass(frozen=True, kw_only=True)
class _Spring(Interaction):
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    NODE: ClassVar[type[ToySpringNode]] = ToySpringNode
    k: float = 0.9

    def build(self, ctx, b):
        node = b.add_response_node(self.NODE(ctx.node_name(), ctx.timestep, k=self.k))
        b.connect(b.read("x"), into=node["x"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class CallsBack(_Spring):
    name = "checking.calls_back"
    NODE = _CallsBackSpringNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Prints(_Spring):
    name = "checking.prints"
    NODE = _PrintingSpringNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Baked(_Spring):
    name = "checking.baked"
    NODE = _BakedSpringNode


def test_every_toy_reports_one_compiled_scan():
    for make in (coupled_toy, full_toy):
        report = make().build().report
        assert report.run.text() == "one compiled scan"
        assert "run: one compiled scan" in report.text()


def test_a_run_of_a_toy_is_one_compiled_scan_and_the_counter_can_see_two():
    built = full_toy().build()
    graph = built.graph
    built.run(s(0.4))
    assert graph.scan_trace_count == 1
    built.run(s(0.4))
    assert graph.scan_trace_count == 1, "a second run of the same length compiled again"
    graph.run_scan(7)
    graph.run_scan(13)
    assert graph.scan_trace_count == 3, "the control: two programs of other lengths are two more traces"


@pytest.mark.parametrize(("entry", "primitive"), [(CallsBack, "pure_callback"), (Prints, "debug_print")],
                         ids=["pure_callback", "debug_print"])
def test_a_host_callback_in_the_step_is_refused_naming_its_node(entry, primitive):
    experiment = coupled_toy()
    experiment.attach(entry(), body="probe")
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "post-build.host-callback" and refusal.category is RefusalClass.INTERNAL
    assert f"node {entry.name}:probe (built by {entry.name}:probe, {entry.name} 0.0.1)" in refusal.message
    assert primitive in refusal.message
    assert "test_compiled_step" in refusal.action


def test_a_build_without_a_host_callback_is_not_refused():
    """The control on the probe: the same spring without the callback builds."""
    experiment = coupled_toy()
    experiment.attach(Baked(), body="probe")
    assert experiment.build().report.refusals == ()


def test_a_leaf_baked_into_the_trace_as_a_constant_is_caught():
    experiment = coupled_toy()
    experiment.attach(Baked(), body="probe")
    failures = param_reach_failures(experiment.build())
    print("\n" + "\n".join(failures))
    assert failures == [
        "writing params['nodes']['checking.baked:probe']['k'] changed nothing over 5 steps: the compiled step does "
        "not read it, so a study changing it changes nothing (a value baked into the trace as a constant)"]


def test_the_toys_leaves_reach_the_step_without_recompiling():
    """The control on the control: the toys' own leaves pass, so the failure above is the baked constant's."""
    assert param_reach_failures(full_toy().build()) == []


def test_jax_reports_the_compilations_the_leaf_check_counts():
    """The check counts compilations with JAX's own monitoring; a new shape compiles, so it would be counted."""
    from mime.testing._saved import _compilations

    with _compilations() as compiled:
        jax.jit(lambda x: x * 3.0)(jnp.ones(17))
    assert compiled
