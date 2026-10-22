"""Cross-effect coupling materialisation (E3).

The v0.2 pilot validated couple() (passes 1-2) but never *wired* an edge. E3
makes it real: couple(target, source) now materialises a graph edge from the
source effect's output CouplingPort to the target effect's input CouplingPort,
after both subgraphs exist and before compile().
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.node import (
    BoundaryFluxSpec,
    BoundaryInputSpec,
    SimulationNode,
)

from mime.effects import (
    BaseEffectModel,
    Body,
    CouplingError,
    CouplingPort,
    EffectHandle,
    Experiment,
    HydrodynamicRegime,
    Medium,
    PortTypeMismatchError,
)

_DT = 1e-3


# ── concrete producer / consumer nodes ──────────────────────────────────────

class _ProducerNode(SimulationNode):
    def initial_state(self) -> dict:
        return {"signal": jnp.ones(3)}

    def update(self, state, boundary_inputs, dt) -> dict:
        return state

    def boundary_flux_spec(self) -> dict:
        return {"signal": BoundaryFluxSpec(
            shape=(3,), description="produced signal", output_units="SI")}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt) -> dict:
        return {"signal": state["signal"]}


class _ConsumerNode(SimulationNode):
    def initial_state(self) -> dict:
        return {"echo": jnp.zeros(3)}

    def boundary_input_spec(self) -> dict:
        return {"signal_in": BoundaryInputSpec(
            shape=(3,), default=jnp.zeros(3), description="consumed signal")}

    def update(self, state, boundary_inputs, dt) -> dict:
        return {"echo": boundary_inputs.get("signal_in", jnp.zeros(3))}

    def boundary_flux_spec(self) -> dict:
        return {"echo": BoundaryFluxSpec(
            shape=(3,), description="echo", output_units="SI")}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt) -> dict:
        return {"echo": state["echo"]}


# ── concrete producer / consumer effects ────────────────────────────────────

class _ProducerEffect(BaseEffectModel):
    def __init__(self, node_name="prod", *, field="signal", shape=(3,),
                 additive=False, transform=None):
        self._name = node_name
        self._port = CouplingPort(node=node_name, field=field, shape=shape,
                                  additive=additive, transform=transform)

    @property
    def coupling_ports(self):
        return {"out": self._port}

    def applicable_regime(self):
        return HydrodynamicRegime()

    def build(self, gm, *, body, medium) -> EffectHandle:
        gm.add_node(_ProducerNode(self._name, _DT))
        return EffectHandle(node_names=(self._name,))


class _ConsumerEffect(BaseEffectModel):
    def __init__(self, node_name="cons", *, field="signal_in", shape=(3,),
                 additive=False):
        self._name = node_name
        self.input_ports = {
            "in": CouplingPort(node=node_name, field=field, shape=shape,
                               additive=additive),
        }

    def applicable_regime(self):
        return HydrodynamicRegime()

    def build(self, gm, *, body, medium) -> EffectHandle:
        gm.add_node(_ConsumerNode(self._name, _DT))
        return EffectHandle(node_names=(self._name,))


def _experiment():
    exp = Experiment(name="coupling")
    exp.set_body(Body(name="body", node=None, properties={}))
    exp.set_medium(Medium(properties={}))
    return exp


# ── tests ────────────────────────────────────────────────────────────────

def test_coupling_materialises_edge_and_compiles():
    exp = _experiment()
    p, c = _ProducerEffect(), _ConsumerEffect()
    exp.attach(p, name="p")
    exp.attach(c, name="c")
    exp.couple(c, target_port="in", source=p, source_port="out")
    gm, _ = exp.build()
    edge = [
        e for e in gm._edges
        if e.source_node == "prod" and e.target_node == "cons"
        and e.source_field == "signal" and e.target_field == "signal_in"
    ]
    assert len(edge) == 1


def test_coupling_carries_additive_flag():
    exp = _experiment()
    p = _ProducerEffect(additive=True)
    c = _ConsumerEffect()
    exp.attach(p, name="p")
    exp.attach(c, name="c")
    exp.couple(c, target_port="in", source=p, source_port="out")
    gm, _ = exp.build()
    edge = next(e for e in gm._edges
                if e.source_node == "prod" and e.target_node == "cons")
    assert edge.additive is True


def test_coupling_carries_transform():
    exp = _experiment()
    p = _ProducerEffect(transform=lambda x: -x)
    c = _ConsumerEffect()
    exp.attach(p, name="p")
    exp.attach(c, name="c")
    exp.couple(c, target_port="in", source=p, source_port="out")
    gm, _ = exp.build()
    edge = next(e for e in gm._edges
                if e.source_node == "prod" and e.target_node == "cons")
    assert edge.transform is not None


def test_shape_mismatch_still_raises_at_pass2():
    exp = _experiment()
    p = _ProducerEffect(shape=(3,))
    c = _ConsumerEffect(shape=(4,))
    exp.attach(p, name="p")
    exp.attach(c, name="c")
    exp.couple(c, target_port="in", source=p, source_port="out")
    with pytest.raises(PortTypeMismatchError):
        exp.build()


def test_non_materialisable_port_raises():
    # A coupling port without a node/field binding (a shape-only stub) is
    # caught at materialisation with a clear error.
    class _BadPort:
        shape = (3,)

    class _BadProducer(BaseEffectModel):
        @property
        def coupling_ports(self):
            return {"out": _BadPort()}

        def applicable_regime(self):
            return HydrodynamicRegime()

        def build(self, gm, *, body, medium):
            gm.add_node(_ProducerNode("prod", _DT))
            return EffectHandle(node_names=("prod",))

    exp = _experiment()
    p = _BadProducer()
    c = _ConsumerEffect()
    exp.attach(p, name="p")
    exp.attach(c, name="c")
    exp.couple(c, target_port="in", source=p, source_port="out")
    with pytest.raises(CouplingError, match="not materialisable"):
        exp.build()
