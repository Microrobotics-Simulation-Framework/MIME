"""The family suites' control. The toy family's laws (the toy drag, both rungs) conform; and a law planted with one
kind of defect fails the case that exists to catch it. The planted sign error runs every case, so that the sign case
is seen to be the one that tells; each other planted law runs the case it plants a defect for.

The planted laws are registered here, under their own namespace, so the registry walk (which lists the entries a
fresh process registers) does not hold them to round-trip or node cases.
"""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, override

import jax
import jax.numpy as jnp
import pytest

from mime.build import InteractionBuilder
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import BuildContext, Delivery, Interaction, Read, RoleKind, register_interaction
from mime.testing import CaseKind, Family, NodeCase, Response, assert_conforms, conformance, register_family
from mime.testing.toys import BODY_VELOCITY, LOAD, POSITION, PROBE_START, ToyBody, ToyDrag, ToySpring, toy_experiment
from mime.testing.toys._nodes import ToyDragNode, ToyLoad, _input
from mime.things import Body, State
from mime.units import Force, Length, Mass, N, Quantity, Time, m, m_per_s
from mime.vocabulary import register_namespace

register_namespace("conforming", "0.0.1")


def linear(read, settings):
    return {"load": -settings["c"] * read["v"]}


def quadratic(read, settings):
    v = read["v"]
    return {"load": -settings["c"] * v * jnp.linalg.norm(v)}


def failed(results) -> set[CaseKind]:
    return {case for case, result in results.items() if not result.passed}


def test_the_toy_drags_linear_rung_conforms():
    assert_conforms(ToyDrag.Linear(), "toy", oracle=linear, swap_with=ToyDrag.Quadratic())


def test_the_toy_drags_quadratic_rung_conforms():
    assert_conforms(ToyDrag.Quadratic(), "toy", oracle=quadratic)


# -- planted laws ------------------------------------------------------------------------------------------------


class _BackwardsNode(ToyDragNode):
    @override
    def law(self, p, given):
        return ToyLoad(load=-super().law(p, given)["load"])


class _ComponentwiseNode(ToyDragNode):
    """The toy drag's quadratic rung at version 0.1.0: each component's magnitude, not the velocity's norm."""

    @override
    def law(self, p, given):
        v = given.get("v", jnp.zeros(3))
        return ToyLoad(load=-p["c"] * v * jnp.abs(v))


class _StoppedNode(ToyDragNode):
    @override
    def law(self, p, given):
        return ToyLoad(load=-p["c"] * jax.lax.stop_gradient(given.get("v", jnp.zeros(3))))


class _LogNode(ToyDragNode):
    @override
    def law(self, p, given):
        v = given.get("v", jnp.zeros(3))
        return ToyLoad(load=-p["c"] * v * jnp.log(jnp.abs(v)))


class _MislabelledNode(ToyDragNode):
    @override
    def boundary_input_spec(self):
        return {"v": _input("m")}


@dataclass(frozen=True, kw_only=True)
class _PlantedDrag(Interaction):
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    reads = {"v": Read("body", BODY_VELOCITY.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    NODE: ClassVar[Any] = ToyDragNode
    c: Annotated[Quantity, Mass / Time] = N(0.37) / m_per_s(1.0)

    def coefficient(self, ctx: BuildContext) -> float:
        return ctx.scalar("c")

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(self.NODE(ctx.node_name(), ctx.timestep, c=self.coefficient(ctx), rung="linear"))
        b.connect(b.read("v"), into=node["v"])
        b.connect(node["load"], into=b.target("load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class BackwardsDrag(_PlantedDrag):
    """The planted sign error: a load along the velocity."""

    name = "conforming.backwards_drag"
    NODE = _BackwardsNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ComponentwiseDrag(_PlantedDrag):
    name = "conforming.componentwise_drag"
    NODE = _ComponentwiseNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class StoppedDrag(_PlantedDrag):
    name = "conforming.stopped_drag"
    NODE = _StoppedNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class LogDrag(_PlantedDrag):
    name = "conforming.log_drag"
    NODE = _LogNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class MislabelledDrag(_PlantedDrag):
    name = "conforming.mislabelled_drag"
    NODE = _MislabelledNode


@register_interaction
@dataclass(frozen=True, kw_only=True)
class BakedDrag(_PlantedDrag):
    """Builds its node with the default coefficient whatever its setting says."""

    name = "conforming.baked_drag"

    @override
    def coefficient(self, ctx: BuildContext) -> float:
        return 0.37


def test_the_planted_sign_error_fails_the_sign_case_and_only_it():
    """The gate's control: a load along the velocity, with its own oracle (so the oracle agrees with it), passes
    every case of the toy family but the sign case."""
    results = conformance(BackwardsDrag(), "toy", oracle=lambda read, settings: {"load": settings["c"] * read["v"]})
    print("\n" + "\n".join(f"{case.value}: {result}" for case, result in results.items()))
    assert failed(results) == {CaseKind.SIGN}
    sign = results[CaseKind.SIGN]
    assert sign.failed and "'sign' violated" in sign.detail and "boundary_inputs" in sign.counterexample
    with pytest.raises(AssertionError, match="does not conform to the toy family:\nsign: sign: FAIL"):
        assert_conforms(BackwardsDrag(), "toy", cases=[CaseKind.SIGN])


@pytest.mark.parametrize(("law", "case"), [
    (ComponentwiseDrag(), CaseKind.FRAME),
    (StoppedDrag(), CaseKind.GRADIENTS),
    (BakedDrag(), CaseKind.MATERIAL_REACH),
    (MislabelledDrag(), CaseKind.UNITS),
    (LogDrag(), CaseKind.NODES),
    (ToySpring(), CaseKind.SWAP),
], ids=lambda value: value.value if isinstance(value, CaseKind) else type(value).__name__)
def test_each_planted_defect_fails_its_case(law, case):
    (result,) = conformance(law, "toy", cases=[case]).values()
    print(f"\n{result}")
    assert result.failed


def test_a_wrong_oracle_fails_the_oracle_case_and_a_missing_one_is_not_a_pass():
    def twice(read, settings):
        return {"load": -2 * settings["c"] * read["v"]}

    (wrong,) = conformance(ToyDrag.Linear(), "toy", oracle=twice, cases=[CaseKind.ORACLE]).values()
    (missing,) = conformance(ToyDrag.Linear(), "toy", cases=[CaseKind.ORACLE]).values()
    assert wrong.failed and missing.failed and "pass oracle=" in missing.detail


def test_a_law_that_reads_nothing_the_sign_rule_names_fails_the_sign_case_without_sampling():
    (result,) = conformance(ToySpring(), "toy", cases=[CaseKind.SIGN]).values()
    assert result.failed and result.n_examples == 0 and "reads no toy.body.velocity of its body" in result.detail


# -- equal and opposite, on a family of two-body laws -------------------------------------------------------------


class _TetherNode(ToyDragNode):
    """``load_a = -k (x_a - x_b)``, and ``load_b`` its reaction (or, planted, the same load again)."""

    def __init__(self, name, timestep, k=1.0, reaction=-1.0):
        super(ToyDragNode, self).__init__(name, timestep, k=k, reaction=reaction)

    @override
    def initial_state(self):
        return {"load_a": jnp.zeros(3), "load_b": jnp.zeros(3)}

    @override
    def boundary_input_spec(self):
        return {"x_a": _input("m"), "x_b": _input("m")}

    @override
    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        load = -p["k"] * (boundary_inputs.get("x_a", jnp.zeros(3)) - boundary_inputs.get("x_b", jnp.zeros(3)))
        return {"load_a": load, "load_b": p["reaction"] * load}


@dataclass(frozen=True, kw_only=True)
class _Tether(Interaction):
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY, "other": RoleKind.BODY}
    reads = {"x_a": Read("body", POSITION.name), "x_b": Read("other", POSITION.name)}
    delivers = {"load_a": Delivery("body", LOAD.name), "load_b": Delivery("other", LOAD.name)}
    k: Annotated[Quantity, Force / Length] = N(1.0) / m(1.0)
    REACTION: ClassVar[float] = -1.0

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(_TetherNode(ctx.node_name(), ctx.timestep, k=ctx.scalar("k"),
                                               reaction=self.REACTION))
        for label in ("x_a", "x_b"):
            b.connect(b.read(label), into=node[label])
        for label in ("load_a", "load_b"):
            b.connect(node[label], into=b.target(label))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Tether(_Tether):
    name = "conforming.tether"


@register_interaction
@dataclass(frozen=True, kw_only=True)
class OneSidedTether(_Tether):
    """The planted defect: the same load on both bodies, which a two-way law never delivers."""

    name = "conforming.one_sided_tether"
    REACTION = 1.0


def pair_rig(law: Interaction) -> Experiment:
    experiment = toy_experiment("pair_rig")
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START))
    experiment.add_body(Body("other", dynamics=ToyBody(), initial=State(position=m((0.0, 0.1, 0.0)),
                                                                        velocity=m_per_s((0.0, 0.0, 0.0)))))
    experiment.attach(law, body="probe", other="other")
    return experiment


PAIRS = register_family(Family(
    name="conforming.pair", rig=pair_rig, cases=(CaseKind.EQUAL_AND_OPPOSITE, CaseKind.FRAME, CaseKind.GRADIENTS),
    pairs=((("body", LOAD.name), ("other", LOAD.name)),)))


def test_a_two_way_law_delivers_equal_and_opposite_and_the_one_sided_one_does_not():
    assert_conforms(Tether(), "conforming.pair")
    results = conformance(OneSidedTether(), "conforming.pair")
    assert failed(results) == {CaseKind.EQUAL_AND_OPPOSITE}


def test_a_response_reads_and_delivers_by_role_and_interface():
    response = Response({"v": jnp.ones(3)}, {"load": -jnp.ones(3)}, {"c": 1.0},
                        {("body", BODY_VELOCITY.name): "v"}, {("body", LOAD.name): "load"})
    assert float(response.reading("body", BODY_VELOCITY.name)[0]) == 1.0
    assert float(response.delivery("body", LOAD.name)[0]) == -1.0


def test_the_nodes_case_takes_the_laws_node_case():
    """Options for verify_node reach the battery: bounds that keep the log drag's velocity away from zero (where
    its load is not finite) let it pass the nodes case its default box fails."""
    case = NodeCase(options={"boundary_bounds": {"v": (0.5, 2.0)}})
    (result,) = conformance(LogDrag(), "toy", cases=[CaseKind.NODES], node_case=case).values()
    assert result.passed, result
