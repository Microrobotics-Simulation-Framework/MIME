"""The build's structural flags (F0-1 M6.5), each with its control, in every profile and with or without the energy
ledger: a law joining two bodies that step at different rates is refused (``law.rate-divided``), and builds at one
rate; a damper between two bodies, neither held in one step nor solved with them in a coupling group, is flagged by a
line the build computes on it, and is not flagged in a group; a coupling group stopped at its cap is reported on the
run's result from MADDENING's own verdict on every run (``RunResult.groups``), and a converged one as converged.

The laws are this file's own, between two of MIME's rigid bodies: a spring between their centres, ``f = -k (x_a - x_b
- rest)``, and a damper, ``f = -c (v_a - v_b)``, each delivering ``f`` to the first and ``-f`` to the second (CG's pair,
F0's spike, at its mild regime)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, override

import pytest

from mime import Body, Experiment, State
from mime.build import BuildRefused
from mime.core import MimeNode, NodeFields, as_run_array, jnp, node_input
from mime.dynamics import RigidInertial
from mime.gradients import GradientStatus
from mime.observations import EnergyLedger
from mime.registry import (
    BuildContext,
    Delivery,
    GroupRequest,
    Interaction,
    Plan,
    PlanContext,
    Read,
    RoleKind,
    register_interaction,
)
from mime.scene import Scene, Z
from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, ms, rad_per_s, s
from mime.vocabulary import APPLIED_WRENCH, POSE, TWIST, EnergyClass, register_namespace

register_namespace("flagged", "0.0.1")

MASS, INERTIA = 8.491116020725203e-05, 1.0331989974018431e-10
#: CG's mild regime at dt 0.5 ms for the reduced mass: omega0 dt 0.05, damping ratio 0.05.
OMEGA0 = 0.05 / 5e-4
STIFFNESS, DAMPING = (MASS / 2) * OMEGA0**2, 2 * 0.05 * (MASS / 2) * OMEGA0
REST = (2e-3, 0.0, 0.0)


class PairNode(MimeNode):
    """``f`` on the first body and ``-f`` on the second: a spring between their centres (reading their poses) or a
    damper (reading their twists)."""

    def __init__(self, name: str, timestep: float, *, kind: str, coefficient: float) -> None:
        super().__init__(name, timestep, kind=kind, coefficient=float(coefficient))

    def _width(self) -> int:
        return 7 if self.constants()["kind"] == "spring" else 6

    @override
    def initial_state(self) -> NodeFields:
        return {"on_first": as_run_array(jnp.zeros(6)), "on_second": as_run_array(jnp.zeros(6))}

    @override
    def boundary_input_spec(self) -> dict[str, Any]:
        units = "m; 1" if self._width() == 7 else "m/s; rad/s"
        return {"first": node_input(units, (self._width(),)), "second": node_input(units, (self._width(),))}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Any],
                dt: float) -> NodeFields:
        a, b = boundary_inputs["first"], boundary_inputs["second"]
        moved = a[:3] - b[:3] - (jnp.asarray(REST, dtype=a.dtype) if self._width() == 7 else 0.0)
        force = -p["coefficient"] * moved
        zero = jnp.zeros_like(force)
        return {"on_first": jnp.concatenate([force, zero]), "on_second": jnp.concatenate([-force, zero])}


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Pair(Interaction):
    differentiability = GradientStatus.EXACT
    builds = {"": (PairNode,)}
    roles = {"first": RoleKind.BODY, "second": RoleKind.BODY}
    delivers = {"on_first": Delivery("first", APPLIED_WRENCH.name), "on_second": Delivery("second", APPLIED_WRENCH.name)}
    #: The cap of a coupling group the pair asks for its two bodies and itself, or none.
    together: int | None = None
    kind = "spring"

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        groups = () if self.together is None else (
            GroupRequest(("first", "second"), "the pair solved together", 1e-12, self.together, own_nodes=True),)
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers), groups=groups)

    @override
    def build(self, ctx: BuildContext, b: Any) -> None:
        node = b.add_response_node(PairNode(ctx.node_name(), ctx.timestep, kind=self.kind,
                                            coefficient=STIFFNESS if self.kind == "spring" else DAMPING))
        b.connect(b.read("first"), into=node["first"])
        b.connect(b.read("second"), into=node["second"])
        b.connect(node["on_first"], into=b.target("on_first"))
        b.connect(node["on_second"], into=b.target("on_second"))


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class PairSpring(_Pair):
    name = "flagged.pair_spring"
    version = "0.1.0"
    energy_class = EnergyClass.POTENTIAL
    reads = {"first": Read("first", POSE.name), "second": Read("second", POSE.name)}


@register_interaction
@dataclasses.dataclass(frozen=True, kw_only=True)
class PairDamper(_Pair):
    name = "flagged.pair_damper"
    version = "0.1.0"
    energy_class = EnergyClass.DISSIPATIVE
    kind = "damper"
    reads = {"first": Read("first", TWIST.name), "second": Read("second", TWIST.name)}


def two_bodies(law: Interaction, *, rate: Any = None, profile: str = "research", ledger: bool = False) -> Experiment:
    """Two of MIME's free rigid bodies 2 mm apart along x and moving apart, ``law`` between them, the second stepping
    every ``rate`` (every step when not given)."""
    exp = Experiment("two_bodies", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(0.5),
                     profile=profile)
    for name, x, v in (("a", 1e-3, 1e-3), ("b", -1.2e-3, -1e-3)):
        dynamics = RigidInertial(rate=rate) if name == "b" and rate is not None else RigidInertial()
        exp.add_body(Body(name, dynamics=dynamics, initial=State(
            position=m((x, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((v, 0.0, 0.0)),
            angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare(f"{name}.mass", kg(MASS), reason="script A's body")
        exp.declare(f"{name}.inertia", kg_m2((INERTIA,) * 3), reason="script A's body, isotropic")
    exp.attach(law, first="a", second="b")
    if ledger:
        exp.observe(EnergyLedger())
    return exp


@pytest.mark.parametrize("profile", ["exploratory", "research"])
@pytest.mark.parametrize("ledger", [False, True])
def test_a_spring_across_a_rate_divider_is_refused_and_builds_at_one_rate(profile, ledger):
    with pytest.raises(BuildRefused) as refused:
        two_bodies(PairSpring(), rate=ms(1.0), profile=profile, ledger=ledger).build()
    assert refused.value.codes() == ["law.rate-divided"], refused.value
    assert "b every 2 steps" in str(refused.value)
    two_bodies(PairSpring(), profile=profile, ledger=ledger).build()


def _flag_lines(built: Any) -> list[str]:
    return [line for owner, lines in built.report.checked.items() if owner.startswith("flagged.pair_damper")
            for line in lines]


def test_a_damper_between_two_receivers_is_flagged_and_in_a_group_is_not():
    loose = two_bodies(PairDamper()).build()
    (line,) = _flag_lines(loose)
    assert line.startswith("a damper between a and b, neither held in one step nor solved with them"), line
    assert "a damper between" in loose.report.text(details=True)
    assert not _flag_lines(two_bodies(PairDamper(together=50)).build())
    assert not _flag_lines(two_bodies(PairSpring()).build()), "a spring is no damper"


@pytest.mark.parametrize("ledger", [False, True])
def test_a_group_stopped_at_its_cap_is_reported_on_every_run_and_a_converged_one_as_converged(ledger):
    capped = two_bodies(PairSpring(together=1), ledger=ledger).build()
    converged = two_bodies(PairSpring(together=50), ledger=ledger).build()
    for _ in range(3):
        (verdict,) = capped.run(s(0.01)).groups.values()
        assert verdict.startswith("not converged at its last step after 1 passes"), verdict
        assert list(converged.run(s(0.01)).groups.values()) == ["converged"]
    result = capped.run(s(0.01))
    assert "coupling group a+b+flagged.pair_spring" in result.text() or "coupling group" in result.text()
    if ledger:
        account = result.energy
        assert account is not None and len(account.groups) == 1
        assert "stopped at its cap" in next(iter(account.groups.values()))
    assert two_bodies(PairSpring()).build().run(s(0.01)).groups == {}
