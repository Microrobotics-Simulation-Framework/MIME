"""Nodes never hard-code an axis, tested on the whole experiment: 4b's body side (the rotating field and the torque,
gravity with its couple, contact against the tube, the swim speed) built in a world turned by a rotation ``Q`` that
Hypothesis draws, with up and every declared direction, position and velocity turned by ``Q``, runs as the first
world turned by ``Q``. Poses and vectors turn by ``Q``, the gradient as a tensor, and scalars (the swim speed along
``tube.axis``, the drives) are unchanged. The body's inertia, moment and centre of buoyancy are declared in its own
frame and are not turned; the inertia is anisotropic and the centre of buoyancy off the centre of mass, so the
gyroscopic term and gravity's couple both act.

**The tolerance, from float rounding.** In float64 each step's arithmetic rounds differently in the turned frame, by a
few ``eps`` of each quantity's scale, and the difference grows at most about linearly over the steps. Measured over 40
rotations at 200 steps, the largest was 1.1 ``eps`` per step (the swim speed, a projection of the velocity, judged on
the velocity's scale); the state and the loads stayed below 0.25 ``eps`` per step. The tolerance is 16 ``eps`` per
step, relative to the scale each value came from.

**The control**: a test-only law that pushes along +z, with the axis written in its code, attached in both worlds.
Hypothesis finds a rotation that shows it, and on fixed rotations the planted error is measured above 1e9 times the
tolerance, so it fails every run, not most."""

from dataclasses import dataclass
from typing import Any, override

import numpy as np
import pytest
from hypothesis import HealthCheck, Phase, find, given, settings
from hypothesis import strategies as st
from rigid_oracles import ConstantNode

from mime import Apparatus, Body, Experiment, Medium, Scene, State, Structure
from mime.build import InteractionBuilder
from mime.dynamics import RigidInertial
from mime.fields import UniformRotatingField
from mime.gradients import GradientStatus
from mime.interactions import Contact, Gravity, Magnetic
from mime.observations import SwimSpeed
from mime.registry import BuildContext, Delivery, Interaction, RoleKind, register_interaction
from mime.scene import Tube
from mime.units import A_m2, Hz, kg, kg_m2, kg_per_m3, m, m3, m_per_s, m_per_s2, mm, mT, rad_per_s, s
from mime.vocabulary import APPLIED_WRENCH, register_namespace

STEPS, DT = 200, 5e-4
EPS = float(np.finfo(np.float64).eps)
TOLERANCE = 16 * EPS * STEPS
IDENTITY = (1.0, 0.0, 0.0, 0.0)
#: The body's orientation in the first world, and where it starts relative to the tube's centre: inside the lumen,
#: 0.85 mm off the axis against a contact radius of 1.0002 mm, so gravity brings it onto the wall within the run.
START_ORIENTATION = np.array([0.9, 0.1, -0.3, 0.2]) / np.linalg.norm([0.9, 0.1, -0.3, 0.2])
CENTRE, OFFSET = np.array([0.002, -0.001, 0.0005]), np.array([0.01, 0.0003, -0.0008])
#: Hypothesis's default example count, kept: each example builds, compiles and runs a fresh graph.
PROPERTY = settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])

register_namespace("frames", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Updraft(Interaction):
    """Planted: a lift written with +z in its code, the defect the turned world exists to catch."""

    name = "frames.updraft"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    delivers = {"wrench": Delivery("body", APPLIED_WRENCH.name)}

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(ConstantNode(ctx.node_name(), ctx.timestep, value=(0.0, 0.0, 4e-4, 0.0, 0.0, 0.0),
                                                shape=(6,)))
        b.connect(node["value"], into=b.target("wrench"))


def matrix(q: Any) -> np.ndarray:
    """The rotation matrix of each unit quaternion ``(w, x, y, z)`` along the last axis."""
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    rows = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return np.moveaxis(np.array(rows), (0, 1), (-2, -1))


def product(a: Any, b: Any) -> np.ndarray:
    """The Hamilton product ``a b`` of quaternions ``(w, x, y, z)``, ``b`` along its last axis."""
    w1, x1, y1, z1 = np.asarray(a, dtype=np.float64)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def body_side(q: Any, *, planted: bool = False) -> Experiment:
    """The body side in the world turned by the unit quaternion ``q``."""
    turn = matrix(q)

    def turned(vector: Any) -> tuple[float, float, float]:
        x, y, z = (float(c) for c in turn @ np.asarray(vector, dtype=np.float64))
        return (x, y, z)

    experiment = Experiment("turned", scene=Scene.new(up=turned((0, 0, 1)), units="m"), gravity=m_per_s2(9.81),
                            dt=s(DT))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(
        axis=turned((1, 0, 0)), inner_radius=mm(3.175), wall=mm(1.6), length=mm(100), centre=m(turned(CENTRE)))))
    water = experiment.add_medium(Medium("water", fills=tube.lumen))
    w, x, y, z = (float(c) for c in product(q, START_ORIENTATION))
    umr = experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m(turned(CENTRE + OFFSET)), orientation=(w, x, y, z), velocity=m_per_s(turned((1e-3, 2e-4, -1e-4))),
        angular_velocity=rad_per_s(turned((3.0, -1.0, 2.0))))))
    field = experiment.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotatingField(rotation_axis=tube.axis, initial_direction=turned((0, 1, 0)))},
        commands={"frequency": Hz(10.0), "amplitude": mT(1.2)}))
    for path, value in (("umr.mass", kg(8.49e-5)), ("umr.inertia", kg_m2((1.0e-10, 1.3e-10, 0.8e-10))),
                        ("umr.buoyant_volume", m3(6.0e-8)), ("umr.magnetic_moment", A_m2((0.2e-3, 1.68e-3, -0.1e-3))),
                        ("umr.extent", mm(2.07)), ("umr.centre_of_buoyancy", m((1e-5, -2e-5, 3e-5))),
                        ("water.density", kg_per_m3(1000.0))):
        experiment.declare(path, value, reason="the turned world's body, in its own frame")
    experiment.attach(Magnetic.Torque(), body=umr, field=field.magnetic)
    experiment.attach(Gravity(), body=umr, medium=water)
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.observe(SwimSpeed(umr, along=tube.axis))
    if planted:
        experiment.attach(Updraft(), body=umr)
    return experiment


type History = dict[str, dict[str, np.ndarray]]


def history(experiment: Experiment) -> History:
    found = experiment.build().run(s(STEPS * DT), history=True).history
    assert found is not None
    return found


def turned_history(found: History, q: Any) -> History:
    """``found`` as the turned world should give it: positions, vectors and wrenches turned by ``q``, orientations
    composed with it, a gradient turned as a tensor, scalars unchanged."""
    turn = matrix(q)
    out: History = {}
    for node, fields in found.items():
        out[node] = {}
        for name, value in fields.items():
            held = np.asarray(value, dtype=np.float64)
            if name == "pose":
                out[node][name] = np.concatenate([held[:, :3] @ turn.T, product(q, held[:, 3:])], axis=1)
            elif held.ndim == 2 and held.shape[1] in (3, 6):
                out[node][name] = np.concatenate([part @ turn.T for part in np.split(held, held.shape[1] // 3, 1)], 1)
            elif held.ndim == 3:
                out[node][name] = np.einsum("ij,tjk,lk->til", turn, held, turn)
            else:
                out[node][name] = held
    return out


def discrepancies(found: History, expected: History) -> dict[str, float]:
    """Each field's largest difference, relative to the scale it came from: its own largest magnitude, but the
    body's velocity for the swim speed and its mean (projections of it) and that over the run for the distance.
    Orientations are compared as rotation matrices, so a quaternion's sign does not count."""
    velocity = float(np.max(np.abs(expected["umr"]["twist"][:, :3])))
    scales = {("swim_speed:umr", "speed"): velocity, ("swim_speed:umr", "mean_speed"): velocity,
              ("swim_speed:umr", "distance"): velocity * STEPS * DT}
    out: dict[str, float] = {}
    for node, fields in expected.items():
        for name, wanted in fields.items():
            got = np.asarray(found[node][name], dtype=np.float64)
            parts = [(name, got, wanted)]
            if name == "pose":
                parts = [("position", got[:, :3], wanted[:, :3]),
                         ("orientation", matrix(got[:, 3:]), matrix(wanted[:, 3:]))]
            for label, a, b in parts:
                scale = scales.get((node, label), float(np.max(np.abs(b))))
                difference = float(np.max(np.abs(a - b)))
                out[f"{node}.{label}"] = difference / scale if scale > 0.0 else difference
    return out


def failures(q: Any, reference: History, *, planted: bool = False) -> list[str]:
    found = discrepancies(history(body_side(q, planted=planted)), turned_history(reference, q))
    return [f"{field}: {value:.3e} of its scale, above {TOLERANCE:.3e}" for field, value in sorted(found.items())
            if not value <= TOLERANCE]


def unit_quaternion(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = float(np.linalg.norm(values))
    w, x, y, z = (c / norm for c in values)
    return (w, x, y, z)


#: A rotation: four components in [-1, 1] at least half a unit long, normalised, so no draw is a near-zero
#: quaternion whose square underflows.
rotations = st.tuples(*[st.floats(-1.0, 1.0)] * 4).filter(lambda v: float(np.linalg.norm(v)) >= 0.5).map(
    unit_quaternion)


@pytest.fixture(scope="module")
def reference() -> History:
    return history(body_side(IDENTITY))


@pytest.fixture(scope="module")
def planted_reference() -> History:
    return history(body_side(IDENTITY, planted=True))


@PROPERTY
@given(q=rotations)
def test_the_body_side_in_a_turned_world_runs_as_the_first_world_turned(q, reference):
    assert failures(q, reference) == []


def test_a_law_with_plus_z_in_its_code_fails_the_turned_world(planted_reference):
    found = find(rotations, lambda q: bool(failures(q, planted_reference, planted=True)),
                 settings=settings(PROPERTY, database=None, phases=[Phase.generate]))
    print(f"Hypothesis's rotation: {found}\n" + "\n".join(failures(found, planted_reference, planted=True)))
    rng = np.random.default_rng(20261023)
    for _ in range(4):
        q = unit_quaternion(tuple(float(c) for c in rng.normal(size=4)))  # type: ignore[arg-type]
        found_errors = discrepancies(history(body_side(q, planted=True)), turned_history(planted_reference, q))
        assert max(found_errors.values()) > 1e9 * TOLERANCE, (q, found_errors)


def test_without_the_planted_law_a_turn_about_up_and_the_identity_change_nothing_but_rounding(reference):
    """The turned world's own control: the identity reproduces the first world bit for bit, so the comparison
    measures the turn and nothing else."""
    found = discrepancies(history(body_side(IDENTITY)), turned_history(reference, IDENTITY))
    assert max(found.values()) == 0.0
