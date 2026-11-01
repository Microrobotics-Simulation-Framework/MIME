"""The body-side laws and the field against their closed forms, each node evaluated directly.

- Gravity: ``(m - rho_f V) g``, the buoyancy acting at the centre of buoyancy, so the couple ``(R_q c) x (-rho_f V g)``.
- The uniform rotating field: constant magnitude, perpendicular to the axis, turning right-handed at the commanded
  frequency, at the time its step count gives.
- The torque: ``m x B``, with ``m`` turned by the body's orientation, and zero when the moment lies along the field;
  both to the rounding of the products, ``4 eps |m| |B|``. Exactly zero is not what XLA's CPU backend computes: it
  fuses each product into the subtraction (an FMA), so ``a x a`` leaves the first product's rounding error, about
  1e-23 for these magnitudes, where NumPy gives zero.

The controls: a torque law planted with its sign flipped fails the magnetic family's sign case and no other case; a
gravity law planted with the couple's sign flipped, its orientation left out, or its offset turned the wrong way is
caught by the couple's check on every draw of a fixed battery.
"""

from dataclasses import dataclass
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mime.build import InteractionBuilder
from mime.core import quaternion
from mime.interactions import MagneticTorque
from mime.nodes.actuation import UniformRotatingNode
from mime.nodes.body import GravityNode
from mime.nodes.magnetic import MagneticTorqueNode
from mime.registry import BuildContext, register_interaction
from mime.testing import CaseKind, conformance
from mime.vocabulary import register_namespace

register_namespace("planted", "0.0.1")

#: The closed forms' own arithmetic, compiled once so that each of Hypothesis's examples costs one call.
exponential, rotation_matrix, rotate = (jax.jit(quaternion.exponential), jax.jit(quaternion.rotation_matrix),
                                        jax.jit(quaternion.rotate))


def vectors(scale: float, size: int = 3):
    """Components up to ``scale``, each zero or above a millionth of it: products of physical magnitudes, never
    within reach of float64's subnormals, where XLA flushes to zero and NumPy does not."""
    component = st.floats(-scale, scale, allow_nan=False).map(lambda x: 0.0 if abs(x) < 1e-6 * scale else x)
    return arrays(np.float64, (size,), elements=component)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


MASS, VOLUME, DENSITY, G = 8.491e-5, 6.022e-8, 1000.0, np.array([0.0, 0.0, -9.81])


def test_gravity_is_weight_minus_buoyancy():
    node = GravityNode("gravity", 1e-3, mass=MASS, buoyant_volume=VOLUME, fluid_density=DENSITY, gravity=G)
    wrench = np.asarray(node.update(node.initial_state(), {}, 1e-3)["wrench"])
    assert np.allclose(wrench[:3], (MASS - DENSITY * VOLUME) * G, rtol=1e-15, atol=0.0)
    assert np.array_equal(wrench[3:], np.zeros(3))
    assert node.boundary_input_spec() == {}


#: One node per law, its drawn constant written through ``params`` as a study would: a node is compiled once.
COUPLED = GravityNode("gravity", 1e-3, mass=MASS, buoyant_volume=VOLUME, fluid_density=DENSITY, gravity=G,
                      centre_of_buoyancy=(0.0, 0.0, 1e-4))
TORQUE = MagneticTorqueNode("torque", 1e-3, moment=(0.0, 1e-3, 0.0))


BUOYANCY = -DENSITY * VOLUME * G

#: The couple's rounding in units of ``eps |c| |B|``, the scale of the terms that cancel in it. The law and the closed
#: form build the same ``R c x B`` with different compiled code (XLA fuses the products into FMAs, NumPy does not), so
#: they differ by rounding alone, and a component of the lever or the couple that is small by cancellation is still
#: rounded at the scale of the terms. A lever component is the dot product of a row of ``R`` (unit length, so
#: ``sum |R_ij c_j| <= |c|``) with ``c``. Two evaluations of a 3-term dot product differ by at most ``2 gamma_3 |c|``,
#: ``3 eps |c|`` (``gamma_3`` = 3 u, u = eps / 2). The entries of ``R`` come from two or three roundings each and may
#: differ by 2 eps between the two, which moves the dot by at most ``2 eps |c|_1 <= 2 sqrt(3) eps |c|``, under 3.5 eps
#: |c|. The cross product with ``B`` adds one rounding, ``eps |lever| |B|`` at most. In all 7.5, taken as 8. Over
#: four million draws the largest difference was 2.7. A relative tolerance on the couple's component judged a 1e-11
#: component, left by cancelling terms of 6e-4, against itself, and failed on about one random draw in 14000.
COUPLE_ROUNDING = 8.0


def couple_error(node: GravityNode, rotation: np.ndarray, centre: np.ndarray) -> float:
    """The node's couple against ``(R c) x B`` in units of ``eps |c| |B|``; also checks the force."""
    q = exponential(jnp.asarray(rotation))
    pose = jnp.concatenate([jnp.asarray([0.1, -0.2, 0.3]), q])
    wrench = np.asarray(node.update(node.initial_state(), {"pose": pose}, 1e-3,
                                    params={"centre_of_buoyancy": jnp.asarray(centre)})["wrench"])
    assert np.allclose(wrench[:3], (MASS - DENSITY * VOLUME) * G, rtol=1e-15, atol=0.0)
    lever = np.asarray(rotation_matrix(q)) @ centre
    difference = float(np.max(np.abs(wrench[3:] - np.cross(lever, BUOYANCY))))
    scale = np.finfo(np.float64).eps * float(np.linalg.norm(centre) * np.linalg.norm(BUOYANCY))
    return 0.0 if difference == 0.0 else difference / scale if scale > 0.0 else np.inf


@settings(deadline=None)
@given(rotation=vectors(3.0), centre=vectors(1e-3))
def test_an_off_centre_buoyancy_gives_the_couple_at_the_bodys_orientation(rotation, centre):
    assert couple_error(COUPLED, rotation, centre) <= COUPLE_ROUNDING


AXIS, START = unit(np.array([0.3, -0.2, 0.9])), np.array([1.0, 0.0, 0.0])


def test_the_field_turns_at_the_commanded_frequency_about_its_axis_at_constant_magnitude():
    dt, frequency, amplitude, steps = 1e-3, 7.0, 1.2e-3, 300
    node = UniformRotatingNode("field", dt, slots=("reader",), rotation_axis=AXIS, initial_direction=START)
    state = node.initial_state()
    commands = {"frequency": jnp.asarray(frequency), "amplitude": jnp.asarray(amplitude)}
    fields = []
    for _ in range(steps):
        state = node.update(state, commands, dt)
        fields.append(np.asarray(state["flux_density_reader"]))
    fields = np.array(fields)
    assert np.allclose(np.linalg.norm(fields, axis=1), amplitude, rtol=1e-14, atol=0.0)
    assert float(np.max(np.abs(fields @ AXIS))) < 1e-15 * amplitude * 10
    first = unit(START - START @ AXIS * AXIS)
    angles = np.unwrap(np.arctan2(fields @ np.cross(AXIS, first), fields @ first))
    times = dt * np.arange(1, steps + 1)
    assert np.allclose(angles, 2 * np.pi * frequency * times, rtol=0.0, atol=1e-11)
    assert int(state["steps"]) == steps
    assert np.array_equal(np.asarray(state["gradient_reader"]), np.zeros((3, 3)))


def test_the_field_keeps_its_time_by_count_not_by_sum():
    """0.2 summed ``dt`` into its time in float32 and drifted by about 1e-4 in phase over 5 s at 10 Hz. Counting
    steps, the phase after 10000 steps is the phase at ``10000 dt`` to the precision of one multiplication."""
    node = UniformRotatingNode("field", 5e-4, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0))
    state = {**node.initial_state(), "steps": jnp.asarray(9999, dtype=jnp.int32)}
    field = np.asarray(node.update(state, {"frequency": jnp.asarray(10.0), "amplitude": jnp.asarray(1.0)},
                                   5e-4)["flux_density_r"])
    assert abs(np.arctan2(field[1], field[0]) - np.mod(2 * np.pi * 10.0 * 10000 * 5e-4 + np.pi, 2 * np.pi)
               + np.pi) < 1e-9


@settings(deadline=None)
@given(rotation=vectors(3.0), field=vectors(1e-2), moment=vectors(2e-3))
def test_the_torque_is_m_cross_b_exactly(rotation, field, moment):
    q = exponential(jnp.asarray(rotation))
    pose = jnp.concatenate([jnp.zeros(3), q])
    wrench = np.asarray(TORQUE.update(TORQUE.initial_state(), {"pose": pose, "flux_density": jnp.asarray(field)},
                                      1e-3, params={"moment": jnp.asarray(moment)})["wrench"])
    turned = np.asarray(rotate(q, jnp.asarray(moment)))
    assert np.array_equal(wrench[:3], np.zeros(3))
    rounding = 4 * np.finfo(np.float64).eps * np.linalg.norm(turned) * np.linalg.norm(field)
    assert float(np.max(np.abs(wrench[3:] - np.cross(turned, field)))) <= rounding


@settings(deadline=None)
@given(rotation=vectors(3.0), moment=vectors(2e-3), scale=st.sampled_from([0.5, 2.0, -4.0]))
def test_the_torque_is_zero_when_the_moment_lies_along_the_field(rotation, moment, scale):
    q = exponential(jnp.asarray(rotation))
    field = scale * rotate(q, jnp.asarray(moment))
    wrench = np.asarray(TORQUE.update(TORQUE.initial_state(), {"pose": jnp.concatenate([jnp.zeros(3), q]),
                                                               "flux_density": field}, 1e-3,
                                      params={"moment": jnp.asarray(moment)})["wrench"])
    rounding = 4 * np.finfo(np.float64).eps * float(np.linalg.norm(field)) * float(np.linalg.norm(moment))
    assert np.array_equal(wrench[:3], np.zeros(3)) and float(np.max(np.abs(wrench[3:]))) <= rounding


# -- the control: a planted sign flip ---------------------------------------------------------------------------------


class _FlippedNode(MagneticTorqueNode):
    @override
    def update(self, state: Any, boundary_inputs: Any, dt: float, *, params: Any = None) -> Any:
        return {"wrench": -super().update(state, boundary_inputs, dt, params=params)["wrench"]}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class FlippedTorque(MagneticTorque):
    """``tau = -m x B``: a torque that turns the moment away from the field."""

    name = "planted.flipped_torque"

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(_FlippedNode(ctx.node_name(), ctx.timestep, moment=ctx.vector_value("moment")))
        b.connect(b.read("pose"), into=node["pose"])
        b.connect(b.read("flux_density"), into=node["flux_density"])
        b.connect(node["wrench"], into=b.target("wrench"))


def flipped_oracle(read, settings):
    pose = np.asarray(read["pose"])
    moment = np.asarray(quaternion.rotate(jnp.asarray(pose[3:]), jnp.asarray(settings["moment"])))
    return {"wrench": np.concatenate([np.zeros(3), -np.cross(moment, np.asarray(read["flux_density"]))])}


def test_a_planted_sign_flip_fails_the_magnetic_familys_sign_case_and_only_it():
    """Given its own oracle, so that the oracle agrees with it, the flipped torque passes every case of the magnetic
    family but the sign case."""
    results = conformance(FlippedTorque(), "magnetic", oracle=flipped_oracle,
                          cases=[c for c in CaseKind if c is not CaseKind.NODES and c in (
                              CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE,
                              CaseKind.GRADIENTS)])
    print("\n" + "\n".join(f"{case.value}: {result}" for case, result in results.items()))
    assert {case for case, result in results.items() if not result.passed} == {CaseKind.SIGN}


class _SignFlippedCouple(GravityNode):
    @override
    def wrench(self, p: Any, pose: Any) -> Any:
        return super().wrench(p, pose).at[3:].multiply(-1.0)


class _OrientationLeftOut(GravityNode):
    @override
    def wrench(self, p: Any, pose: Any) -> Any:
        return super().wrench(p, pose.at[3:].set(jnp.asarray([1.0, 0.0, 0.0, 0.0])))


class _OffsetInTheWorldFrameTurnedBack(GravityNode):
    @override
    def wrench(self, p: Any, pose: Any) -> Any:
        return super().wrench(p, pose.at[3:].set(quaternion.conjugate(pose[3:])))


PLANTED_COUPLES = [_SignFlippedCouple, _OrientationLeftOut, _OffsetInTheWorldFrameTurnedBack]


def test_a_planted_error_in_the_couple_fails_the_check_on_every_draw_of_a_fixed_battery():
    """Each plant moves the couple by its own size, 1e12 times ``COUPLE_ROUNDING`` at the battery's smallest. On the
    property's own draws, with their zero components, the sign flip is missed where the couple is zero (the offset
    along the buoyancy, or none), and the other two where the rotation leaves the offset in place: a battery of
    generic draws is what makes this control fail on every run."""
    rng = np.random.default_rng(20261101)
    battery = [(rng.uniform(-3.0, 3.0, 3), rng.uniform(-1e-3, 1e-3, 3)) for _ in range(64)]
    for planted in PLANTED_COUPLES:
        node = planted("gravity", 1e-3, mass=MASS, buoyant_volume=VOLUME, fluid_density=DENSITY, gravity=G,
                       centre_of_buoyancy=(0.0, 0.0, 1e-4))
        errors = [couple_error(node, rotation, centre) for rotation, centre in battery]
        assert min(errors) > 1e3 * COUPLE_ROUNDING, (planted.__name__, min(errors))
