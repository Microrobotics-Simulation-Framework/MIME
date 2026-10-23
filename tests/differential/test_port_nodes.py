"""Node-level differentials against the 0.2 port (optional in phase 4b; the weakest oracle). The port's nodes ran once,
in the port's own environment, on fixed random inputs (``generate_port_references.py``; ``references/`` holds the
outputs and their provenance: MIME-port041 at 6485516, MADDENING 0.4.1, float64, CPU); MIME 0.3's nodes run here on
the same inputs, mapped onto 0.3's ports. The port's ``mime`` is never imported beside this one.

**Declared before the first run.** The two sides compute the same formulas in other orders, so every field agrees
within 1e-12 of its largest magnitude, and these differences are the only ones allowed, each a declared fix or a
change of scope:

- the torque law delivers no force (a field gradient's force is another law): compared with a zero gradient, where
  the port's force is zero too;
- the confinement acts on the tube's inner surface, not its ends: the bodies are placed away from the ends, where the
  port's axial clamp does nothing;
- the quaternion's increment is finite with finite derivatives at rest, a fix to derivatives, not to values: four
  bodies at rest are compared like the others;
- the field keeps its time by count: the port (unlike the 0.2 tag) already did, so its values agree too.

**Found on the first run, not declared before it**, and traced to the port (the old code, the weakest oracle): each
is logged as an anomaly of the 0.2 line, and the 0.3 node agrees with its analytic oracle (``tests/oracles``).

- MIME-ANO-110. The port's permanent moment is float32 under x64 (its moment axis is cast to float32, and the moment
  takes its dtype): its torque is off by the moment's float32 rounding, 7.9e-9 relative. With the moment rounded as
  the port rounds it, the two agree to 1.6e-16.
- MIME-ANO-111. Where the port projects a body back into the vessel, it removes ``(R / r)^2`` of the outward velocity
  rather than all of it, ``r`` being the radius before the projection: its unit normal is the projected position
  divided by the unprojected radius. That model matches the port to 1e-18; 0.3 removes all of it, as the
  ``remove_outward`` rule and the projection property require. Elsewhere the two agree to 1e-18.

What the two share: the formulas, the units (SI; the port's field strength is in mT), and the declared values of A's
reference. The control: a planted mutation (the moment reversed; the mass off by one part in a million) is caught.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.actuation import UniformRotatingFieldNode
from mime.nodes.body import GravityNode, RigidInertialNode
from mime.nodes.magnetic import MagneticTorqueNode

PORT = np.load(Path(__file__).parent / "references" / "port_nodes.npz")
DT = 5e-4
M_EFF, I_EFF, MOMENT, CONSTRAINT_RADIUS = 8.491116020725203e-05, 1.0331989974018431e-10, 2 * 8.4e-4, 1.0002e-3
RTOL = 1e-12


def moved(found: np.ndarray, port: np.ndarray) -> float:
    """The largest difference, relative to the port's largest magnitude in the field."""
    return float(np.max(np.abs(np.asarray(found) - port))) / float(np.max(np.abs(port)))


def field_vectors(node: UniformRotatingFieldNode) -> np.ndarray:
    initial = node.initial_state()
    return np.array([np.asarray(node.update({**initial, "steps": jnp.asarray(s, dtype=jnp.int32)},
                                            {"frequency": jnp.asarray(f), "amplitude": jnp.asarray(b * 1e-3)},
                                            DT)["flux_density_r"])
                     for s, f, b in zip(PORT["field_steps"], PORT["field_frequency_hz"], PORT["field_strength_mt"],
                                        strict=True)])


def torques(node: MagneticTorqueNode) -> np.ndarray:
    return np.array([np.asarray(node.update(node.initial_state(), {
        "pose": jnp.concatenate([jnp.zeros(3), jnp.asarray(q)]), "flux_density": jnp.asarray(b)}, DT)["wrench"])
        for q, b in zip(PORT["magnet_orientation"], PORT["magnet_field"], strict=True)])


def body_steps(node: RigidInertialNode) -> dict[str, np.ndarray]:
    out = [node.update({"pose": jnp.concatenate([jnp.asarray(p), jnp.asarray(o)]),
                        "twist": jnp.concatenate([jnp.asarray(v), jnp.asarray(w)])},
                       {"applied_wrench": jnp.concatenate([jnp.asarray(f), jnp.asarray(t)])}, DT)
           for p, o, v, w, f, t in zip(PORT["body_position"], PORT["body_orientation"], PORT["body_velocity"],
                                       PORT["body_omega"], PORT["body_force"], PORT["body_torque"], strict=True)]
    pose, twist = np.array([np.asarray(s["pose"]) for s in out]), np.array([np.asarray(s["twist"]) for s in out])
    return {"position": pose[:, :3], "orientation": pose[:, 3:], "velocity": twist[:, :3],
            "angular_velocity": twist[:, 3:]}


def body(mass: float = M_EFF) -> RigidInertialNode:
    return RigidInertialNode("body", DT, mass=mass, inertia=(I_EFF,) * 3, initial_pose=(0, 0, 0, 1, 0, 0, 0),
                             initial_twist=(0.0,) * 6, surface_shape="cylinder", extent=0.0, margin=0.0,
                             velocity_rule="remove_outward", surface_centre=(0.0, 0.0, 0.0),
                             surface_axis=(1.0, 0.0, 0.0), surface_radius=CONSTRAINT_RADIUS)


def test_the_field_agrees_with_the_ports():
    node = UniformRotatingFieldNode("field", DT, slots=("r",), rotation_axis=(0, 0, 1), initial_direction=(1, 0, 0))
    assert moved(field_vectors(node), PORT["field_vector"]) <= RTOL


def test_the_torque_agrees_with_the_ports_once_the_moment_is_rounded_as_the_port_rounds_it():
    """MIME-ANO-110: the port's moment is float32. With the moment rounded to float32 the torques agree; with the
    exact moment they differ by that rounding and no more, and both deliver no force where the gradient is zero."""
    rounded = float(np.float32(MOMENT))
    wrench = torques(MagneticTorqueNode("torque", DT, moment=(rounded, 0.0, 0.0)))
    assert moved(wrench[:, 3:], PORT["magnet_torque"]) <= RTOL
    exact = torques(MagneticTorqueNode("torque", DT, moment=(MOMENT, 0.0, 0.0)))
    assert RTOL < moved(exact[:, 3:], PORT["magnet_torque"]) <= 1.01 * abs(rounded - MOMENT) / MOMENT
    assert np.array_equal(PORT["magnet_force"], np.zeros_like(PORT["magnet_force"]))
    assert np.array_equal(exact[:, :3], np.zeros_like(exact[:, :3]))


def test_gravity_agrees_with_the_ports():
    volume = 6.022068099805108e-08
    node = GravityNode("gravity", DT, mass=1410.0 * volume, buoyant_volume=volume, fluid_density=1000.0,
                       gravity=(0.0, -9.81, 0.0))
    found = np.asarray(node.update(node.initial_state(), {}, DT)["wrench"])
    assert moved(found[:3], PORT["gravity_force"]) <= RTOL and np.array_equal(found[3:], np.zeros(3))


def held_in_the_solve(position, reached):
    """rigid_inertial 0.2.0 where a body starts on its surface and moves outward (declared change 2 of
    4c3): with no resistance, ``M/dt`` is isotropic in translation, so the hold is ``remove_outward`` at the start's
    normal, then the step and the radial projection with the rule at the end's normal. Which bodies, and their
    position and velocity after the step."""
    radius0 = np.linalg.norm(position[:, 1:], axis=1)
    normal0 = np.zeros_like(position)
    normal0[:, 1:] = position[:, 1:] / radius0[:, None]
    resolution = 64 * np.finfo(np.float64).eps * np.linalg.norm(position, axis=1)
    outward0 = np.sum(reached * normal0, axis=1)
    held = (np.abs(radius0 - CONSTRAINT_RADIUS) <= resolution) & (outward0 > 0)
    velocity = reached - np.maximum(outward0, 0.0)[:, None] * normal0
    stepped = position + DT * velocity
    radius = np.linalg.norm(stepped[:, 1:], axis=1)
    beyond = radius > CONSTRAINT_RADIUS
    stepped[beyond, 1:] *= (CONSTRAINT_RADIUS / radius[beyond])[:, None]
    normal = np.zeros_like(stepped)
    normal[:, 1:] = stepped[:, 1:] / np.linalg.norm(stepped[:, 1:], axis=1)[:, None]
    out = np.maximum(np.sum(velocity * normal, axis=1), 0.0)
    velocity = np.where(beyond[:, None], velocity - out[:, None] * normal, velocity)
    return held, stepped, velocity


def test_the_rigid_body_agrees_with_the_ports_inertial_body_where_nothing_resists_it():
    """With no resistance delivered, the implicit solve is the port's explicit update, and both project radially onto
    the same radius. Positions, orientations and angular velocities agree everywhere a body does not start on its
    surface moving outward; velocities agree where nothing is projected, and where a body is, the port removes
    ``(R / r)^2`` of its outward velocity (MIME-ANO-111) and 0.3 all of it. Where a body starts on its
    surface moving outward, rigid_inertial 0.2.0 holds the contact inside the solve (a declared difference), and
    agrees with that rule's closed form instead."""
    found = body_steps(body())
    reached = PORT["body_velocity"] + DT * PORT["body_force"] / M_EFF
    free = PORT["body_position"] + DT * reached
    radius = np.linalg.norm(free[:, 1:], axis=1)
    projected = radius > CONSTRAINT_RADIUS
    held, held_position, held_velocity = held_in_the_solve(PORT["body_position"], reached)
    print(f"\n{int(held.sum())} of {len(held)} bodies start on the surface moving outward")
    assert 8 < int(projected.sum()) < 56, "some bodies are projected and some are not"
    for name in ("position", "orientation", "angular_velocity"):
        assert moved(found[name][~held], PORT[f"body_out_{name}"][~held]) <= RTOL, name
    if held.any():  # the drawn bodies start inside or beyond their surface, not on it: none is held
        assert moved(found["position"][held], held_position[held]) <= RTOL
        assert moved(found["velocity"][held], held_velocity[held]) <= RTOL
    assert moved(found["velocity"][~projected], PORT["body_out_velocity"][~projected]) <= RTOL
    normal = np.zeros_like(free)
    normal[:, 1:] = free[:, 1:] / radius[:, None]
    outward = np.maximum(np.sum(reached * normal, axis=1), 0.0)
    port_rule = reached - ((CONSTRAINT_RADIUS / radius) ** 2 * outward)[:, None] * normal
    full_rule = reached - outward[:, None] * normal
    assert moved(port_rule[projected], PORT["body_out_velocity"][projected]) <= RTOL
    assert moved(found["velocity"][projected & ~held], full_rule[projected & ~held]) <= RTOL
    assert moved(found["velocity"][projected], PORT["body_out_velocity"][projected]) > 1e-3


@pytest.mark.parametrize("mutation", ["the moment reversed", "the mass off by one part in a million"])
def test_a_planted_mutation_is_caught(mutation):
    if mutation == "the moment reversed":
        wrench = torques(MagneticTorqueNode("torque", DT, moment=(-MOMENT, 0.0, 0.0)))
        assert moved(wrench[:, 3:], PORT["magnet_torque"]) > RTOL
    else:
        found = body_steps(body(M_EFF * (1 + 1e-6)))
        assert moved(found["velocity"], PORT["body_out_velocity"]) > RTOL
