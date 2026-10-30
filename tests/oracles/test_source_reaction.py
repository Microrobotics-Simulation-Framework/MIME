"""The magnetic laws' reaction on the field's source, delivered to the frame the source is mounted on (float64, per
module), on A0's AR4 on its dynamic rung with the magnet on its flange (``chain_cases.dynamic_rpm(motor=False)``), its
nodes as the build makes them, evaluated as the graph's step does, over drawn poses of the body below.

- **Equal and opposite**: what ``Magnetic.Torque`` and ``Magnetic.Force`` deliver on the body, moved from its centre
  of mass to the flange's origin, and the reactions they deliver in the flange's row of the arm's loads sum to zero,
  and every other row is zero. The control: the laws with their reactions dropped fail on every draw.
- **Two point dipoles**: the two reactions together are the force and torque the body's dipole exerts on the source's,
  computed from the source's side in closed form, ``F_M = grad(m_M . B_b)`` and ``tau_M = m_M x B_b`` at the magnet's
  centre, ``B_b`` the body's field, moved to the flange's origin: the pair's torques balance only with both laws'
  reactions. The control: the force's reaction alone fails on every draw.
- **The reaction's node** passes MADDENING's battery.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import BODY_MOMENT, MAGNET_MOMENT, dynamic_rpm
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mime.core import quaternion
from mime.testing import node_results
from mime.units import VACUUM_PERMEABILITY

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 64
DT = 1e-3
K = VACUUM_PERMEABILITY / (4 * np.pi)
LAWS = ("force", "torque")


@pytest.fixture(scope="module")
def nodes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = dynamic_rpm("flange_magnet", motor=False).build()
    graph = built.graph
    found = {name: graph.get_node(name) for name in ("rpm.arm", "rpm.magnetic", *(f"magnetic.{law}:umr" for law in LAWS),
                                                     *(f"magnetic.{law}:umr.reaction" for law in LAWS))}
    return found


def step(nodes, pose):
    """One step of the provider, the two laws and their reactions, from the body's pose and the arm's frames at its
    start: the body's wrenches, the reactions' rows, the magnet's centre and moment, and the flange's origin."""
    frames = nodes["rpm.arm"].initial_state()["frame_poses"]
    provider = nodes["rpm.magnetic"]
    slots = [name.removeprefix("points_") for name in provider.boundary_input_spec() if name.startswith("points_")]
    field = provider.update(provider.initial_state(), {"mount_poses": frames, **{f"points_{s}": pose for s in slots}},
                            DT)
    force_slot = next(s for s in slots if "force" in s)
    torque_slot = next(s for s in slots if "torque" in s)
    wrenches = {"force": nodes["magnetic.force:umr"].update({}, {"pose": pose, "gradient":
                                                                  field[f"gradient_{force_slot}"]}, DT)["wrench"],
                "torque": nodes["magnetic.torque:umr"].update({}, {"pose": pose, "flux_density":
                                                                    field[f"flux_density_{torque_slot}"]}, DT)["wrench"]}
    rows = {law: nodes[f"magnetic.{law}:umr.reaction"].update({}, {"pose": pose, "wrench": wrenches[law],
                                                                    "mount_poses": frames}, DT)["link_wrench"]
            for law in LAWS}
    index = nodes["magnetic.force:umr.reaction"].constants()["mount_index"]
    centre = np.asarray(field["centre"])
    moment = np.asarray(quaternion.rotate(field["orientation"], jnp.asarray((MAGNET_MOMENT, 0.0, 0.0))))
    return ({law: np.asarray(w) for law, w in wrenches.items()}, {law: np.asarray(r) for law, r in rows.items()},
            index, centre, moment, np.asarray(frames)[index, :3])


def unit(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-3 else np.array([1.0, 0.0, 0.0, 0.0])


#: The body's pose: anywhere within 3 mm of its start in A0's tube (1 mm below the origin), turned any way.
POSES = st.tuples(arrays(np.float64, (3,), elements=st.floats(-3e-3, 3e-3, allow_nan=False)),
                  arrays(np.float64, (4,), elements=st.floats(-1.0, 1.0, allow_nan=False)).map(unit)).map(
    lambda drawn: jnp.asarray(np.concatenate([drawn[0] + np.array([0.0, 0.0, -1e-3]), drawn[1]])))


def moved(wrench, centre, origin):
    """A wrench about ``centre`` as one about ``origin``."""
    return np.concatenate([wrench[:3], wrench[3:] + np.cross(centre - origin, wrench[:3])])


def balance(wrenches, rows, index, origin, centre, dropped: bool = False) -> tuple[float, float]:
    on_body = sum(moved(w, centre, origin) for w in wrenches.values())
    on_source = 0.0 if dropped else sum(r[index] for r in rows.values())
    others = 0.0 if dropped else max(float(np.max(np.abs(np.delete(r, index, axis=0)))) for r in rows.values())
    scale = max(float(np.max(np.abs(on_body))), 1e-300)
    return max(float(np.max(np.abs(on_body + on_source))), others), ROUNDINGS * EPS * scale


@PROPERTY
@given(POSES)
def test_the_laws_and_their_reactions_on_the_source_are_equal_and_opposite(nodes, pose):
    wrenches, rows, index, _, _, origin = step(nodes, pose)
    misfit, bound = balance(wrenches, rows, index, origin, np.asarray(pose[:3]))
    assert misfit <= bound, f"{misfit:.3g} > {bound:.3g}"


@PROPERTY
@given(POSES)
def test_control_the_laws_with_their_reactions_dropped_fail_on_every_draw(nodes, pose):
    wrenches, rows, index, _, _, origin = step(nodes, pose)
    misfit, bound = balance(wrenches, rows, index, origin, np.asarray(pose[:3]), dropped=True)
    assert misfit > bound


def body_field(moment, centre, point):
    r = point - centre
    d = np.linalg.norm(r)
    return K * (3 * np.dot(moment, r) * r / d**2 - moment) / d**3


def on_the_source(pose, magnet_centre, magnet_moment, origin):
    """The force and the torque the body's dipole exerts on the source's dipole, about the flange's origin: the force
    the gradient of ``m_M . B_b`` by central differences of the closed form, refined to rounding by Richardson."""
    body_moment = np.asarray(quaternion.rotate(pose[3:], jnp.asarray(BODY_MOMENT)))
    centre = np.asarray(pose[:3])

    def energy(point):
        return float(np.dot(magnet_moment, body_field(body_moment, centre, point)))

    def gradient(h):
        return np.array([(energy(magnet_centre + h * e) - energy(magnet_centre - h * e)) / (2 * h)
                         for e in np.eye(3)])

    h = 1e-4 * np.linalg.norm(magnet_centre - centre)
    force = (4 * gradient(h / 2) - gradient(h)) / 3
    torque = np.cross(magnet_moment, body_field(body_moment, centre, magnet_centre))
    return np.concatenate([force, torque + np.cross(magnet_centre - origin, force)])


#: How near the closed form the reactions lie, relative to its size: the force by Richardson's central differences,
#: whose truncation at a step of 1e-4 of the distance is about 1e-14 of it and whose rounding about 1e-10 (the energy's
#: rounding over the step), far below the force's reaction alone being off by the torque's whole share.
PAIR_TOLERANCE = 1e-7


def pair_misfit(rows, index, expected, laws=LAWS) -> float:
    found = sum(rows[law][index] for law in laws)
    return float(np.max(np.abs(found - expected))) / float(np.max(np.abs(expected)))


@PROPERTY
@given(POSES)
def test_the_reactions_are_the_force_and_torque_of_the_bodys_dipole_on_the_sources(nodes, pose):
    _, rows, index, centre, moment, origin = step(nodes, pose)
    expected = on_the_source(pose, centre, moment, origin)
    assert pair_misfit(rows, index, expected) <= PAIR_TOLERANCE


@PROPERTY
@given(POSES)
def test_control_the_forces_reaction_alone_fails_the_pair_on_every_draw(nodes, pose):
    _, rows, index, centre, moment, origin = step(nodes, pose)
    expected = on_the_source(pose, centre, moment, origin)
    assert pair_misfit(rows, index, expected, laws=("force",)) > PAIR_TOLERANCE


def test_the_reactions_node_passes_maddenings_battery(nodes):
    node = nodes["magnetic.force:umr.reaction"]
    options = {"boundary_bounds": {"pose": (-1.0, 1.0), "wrench": (-1e-3, 1e-3), "mount_poses": (-1.0, 1.0)},
               "bounds": {"link_wrench": (-1e-3, 1e-3)}}
    results = node_results(node, options)
    print("\n" + "\n".join(str(r) for r in results.values()))
    assert all(r.passed for r in results.values())

