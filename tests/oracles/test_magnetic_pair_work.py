"""``Magnetic.Torque`` 0.1.0 and ``Magnetic.Force`` 0.1.0 as a pair, round a closed loop of the body's pose
(MIME-ANO-137), float64, per module. A characterisation of a known defect, not a pass for the pair.

Both laws read the field, and the force law its gradient, at the body's centre of mass. The force acts at the magnet's
centroid, the body's ``magnet_centre`` ``l``, and delivers ``[F; l x F]``; the torque delivers ``m x B`` with ``B`` at
the centre of mass. A point dipole at the magnet would feel ``m x B(x + R l)``, and the pair misses ``m x (G R l)``,
first order in ``l |G| / |B|``. So in a static field that varies in space the pair is not the derivative of an energy,
and a body turned round a closed loop of its pose gains net work, whatever the step.

- **What the nodes do** (the first two tests): the body is held at ``x`` and turned once about a fixed axis, the nodes
  fed as the interactions feed them, ``B(x)`` and ``G(x)``. The work round the loop, ``sum tau . axis dtheta``, equals
  ``-oint (axis x m) . G l dtheta``, which is ``oint m . G (axis x l) dtheta``: exact for these nodes at any step count
  (the field is constant, so what is left is a total derivative's loop), and ``m l pi G_xx`` for the moment and lever
  at right angles to a turn about x on a dipole's axis. It is computed here by rotation matrices and a product, not
  by the nodes.
- **The control**: the same nodes fed ``B`` and ``G`` at the magnet's centroid are the derivative of
  ``U = -m . B(x + R l)``, and the loop's work is zero to rounding.
- **The laws as built** (the last): a graph that attaches both laws to a body with its magnet off the centre of mass
  gives the wrench of the nodes fed at the centre of mass, and not of the nodes fed at the centroid.

A phase that reads the field at the magnet's centroid (planned in the anomaly's record) inverts the last test and the
first two's expectation; the control, and the property that the pair's loop work is zero, then hold of the laws.

Scale of every comparison: the sum, over the loop, of the absolute work of the two nodes taken apart, which is what
their terms cancel against (a step's total can be smaller than the rounding of its parts: a turn about the source's own
axis does no work at all). At the
operating point below (a ten ampere-square-metre source 50 mm away, a 1.7 mA m^2 moment, a 1 mm lever), the 0.1.0 pair
leaves 2.35 % of the loop's summed absolute work."""

import jax
import jax.numpy as jnp
import numpy as np
from chain_cases import experiment
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mime.build._maddening import run_scan
from mime.dynamics import RigidInertial
from mime.fields import PointDipole
from mime.interactions import Magnetic
from mime.nodes.actuation import point_dipole
from mime.nodes.magnetic import MagneticForceNode, MagneticTorqueNode
from mime.scene import Cylinder
from mime.things import Apparatus, Body, Part, PermanentMagnet, State
from mime.units import A_m2, kg, kg_m2, m, m_per_s, mm, rad_per_s

EPS = float(np.finfo(np.float64).eps)
#: Rounding of a loop's sum of STEPS terms, in units of eps times the scale of its terms: 2,500 draws of the property
#: below measured at most 217 for the pair against its prediction and 73 for the control, a few times the step count.
ROUNDINGS = 1024
#: The smallest part of the inputs' own scale, ``2 pi |m| |l| |G|``, that a draw's predicted loop work must be: a draw
#: whose lever, moment and axis make the work vanish (a lever along the moment, on an axisymmetric gradient) has nothing
#: to detect, and the comparison there is between roundings. Above it the defect stands clear of ROUNDINGS by orders of
#: magnitude on every draw.
GENERIC = 1e-3
PROPERTY = settings(deadline=None)
STEPS = 360

#: The coordinator's operating point: a source of 10 A m^2 along z at the origin, a body 50 mm up its axis, a 1.68 mA m^2
#: moment along y, a 1 mm lever along z, the pose turned once about x.
SOURCE_MOMENT, SOURCE_AT = np.array([0.0, 0.0, 10.0]), np.zeros(3)
ON_THE_AXIS = np.array([0.0, 0.0, 0.05])
MOMENT, LEVER, AXIS_X = np.array([0.0, 1.68e-3, 0.0]), np.array([0.0, 0.0, 1e-3]), np.array([1.0, 0.0, 0.0])


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula: ``axis`` a unit vector."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])


def loop_work(source_moment, source_at, body_at, moment, lever, axis, *, at_centroid: bool, steps: int = STEPS):
    """The work of the pair's torque on the body's pose turned once about ``axis`` (through the centre of mass,
    which stays at ``body_at``), as ``(sum, sum of the steps' absolute work, sum of the absolute work of the torque
    node and of the force node taken apart)``; the last is the scale the two cancel against. ``at_centroid`` feeds the nodes the field and
    gradient where the magnet's centroid stands; otherwise where the centre of mass does, as the interactions read
    them."""
    torque = MagneticTorqueNode("torque", 1.0, moment=tuple(moment))
    force = MagneticForceNode("force", 1.0, moment=tuple(moment), magnet_centre=tuple(lever))
    angles = np.arange(steps) * 2.0 * np.pi / steps
    rotations = np.stack([rotation(axis, a) for a in angles])
    poses = jnp.asarray(np.concatenate([np.tile(body_at, (steps, 1)), np.stack([quaternion(axis, a) for a in angles])],
                                       axis=1))
    where = body_at + rotations @ lever if at_centroid else np.tile(body_at, (steps, 1))
    flux, gradient = jax.vmap(lambda p: point_dipole(jnp.asarray(source_moment), jnp.asarray(source_at), p))(
        jnp.asarray(where))
    torques = jax.vmap(lambda pose, b: torque.advance({"moment": jnp.asarray(moment)}, {},
                                                      {"pose": pose, "flux_density": b}, 1.0)["wrench"])(poses, flux)
    forces = jax.vmap(lambda pose, g: force.advance(
        {"moment": jnp.asarray(moment), "magnet_centre": jnp.asarray(lever)}, {}, {"pose": pose, "gradient": g},
        1.0)["wrench"])(poses, gradient)
    each = (2.0 * np.pi / steps)
    terms = np.asarray((torques[:, 3:] + forces[:, 3:]) @ axis) * each
    parts = (np.abs(np.asarray(torques[:, 3:] @ axis)) + np.abs(np.asarray(forces[:, 3:] @ axis))) * each
    return float(terms.sum()), float(np.abs(terms).sum()), float(parts.sum())


def predicted_work(source_moment, source_at, body_at, moment, lever, axis, steps: int = STEPS):
    """``oint m . G (axis x l) dtheta`` with ``G`` at the centre of mass, ``m`` and ``l`` turned by Rodrigues' formula:
    what the pair leaves, with no node in it. Returns it and ``G``."""
    gradient = np.asarray(point_dipole(jnp.asarray(source_moment), jnp.asarray(source_at), jnp.asarray(body_at))[1])
    angles = np.arange(steps) * 2.0 * np.pi / steps
    work = sum((rotation(axis, a) @ moment) @ gradient @ np.cross(axis, rotation(axis, a) @ lever) for a in angles)
    return work * (2.0 * np.pi / steps), gradient


def directions() -> st.SearchStrategy[np.ndarray]:
    return st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))).map(np.asarray).filter(
        lambda v: float(np.linalg.norm(v)) > 0.1).map(lambda v: v / np.linalg.norm(v))


def cases() -> st.SearchStrategy[tuple]:
    """A source of 1 to 50 A m^2 between 3 and 20 cm from a body at the origin, a body moment of 0.1 to 10 mA m^2,
    a lever of 0.2 to 2 mm, a turn about any axis: millimetre bodies and centimetre magnets, as in the family's rig."""
    return st.tuples(directions(), st.floats(1.0, 50.0), directions(), st.floats(0.03, 0.2), directions(),
                     st.floats(1e-4, 1e-2), directions(), st.floats(2e-4, 2e-3), directions()).map(
        lambda d: (d[0] * d[1], d[2] * d[3], d[4] * d[5], d[6] * d[7], d[8]))


def test_the_pair_does_net_work_round_a_turn_about_x_and_it_is_m_l_pi_gxx_whatever_the_step():
    """The coordinator's operating point (MIME-ANO-137): 2.5334e-6 J, 2.35 % of the loop's summed absolute work,
    at every step count."""
    expected_work, gradient = predicted_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, AXIS_X)
    closed_form = MOMENT[1] * LEVER[2] * np.pi * gradient[0, 0]
    assert abs(expected_work - closed_form) <= ROUNDINGS * EPS * abs(closed_form)
    for steps in (64, 360, 3600):
        work, summed, scale = loop_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, AXIS_X,
                                        at_centroid=False, steps=steps)
        assert abs(work - closed_form) <= ROUNDINGS * EPS * scale, steps
        assert work / summed > 0.02, steps


def test_control_the_nodes_fed_at_the_magnets_centroid_do_no_work_round_the_same_turn():
    for steps in (64, 360, 3600):
        work, _, scale = loop_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, AXIS_X, at_centroid=True,
                                   steps=steps)
        assert abs(work) <= ROUNDINGS * EPS * scale, steps


@PROPERTY
@given(cases())
def test_the_pair_does_the_predicted_net_work_and_the_control_none_on_every_draw(drawn):
    source_moment, source_at, moment, lever, axis = drawn
    expected_work, gradient = predicted_work(source_moment, source_at, np.zeros(3), moment, lever, axis)
    work, _, scale = loop_work(source_moment, source_at, np.zeros(3), moment, lever, axis, at_centroid=False)
    assume(abs(expected_work) >= GENERIC * 2.0 * np.pi * np.linalg.norm(moment) * np.linalg.norm(lever)
           * np.linalg.norm(gradient))
    assert abs(work - expected_work) <= ROUNDINGS * EPS * scale
    assert abs(work) > 1e4 * ROUNDINGS * EPS * scale, "the defect stands clear of the control's rounding"
    control_work, _, control_scale = loop_work(source_moment, source_at, np.zeros(3), moment, lever, axis,
                                               at_centroid=True)
    assert abs(control_work) <= ROUNDINGS * EPS * control_scale


def test_the_laws_as_built_read_the_field_at_the_centre_of_mass_and_not_at_the_magnets_centroid():
    """A body with its magnet 0.5 mm off its centre of mass along x, a fixed magnet 10 cm above: the built wrench is
    the nodes' fed ``B(x)`` and ``G(x)``, and differs from what they give fed at the centroid by ``m x (G l)``."""
    lever = np.array([5e-4, 0.0, 0.0])
    moment = np.array([0.0, 1.7e-3, 0.4e-3])
    body_at = np.array([0.0, 0.0, -1e-3])
    exp = experiment("pair_work")
    exp.add_apparatus(Apparatus("stand", parts={"magnet": Part(Cylinder(radius=m(17.5e-3), length=m(20e-3),
                                                                        axis=(1.0, 0.0, 0.0)), mount=None,
                                                               at=m((0.0, 0.0, 0.1)))},
                                magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
                                fields={"magnetic": PointDipole(part="magnet")}))
    exp.declare("stand.magnetic_moment", A_m2(18.89), reason="a fixed magnet above the body")
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m(tuple(body_at)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    exp.declare("umr.mass", kg(8.5e-5), reason="a millimetre body")
    exp.declare("umr.inertia", kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), reason="a millimetre body")
    exp.declare("umr.magnetic_moment", A_m2(tuple(moment)), reason="a millimetre body's magnet")
    exp.declare("umr.magnet_centre", m(tuple(lever)), reason="its magnet off its centre of mass")
    exp.declare("umr.extent", mm(2.0), reason="a millimetre body")
    exp.attach(Magnetic.Torque(), body="umr", field="stand.magnetic")
    exp.attach(Magnetic.Force(), body="umr", field="stand.magnetic")
    graph = exp.build().graph
    graph.reset_state()
    state = run_scan(graph, 1)
    built = np.asarray(state["magnetic.torque:umr"]["wrench"]) + np.asarray(state["magnetic.force:umr"]["wrench"])

    source, source_at = jnp.asarray((0.0, 0.0, 18.89)), jnp.asarray((0.0, 0.0, 0.1))
    torque = MagneticTorqueNode("torque", 1.0, moment=tuple(moment))
    force = MagneticForceNode("force", 1.0, moment=tuple(moment), magnet_centre=tuple(lever))
    pose = jnp.asarray((*body_at, 1.0, 0.0, 0.0, 0.0))

    def fed_at(where: np.ndarray) -> np.ndarray:
        flux, gradient = point_dipole(source, source_at, jnp.asarray(where))
        return (np.asarray(torque.advance({"moment": jnp.asarray(moment)}, {}, {"pose": pose, "flux_density": flux},
                                          1.0)["wrench"])
                + np.asarray(force.advance({"moment": jnp.asarray(moment), "magnet_centre": jnp.asarray(lever)}, {},
                                           {"pose": pose, "gradient": gradient}, 1.0)["wrench"]))

    at_mass, at_centroid = fed_at(body_at), fed_at(body_at + lever)
    scale = float(np.max(np.abs(at_mass[3:])))
    assert np.allclose(built, at_mass, rtol=1e-9, atol=1e-9 * scale)
    gradient = np.asarray(point_dipole(source, source_at, jnp.asarray(body_at))[1])
    missing = np.cross(moment, gradient @ lever)
    assert np.linalg.norm(missing) > 1e-3 * scale, "the case's lever matters"
    assert np.linalg.norm((at_centroid - at_mass)[3:] - missing) <= 0.05 * np.linalg.norm(missing), "first order in l"
    assert np.linalg.norm((built - at_centroid)[3:]) > 1e-3 * scale
