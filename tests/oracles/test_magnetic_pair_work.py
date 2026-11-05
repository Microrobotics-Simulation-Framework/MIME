"""``Magnetic.Torque`` 0.1.0 and ``Magnetic.Force`` 0.1.0, as an experiment builds them, as a pair round a closed loop of
the body's pose (MIME-ANO-137), float64, per module. A characterisation of a known defect, not a pass for the pair.

Both laws read the field, and the force law its gradient, at the body's centre of mass. The force acts at the magnet's
centroid, the body's ``magnet_centre`` ``l``, and delivers ``[F; l x F]``; the torque delivers ``m x B`` with ``B`` at
the centre of mass. A point dipole at the magnet would feel ``m x B(x + R l)``, and the pair misses ``m x (G R l)``,
first order in ``l |G| / |B|``. So in a static field that varies in space the pair is not the derivative of an energy,
and a body turned round a closed loop of its pose gains net work, whatever the step.

**What each test pins.** Every test builds the two laws as registered, ``magnetic.torque`` 0.1.0 and ``magnetic.force``
0.1.0 looked up in the registry by name and version (not through ``Magnetic.Torque``, which names the newest), attached
to a body that carries a magnet off its centre of mass in the field of a fixed point dipole, and reads the wrench of the
built graph. The read point is therefore the registered laws' own wiring, and it is not restated here. The body is
turned to a general orientation (not the identity, and a lever and moment not along an axis) before it is turned round
the loop. The 0.1.0 entries stay registered after the fix the anomaly plans (0.2.0 reading at the centroid), so these
tests stay valid for 0.1.0 then; they fail on a graph whose read point is moved to the centroid, which the last test
shows with that graph.

- **The pair's work** (the first three tests and the property): the pose is held at ``x`` and turned once about a fixed
  axis. The work round the loop, ``sum tau . axis dtheta``, equals ``oint m . G (axis x l) dtheta``: exact for these
  laws at any step count (the field is constant, so what is left is a total derivative's loop). It is computed here by
  rotation matrices and a product, not by the laws. For the moment and lever at right angles to a turn about x, at a
  point on a dipole's axis, it is ``m l pi G_xx`` when ``m x l`` points along +x (the operating point: ``m`` along y,
  ``l`` along z); it is ``-m l pi G_xx`` for ``l x m`` along +x, and zero for ``m`` and ``l`` parallel.
- **The control**: the same two nodes (taken from the built graph) fed ``B`` and ``G`` at the magnet's centroid are the
  derivative of ``U = -m . B(x + R l)``, and the loop's work is zero to rounding. A later phase that reads there makes
  the registered pair this control.
- **The read point** (the fourth test): the built wrench is that of the nodes fed at the centre of mass, and not that
  of the nodes fed at the centroid, at a general orientation, and the difference is ``m x (G l)`` to first order.

Two quantities, kept apart. ``scale`` is the inputs' scale, ``2 pi |m| (|B| + |l| |G|)``: what the nodes' terms are
computed from, and the scale of every tolerance here. A result is never judged on its own size, which can be rounding
itself where the geometry makes the work vanish (a moment along the field, a turn about the moment). ``summed`` is the
sum of the absolute work of the pair's steps, which is what the 2.35 % below is of. At the operating point below (a ten
ampere-square-metre source 50 mm away, a 1.68 mA m^2 moment, a 1 mm lever) the 0.1.0 pair's loop work is 2.35 % of
``summed`` and 1.40 % of ``scale``."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import experiment
from hypothesis import event, given, settings
from hypothesis import strategies as st

from mime.core import quaternion as quaternions
from mime.dynamics import RigidInertial
from mime.fields import PointDipole
from mime.nodes.actuation import PointDipoleNode, point_dipole
from mime.registry import EntryKind, lookup
from mime.scene import Cylinder
from mime.things import Apparatus, Body, Part, PermanentMagnet, State
from mime.units import A_m2, kg, kg_m2, m, m_per_s, mm, rad_per_s

EPS = float(np.finfo(np.float64).eps)
#: Rounding of a loop's sum of STEPS terms, in units of eps times ``scale`` (the inputs' scale). Measured over 3,000
#: unfiltered draws of the property below (any direction, any starting orientation, 360 steps): at most 1.54 roundings
#: for the pair against its prediction and 1.57 for the control (medians 0.12 and 0.13), and at most 0.27 at the
#: operating point at 64 to 10,000 steps; 64 stands 40 times clear of the worst draw.
ROUNDINGS = 64
#: How far above ROUNDINGS' rounding the defect must stand, for a draw whose predicted work does not vanish: the same
#: 3,000 draws had it at least 2e10 roundings above, so 1e4 times ROUNDINGS (6.4e5 roundings) leaves 3e4 to spare.
VISIBLE = 1e4
#: The smallest part of the inputs' own scale, ``2 pi |m| |l| |G|``, that a draw's predicted loop work must be to count
#: as one the defect shows in: a draw whose lever, moment and axis make the work vanish (a lever along the moment, on an
#: axisymmetric gradient) has nothing to detect, and the comparison there is between roundings.
GENERIC = 1e-3
PROPERTY = settings(deadline=None)
STEPS = 360

#: The operating point of the design pass's red-team script: a source of 10 A m^2 along z at the origin, a body 50 mm up
#: its axis, a 1.68 mA m^2 moment along y, a 1 mm lever along z, the pose turned once about x.
SOURCE_MOMENT, SOURCE_AT = np.array([0.0, 0.0, 10.0]), np.zeros(3)
ON_THE_AXIS = np.array([0.0, 0.0, 0.05])
MOMENT, LEVER, AXIS_X = np.array([0.0, 1.68e-3, 0.0]), np.array([0.0, 0.0, 1e-3]), np.array([1.0, 0.0, 0.0])
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
TORQUE, FORCE = "magnetic.torque:umr", "magnetic.force:umr"


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])


def product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The Hamilton product of two quaternions (scalar first): ``a`` applied after ``b``."""
    return np.array([a[0] * b[0] - a[1:] @ b[1:], *(a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]))])


def matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rodrigues' formula for each angle: ``axis`` a unit vector."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angles)[:, None, None] * k + (1.0 - np.cos(angles))[:, None, None] * (k @ k)


class Work(NamedTuple):
    """A loop's work: the ``net`` sum, ``summed`` (the sum of the absolute work of each step's total) and ``scale``, the
    inputs' scale ``2 pi |m| (|B| + |l| |G|)``, which the nodes' terms are computed from and rounded against."""

    net: float
    summed: float
    scale: float


def loop_poses(body_at: np.ndarray, start: np.ndarray, axis: np.ndarray, steps: int) -> np.ndarray:
    """The pose (position, quaternion) of a body whose centre of mass stays at ``body_at``, started at orientation
    ``start`` and turned once about the world's ``axis`` through that point, in ``steps`` equal angles."""
    angles = np.arange(steps) * 2.0 * np.pi / steps
    return np.stack([np.concatenate([body_at, product(quaternion(axis, a), start)]) for a in angles])


def work_of(torque: np.ndarray, force: np.ndarray, axis: np.ndarray, scale: float) -> Work:
    """The work, about ``axis``, of the two nodes' wrenches at each pose of a loop of equal angles."""
    total = (torque[:, 3:] + force[:, 3:]) @ axis * (2.0 * np.pi / len(torque))
    return Work(float(total.sum()), float(np.abs(total).sum()), scale)


def inputs_scale(source_moment, source_at, body_at, moment, lever) -> float:
    """``2 pi |m| (|B| + |l| |G|)``, with ``B`` and ``G`` at the centre of mass: what the torque node's ``m x B`` and the
    force node's ``l x G^T m`` are computed from. A result is judged on this and never on its own size, which can be
    rounding itself where the geometry makes the work vanish (a moment along the field)."""
    flux, gradient = point_dipole(jnp.asarray(source_moment), jnp.asarray(source_at), jnp.asarray(body_at))
    return 2.0 * np.pi * float(np.linalg.norm(moment)) * (float(np.linalg.norm(flux))
                                                            + float(np.linalg.norm(lever) * np.linalg.norm(gradient)))


def predicted_work(source_moment, source_at, body_at, moment, lever, start, axis, steps: int = STEPS):
    """``oint m . G (axis x l) dtheta`` with ``G`` at the centre of mass and ``m`` and ``l`` (body frame) turned by the
    starting orientation and then by Rodrigues' formula: what the pair leaves, with no law or node in it. Returns it and
    ``G``."""
    gradient = np.asarray(point_dipole(jnp.asarray(source_moment), jnp.asarray(source_at), jnp.asarray(body_at))[1])
    angles = np.arange(steps) * 2.0 * np.pi / steps
    turns = rotations(axis, angles)
    moments, levers = turns @ (matrix(start) @ moment), turns @ (matrix(start) @ lever)
    work = np.einsum("ni,ij,nj->", moments, gradient, np.cross(axis, levers))
    return float(work) * (2.0 * np.pi / steps), gradient


class Laws:
    """The two laws as an experiment builds them: a body with a magnet off its centre of mass, and a fixed point dipole
    above it. The values the laws read (the source's moment and place, the body's moment and lever) are the graph's
    parameters, so one built and compiled graph serves every draw; its sweep gives the wrench of each law at each of a
    batch of poses, one step from each, as the laws read the pose at the start of the step."""

    def __init__(self) -> None:
        torque, force = (lookup(EntryKind.INTERACTION, name, "0.1.0") for name in ("magnetic.torque", "magnetic.force"))
        assert torque is not None and force is not None, "0.1.0 of both laws stays registered"
        exp = experiment("pair_work")
        exp.add_apparatus(Apparatus("stand", parts={"magnet": Part(Cylinder(radius=m(17.5e-3), length=m(20e-3),
                                                                            axis=(1.0, 0.0, 0.0)), mount=None,
                                                                   at=m((0.0, 0.0, 0.0)))},
                                    magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))},
                                    fields={"magnetic": PointDipole(part="magnet")}))
        exp.declare("stand.magnetic_moment", A_m2(10.0), reason="a fixed magnet above the body")
        exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
            position=m(tuple(ON_THE_AXIS)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
            angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare("umr.mass", kg(8.5e-5), reason="a millimetre body")
        exp.declare("umr.inertia", kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), reason="a millimetre body")
        exp.declare("umr.magnetic_moment", A_m2(tuple(MOMENT)), reason="a millimetre body's magnet")
        exp.declare("umr.magnet_centre", m(tuple(LEVER)), reason="its magnet off its centre of mass")
        exp.declare("umr.extent", mm(2.0), reason="a millimetre body")
        exp.attach(torque.cls(), body="umr", field="stand.magnetic")
        exp.attach(force.cls(), body="umr", field="stand.magnetic")
        self.graph = exp.build().graph
        self.graph.reset_state()
        self.state = {name: self.graph.get_node_state(name) for name in self.graph.params["nodes"]}
        torque_node, force_node = self.graph.get_node(TORQUE), self.graph.get_node(FORCE)

        def fed(pose, source_moment, source_at, moment, lever, at_centroid):
            where = pose[:3] + at_centroid * quaternions.rotate(pose[3:], lever)
            flux, gradient = point_dipole(source_moment, source_at, where)
            return (torque_node.advance({"moment": moment}, {}, {"pose": pose, "flux_density": flux}, 1.0)["wrench"],
                    force_node.advance({"moment": moment, "magnet_centre": lever}, {},
                                       {"pose": pose, "gradient": gradient}, 1.0)["wrench"])

        self._fed = jax.jit(jax.vmap(fed, in_axes=(0, None, None, None, None, None)))

    def built(self, source_moment, source_at, moment, lever, poses) -> tuple[np.ndarray, np.ndarray]:
        """The wrench of the torque law and of the force law, one step from each of ``poses``, as the graph gives it."""
        nodes = self.graph.params["nodes"]
        params = {**self.graph.params, "nodes": {
            **nodes,
            "stand.magnetic": {**nodes["stand.magnetic"], "moment": jnp.asarray(source_moment),
                               "placement": jnp.asarray([*source_at, 1.0, 0.0, 0.0, 0.0])},
            TORQUE: {**nodes[TORQUE], "moment": jnp.asarray(moment)},
            FORCE: {**nodes[FORCE], "moment": jnp.asarray(moment), "magnet_centre": jnp.asarray(lever)}}}
        batch = {name: {key: jnp.broadcast_to(value, (len(poses), *value.shape)) for key, value in fields.items()}
                 for name, fields in self.state.items()}
        batch["umr"] = {**batch["umr"], "pose": jnp.asarray(poses)}
        final = self.graph.run_sweep(1, batch, params=params)
        return np.asarray(final[TORQUE]["wrench"]), np.asarray(final[FORCE]["wrench"])

    def fed(self, source_moment, source_at, moment, lever, poses, *, at_centroid: bool):
        """The same two nodes (the graph's own), fed ``B`` and ``G`` where the centre of mass stands or, with
        ``at_centroid``, where the magnet's centroid does."""
        torque, force = self._fed(jnp.asarray(poses), jnp.asarray(source_moment), jnp.asarray(source_at),
                                  jnp.asarray(moment), jnp.asarray(lever), 1.0 if at_centroid else 0.0)
        return np.asarray(torque), np.asarray(force)

    def pair_work(self, source_moment, source_at, body_at, moment, lever, start, axis, *, steps: int = STEPS) -> Work:
        return work_of(*self.built(source_moment, source_at, moment, lever, loop_poses(body_at, start, axis, steps)),
                       axis, inputs_scale(source_moment, source_at, body_at, moment, lever))

    def control_work(self, source_moment, source_at, body_at, moment, lever, start, axis, *,
                     steps: int = STEPS) -> Work:
        return work_of(*self.fed(source_moment, source_at, moment, lever, loop_poses(body_at, start, axis, steps),
                                 at_centroid=True), axis, inputs_scale(source_moment, source_at, body_at, moment, lever))


@pytest.fixture(scope="module")
def laws(x64) -> Laws:
    return Laws()


#: A general orientation, a moment and a lever not along an axis, and a turn about an oblique axis, for the cases that
#: are not the operating point.
GENERAL_START = quaternion(unit([1.0, -2.0, 0.5]), 1.1)
GENERAL_MOMENT, GENERAL_LEVER = np.array([0.4e-3, 1.5e-3, -0.8e-3]), np.array([-6e-4, 4e-4, 5e-4])
GENERAL_AXIS = unit([-0.4, 0.2, 1.0])
GENERAL_SOURCE, GENERAL_SOURCE_AT = np.array([2.0, -3.0, 12.0]), np.array([0.02, 0.01, 0.09])


def the_pair_shows_the_defect_at_the_operating_point(laws: Laws, steps: int) -> None:
    """The first test's check, a function so that the last can run it on a graph with the fix."""
    expected, gradient = predicted_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, IDENTITY, AXIS_X)
    closed_form = MOMENT[1] * LEVER[2] * np.pi * gradient[0, 0]
    assert abs(expected - closed_form) <= ROUNDINGS * EPS * abs(closed_form)
    work = laws.pair_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, IDENTITY, AXIS_X, steps=steps)
    assert abs(work.net - closed_form) <= ROUNDINGS * EPS * work.scale, steps
    assert work.net / work.summed > 0.02, steps


def test_the_registered_pair_does_net_work_round_a_turn_about_x_and_it_is_m_l_pi_gxx_whatever_the_step(laws):
    """The operating point (MIME-ANO-137): 2.5334e-6 J, 2.35 % of the loop's summed absolute work, at every step count,
    from the built graph's wrench, and the closed form ``m l pi G_xx`` (``m x l`` along +x)."""
    for steps in (64, 360, 3600):
        the_pair_shows_the_defect_at_the_operating_point(laws, steps)


def test_the_registered_pair_does_the_predicted_work_at_a_general_orientation(laws):
    """The same loop with the body first turned to a general orientation, a moment and a lever not along an axis and an
    oblique axis (the case the anomaly is about: a body turned about its lever), against ``oint m . G (axis x l)``."""
    for steps in (64, 360, 3600):
        expected, gradient = predicted_work(GENERAL_SOURCE, GENERAL_SOURCE_AT, np.zeros(3), GENERAL_MOMENT,
                                            GENERAL_LEVER, GENERAL_START, GENERAL_AXIS, steps)
        work = laws.pair_work(GENERAL_SOURCE, GENERAL_SOURCE_AT, np.zeros(3), GENERAL_MOMENT, GENERAL_LEVER,
                              GENERAL_START, GENERAL_AXIS, steps=steps)
        assert abs(work.net - expected) <= ROUNDINGS * EPS * work.scale, steps
        own = 2.0 * np.pi * np.linalg.norm(GENERAL_MOMENT) * np.linalg.norm(GENERAL_LEVER) * np.linalg.norm(gradient)
        assert abs(expected) > 0.1 * own, "the case's work does not vanish"
        assert abs(work.net) > VISIBLE * ROUNDINGS * EPS * work.scale, steps


def test_control_the_nodes_fed_at_the_magnets_centroid_do_no_work_round_the_same_turns(laws):
    for steps in (64, 360, 3600):
        work = laws.control_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, IDENTITY, AXIS_X, steps=steps)
        assert abs(work.net) <= ROUNDINGS * EPS * work.scale, steps
        work = laws.control_work(GENERAL_SOURCE, GENERAL_SOURCE_AT, np.zeros(3), GENERAL_MOMENT, GENERAL_LEVER,
                                 GENERAL_START, GENERAL_AXIS, steps=steps)
        assert abs(work.net) <= ROUNDINGS * EPS * work.scale, steps


def directions() -> st.SearchStrategy[np.ndarray]:
    """A unit vector from three components, each of magnitude 0.1 to 1 and either sign: no component is subnormal or
    near zero (a body's frames are not drawn at 1e-227 of an axis), and every octant is reached."""
    component = st.builds(lambda sign, size: sign * size, st.sampled_from((-1.0, 1.0)), st.floats(0.1, 1.0))
    return st.tuples(component, component, component).map(unit)


def orientations() -> st.SearchStrategy[np.ndarray]:
    """A unit quaternion: a turn of 0.1 to 6 radians about any direction (the identity is the operating point's)."""
    return st.builds(quaternion, directions(), st.floats(0.1, 6.0))


def cases() -> st.SearchStrategy[tuple]:
    """A source of 1 to 50 A m^2 between 3 and 20 cm from a body at the origin, a body moment of 0.1 to 10 mA m^2,
    a lever of 0.2 to 2 mm in any direction of the body's frame, the body in any orientation, a turn about any axis:
    millimetre bodies and centimetre magnets, as in the family's rig."""
    return st.tuples(directions(), st.floats(1.0, 50.0), directions(), st.floats(0.03, 0.2), directions(),
                     st.floats(1e-4, 1e-2), directions(), st.floats(2e-4, 2e-3), directions(), orientations()).map(
        lambda d: (d[0] * d[1], d[2] * d[3], d[4] * d[5], d[6] * d[7], d[8], d[9]))


class Draw(NamedTuple):
    """What one drawn case measured, in roundings (units of ``EPS * scale``)."""

    pair_error: float
    control: float
    work: float
    predicted_fraction: float


def measured(laws: Laws, drawn: tuple) -> Draw:
    """One draw's pair work against its prediction, its control, and the pair's work in roundings; the predicted work as a
    fraction of the inputs' own scale ``2 pi |m| |l| |G|``."""
    source_moment, source_at, moment, lever, axis, start = drawn
    expected, gradient = predicted_work(source_moment, source_at, np.zeros(3), moment, lever, start, axis)
    work = laws.pair_work(source_moment, source_at, np.zeros(3), moment, lever, start, axis)
    control = laws.control_work(source_moment, source_at, np.zeros(3), moment, lever, start, axis)
    rounding = EPS * work.scale
    return Draw(abs(work.net - expected) / rounding, abs(control.net) / (EPS * control.scale), abs(work.net) / rounding,
                abs(expected) / (2.0 * np.pi * np.linalg.norm(moment) * np.linalg.norm(lever)
                                 * np.linalg.norm(gradient)))


@PROPERTY
@given(cases())
def test_the_pair_does_the_predicted_net_work_and_the_control_none_on_every_draw(laws, drawn):
    """On every draw, with no filter: the built pair's work is the prediction and the control's is zero, to rounding.
    Where the predicted work is not a small part of the inputs' scale the defect is also far above that rounding."""
    draw = measured(laws, drawn)
    assert draw.pair_error <= ROUNDINGS
    assert draw.control <= ROUNDINGS
    if draw.predicted_fraction >= GENERIC:
        assert draw.work > VISIBLE * ROUNDINGS, "the defect stands clear of the control's rounding"
    else:
        event("the work vanishes by the geometry")


def the_pair_reads_at_the_centre_of_mass(laws: Laws) -> None:
    """The fourth test's check, a function so that the last can run it on a graph with the fix: a body at a general
    orientation, 4 mm and 2 mm off the source's axis, 0.1 m below a fixed magnet, its lever LEVER."""
    body_at, source_at = np.array([4e-3, -2e-3, 0.0]), np.array([0.0, 0.0, 0.1])
    pose = np.concatenate([body_at, GENERAL_START])[None]
    built = sum(laws.built(GENERAL_SOURCE, source_at, GENERAL_MOMENT, LEVER, pose))[0]
    at_mass = sum(laws.fed(GENERAL_SOURCE, source_at, GENERAL_MOMENT, LEVER, pose, at_centroid=False))[0]
    at_centroid = sum(laws.fed(GENERAL_SOURCE, source_at, GENERAL_MOMENT, LEVER, pose, at_centroid=True))[0]
    scale = float(np.max(np.abs(at_mass[3:])))
    assert np.allclose(built, at_mass, rtol=1e-9, atol=1e-9 * scale), "the built wrench is the nodes fed at the centre"
    gradient = np.asarray(point_dipole(jnp.asarray(GENERAL_SOURCE), jnp.asarray(source_at), jnp.asarray(body_at))[1])
    moment, lever = matrix(GENERAL_START) @ GENERAL_MOMENT, matrix(GENERAL_START) @ LEVER
    missing = np.cross(moment, gradient @ lever)
    assert np.linalg.norm(missing) > 1e-3 * scale, "the case's lever matters"
    assert np.linalg.norm((at_centroid - at_mass)[3:] - missing) <= 0.05 * np.linalg.norm(missing), "first order in l"
    assert np.linalg.norm((built - at_centroid)[3:]) > 1e-3 * scale, "the built wrench is not the centroid's"


def test_the_registered_laws_read_the_field_at_the_centre_of_mass_and_not_at_the_magnets_centroid(laws):
    """At a general orientation the built wrench is the nodes' fed ``B(x)`` and ``G(x)`` (to 1e-9 of the torque), differs
    from what they give fed at the centroid by ``m x (G l)`` to first order in ``l`` (5 %: the second order is 4 %
    here), and is not the centroid's."""
    the_pair_reads_at_the_centre_of_mass(laws)


@dataclass(frozen=True)
class MovedToTheCentroid:
    """The fix the anomaly plans, planted: the field provider reads at the body's pose moved by its (body-frame) lever,
    ``x + R l``, for both laws."""

    lever: np.ndarray

    def apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        advance, lever = PointDipoleNode.advance, jnp.asarray(self.lever)

        def moved(node, p, state, boundary_inputs, dt):
            shifted = {key: value.at[:3].add(quaternions.rotate(value[3:], lever)) if key.startswith("points_")
                       else value for key, value in boundary_inputs.items()}
            return advance(node, p, state, shifted, dt)

        monkeypatch.setattr(PointDipoleNode, "advance", moved)


def test_control_with_the_read_point_moved_to_the_centroid_the_pins_fail_and_the_pair_does_no_work(monkeypatch):
    """The planned fix, planted in a fresh graph: the pins above fail on it (so neither restates the read point), and
    its pair does zero work round the operating point's loop and a general one, as the fix intends."""
    MovedToTheCentroid(LEVER).apply(monkeypatch)
    fixed = Laws()
    with pytest.raises(AssertionError):
        the_pair_shows_the_defect_at_the_operating_point(fixed, 360)
    with pytest.raises(AssertionError):
        the_pair_reads_at_the_centre_of_mass(fixed)
    work = fixed.pair_work(SOURCE_MOMENT, SOURCE_AT, ON_THE_AXIS, MOMENT, LEVER, IDENTITY, AXIS_X)
    assert abs(work.net) <= ROUNDINGS * EPS * work.scale
    work = fixed.pair_work(GENERAL_SOURCE, GENERAL_SOURCE_AT, np.zeros(3), GENERAL_MOMENT, LEVER, GENERAL_START,
                           GENERAL_AXIS)
    assert abs(work.net) <= ROUNDINGS * EPS * work.scale
