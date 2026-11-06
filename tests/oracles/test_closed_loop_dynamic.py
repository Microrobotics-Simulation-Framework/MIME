"""The magnetic laws through the build, over whole turns (float64, per module): the closed-loop case's dynamic
counterpart. A body spun at 400 rad/s about x, free (no gravity, no drag, an isotropic inertia, so no gyroscopic
term), its moment across the spin axis and its magnet's centroid off its centre of mass, 15 cm below a fixed magnet of
18.89 A m^2 on the magnet's axis (F0's ledger spike FU3's operating point, its source fixed in the world), both laws
attached, run as the graph runs for 0.1 s (about six turns) at dt 0.2, 0.1 and 0.05 ms. The exact motion conserves
the body's energy, ``E = (1/2) m v^2 + (1/2) I omega^2 + U``, ``U = -m . B(x + R l)`` at the magnet's centroid (in
NumPy, from the run's history, by the dipole's closed form); the drift ``R = E - E_0`` is judged at whole turns of the
body (its y axis's angle about x, crossed by interpolation between steps).

- **The judgement.** The step-independent part of the drift, Richardson's limit from the two finest steps,
  ``L = 2 R(dt/4) - R(dt/2)``, must lie within its own error estimate ``|L(dt/2, dt/4) - L(dt, dt/2)|`` plus 1e-6 of the
  run's summed ``|W . V| dt`` (the laws' work, what the energy exchange is made of); no absolute part.
- **0.2.0** converges at first order to no step-independent part. Measured over 60 configurations drawn before this
  was written (the lever 0.2 to 2 mm in any direction, the moment at any angle across the spin): the limit stayed
  within 2.0e-7 of the summed work, five times inside the floor.
- **The control**: 0.1.0, built through its registered entries, keeps a part independent of the step, about ``pi
  (a . G a) a . (l x m)`` a turn (the static case's closed form; ``m l pi G_xx`` at FU3's point, where it is 5.91e-8 J a
  turn, 1.00 of the closed form at one turn), and fails on every configuration whose axis, lever and moment are not
  coplanar: measured on the same 60, all failed, the least by 3.3e-5 of the summed work, 33 times the floor, and the
  limit was 1.00 times the closed form summed over the turns (median).
- **Whole turns, and fractions of one.** At whole turns 0.2.0's drift is about 1e-11 J against 7.5e-6 J of summed work
  a turn. Judged at fractions of the first turn at FU3's point, 0.1.0 still fails at every fraction from 1 down to
  0.02 (its partial-turn work is independent of the step too), and 0.2.0 passes down to 0.05 and fails at 0.02: below
  about a twentieth of a turn the judgement no longer tells them apart, so it is held at whole turns only. (A two-step
  judgement by the ratio of drifts alone, F0's FU3, passed 0.1.0 at 0.3 turns at another point; three steps and the
  run's scale do not.)
"""

import functools
import warnings

import numpy as np
import pytest

from mime import Apparatus, Body, Experiment, State
from mime.build._maddening import _held, fed_inputs
from mime.dynamics import RigidInertial
from mime.fields import PointDipole
from mime.interactions import KEPT, Magnetic
from mime.scene import Cylinder, Scene, Z
from mime.things import Part, PermanentMagnet
from mime.units import VACUUM_PERMEABILITY, A_m2, kg, kg_m2, m, m_per_s, m_per_s2, mm, rad_per_s, s

K = VACUUM_PERMEABILITY / (4 * np.pi)
MASS, INERTIA, SPIN, MOMENT = 8.5e-5, 1e-10, 400.0, 1.68e-3
SOURCE, DISTANCE = 18.89, 0.15
STEPS = (2e-4, 1e-4, 5e-5)
DURATION = 0.1
#: The floor's share of the run's summed ``|W . V| dt``: five times 0.2.0's largest limit measured, and a thirty-third of
#: 0.1.0's least (the module's docstring).
FLOOR = 1e-6
KEPT_LAWS = (KEPT["magnetic.torque", "0.1.0"][0], KEPT["magnetic.force", "0.1.0"][0])
LAWS = {"0.2.0": (Magnetic.Torque, Magnetic.Force), "0.1.0": KEPT_LAWS}
#: Each configuration: the lever (direction and length, body frame) and the moment's angle across the spin axis (from
#: body y towards body z). The first is FU3's; none has its axis, lever and moment coplanar (``a . (l x m)`` from 0.58
#: to 0.93 of ``|l| |m|``).
CONFIGURATIONS = {
    "fu3": ((0.0, 0.0, 1.0), 1e-3, 0.0),
    "a": ((0.6, -0.4, 0.9), 1e-3, 1.1),
    "b": ((-0.3, 0.8, 0.5), 2e-4, 2.5),
    "c": ((0.7, 0.2, -0.6), 2e-3, 4.0),
    "d": ((0.1, -0.9, -0.4), 5e-4, 5.5),
    "e": ((-0.8, -0.3, 0.5), 1.4e-3, 0.6),
}


def lever_and_moment(name: str) -> tuple[np.ndarray, np.ndarray]:
    direction, length, angle = CONFIGURATIONS[name]
    lever = length * np.asarray(direction) / np.linalg.norm(direction)
    return lever, MOMENT * np.array([0.0, np.cos(angle), np.sin(angle)])


def spun_body(version: str, name: str, dt: float) -> Experiment:
    lever, moment = lever_and_moment(name)
    exp = Experiment("spun_body", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(dt))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((SPIN, 0.0, 0.0)))))
    exp.declare("umr.mass", kg(MASS), reason="a millimetre body")
    exp.declare("umr.inertia", kg_m2((INERTIA,) * 3), reason="isotropic, so the spin keeps no gyroscopic term")
    exp.declare("umr.magnetic_moment", A_m2(tuple(moment)), reason="across the spin axis")
    exp.declare("umr.magnet_centre", m(tuple(lever)), reason="its magnet off its centre of mass")
    exp.add_apparatus(Apparatus(
        "stand", parts={"magnet": Part(Cylinder(radius=mm(4.0), length=mm(6.0), axis=(0.0, 0.0, 1.0)),
                                       at=m((0.0, 0.0, DISTANCE)))},
        magnets={"magnet": PermanentMagnet(direction=(0.0, 0.0, 1.0))}, fields={"magnetic": PointDipole(part="magnet")}))
    exp.declare("stand.magnetic_moment", A_m2(SOURCE), reason="FU3's magnet, fixed")
    for law in LAWS[version]:
        exp.attach(law(), body="umr", field="stand.magnetic")
    return exp


def turned(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


@functools.cache
def drift(version: str, name: str, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The run's energy drift ``E - E_0`` at each step, the body's angle about x (unwrapped), and the summed ``|W . V|
    dt`` of the laws up to each step."""
    lever, moment = lever_and_moment(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = spun_body(version, name, dt).build().graph
    graph.reset_state()
    start = graph.get_node_state("umr")
    _, history = _held(graph).run_scan_with_history(round(DURATION / dt), fed_inputs(graph, None))
    pose = np.concatenate([np.asarray(start["pose"])[None], np.asarray(history["umr"]["pose"])])
    twist = np.concatenate([np.asarray(start["twist"])[None], np.asarray(history["umr"]["twist"])])
    rotations = turned(pose[:, 3:])
    centroid = pose[:, :3] + rotations @ lever
    offset = centroid - np.array([0.0, 0.0, DISTANCE])
    source, distance = np.array([0.0, 0.0, SOURCE]), np.linalg.norm(offset, axis=1)[:, None]
    field = K * (3 * np.sum(source * offset, 1)[:, None] * offset / distance**5 - source / distance**3)
    energy = (0.5 * MASS * np.sum(twist[:, :3] ** 2, 1) + 0.5 * INERTIA * np.sum(twist[:, 3:] ** 2, 1)
              - np.sum((rotations @ moment) * field, 1))
    angle = np.unwrap(np.arctan2(rotations[:, 2, 1], rotations[:, 1, 1]))
    wrench = sum(np.asarray(history[f"magnetic.{law}:umr"]["wrench"]) for law in ("torque", "force"))
    summed = np.concatenate([[0.0], np.cumsum(np.abs(np.sum(wrench * twist[:-1], 1)) * dt)])
    return energy - energy[0], angle, summed


def at_turn(values: np.ndarray, angle: np.ndarray, turns: float) -> float:
    """``values`` where the body's angle first reaches ``turns`` turns, interpolated between the steps either side."""
    target = 2 * np.pi * turns
    index = int(np.argmax(angle >= target))
    assert index > 0, f"the run does not reach {turns} turns"
    share = (target - angle[index - 1]) / (angle[index] - angle[index - 1])
    return float(values[index - 1] + share * (values[index] - values[index - 1]))


def judged(version: str, name: str, turns: float) -> tuple[bool, float, float]:
    """Whether the drift at ``turns`` keeps no part independent of the step, its limit, and the floor's scale."""
    drifts = [at_turn(*drift(version, name, dt)[:2], turns) for dt in STEPS]
    finest = drift(version, name, STEPS[-1])
    summed = at_turn(finest[2], finest[1], turns)
    coarse, fine = 2 * drifts[1] - drifts[0], 2 * drifts[2] - drifts[1]
    return abs(fine) <= abs(fine - coarse) + FLOOR * summed, fine, summed


def whole_turns(name: str) -> int:
    return min(int(drift(version, name, dt)[1][-1] // (2 * np.pi)) for version in LAWS for dt in STEPS)


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_the_pair_0_2_0_keeps_no_part_of_its_energy_drift_independent_of_the_step_over_whole_turns(name):
    turns = whole_turns(name)
    assert turns >= 5
    holds, limit, summed = judged("0.2.0", name, turns)
    print(f"\n{name}: {turns} turns, limit {limit:.3g} J, {abs(limit) / summed:.3g} of the summed work {summed:.3g} J")
    assert holds


def closed_form(name: str) -> float:
    """``-pi (a . G a) a . (l x m)`` a turn about x at the start, ``G`` the dipole's gradient at the body's start: on
    the magnet's axis below it, ``G_xx = -3 K M / d^4``."""
    lever, moment = lever_and_moment(name)
    gradient_xx = -3 * K * SOURCE / DISTANCE**4
    return -np.pi * gradient_xx * float(np.cross(lever, moment)[0])


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_control_the_pair_0_1_0_keeps_its_closed_form_a_turn_whatever_the_step_and_fails(name):
    """Its limit is the static closed form summed over the turns, within 5 % (the body moves a few millimetres toward
    the magnet over the run, which the form at the start leaves out)."""
    turns = whole_turns(name)
    holds, limit, summed = judged("0.1.0", name, turns)
    print(f"\n{name}: {turns} turns, limit {limit:.4g} J against {turns * closed_form(name):.4g} J, "
          f"{abs(limit) / summed:.3g} of the summed work")
    assert not holds
    assert abs(limit / (turns * closed_form(name)) - 1.0) <= 0.05


@pytest.mark.parametrize(("turns", "discriminates"), [(1.0, True), (0.3, True), (0.05, True), (0.02, False)])
def test_at_fractions_of_a_turn_the_judgement_tells_the_versions_apart_down_to_a_twentieth(turns, discriminates):
    """FU3's point, the first turn: 0.1.0 fails at each fraction; 0.2.0 passes down to 0.05 of a turn and fails at
    0.02, where the judgement cannot discriminate."""
    old, new = judged("0.1.0", "fu3", turns), judged("0.2.0", "fu3", turns)
    print(f"\n{turns} turns: 0.1.0 limit {old[1]:.3g} J, 0.2.0 limit {new[1]:.3g} J, summed {new[2]:.3g} J")
    assert not old[0]
    assert new[0] is discriminates
