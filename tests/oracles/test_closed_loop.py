"""The magnetic family's closed-loop case (float64, per module): the laws that together are the derivative of a point
dipole's energy, ``U = -m . B(x + R l)``, do no net work round a closed loop of the body's pose in a static field
(MIME-ANO-137's oracle). The family walk runs the case on every magnetic law in float32; here its parts and its
control, in float64, where its resolution is finest.

- **The anomaly's operating point** (a 10 A m^2 source 50 mm away on its axis, a 1.68 mA m^2 moment, a 1 mm lever,
  one turn about x): ``magnetic.torque`` and ``magnetic.force`` 0.1.0, which read the field at the centre of mass, do
  ``m l pi G_xx`` (2.5334e-6 J, 2.35 % of the loop's summed ``|W|``), and 0.2.0, which read it at the magnet's
  centroid, none.
- **The quadrature's order.** The case sums each loop's work by the periodic trapezoid rule at ``LOOP_POINTS`` (64)
  points, and bounds its error by the difference from every other point. A loop's work is smooth and periodic, so the
  rule converges geometrically: on the draws' closest corner (a 50 A m^2 source 3 cm away, a 2 mm lever, a 5 mm
  translation) 0.2.0's work round a translation loop falls from 5.7e-3 of its summed ``|W|`` at 4 points to 1.3e-5 at
  8 and 3e-13 at 16, about tenfold a point, and is at rounding (2e-16) from 24, where the case's floor is 2.5e-13.
- **The rounding floor**, measured: four times the work's movement when every input of the laws' nodes moves by 64
  units in its last place. Over 300 draws of four loops each, 0.2.0's work stayed within 0.0067 of it in float64 and
  0.0063 in float32.
- **The control**: the pair 0.1.0 fails the case on every draw whose turns carry its lever, by an independent
  prediction of the work it leaves (``oint m . G (a x l) dtheta``, in NumPy) of at least a millionth of ``2 pi |m| |l|
  |G|``; draws whose lever lies along the turns' axes, or whose moment and gradient make the work vanish, have no
  defect to find. Measured before it was written: 149 such draws of 149 failed (300 draws, 164 with a lever).
- **The smallest lever** 0.1.0 still fails at, at the operating point: 1e-14 m in float64, which passes at 5.6e-15 m;
  in float32, 5.6 micrometres, which passes at 3.2 micrometres.
- **A law alone**: 0.2.0's torque without its force is not the derivative of an energy, and fails a turn; the case
  needs ``together=``.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings

from mime.interactions import KEPT, Magnetic
from mime.testing import CaseKind, conformance
from mime.testing._families import Loop, loop_works
from mime.testing._mime_families import PROBE, MagneticLoopDraw, magnetic_loop_draws, magnetic_loop_rig
from mime.units import VACUUM_PERMEABILITY

PROPERTY = settings(deadline=None)
K = VACUUM_PERMEABILITY / (4 * np.pi)
CentreMagneticTorque, _ = KEPT["magnetic.torque", "0.1.0"]
CentreMagneticForce, _ = KEPT["magnetic.force", "0.1.0"]
ZERO, IDENTITY = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
#: The least share of ``2 pi |m| |l| |G|`` the work 0.1.0 leaves round a draw's turns, by the independent prediction,
#: for the draw to carry a defect: three orders of magnitude above the case's float64 resolution at the draws' farthest
#: corner (3e-11 of it, a 0.2 mm lever 20 cm from the source).
GENERIC = 1e-6


def pair(version: str) -> tuple:
    return (Magnetic.Torque(), Magnetic.Force()) if version == "0.2.0" else (CentreMagneticTorque(),
                                                                           CentreMagneticForce())


def anomalys_point(lever: float) -> MagneticLoopDraw:
    """MIME-ANO-137's operating point, with the source below the body on its axis, and one turn about x."""
    return MagneticLoopDraw((0.0, 0.0, -0.05), (0.0, 0.0, 1.0), 10.0, (0.0, 1.68e-3, 0.0), (0.0, 0.0, lever),
                            (Loop(ZERO, ZERO, ZERO, (1.0, 0.0, 0.0), IDENTITY),))


def works(version: str, drawn: MagneticLoopDraw, points: int | None = None):
    laws = pair(version)
    built = magnetic_loop_rig(laws, drawn).build()
    return loop_works(built, laws, PROBE, drawn.loops) if points is None else loop_works(
        built, laws, PROBE, drawn.loops, points=points)


def dipole_gradient(moment: np.ndarray, centre: np.ndarray, point: np.ndarray) -> np.ndarray:
    r = point - centre
    d = float(np.linalg.norm(r))
    along = float(moment @ r)
    return 3 * K / d**5 * (along * np.eye(3) + np.outer(moment, r) + np.outer(r, moment) - 5 * along * np.outer(r, r)
                           / d**2)


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def turned(q: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def predicted_defect(drawn: MagneticLoopDraw, loop: Loop, steps: int = 64) -> tuple[float, float]:
    """The work the pair 0.1.0 leaves round a turn of the body about its centre of mass at the origin, ``oint m . G
    (a x l) dtheta`` with ``G`` at the centre of mass, by Rodrigues' formula in NumPy and no node; and the scale it is
    judged on, ``2 pi |m| |l| |G|``. The torque's own share is exact at a fixed point and leaves nothing, and the
    force's lever ``l x F`` leaves this."""
    source = drawn.source_moment * np.asarray(drawn.source_direction)
    gradient = dipole_gradient(source, np.asarray(drawn.source_at), np.zeros(3))
    axis, start = np.asarray(loop.axis), turned(loop.start)
    moment, lever = np.asarray(drawn.moment), np.asarray(drawn.lever)
    angles = 2 * np.pi * np.arange(steps) / steps
    work = sum((rotation(axis, a) @ start @ moment) @ gradient @ np.cross(axis, rotation(axis, a) @ start @ lever)
               for a in angles) * 2 * np.pi / steps
    return float(work), 2 * np.pi * float(np.linalg.norm(moment) * np.linalg.norm(lever) * np.linalg.norm(gradient, 2))


def test_at_the_anomalys_operating_point_0_1_0_does_m_l_pi_gxx_and_0_2_0_none():
    (old,), (new,) = works("0.1.0", anomalys_point(1e-3)), works("0.2.0", anomalys_point(1e-3))
    gradient = dipole_gradient(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -0.05]), np.zeros(3))
    closed_form = 1.68e-3 * 1e-3 * np.pi * gradient[0, 0]
    print(f"\n0.1.0: {old.describe()}\n0.2.0: {new.describe()}\nm l pi G_xx = {closed_form:.6g} J")
    assert abs(old.work - closed_form) <= old.floor and not old.holds
    assert 0.023 < old.work / old.summed < 0.024
    assert new.holds and abs(new.work) <= 1e-12 * new.summed


#: The draws' closest corner: a 50 A m^2 source 3 cm away, a 2 mm lever, a 5 mm translation loop, a generic axis.
CLOSE = MagneticLoopDraw((0.018, -0.012, 0.02127), tuple(np.ones(3) / np.sqrt(3)), 50.0, (2e-3, -5e-3, 7e-3),
                         (1.2e-3, 0.8e-3, -1.4e-3),
                         (Loop(ZERO, ZERO, ZERO, (0.6, 0.0, 0.8), IDENTITY),
                          Loop(ZERO, (5e-3, 0.0, 0.0), (0.0, 3e-3, 4e-3), ZERO, IDENTITY),
                          Loop(ZERO, (5e-3, 0.0, 0.0), (0.0, 3e-3, 4e-3), (0.6, 0.0, 0.8), IDENTITY)))


def test_the_loops_quadrature_converges_geometrically_and_is_at_rounding_well_below_the_cases_count():
    """0.2.0's work round each of the closest corner's loops, whose exact value is zero, so that what is left is the
    rule's error: it falls by more than a hundred from 8 to 12 points and from 12 to 16, and from 24 points it is below
    the case's rounding floor, as at the case's 64 and at 128."""
    laws = pair("0.2.0")
    built = magnetic_loop_rig(laws, CLOSE).build()
    found = {n: loop_works(built, laws, PROBE, CLOSE.loops, points=n) for n in (8, 12, 16, 24, 64, 128)}
    errors = {n: [abs(w.work) / w.summed for w in held] for n, held in found.items()}
    print("\n" + "\n".join(f"{n:4d} points: " + ", ".join(f"{e:.2g}" for e in held) for n, held in errors.items()))
    translation = 1
    assert errors[8][translation] > 100 * errors[12][translation] > 100 * 100 * errors[16][translation]
    for n in (24, 64, 128):
        assert all(abs(w.work) <= case.floor for w, case in zip(found[n], found[64]))
    assert all(1e-14 < w.floor / w.summed < 1e-12 for w in found[64])


@PROPERTY
@given(magnetic_loop_draws())
def test_control_the_pair_0_1_0_fails_the_case_on_every_draw_whose_turns_carry_its_lever(drawn):
    assume(float(np.linalg.norm(drawn.lever)) > 0.0)
    assume(max(abs(work) / scale for work, scale in (predicted_defect(drawn, loop) for loop in drawn.loops[:2]))
           >= GENERIC)
    found = works("0.1.0", drawn)
    assert not all(work.holds for work in found), "; ".join(work.describe() for work in found)


@pytest.mark.parametrize("lever, fails", [(1e-14, True), (5.6e-15, False)])
def test_the_smallest_lever_the_pair_0_1_0_still_fails_at_is_a_hundredth_of_a_picometre(lever, fails):
    """At the anomaly's operating point, in float64: the case's resolution there, where its floor (2.2e-17 J) meets the
    work 0.1.0 leaves, 2.5e-3 J per metre of lever."""
    (found,) = works("0.1.0", anomalys_point(lever))
    print(f"\n{lever:.3g} m: {found.describe()}")
    assert found.holds is not fails


def test_the_pair_0_2_0_passes_the_closed_loop_case():
    (result,) = conformance(Magnetic.Torque(), "magnetic", together=(Magnetic.Force(),),
                            cases=[CaseKind.CLOSED_LOOP]).values()
    print(f"\n{result}")
    assert result.passed


def test_control_the_torque_0_2_0_alone_is_not_conservative_and_fails_a_turn():
    """A turn about the closest corner's axis with the torque alone: the work it leaves is the force's lever's share,
    which only its partner returns."""
    laws = (Magnetic.Torque(),)
    alone = loop_works(magnetic_loop_rig(laws, CLOSE).build(), laws, PROBE, CLOSE.loops[:1])
    print(f"\n{alone[0].describe()}")
    assert not alone[0].holds
