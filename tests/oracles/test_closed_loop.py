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
  units in its last place. The draws are at a body's scale: each direction from three components 0.1 to 1 in size,
  each signed, so none has a component that underflows (a box's draw offers 5e-227). Over 1,100 unfiltered draws of
  four loops each, 0.2.0's work stayed within 0.0076 of its tolerance in float64 (median 3.3e-4) and 0.0039 in float32
  (median 1.0e-4).
- **The control**: the pair 0.1.0, built through its registered entries (``mime.interactions.KEPT``) and evaluated as
  the case evaluates any law, fails the case on every draw with a lever, but the geometry that leaves it exactly no
  work: the work it leaves round a turn about the centre of mass averages to ``-pi (a . G a) a . (l x m)``
  (``turn_defect``, against the quadrature ``predicted_defect``), zero where the axis, lever and moment are coplanar
  (Hypothesis draws the lever along the moment, a real geometry) or ``a . G a`` is zero. Measured before it was
  written, over the same 1,100 draws: in float64, 505 had a lever, 15 of them that geometry, and all 490 others failed
  (by at least 1.4e8 times the tolerance); in float32, 446, 24 and all 422 (by at least 1.23 times it). The control
  draws a lever on every draw, so its filter removes that geometry alone (about 3 %). **Float32 is not every draw**: on
  a second, larger sample (1,200 draws, fresh seeds, 524 with a lever outside that geometry), 0.1.0 failed 521 and
  passed 3, a weak field and a short lever (0.2 to 0.64 mm at 14.5 to 17 cm), their work 0.59, 0.77 and 0.80 times the
  tolerance: below float32's resolution, which a lever up to 2 mm at 20 cm cannot always clear (the work goes as the
  lever over the distance). The float32 draws' control is therefore "fails somewhere in the walk's 200", and the
  every-draw control is this module's float64 one.
- **The smallest lever** 0.1.0 still fails at, at the operating point: 1e-14 m in float64, which passes at 5.6e-15 m;
  in float32, 5.6 micrometres (1.21 times the tolerance), which passes at 4.5 (0.97), both pinned below.
- **A law alone**: 0.2.0's torque without its force is not the derivative of an energy, and fails a turn; the case
  needs ``together=``.
- **The pair with its reactions** (``pair_loop_works``): the laws and their reactions on the field's source, as one
  potential law, do no net work round closed loops of the source's pose with the body held (turns about its centre,
  a translation, both) and of the two together (each translated a quarter turn apart). At the anomaly's operating
  point, 0.2.0's pair does none round a turn of the source about its axis's normal; 0.1.0's, whose reaction carries
  the lever's couple ``l x F`` where the field was read at the centre of mass, does, and fails every drawn source loop
  with a lever (the control).
- **Closed loops only.** Every loop the case drives returns the pose to its start (whole turns, a closed translation):
  over an open arc a conservative pair's work is the energy's change, not zero, so zero work cannot judge it. The
  case's dynamic counterpart through the build, over whole turns, is ``test_closed_loop_dynamic.py``.
"""

import jax
import numpy as np
import pytest
from hypothesis import assume, given, settings

from mime.interactions import KEPT, Magnetic
from mime.testing import CaseKind, conformance
from mime.testing._families import Loop, loop_works, pair_loop_works
from mime.testing._mime_families import PROBE, MagneticLoopDraw, magnetic_loop_draws, magnetic_loop_rig
from mime.units import VACUUM_PERMEABILITY

PROPERTY = settings(deadline=None)
K = VACUUM_PERMEABILITY / (4 * np.pi)
CentreMagneticTorque, _ = KEPT["magnetic.torque", "0.1.0"]
CentreMagneticForce, _ = KEPT["magnetic.force", "0.1.0"]
ZERO, IDENTITY = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
#: How near zero, as a share of ``2 pi |m| |l| |G|``, the closed form of the work 0.1.0 leaves round a turn is taken
#: to be zero: its rounding is about 1e-16 of that scale, and the smallest work of a draw not exactly degenerate,
#: measured over 1,100 draws, is far above it (the module's docstring).
NO_WORK = 1e-12


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


def turn_defect(drawn: MagneticLoopDraw, loop: Loop) -> tuple[float, float]:
    """The work the pair 0.1.0 leaves round a turn of the body about its centre of mass at the origin, in closed form,
    and the scale it is judged on, ``2 pi |m| |l| |G|``: ``oint m . G (a x l) dtheta`` with ``m`` and ``l`` turned
    together about the axis ``a`` averages, over the turn, to ``-pi (a . G a) a . (l x m)``, since ``G`` is symmetric
    and its trace is zero (the field is free of curl and divergence). It is zero exactly where the axis, the lever and
    the moment are coplanar (the lever along the moment, among them) or ``a . G a`` is zero."""
    source = drawn.source_moment * np.asarray(drawn.source_direction)
    gradient = dipole_gradient(source, np.asarray(drawn.source_at), np.zeros(3))
    axis, start = np.asarray(loop.axis), turned(loop.start)
    moment, lever = start @ np.asarray(drawn.moment), start @ np.asarray(drawn.lever)
    work = -np.pi * float(axis @ gradient @ axis) * float(axis @ np.cross(lever, moment))
    return work, 2 * np.pi * float(np.linalg.norm(moment) * np.linalg.norm(lever) * np.linalg.norm(gradient, 2))


def degenerate(drawn: MagneticLoopDraw) -> bool:
    """Whether the draw's turns leave 0.1.0 exactly no work: no lever, or each turn's closed form zero to its rounding
    (``turn_defect``)."""
    if float(np.linalg.norm(drawn.lever)) == 0.0:
        return True
    return all(abs(work) <= NO_WORK * scale for work, scale in (turn_defect(drawn, loop) for loop in drawn.loops[:2]))


@PROPERTY
@given(magnetic_loop_draws())
def test_the_turns_closed_form_is_the_independent_quadratures(drawn):
    """``turn_defect``'s closed form against ``predicted_defect``'s quadrature of the integrand, by Rodrigues' formula,
    on every draw with a lever: they agree to 1e-14 of the scale (measured: 2.3e-16 over 300 draws)."""
    assume(float(np.linalg.norm(drawn.lever)) > 0.0)
    for loop in drawn.loops[:2]:
        (closed, scale), (quadrature, _) = turn_defect(drawn, loop), predicted_defect(drawn, loop)
        assert abs(closed - quadrature) <= 1e-14 * scale


@PROPERTY
@given(magnetic_loop_draws(with_zero_lever=False))
def test_control_the_pair_0_1_0_fails_the_case_on_every_draw_with_a_lever(drawn):
    """Filtered only of the draws whose turns leave it exactly no work (``degenerate``: no lever, or each turn's axis,
    lever and moment coplanar, or ``a . G a`` zero), which can pass; every other draw fails, as measured before it was
    written (the module's docstring)."""
    assume(not degenerate(drawn))
    found = works("0.1.0", drawn)
    assert not all(work.holds for work in found), "; ".join(work.describe() for work in found)


@pytest.mark.parametrize("lever, fails", [(1e-14, True), (5.6e-15, False)])
def test_the_smallest_lever_the_pair_0_1_0_still_fails_at_is_a_hundredth_of_a_picometre(lever, fails):
    """At the anomaly's operating point, in float64: the case's resolution there, where its floor (2.2e-17 J) meets the
    work 0.1.0 leaves, 2.5e-3 J per metre of lever."""
    (found,) = works("0.1.0", anomalys_point(lever))
    print(f"\n{lever:.3g} m: {found.describe()}")
    assert found.holds is not fails


@pytest.fixture
def float32():
    """The module's float64, switched off for one test and restored: the family walk runs the case in float32."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.mark.parametrize("lever, fails", [(1.1e-5, True), (5.6e-6, True), (4.5e-6, False), (2.2e-6, False)])
def test_the_smallest_lever_the_pair_0_1_0_still_fails_at_in_float32_is_a_few_micrometres(float32, lever, fails):
    """At the anomaly's operating point, in float32 (the family walk's dtype): the case's resolution there, where its
    floor meets the work 0.1.0 leaves. The work is 2.4 times the tolerance at 11 micrometres, 1.21 at 5.6 (fails),
    0.97 at 4.5 and 0.48 at 2.2 (pass): a lever below that is not seen in float32. That is why the every-draw control
    is the float64 one above, and the family walk asks of this one that 0.1.0 fail the case somewhere in its draws."""
    assert not jax.config.jax_enable_x64
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


def pair_works(version: str, drawn: MagneticLoopDraw):
    laws = pair(version)
    built = magnetic_loop_rig(laws, drawn).build()
    placement = built.graph.get_node("stand.magnetic").constants()["placement"]
    return pair_loop_works(built, laws, PROBE, drawn.pair_loops(placement))


def source_turn_defect(drawn: MagneticLoopDraw, axis) -> tuple[float, float]:
    """The work the pair 0.1.0's reactions leave round a turn of the source about its centre, the body held, in closed
    form, and the scale it is judged on, ``2 pi |l| |m| |G|``: 0.1.0's reaction is the two point dipoles' own, the
    body's read at its centre of mass, but for the couple ``-l x F`` of the lever, whose work round the turn is
    ``-oint (l x F) . a dtheta``; ``F`` is linear in the source's moment, which averages over the turn to its part
    along the axis, so the work is ``-2 pi (l x F[(m_s . a) a]) . a``. Zero where the source's moment is across the
    axis, or the lever, the force and the axis are coplanar."""
    a = np.asarray(axis) / np.linalg.norm(axis)
    source = drawn.source_moment * np.asarray(drawn.source_direction)
    start = turned(drawn.loops[0].start)
    moment, lever = start @ np.asarray(drawn.moment), start @ np.asarray(drawn.lever)
    gradient = dipole_gradient(float(source @ a) * a, np.asarray(drawn.source_at), np.zeros(3))
    force = gradient.T @ moment
    full = dipole_gradient(source, np.asarray(drawn.source_at), np.zeros(3))
    return (-2 * np.pi * float(np.cross(lever, force) @ a),
            2 * np.pi * float(np.linalg.norm(lever) * np.linalg.norm(moment) * np.linalg.norm(full, 2)))


def source_degenerate(drawn: MagneticLoopDraw) -> bool:
    """Whether the draw's pure turns of the source leave 0.1.0 exactly no work (``source_turn_defect``)."""
    turns = [shape[4] for shape in drawn.pairs if not any(np.any(shape[k]) for k in range(4))]
    return all(abs(work) <= NO_WORK * scale for work, scale in (source_turn_defect(drawn, axis) for axis in turns))


#: A turn of the source about an axis with a part along its moment and a part across it, at the anomaly's point.
SLANTED = tuple(np.array([0.6, 0.0, 0.8]))


def test_the_pair_with_its_reactions_does_no_work_round_the_source_s_loops_at_the_anomalys_point():
    held = anomalys_point(1e-3)
    drawn = MagneticLoopDraw(*[getattr(held, f) for f in ("source_at", "source_direction", "source_moment",
                                                          "moment", "lever", "loops")],
                             pairs=((ZERO, ZERO, ZERO, ZERO, SLANTED),))
    (new,), (old,) = pair_works("0.2.0", drawn), pair_works("0.1.0", drawn)
    predicted, _ = source_turn_defect(drawn, SLANTED)
    print(f"\n0.2.0: {new.describe()}\n0.1.0: {old.describe()}; its closed form {predicted:.6g} J")
    assert new.holds and not old.holds
    assert abs(old.work - predicted) <= 1e-9 * abs(predicted)


@PROPERTY
@given(magnetic_loop_draws(with_zero_lever=False))
def test_control_the_pair_0_1_0_fails_a_loop_of_its_source_on_every_draw_with_a_lever(drawn):
    """Filtered only of the draws whose source turns leave it exactly no work (``source_degenerate``)."""
    assume(not source_degenerate(drawn))
    found = pair_works("0.1.0", drawn)
    assert not all(work.holds for work in found), "; ".join(work.describe() for work in found)


@PROPERTY
@given(magnetic_loop_draws())
def test_the_pair_0_2_0_with_its_reactions_does_no_work_round_every_drawn_pair_loop(drawn):
    found = pair_works("0.2.0", drawn)
    assert all(work.holds for work in found), "; ".join(work.describe() for work in found if not work.holds)
