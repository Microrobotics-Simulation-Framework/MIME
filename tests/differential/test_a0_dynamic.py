"""A0 on its dynamic rungs through the public API, in float32, against the pre-0.3 tag's A0 with its drag surrogate
given the tube's axis: the gate of dev1's phase 7c2.

0.3's A0 here runs ``Arm.Dynamic`` and ``Motor.Dynamic``, closed by one controller (``a0_loops.A0Loops``, inside the
compiled step, every step) that carries the tag's computed-torque joint loop on ``arm_tracking``'s targets and its
rotor's PI velocity loop. Every expected difference was declared before this comparison first ran, with its predicted
size, its cause and how it is checked, in ``a0_dynamic_expected_differences.json``, written by
``predict_a0_dynamic.py`` from the reference alone. 7c1's two differences the kinematic rungs could not close (the
arm's 20 ms trail and the rotor's PI loop) are closed here; what is left of them, and what the dynamic rungs add (the
rotor's inertia and the magnet's reaction on the arm, the loop's read of the rotor's speed, the field reading the
rotor a step late), is declared. A difference found that is not on the list is a finding, reported, and is not added to
the list after the fact.

**What this bounds.** That 0.3's A0 on its dynamic rungs is the tag's with exactly the declared differences, at their
predicted sizes, at these two operating points, in float32. **What it does not.** The drag surrogate's weights are the
reference's (``dejongh_mlp_v2``) on both sides, and MIME-ANO-117 records that they inherit the old wall tables' defect:
this checks the framework against its reference on the same weights, not the surrogate's physics. A0 is as declared:
its magnet has no mass (the arm with the magnet's mass is reported as data, never compared with the 0.2 reference).
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from a0_dynamic_compare import (
    DECLARED,
    FREQUENCY,
    GATED,
    REFERENCE,
    TOLERANCES,
    compare,
    measure,
    missed,
    reference_poses,
    table,
    transient,
)
from a0_runs import REPORTED
from script_a0_dynamic import a0_dynamic

from mime.data import asset

HERE = Path(__file__).parent
#: The finding not on the declaration, with its control (``measure_a0_dynamic_findings.py``).
FINDINGS = json.loads((HERE / "references" / "a0_dynamic_findings.json").read_text())


def units_apart(key: str, run) -> tuple[float, float]:
    """How many units in its last place 0.3's single-precision rotor angle is from the emulation's at the window's
    start and at its end: the finding's measure (``measure_a0_dynamic_findings.py``)."""
    held = FINDINGS["by_frequency"][key]["emulated_angle"]
    found = []
    for end in ("window_start", "window_end"):
        angle = held[end]["angle"]
        found.append(abs(float(run.history.rotor_angle[held[end]["step"]]) - angle)
                     / float(np.spacing(np.float32(abs(angle)))))
    return found[0], found[1]


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32: it overrides the folder's float64 fixture, and restores the mode after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module", params=list(FREQUENCY))
def measured(request):
    found, run = measure(request.param)
    return request.param, found, run


def test_a0_on_its_dynamic_rungs_matches_its_reference_with_exactly_the_declared_differences(measured):
    """Every gated quantity holds its declared tolerance about its prediction, but those the finding moves (the
    rotation rates by phase and the motor's angle rate, which read the rotor's single-precision angle), which hold
    their tolerance and the move of one unit in the angle's last place as the window opens: the finding's bound,
    measured by its control (``measure_a0_dynamic_findings.py``). The declaration is unchanged."""
    key, found, run = measured
    rows = compare(found, key)
    print(f"\nfloat32, dynamic rungs, {FREQUENCY[key]:g} Hz: the reference, 0.3, the declared prediction, and 0.3 "
          f"against it\n" + table(rows) + "\n" + "\n".join(
              f"{name}: {found[name]!r} (reported, not gated: {why})" for name, why in REPORTED.items()
              if name in found))
    one_unit = FINDINGS["by_frequency"][key]["one_unit_moves"]
    apart = sum(units_apart(key, run))
    beyond = [row.quantity for row in rows if not row.holds]
    print(f"beyond the declared tolerance: {beyond}; the rotor's angle {apart:g} units from the emulation's at the "
          f"window's ends, one unit's moves {one_unit}")
    assert set(beyond) <= set(one_unit), beyond
    for row in rows:
        if row.quantity in beyond:
            assert abs(row.miss) <= row.tolerance + apart * one_unit[row.quantity], (row.quantity, row.miss, apart)


def test_the_rotors_spin_up_turns_the_wrist_as_predicted(measured):
    """The rotor's inertia on the arm (the declaration's ``rotor-axial-inertia``): as the PI loop spins the rotor up,
    the drive's reaction turns joint 6 backwards and the joint loop pulls it back. Its largest excursion and speed are
    those the emulation predicted, within the declared transient tolerance; the tag's arm, and 0.3's kinematic one, do
    not move at all there."""
    key, _, run = measured
    found = transient(run)
    predicted = DECLARED["transient"]["predicted"][key]
    tolerance = DECLARED["transient"]["tolerance"]
    print(f"\n{key}: {found} against {predicted}, tolerance {tolerance:.3g}")
    for name, value in found.items():
        assert abs(value / predicted[name] - 1) <= tolerance, name


def test_no_joint_reaches_a_stop(measured):
    """The declaration's ``stops``: no joint reaches a limit, so the stops' impulses (0.2 clipped and kept the velocity)
    never act."""
    from chain_cases import AR4
    from mime.core.multibody import read_urdf

    key, _, run = measured
    tree = read_urdf(AR4).tree
    lower, upper = np.asarray(tree.articulation.lower), np.asarray(tree.articulation.upper)
    margin = float(np.min(np.minimum(run.joints - lower, upper - run.joints)))
    print(f"\n{key}: least margin to a limit {margin:.4g} rad")
    assert margin > 0.0


def test_what_cannot_discriminate_is_reported_and_why(measured):
    """The radial offset sits on the projection's constraint, and the locked ratio of the body's rotation to the field's
    is 1 whatever the frequency."""
    _, found, _ = measured
    assert abs(found["radial_offset_mean_m"] / 1.0002e-3 - 1) <= 1e-7
    assert abs(found["rotation_ratio_body_over_field"] - 1) <= 1e-4


def test_the_comparison_tells_one_frequency_from_the_other(measured):
    """The control for the whole comparison: held to the other frequency's prediction, 0.3 misses every gated quantity
    but the distance from the body, which the controller holds at either frequency."""
    key, found, _ = measured
    other = "10hz" if key == "3hz" else "3hz"
    rows = compare(found, key, against=other)
    print("\nheld to the other frequency's prediction\n" + table(rows))
    assert [row.quantity for row in rows if row.holds] == ["magnet_body_distance_minus_standoff_mean_m"]


def test_the_magnets_pose_through_its_mount_is_the_references_moved_by_the_prediction(measured):
    """The rotor's pose (the flange's, the motor on it, the rotor turned about its axis) at the reference's sampled
    steps in the window: its position, its spin axis and its phase about that axis are the reference's moved by the
    emulation's predicted difference, within the declared tolerances; the phase, which the finding moves, within its
    tolerance and the move of one unit of the rotor's angle."""
    from a0_poses import difference, misses

    key, _, run = measured
    time, reference = reference_poses(key)
    poses = np.load(HERE / "a0_dynamic_poses.npz")
    predicted = difference(poses[f"{key}_new"], poses[f"{key}_tag"])
    found = misses(run.rotor_poses[::10], reference, predicted, time, FREQUENCY[key])
    tolerances = dict(DECLARED["magnet_pose"]["tolerances"]["values"])
    print(f"\n{key}: " + ", ".join(f"{name} {value:.3g} (tolerance {tolerances[name]:.3g})"
                                     for name, value in found.items()))
    # the finding: the phase reads the rotor's single-precision angle, which holds its tolerance and the units the angle
    # is apart from the emulation's at the window's ends
    held = FINDINGS["by_frequency"][key]["emulated_angle"]
    tolerances["magnet_phase_rad"] += sum(n * float(np.spacing(np.float32(abs(held[end]["angle"]))))
                                          for n, end in zip(units_apart(key, run), ("window_start", "window_end"),
                                                            strict=True))
    assert all(found[name] <= tolerances[name] for name in found), found


def test_the_build_is_the_declared_one():
    """What the declaration's entries rest on: the controller reads the state after the step before and writes this
    step, every step, with no latency; the field reads the rotor's pose a step late (``field-reads-the-rotor-late``)
    while each law's reaction reaches the rotor within the step; the motor reads the arm's new pose and rates, and the
    arm reads the motor's reaction and momentum a step late; the joint loop read the arm's built tree; the arm has no
    damping and no friction; the drag names the reference's weights."""
    from mime.build import Lag

    built = a0_dynamic().build()
    lags = {(edge.source[0], edge.source[1], edge.target[0]): edge.lag for edge in built.report.lags}
    for source in (("umr", "pose"), ("rpm.arm", "joint_positions"), ("rpm.arm", "joint_velocities"),
                   ("rpm.motor", "frame_poses"), ("rpm.motor", "frame_angular_velocities")):
        assert lags[(*source, "controller")] is Lag.PREVIOUS_STEP, source
    assert lags[("controller", "write_joint_torques", "rpm.arm")] is Lag.THIS_STEP
    assert lags[("controller", "write_torque", "rpm.motor")] is Lag.THIS_STEP
    assert lags[("rpm.motor", "frame_poses", "rpm.magnetic")] is Lag.PREVIOUS_STEP
    assert lags[("magnetic.torque:umr.reaction", "link_wrench", "rpm.motor")] is Lag.THIS_STEP
    assert lags[("magnetic.force:umr.reaction", "link_wrench", "rpm.motor")] is Lag.THIS_STEP
    assert lags[("rpm.arm", "frame_poses", "rpm.motor")] is Lag.THIS_STEP
    assert lags[("rpm.arm", "frame_angular_velocities", "rpm.motor")] is Lag.THIS_STEP
    assert lags[("rpm.motor", "reaction", "rpm.arm")] is Lag.PREVIOUS_STEP
    assert lags[("rpm.motor", "momentum", "rpm.arm")] is Lag.PREVIOUS_STEP
    schedule = built.graph.schedule
    assert schedule.index("controller") < schedule.index("rpm.magnetic") < schedule.index("rpm.arm") < \
        schedule.index("rpm.motor") and not built.graph.is_multirate
    assert any(line.strip().startswith("models: its law read the built tree of rpm.arm")
               for line in built.report.controller), built.report.controller
    arm = built.graph.get_node("rpm.arm")
    assert not any(np.asarray(arm.params["damping"])) and arm.constants()["rubbing"] == ()
    document = json.loads(json.dumps(a0_dynamic().document(), default=str))
    assert '"dejongh_mlp_v2"' in json.dumps(document) and asset("dejongh_mlp_v2").sha256 in json.dumps(document)


def test_the_extended_emulation_is_7c1s_with_every_change_off():
    """The control on the prediction's tool: with every change of the dynamic rungs off, the extended emulation is
    7c1's emulation of the tag's scheme, to the bit, over the first second."""
    jax.config.update("jax_enable_x64", True)
    try:
        import predict_a0 as P
        import predict_a0_dynamic as D

        old = P.emulate(P.TAG, 3.0, duration=1.0)
        new, _, _ = D.emulate(D.TAG_D, 3.0, duration=1.0)
    finally:
        jax.config.update("jax_enable_x64", False)
    for name in ("position", "velocity", "orientation", "angular_velocity", "field"):
        assert np.array_equal(getattr(old.body, name), getattr(new.body, name)), name
    assert np.array_equal(old.rotor_position, new.rotor_position)
    assert np.array_equal(old.rotor_angular_velocity, new.rotor_angular_velocity)


def test_the_emulation_is_the_declared_prediction():
    """The prediction's basis, rerun at 10 Hz: 0.3's dynamic scheme, emulated, gives the quantities recorded when the
    differences were declared."""
    jax.config.update("jax_enable_x64", True)
    try:
        import predict_a0_dynamic as D

        history, _, _ = D.emulate(D.NEW_D, 10.0)
        found = D.steady(history, 10.0, D.NEW_D)
    finally:
        jax.config.update("jax_enable_x64", False)
    recorded = json.loads((HERE / "a0_dynamic_emulation.json").read_text())["new"]["10hz"]
    for name in GATED:
        assert abs(D.P.difference(name, found[name], recorded[name])) <= 1e-9, name


def test_every_tolerance_is_the_declarations_and_7c1s_where_it_carries_over():
    """The body's and the tracking quantities are held to 7c1's tolerances (its rule does not look at 0.3); the
    rotor's speed to the larger of one float32 epsilon and three times its read's rounding; the motor's angle rate to
    one epsilon."""
    seven = DECLARED["tolerances"]["from_7c1"]["values"]
    for name in GATED:
        if name in seven and name != "rotor_rate_mean_hz":
            assert TOLERANCES[name] == seven[name], name
    assert TOLERANCES["rotor_rate_mean_hz"] >= 1e-7 and TOLERANCES["rotor_angle_rate_mean_hz"] == 1e-7
    assert set(REFERENCE["3hz"]) >= set(GATED) - {"rotor_angle_rate_mean_hz"}


def test_the_findings_control_explains_it():
    """The finding's control, from its record: the emulation with its rotor angle set to 0.3's one step before the
    window opens gives 0.3's rotation rates, motor's angle rate and magnet's phase within their declared tolerances,
    where with its own angle it may not."""
    poses = DECLARED["magnet_pose"]["tolerances"]["values"]
    for key, held in FINDINGS["by_frequency"].items():
        against = held["runs"]["0.3's angle"]["0.3_against_it"]
        for name in ("body_rotation_rate_hz", "field_rotation_rate_hz", "rotor_angle_rate_mean_hz"):
            assert abs(against[name]) <= TOLERANCES[name], (key, name, against[name])
        assert abs(against["magnet_phase_rad"]) <= poses["magnet_phase_rad"], (key, against["magnet_phase_rad"])
