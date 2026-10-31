"""A0 on its dynamic rungs through the public API, in float32, against the pre-0.3 tag's A0 with its drag surrogate
given the tube's axis: the gate of dev1's phase 7c2, by two declarations.

0.3's A0 here runs ``Arm.Dynamic`` and ``Motor.Dynamic``, closed by one controller (``a0_loops.A0Loops``, inside the
compiled step, every step) that carries the tag's computed-torque joint loop on ``arm_tracking``'s targets and its
rotor's PI velocity loop. Every expected difference was declared before this comparison first ran, with its predicted
size, its cause and how it is checked, in ``a0_dynamic_expected_differences.json``, written by
``predict_a0_dynamic.py`` from the reference alone. 7c1's two differences the kinematic rungs could not close (the
arm's 20 ms trail and the rotor's PI loop) are closed here; what is left of them, and what the dynamic rungs add (the
rotor's inertia and the magnet's reaction on the arm, the loop's read of the rotor's speed, the field reading the
rotor a step late), is declared. A difference found that is not on the list is a finding, reported, and is not added to
the list after the fact.

**Two declarations.** The first held at 3 Hz on the build's machine; at 10 Hz, and at 3 Hz on some of CI's runners,
its gates that read the rotor's single-precision angle met the finding (0.3's angle and the emulation's reach the window
a unit of its last place apart, ``references/a0_dynamic_findings.json``), and they are reported as measured, not
gated. The second
(``a0_dynamic_expected_differences_b.json``, ``predict_a0_dynamic_b.py``, the maintainer's option A), written after
that run was seen and before its own, reads every phase against the angle of the same run, gates the absolute phase
where the angle's rounding does not reach it, and holds the angle's own rounding to a derived bound: it is the gate.

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
    ANGLE_READING,
    DECLARED,
    FINDINGS,
    FREQUENCY,
    SECOND,
    failures,
    second,
    second_failures,
    second_table,
    units_apart,
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


def test_a0_on_its_dynamic_rungs_matches_its_reference_with_exactly_the_second_declarations_differences(measured):
    """The gate: every quantity of the second declaration holds its tolerance about its prediction, at both
    frequencies (``a0_dynamic_compare.second``): the first's gates that never read the angle, carried; the phases read
    against the angle of the same run; the rotor frame's phase against its angle; the motor's angle offset from the
    emulation's across the window, to one unit below each power of two crossed; the angle at the start, exactly; and
    joint 6's kick, to ten times the single-precision twin's gap."""
    key, found, run = measured
    gates = second(key, found, run)
    print(f"\nfloat32, dynamic rungs, {FREQUENCY[key]:g} Hz, against the second declaration\n" + second_table(gates))
    assert second_failures(gates) == []


def test_the_first_declaration_holds_but_where_the_finding_moves_it(measured):
    """The first declaration, as committed: every gated quantity holds its declared tolerance about its prediction
    but those that read the rotor's single-precision angle (``ANGLE_READING``), which the finding moves and which are
    reported here as measured, not gated, at both frequencies: at 10 Hz everywhere, and at 3 Hz on some of CI's
    runners, where the angle reaches the window a unit off too (``references/a0_dynamic_findings.json``). The finding's
    allowance, a widening of those tolerances after the 10 Hz miss was seen, is not a gate."""
    key, found, run = measured
    rows = compare(found, key)
    print(f"\nfloat32, dynamic rungs, {FREQUENCY[key]:g} Hz: the reference, 0.3, the declared prediction, and 0.3 "
          f"against it\n" + table(rows) + "\n" + "\n".join(
              f"{name}: {found[name]!r} (reported, not gated: {why})" for name, why in REPORTED.items()
              if name in found))
    print(f"the rotor's angle {units_apart(key, run)} units from the emulation's at the window's start and end")
    assert [name for name in failures(rows) if name not in ANGLE_READING] == []


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


def test_the_second_declaration_tells_one_frequency_from_the_other(measured):
    """Its control for the whole comparison: held to the other frequency's prediction, every carried gate but the
    distance misses, and so does every phase gate the declaration said would (each whose two predictions are further
    apart than its tolerance)."""
    key, found, run = measured
    other = "10hz" if key == "3hz" else "3hz"
    gates = {gate.quantity: gate for gate in second(key, found, run, against=other)}
    print("\nheld to the other frequency's prediction\n" + second_table(list(gates.values())))
    carried = [name for name in SECOND["carried"]["tolerances"] if name != "magnet_body_distance_minus_standoff_mean_m"]
    expected = SECOND["frequency_control"]["held_to_the_other_frequencys_prediction_these_miss"]
    assert [name for name in [*carried, *expected] if gates[name].holds] == []
    assert gates["magnet_body_distance_minus_standoff_mean_m"].holds


def test_the_magnets_pose_through_its_mount_is_the_references_moved_by_the_prediction(measured):
    """The rotor's pose (the flange's, the motor on it, the rotor turned about its axis) at the reference's sampled
    steps in the window: its position and its spin axis are the reference's moved by the emulation's predicted
    difference, within the first declaration's tolerances. Its phase about the axis reads the rotor's angle and is the
    finding's, reported at both frequencies (at 3 Hz it holds here and misses on some of CI's runners); the second
    declaration gates it against the angle (``magnet_phase_against_rotor_angle_rad``)."""
    from a0_poses import difference, misses

    key, _, run = measured
    time, reference = reference_poses(key)
    poses = np.load(HERE / "a0_dynamic_poses.npz")
    predicted = difference(poses[f"{key}_new"], poses[f"{key}_tag"])
    found = misses(run.rotor_poses[::10], reference, predicted, time, FREQUENCY[key])
    tolerances = dict(DECLARED["magnet_pose"]["tolerances"]["values"])
    print(f"\n{key}: " + ", ".join(f"{name} {value:.3g} (tolerance {tolerances[name]:.3g})"
                                     for name, value in found.items()))
    assert all(found[name] <= tolerances[name] for name in found if name != "magnet_phase_rad"), found


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
    """The predictions' basis, rerun at 10 Hz: 0.3's dynamic scheme, emulated (its trace on, which changes nothing it
    computes), gives the quantities recorded when each declaration was written: the first's, and the second's phases
    against the angle and the angles its offset is read from."""
    jax.config.update("jax_enable_x64", True)
    try:
        import predict_a0_dynamic as D
        from a0_phases import EVERY, Phases, quantities
        from a0_runs import SCRIPT_A_FRAME

        trace: dict[str, list[float]] = {}
        history, _, poses = D.emulate(D.NEW_D, 10.0, trace=trace)
        found = D.steady(history, 10.0, D.NEW_D)
        body, angle = history.body, np.asarray(history.rotor_angle)
        integral = np.concatenate([[0.0], np.asarray(trace["integral"])])
        second_found = quantities(Phases(body.time[::EVERY], body.field[::EVERY], body.orientation[::EVERY],
                                         body.position[::EVERY], angle[::EVERY], poses, integral[::EVERY],
                                         SCRIPT_A_FRAME), 10.0)
    finally:
        jax.config.update("jax_enable_x64", False)
    recorded = json.loads((HERE / "a0_dynamic_emulation.json").read_text())["new"]["10hz"]
    for name in GATED:
        assert abs(D.P.difference(name, found[name], recorded[name])) <= 1e-9, name
    second_recorded = json.loads((HERE / "a0_dynamic_emulation_b.json").read_text())
    for name, value in second_found.items():
        tolerance = SECOND["gates"][name]["tolerance"]["value"]
        assert abs(value - second_recorded["new"]["10hz"][name]) <= 1e-3 * tolerance, name
    assert [float(angle[step]) for step in second_recorded["angle"]["10hz"]["steps"]] == \
        second_recorded["angle"]["10hz"]["angles"]


def test_the_second_declarations_floors_are_its_rule_on_the_reference_pair():
    """The second declaration's rounding floor, recomputed from the reference's copied arrays: for each phase read
    against the angle, ten times the gap between the reference's float32 and mixed runs (the angle cancels, so the
    rule is no longer blind: the pair share the angle bit for bit), and the value its tolerance is the largest of."""
    from a0_phases import quantities, reference_phases

    for name, gate in SECOND["gates"].items():
        floor = max(10 * abs(quantities(reference_phases(f"f32_{key}"), f)[name]
                             - quantities(reference_phases(f"mixed_f64_{key}"), f)[name])
                    for key, f in FREQUENCY.items())
        tolerance = gate["tolerance"]
        assert floor == pytest.approx(tolerance["rounding_floor_reference_pair"], rel=1e-9, abs=1e-300), name
        assert tolerance["value"] == max(tolerance["rounding_floor_reference_pair"], tolerance["emulation_resolution"],
                                         tolerance["rounding_floor_single_precision_arm"]), name


def test_the_copied_reference_arrays_are_the_ones_recorded():
    """Every array copied from phase 0's runs into ``references/a0/`` for 7c2 (the rotor poses and the phases) is the
    file whose sha256 ``freeze_a0_phase_arrays.py`` recorded, and each names the run it came from."""
    import hashlib

    folder = HERE / "references" / "a0"
    record = json.loads((folder / "copied_arrays.json").read_text())
    assert len(record) == 6
    for name, entry in record.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == entry["sha256"], name
        assert entry["from"].removesuffix(".npz") == name.split("_phases_")[0].split("_rotor_")[0], name


def test_the_angle_offsets_bound_rests_on_its_premise_and_holds_on_the_twin():
    """The motor's angle offset may move across the window by one unit of its last place below each power of two the
    angle crosses (128 rad at 3 Hz, 512 rad at 10 Hz), and nothing between: that rests on the increments rounding
    alike, which the emulation's least distance from a rounding midpoint over the window (hundreds of the increment's
    units) makes certain for increments that agree to within the swap test's 8 a step. The single-precision twin,
    whose angle parts from the emulation's as 0.3's did, holds the bound (at it, at 10 Hz)."""
    record = json.loads((HERE / "a0_dynamic_emulation_b.json").read_text())
    offset = SECOND["motor_angle_offset_change_rad"]
    for key in FREQUENCY:
        angle = record["angle"][key]
        assert angle["midpoint_margin_in_increment_units"] > 8, key
        (power,) = angle["crossed"]
        assert offset["tolerance"][key] == float(np.spacing(np.float32(power / 2))), key
        assert abs(offset["checked_on_the_twin"][key]) <= offset["tolerance"][key], key
    assert SECOND["the_twin_reproduces_the_finding"]["3hz"] == [0.0, 0.0]
    assert [abs(v) for v in SECOND["the_twin_reproduces_the_finding"]["10hz"]] == [2.0 ** -15, 2.0 ** -14]


def test_the_second_transient_tolerance_is_the_twins_and_a_4_percent_j_is_far_beyond_it():
    """The kick's tolerance is ten times the gap between the emulation and its single-precision twin, and a 4 % error
    in the rotor's J (0.66 of the excursion per unit, the review's hand model) is beyond it by more than 1.5 times: the
    controls plant it (``test_a0_dynamic_b_controls.py``). The first declaration's 2.8e-2 was not."""
    transient = SECOND["transient"]
    assert transient["tolerance"] == pytest.approx(10 * transient["twin_gap"], rel=1e-12)
    assert transient["a_4_percent_error_in_J"]["in_tolerances"] > 1.5
    assert DECLARED["transient"]["tolerance"] > transient["a_4_percent_error_in_J"]["moves_the_excursion_by"]


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


def test_with_the_magnets_mass_the_joint_loop_reads_the_tree_the_magnet_is_grafted_on():
    """A0 with its magnet's mass (reported as data, never compared): the build grafts the magnet's 0.144 kg on the
    flange, the rotor's moment about its axis is the motor's and the magnet's, and the joint loop reads that built tree
    (``ctx.models``), as the build report says, so the weight it compensates is the arm's with the magnet."""
    import math

    from chain_cases import AR4, MAGNET_DENSITY, MAGNET_LENGTH, MAGNET_RADIUS
    from mime.core.multibody import read_urdf

    built = a0_dynamic(density=MAGNET_DENSITY, top_speed=2 * math.pi * 10).build()
    assert any(line.strip().startswith("models: its law read the built tree of rpm.arm")
               for line in built.report.controller), built.report.controller
    bare = float(np.sum(np.asarray(read_urdf(AR4).tree.articulation.masses)))
    carried = float(np.sum(np.asarray(built.plans.models["rpm.arm"].articulation.masses)))
    magnet = MAGNET_DENSITY * math.pi * MAGNET_RADIUS ** 2 * MAGNET_LENGTH
    assert carried - bare == pytest.approx(magnet, rel=1e-9) and magnet == pytest.approx(0.144, rel=0.01)
    arm = built.graph.get_node("rpm.arm")
    assert float(np.sum(np.asarray(arm.params["masses"]))) == pytest.approx(carried, rel=1e-6)
