"""A0 through the public API, in float32, against the pre-0.3 tag's A0 with its drag surrogate given the tube's axis:
the gate of dev1's phase 7c1.

The reference (``references/a0/``) is that tag's run at 3 Hz and 10 Hz (MADDENING 0.3.1, CPU), frozen once in its own
environment; 0.3 never imports it. Every expected difference was declared before this comparison first ran, with its
predicted size, its cause and how it is checked, in ``a0_expected_differences.json``, written by ``predict_a0.py`` from
the reference alone. A quantity is held to the reference moved by those predictions, within the declared tolerance;
one that cannot discriminate is reported and not gated, with the reason. A difference found that is not on the list is
a finding, reported, and is not added to the list after the fact.

**What this bounds.** That 0.3's A0 is the tag's with exactly the declared differences, at their predicted sizes, at
these two operating points, and in float32 (the reference's "float64" is mixed precision, the controller computing in
float32, so it is not compared). **What it does not.** The drag surrogate's weights are the reference's
(``dejongh_mlp_v2``) on both sides, and MIME-ANO-117 records that they inherit the old wall tables' defect: this checks
the framework against its reference on the same weights, not the surrogate's physics, and the 0.2 code is the weakest
oracle. The arm is kinematic here; the dynamic arm is phase 7a3's, and the 20 ms its lag adds is the first declared
difference.

**Limits of the gates** (measured: ``references/a0_controls_measured.json``). Eight of the nine declared quantities are
gated, and the distance is effectively not: its tolerance (1.3e-2 of the excess over the standoff, about 1.3e-5 m) is
ten times the reference's own float32-to-mixed gap in it, which is what the tolerance rule gives it, and at that
tolerance it cannot tell 3 Hz from 10 Hz (``test_the_comparison_tells_one_frequency_from_the_other``): it is reported
beside the others and tells nothing the other eight do not. The gradient force alone (the field, and so the torque,
unmoved) is not seen below about 2e-3 of itself, since it is about 1 % of the swim speed (the declaration's 0.81 %): a
force error smaller than that is not covered. ``rotor_rate_mean_hz`` reads the speed the drive commands, which the
kinematic motor reports no other way, and so tests that float32(2 pi f) is fed and cannot fail on a fault of the
motor; the motor's output, its angle, is gated by ``rotor_angle_rate_mean_hz`` (see ``a0_compare.py``), added in the
review of this phase and not declared before the comparison, with the rotor's declared tolerance and the declared
emulation's accumulator line as its prediction. That angle is a single-precision sum whose step is rounded to a whole
number of units in the last place, so it shows a speed fault only above about 1e-3 of the speed (the rotation rates
see the same motion).

**Findings** (not on the list; none moves a gated quantity beyond its tolerance). The controller's target filter,
computed in single precision in both lines, stops short of the standoff by 3.3e-7 m in 0.3 and by 1.3e-6 m in the
tag's run: XLA fused the filter's multiply by ``1 - alpha`` into the sum there and does not here. It moves the distance
from the body (the standoff's excess) by 9.2e-7 m, 9.2e-4 of itself at 3 Hz, inside the 1.3e-2 the reference's own
float32 gap sets for that quantity.
"""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from a0_compare import (
    DECLARED,
    FREQUENCY,
    GATED,
    MOTOR_OUTPUT,
    REFERENCE,
    TOLERANCES,
    compare,
    difference,
    measure,
    missed,
    table,
)
from a0_runs import REPORTED
from script_a0 import a0

from mime.data import asset


@pytest.fixture(autouse=True, scope="module")
def x64():
    """This module runs in float32: it overrides the folder's float64 fixture, and restores the mode after."""
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(scope="module", params=list(FREQUENCY))
def measured(request):
    return request.param, measure(request.param)


def test_a0_matches_its_reference_with_exactly_the_declared_differences(measured):
    key, found = measured
    rows = compare(found, key)
    print(f"\nfloat32, {FREQUENCY[key]:g} Hz: the reference, 0.3, the declared prediction, and 0.3 against it\n"
          + table(rows) + "\n" + "\n".join(f"{name}: {found[name]!r} (reported, not gated: {why})"
                                          for name, why in REPORTED.items() if name in found))
    assert missed(rows) == []
    assert MOTOR_OUTPUT in [row.quantity for row in rows], "the motor's output is gated beside the commanded speed"


def test_what_cannot_discriminate_is_reported_and_why(measured):
    """The radial offset sits on the projection's constraint, and the locked ratio of the body's rotation to the field's
    is 1 whatever the frequency."""
    key, found = measured
    assert abs(found["radial_offset_mean_m"] / 1.0002e-3 - 1) <= 1e-7
    assert abs(found["rotation_ratio_body_over_field"] - 1) <= 1e-4


def test_the_comparison_tells_one_frequency_from_the_other(measured):
    """The control for the whole comparison: held to the other frequency's prediction, 0.3 misses every gated quantity
    but one. The distance from the body (its excess over the standoff) is a function of the standoff, which the
    controller holds at either frequency: it cannot tell them, and that is why it is held to a loose tolerance."""
    key, found = measured
    other = "10hz" if key == "3hz" else "3hz"
    rows = compare(found, key, against=other)
    print("\nheld to the other frequency's prediction\n" + table(rows))
    assert [row.quantity for row in rows if row.holds] == ["magnet_body_distance_minus_standoff_mean_m"]


def test_every_declared_difference_is_larger_than_the_tolerance_of_the_quantity_that_gates_it():
    """A difference smaller than its tolerance would be one the comparison cannot see: a null result without evidence.
    Each declared difference moves the quantity it names by more than that quantity's tolerance, at both frequencies,
    and the one declared negligible moves none."""
    moved = {"arm-dynamics": "tracking_error_axial_mean_m", "rotor-motor": "rotor_rate_mean_hz",
             "surrogate-about-the-centre": "swim_speed_axial_m_s", "body-step": "phase_lag_field_minus_moment_rad"}
    by_id = {entry["id"]: entry for entry in DECLARED["differences"]}
    for identity, name in moved.items():
        for key in FREQUENCY:
            size = abs(by_id[identity]["predicted_alone"][key][name])
            assert size > 3 * TOLERANCES[name], (identity, key, name, size, TOLERANCES[name])
    for key in FREQUENCY:
        negligible = by_id["vacuum-permeability"]["predicted_alone"][key]
        assert max(abs(size) / TOLERANCES[name] for name, size in negligible.items()) < 1e-6


def test_the_build_is_the_declared_one():
    """The things the declaration's 'none expected' entries rest on: the controller reads the body and the joints at the
    previous step and writes the arm at this one, runs before the arm, every step; and the drag names the
    reference's weights."""
    from mime.build import Lag

    built = a0().build()
    lags = {(edge.source[0], edge.target[0]): edge.lag for edge in built.report.lags}
    assert lags[("umr", "controller")] is Lag.PREVIOUS_STEP
    assert lags[("rpm.arm", "controller")] is Lag.PREVIOUS_STEP
    assert lags[("controller", "rpm.arm")] is Lag.THIS_STEP
    assert built.graph.schedule.index("controller") < built.graph.schedule.index("rpm.arm") and not \
        built.graph.is_multirate
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("name") == "drag":
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(json.dumps(a0().document(), default=str)))
    (drag,) = found
    assert drag["settings"]["weights"] == {"asset": "dejongh_mlp_v2", "sha256": asset("dejongh_mlp_v2").sha256}
    assert asset("dejongh_mlp_v2").sha256.startswith("ac1fca5c") and asset("dejongh_mlp_v2").sha256.endswith("0a345")


def test_the_references_recorded_weights_are_the_ones_the_drag_names():
    """The surrogate's weights are pinned on 0.3's side by the asset the drag names; on the reference's side they are
    recorded in each of its provenance files (the sha256 of the weights file its run read). The two are the same file's
    hash, so the comparison is on one set of weights, not on two that were once the same."""
    path = "data/dejongh_benchmark/mlp_cholesky_weights_v2.npz"
    references = Path(__file__).parent / "references" / "a0"
    recorded = {label: json.loads((references / f"{label}_provenance.json").read_text())["source_sha256"][path]
                for label in ("f32_3hz", "f32_10hz", "mixed_f64_3hz", "mixed_f64_10hz")}
    assert set(recorded.values()) == {asset("dejongh_mlp_v2").sha256}, recorded


def test_the_controller_and_the_rotor_magnet_are_declared_with_the_references_values():
    """The reference's own parameters, read back from its provenance: the controller's settings (the defaults of
    ``arm_tracking`` are the 0.2 line's), the arm's base and home pose, the tick, and the rotor magnet's moment."""
    from chain_cases import tracking

    parameters = json.loads((Path(__file__).parent / "references" / "a0" / "f32_3hz_provenance.json").read_text())[
        "parameters"]["params_py_namespace_as_runner_passes_it"]
    controller = tracking()
    assert (controller.smoothing, controller.gain, controller.damping, controller.iterations) == (
        parameters["CONTROL_TARGET_ALPHA"], parameters["CONTROL_K_IK"], parameters["CONTROL_IK_LAMBDA"],
        parameters["CONTROL_IK_INNER_ITERS"])
    assert controller.standoff.value == parameters["CONTROL_STANDOFF_M"]
    assert controller.reach.value == (parameters["CONTROL_X_MIN_M"], parameters["CONTROL_X_MAX_M"])
    assert controller.orientation_feedback is parameters["ENABLE_ORIENTATION_FEEDBACK"] is False
    assert controller.home.value == tuple(parameters["ARM_HOME_RAD"])
    assert controller.at.value == tuple(parameters["BASE_POSE_WORLD"][:3])
    document = a0().document()
    assert document["declared"]["rpm.magnetic_moment"]["value"] == parameters["MAGNET_DIPOLE_A_M2"]
    assert document["controller"]["rate"]["value"] == parameters["DT_PHYS"]


#: The emulation's two schemes, each a full run of the predictor, against its record in the declaration.
@pytest.mark.parametrize("scheme", ["TAG", "NEW"])
def test_the_emulation_reproduces_the_reference_and_is_the_declared_prediction(scheme):
    """The predictions' basis, rerun at 10 Hz: the tag's scheme, emulated, is the reference within the emulation's
    resolution; and each scheme's quantities are those recorded when the differences were declared."""
    jax.config.update("jax_enable_x64", True)
    try:
        import predict_a0 as predictor

        record = json.loads((Path(__file__).parent / "a0_emulation.json").read_text())
        chosen = predictor.TAG if scheme == "TAG" else predictor.NEW
        found = predictor.steady(predictor.emulate(chosen, 10.0), 10.0, chosen)
    finally:
        jax.config.update("jax_enable_x64", False)
    recorded = record["tag" if scheme == "TAG" else "new"]["10hz"]
    for name in GATED:
        assert abs(difference(name, found[name], recorded[name])) <= 1e-9, name
    if scheme == "TAG":
        for name in GATED:
            miss = abs(difference(name, found[name], REFERENCE["10hz"][name]))
            assert miss <= DECLARED["tolerances"]["emulation_resolution"][name] or name == "rotor_rate_mean_hz", name
        assert np.isclose(found["rotor_rate_mean_hz"], REFERENCE["10hz"]["rotor_rate_mean_hz"], rtol=1e-12)
