"""Script A, built through the public API, against the frozen 0.2 reference: the gate of dev1's phase 4.

The reference (``references/script_a/``) is the 0.2 port's run of A (MIME-port041 at 6485516, MADDENING 0.4.1,
CPU), frozen once in its own environment; 0.3 never imports it. Every expected difference was declared before this
comparison first ran, with its predicted size and its reasoning, in ``script_a_expected_differences.json``, written
by ``predict_script_a.py`` from the reference alone. A quantity is held to the reference moved by those
predictions, within the declared tolerance; one that cannot discriminate is reported and not gated, with the reason.

**What this bounds.** That 0.3's A is 0.2's A with exactly the declared changes, at their predicted sizes, at this
operating point. The 0.2 code is the weakest oracle: agreement here says nothing about the flow or the hardware.

**4c3's declared changes** (``declared_changes`` in the file) land one at a time, each predicted by the same emulator
and committed before it is measured; 0.3 is held to the predictions of the last one made (``script_a_runs.IN_FORCE``).
The velocity-increment form (MIME-ANO-113) moves float64 by rounding only, and that is checked against the old form,
planted.

**Controls.** At 11 Hz, 0.3 matches the 11 Hz reference; at 10 Hz it misses that reference on every gated
quantity, so the comparison can tell a 10% change in frequency. At half the step, 0.3 matches the prediction there,
and the integrator's gap to 0.2's scheme halves: the declared difference is first order in dt, as its mechanism
says.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from script_a import reproduction_arm
from script_a_runs import (
    DECLARED,
    GATED,
    IN_FORCE,
    PREDICTED,
    REFERENCE,
    TRANSIENT,
    compare,
    declared,
    planted_out,
    quantities,
    reported,
    run,
    table,
)

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES = DECLARED["tolerances"]["f64"]
GATED_TOLERANCES = {name: TOLERANCES[name] for name in GATED}


def reference_transient() -> float:
    """The reference's speed along its tube (z) over the first 0.5 s, from its trajectory's every tenth step."""
    trajectory = np.load(Path(__file__).parent / "references" / "script_a" / "f64_trajectory_every_10th_step.npz")
    time, along = trajectory["time_s"], trajectory["state/body/position"][:, 2]
    end = int(np.searchsorted(time, 0.5 - 1e-12))
    return float((along[end] - along[0]) / (time[end] - time[0]))


@pytest.fixture(scope="module")
def at_10_hz():
    return quantities(run(10.0), 10.0)


def test_script_a_matches_its_reference_with_exactly_the_declared_differences(at_10_hz):
    reference = {**REFERENCE["f64"], TRANSIENT: reference_transient()}
    rows = compare(at_10_hz, reference, PREDICTED["f64_10hz"], TOLERANCES)
    print("\nfloat64, 10 Hz: the reference, 0.3, their difference, the declared difference, and 0.3 against the "
          "prediction\n" + table(rows) + "\n" + reported(at_10_hz))
    assert [row.quantity for row in rows if not row.holds] == []


def test_what_cannot_discriminate_is_reported_and_why(at_10_hz):
    """The radial offset sits on the projection's constraint and the locked ratio is 1 at any frequency: reported, not
    gated, and this says why with numbers."""
    constraint = DECLARED["operating_point"]["offset_m"]
    assert abs(at_10_hz["radial_offset_max_m"] - at_10_hz["radial_offset_min_m"]) <= 1e-12 * constraint
    assert abs(at_10_hz["radial_offset_mean_m"] / constraint - 1) <= 1e-12
    eleven = REFERENCE["f64_freq11hz"]["rotation_ratio_body_over_field"]
    assert abs(at_10_hz["rotation_ratio_body_over_field"] - eleven) < 1e-6


def test_at_11_hz_it_matches_the_11_hz_reference_and_at_10_hz_it_misses_it_everywhere(at_10_hz):
    predicted = PREDICTED["f64_11hz"]
    eleven = compare(quantities(run(11.0), 11.0), REFERENCE["f64_freq11hz"], predicted, GATED_TOLERANCES)
    ten = compare(at_10_hz, REFERENCE["f64_freq11hz"], predicted, GATED_TOLERANCES)
    print("\nfloat64, 11 Hz against the 11 Hz reference\n" + table(eleven)
          + "\nfloat64, 10 Hz against the 11 Hz reference (the control)\n" + table(ten))
    assert [row.quantity for row in eleven if not row.holds] == []
    assert [row.quantity for row in ten if row.holds] == []


def test_at_half_the_step_it_matches_the_prediction():
    predicted = PREDICTED["f64_10hz_dt_2.5e-4"]
    half = quantities(run(10.0, dt=2.5e-4), 10.0)
    rows = compare(half, predicted, predicted, GATED_TOLERANCES)
    print("\nfloat64, 10 Hz, dt = 2.5e-4 s, against its prediction\n" + table(rows))
    assert [row.quantity for row in rows if not row.holds] == []


def test_the_example_declares_the_reproduction_arm():
    """``examples/script_a.py``, written as a user writes it, declares this experiment: the same things, laws, settings
    and values (only the reasons' wording differs), so the same derived graph."""
    spec = importlib.util.spec_from_file_location("example_script_a", ROOT / "examples" / "script_a.py")
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    theirs, ours = example.script_a(), reproduction_arm()

    def without_reasons(document):
        declared = {path: {k: v for k, v in value.items() if k != "reason"}
                    for path, value in document["declared"].items()}
        return {**document, "declared": declared}

    assert without_reasons(theirs.document()) == without_reasons(ours.document())
    assert theirs.build().record.hashes()["derived"] == ours.build().record.hashes()["derived"]


def test_the_emulation_reproduces_the_reference_and_gives_the_declared_prediction():
    """The predictions' basis, rerun: 0.2's scheme, emulated, is the reference to rounding; 4c2's 0.3 scheme is 4c2's
    declared prediction, and the scheme with the changes made so far is the declaration in force."""
    import predict_script_a as predictor

    old = predictor.quantities(predictor.emulate(predictor.SCHEME_0_2), 10.0)
    new = predictor.quantities(predictor.emulate(predictor.SCHEME_0_3), 10.0)
    current = predictor.quantities(predictor.emulate(predictor.CHANGES[IN_FORCE]), 10.0)
    for name in GATED:
        assert abs(predictor.difference(name, old[name], REFERENCE["f64"][name])) <= 1e-12, name
        assert abs(predictor.difference(name, new[name], DECLARED["predicted"]["f64_10hz"][name])) <= 1e-12, name
        assert abs(predictor.difference(name, current[name], PREDICTED["f64_10hz"][name])) <= 1e-12, name


def test_in_float64_the_increment_form_moves_a_by_rounding_only(at_10_hz, monkeypatch):
    """Declared change 1 (MIME-ANO-113): the same equation solved for the change in velocity. Against rigid_inertial
    0.1.0's form, planted, every float64 quantity moves by less than the declared bound."""
    change = declared("velocity-increment-form")
    bound = change["float64_rounding"]["declared_bound"]
    planted_out(monkeypatch, "velocity-increment-form")
    old = quantities(run(10.0), 10.0)
    moved = {name: abs((at_10_hz[name] - old[name]) / (1.0 if "lag" in name else abs(old[name]))) for name in GATED}
    print(f"\nfloat64, the increment form against 0.1.0's form: {moved}\n"
          f"emulated: {change['float64_rounding']['emulated_change']}")
    assert all(moved[name] <= bound.get(name, bound["relative"]) for name in GATED), moved


def test_holding_the_contact_in_the_solve_shrinks_the_dt_gap_as_declared(monkeypatch):
    """Declared change 2, on its own scheme (the surrogate about the end, as change 3 is planted out): 4c2 measured
    A's speed falling 1.18% from dt to dt/2 and 0.60% from dt/2 to dt/4, the first-order error of a solve that resisted
    a fall the projection discarded. Held in the solve, the change is the declared one, far smaller, and the speed at
    every step is the emulated one. What remains against 0.2's scheme, the implicit-explicit difference, is still
    first order: its gap halves with dt."""
    planted_out(monkeypatch, "surrogate-about-the-centre")
    speeds_by_step = {f"{dt:g}": quantities(run(10.0, dt=dt), 10.0)["swim_speed_axial_m_s"]
                      for dt in (5e-4, 2.5e-4, 1.25e-4)}
    gap = declared("contact-in-the-solve")["dt_gap"]
    for step, speed in speeds_by_step.items():
        assert abs(speed / gap["speeds_m_s"][step]["after"] - 1) <= TOLERANCES["swim_speed_axial_m_s"], step
    steps = list(speeds_by_step.values())
    first, second = steps[0] / steps[1] - 1, steps[1] / steps[2] - 1
    before = gap["change_between_dt_and_dt_over_2"]["before"]
    print(f"\nthe speed's change from dt to dt/2: {first:+.4e} (4c2: {before:+.4e}, {abs(before / first):.1f} times "
          f"larger); from dt/2 to dt/4: {second:+.4e} (4c2: {gap['change_between_dt_over_2_and_dt_over_4']['before']:+.4e})")
    assert abs(first - gap["change_between_dt_and_dt_over_2"]["after"]) <= 1e-7
    assert abs(second - gap["change_between_dt_over_2_and_dt_over_4"]["after"]) <= 1e-7
    assert abs(before / first) > 20
    to_0_2 = [speeds_by_step[step] / gap["speeds_m_s"][step]["0.2"] - 1 for step in speeds_by_step]
    print(f"gap to 0.2's scheme at dt, dt/2, dt/4: {to_0_2}, ratios {to_0_2[0] / to_0_2[1]:.4f}, "
          f"{to_0_2[1] / to_0_2[2]:.4f}")
    assert 1.9 <= to_0_2[0] / to_0_2[1] <= 2.1 and 1.9 <= to_0_2[1] / to_0_2[2] <= 2.1


def test_with_the_contact_taken_out_of_the_solve_the_old_gap_returns(monkeypatch):
    """The control: the hold planted out (the confinement projected after the step only, as rigid_inertial 0.1.0
    did), with the surrogate about the end, gives 4c2's prediction, +2.35% against the reference, again."""
    planted_out(monkeypatch, "contact-in-the-solve", "surrogate-about-the-centre")
    found = quantities(run(10.0), 10.0)
    rows = compare(found, REFERENCE["f64"], DECLARED["predicted"]["f64_10hz"], GATED_TOLERANCES)
    print("\nthe contact out of the solve, against 4c2's prediction\n" + table(rows))
    assert [row.quantity for row in rows if not row.holds] == []
    held = compare(found, REFERENCE["f64"], declared("contact-in-the-solve")["predicted"]["f64_10hz"],
                   GATED_TOLERANCES)
    assert not any(row.holds for row in held if row.quantity.startswith("swim_speed"))
