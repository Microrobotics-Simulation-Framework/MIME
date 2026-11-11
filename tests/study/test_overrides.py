"""Gate 8: a run's overrides. ``run(override=...)`` is the point the study's member is, run alone: within gate 1's
tolerance of the member, and bit for bit the base's own run where it overrides nothing that moves. Each refusal names its
route: a baked leaf and a structural setting need a rebuild, a numerical setting is a variant. ``run_to`` records the
overrides in ``run.json``, writes the point's report, and stamps the outputs; ``mime verify --run`` replays them, and
fails once an override is dropped from ``run.json``."""

from __future__ import annotations

import json

import numpy as np
import pytest
from compare import TOLERANCE_EPS, misses, worst
from planted import planted
from script_a import EPSILON, REFERENCE_START, reproduction_arm, script_a

from mime import Study
from mime.build import BuildRefused
from mime.cli import main
from mime.data import asset
from mime.interactions import Lubrication
from mime.interactions.hydrodynamic import UncoupledDrag
from mime.study import DT, grid
from mime.study._override import verify_run
from mime.units import Hz, N, kg, m, ms, s

SPEED = "swim_speed:umr.speed"


def test_an_overridden_run_is_within_the_tolerance_of_the_study_member_it_is():
    exp = reproduction_arm()
    field = exp.handle("field")
    values = {field.commands.frequency: Hz(25.0), "umr.mass": kg(8.8e-5)}  # type: ignore[union-attr]
    member = Study(exp).sweep(grid({k: [v] for k, v in values.items()}), duration=s(0.2), record=[SPEED]).run()
    alone = reproduction_arm().build().run(s(0.2), history=True, override=values)
    where, miss = worst(misses(alone, member.member(0), [SPEED]))
    assert miss <= TOLERANCE_EPS["float32"], (where, miss)
    assert alone.overrides and "OVERRIDDEN for this run" in alone.text()


def test_an_override_at_the_bases_own_value_is_bit_for_bit_the_bases_run():
    field = reproduction_arm().handle("field")
    base = reproduction_arm().build().run(s(0.05))
    same = reproduction_arm().build().run(s(0.05), override={field.commands.frequency: Hz(10.0)})  # type: ignore[union-attr]
    assert all(np.array_equal(base.final[n][f], same.final[n][f]) for n in base.final for f in base.final[n])


def test_a_moved_override_is_not_the_bases_run():
    """The control for the bit-for-bit claim above: the comparison sees a frequency moved by a part in a million."""
    field = reproduction_arm().handle("field")
    base = reproduction_arm().build().run(s(0.05))
    moved = reproduction_arm().build().run(s(0.05), override={field.commands.frequency: Hz(10.0 * (1 + 1e-6))})  # type: ignore[union-attr]
    assert not all(np.array_equal(base.final[n][f], moved.final[n][f]) for n in base.final for f in base.final[n])


@pytest.mark.parametrize(("kind", "code"), [("static", "override.baked-leaf"), ("baked", "override.leaf-unread")])
def test_a_leaf_the_compiled_step_does_not_read_is_refused_with_a_rebuild_as_its_route(kind, code):
    exp, spring = planted(kind)
    with pytest.raises(BuildRefused) as refused:
        exp.build().run(s(0.1), override={spring.k: N(2.5) / m(1.0)})  # type: ignore[union-attr]
    assert refused.value.codes() == [code]
    assert "Experiment.based_on" in refused.value.refusals[0].action


def test_a_structural_setting_and_a_numerical_one_are_refused_with_their_routes():
    """Drag 0.4.0's lubrication (the surrogate's blend, a design setting there) is a setting the program is built
    from; drag 0.5.0's near_contact is a numerical choice (the mrs rung's hand-over, the MRS's completion in the gap,
    whose validity line is its near-wall error), so overriding it is a variant, as the step and the rung are (script A
    itself keeps drag 0.4.0, whose blend 0.5.0 retired)."""
    kept = script_a("script_a", start=REFERENCE_START, drag=UncoupledDrag.Surrogate(
        weights=asset("dejongh_mlp_v2"), lubrication=Lubrication(epsilon=EPSILON))).build()
    with pytest.raises(BuildRefused) as structural:
        kept.run(s(0.1), override={"drag:umr.lubrication": None})
    assert structural.value.codes() == ["override.structural"]
    assert "Study" in structural.value.refusals[0].action
    from mime.interactions import Drag

    built = script_a("script_a_050", start=REFERENCE_START, drag=Drag.Surrogate(weights=asset("dejongh_mlp_v2"))).build()
    for override in ({"drag:umr.near_contact": True}, {DT: ms(0.25)}, {"drag:umr.rung": "mrs"}):
        with pytest.raises(BuildRefused) as variant:
            built.run(s(0.1), override=override)
        assert variant.value.codes() == ["override.variant"] and "variant" in variant.value.refusals[0].message


def test_run_to_records_the_overrides_and_verify_run_replays_them(tmp_path, capsys):
    field = reproduction_arm().handle("field")
    out = tmp_path / "run"
    written = reproduction_arm().build().run_to(out, s(0.05), record=[SPEED],
                                                 override={field.commands.frequency: Hz(15.0)})  # type: ignore[union-attr]
    record = json.loads((out / "run.json").read_text())
    assert [held["path"] for held in record["overrides"]] == ["field.commands.frequency"]
    assert record["overrides"][0]["si"] == 15.0 and record["overrides"][0]["unit"] == "Hz"
    assert record["overrides"][0]["leaves"] == ["drive:field.frequency.value"]
    assert written.result.overrides and json.loads((out / "outputs.json").read_text())["overrides"]
    assert verify_run(out).reproduced
    assert main(["verify", "--run", str(out)]) == 0
    record["overrides"] = []
    (out / "run.json").write_text(json.dumps(record))
    assert not verify_run(out).reproduced
    capsys.readouterr()


def test_mime_run_takes_an_override(tmp_path):
    exp = reproduction_arm()
    exp.build().save(tmp_path / "saved", steps=2)
    assert main(["run", str(tmp_path / "saved"), "--duration", "5ms", "--out", str(tmp_path / "out"),
                 "--override", "field.commands.frequency=15Hz"]) == 0
    record = json.loads((tmp_path / "out" / "run.json").read_text())
    assert record["overrides"][0]["value"] == "15.0 Hz"
    assert verify_run(tmp_path / "out").reproduced


def test_an_override_runs_the_nodes_own_checks_at_the_build_as_a_study_does():
    """The MRS rung's own check of the start, judged on the overridden point by the code that judges a study's: a ball
    started 4.45 mm off the axis of a 5 mm tube is refused when the run is asked for (``drag.mrs-resistance``, as its
    own build and a study's point are), not at the run's first step, and the build has taken no step. The control: an
    override of a start inside the rung's range is planned, its check judged on its own node and passed."""
    from mrs_cases import mrs_tube

    from mime.study._override import overridden
    from mime.units import mm

    built = mrs_tube().build()
    with pytest.raises(BuildRefused) as refused:
        built.run(ms(2), override={"ball.initial.position": mm((4.45, 0.0, 0.0))})
    assert "drag.mrs-resistance" in refused.value.codes(), refused.value.codes()
    assert built.steps_taken == 0
    inside = overridden(built, {"ball.initial.position": mm((0.1, 0.0, 0.0))})
    assert inside.starting == ("ball",) and inside.point.computed is not None
