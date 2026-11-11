"""A declared range (``exp.declare_range``) as phase 1 plans it: held by its path, reaching the hull of the body whose
kind's number it ranges, and refused where it cannot hold (a variant, another dimension, ends out of order, the
experiment's own value outside it, twice, too many, a corner the plan refuses, no reason in research). A geometry axis
names a kind's number on a body's part; a vector setting of a kind is refused as an axis. Every rig here is planned, not
built, but the corner's, which the build judges."""

from __future__ import annotations

import pytest
from ranged_rigs import HIGH, LOW, RADIUS, REASON, planted

from mime.build import BuildRefused, RangeDeclaration
from mime.build._refusals import ArgumentRefused
from mime.build._ranges import MAX_RANGES
from mime.experiment import Experiment
from mime.saved import dump_document
from mime.scene import Ellipsoid, Sphere
from mime.study import DT, PRECISION
from mime.study._axes import AxisKind, base_value, candidates, declared, resolve
from mime.units import Hz, mm, mPa_s, ms


def codes(error: BuildRefused | ArgumentRefused) -> list[str]:
    return [refusal.code for refusal in error.refusals]


def test_a_range_of_a_kinds_number_is_held_by_its_path_and_reaches_the_bodys_hull():
    plans = planted("clean").plan()
    (held,) = plans.ranges.values()
    assert (held.path, held.kind, held.low_si, held.high_si, held.unit) == (RADIUS, "geometry", 1e-3, 1.15e-3, "m")
    (entry,) = [plan for plan in plans.entries if plan.owner == "rangetoy.shares:ball"]
    assert entry.hulls["body"].ranges == {"radius": (1e-3, 1.15e-3)}
    assert planted("clean", ranged=None).plan().ranges == {}


def test_a_geometry_axis_names_a_kinds_number_and_a_parts_value_stays_a_value():
    _, declarations, _ = planted("clean").declared_as()
    axis = resolve(RADIUS, declarations)
    assert (axis.kind, axis.owner, axis.name) == (AxisKind.GEOMETRY, "ball", "parts.shell.radius")
    assert resolve("ball.parts.shell.density", declarations).kind is AxisKind.VALUE
    assert RADIUS in candidates(declarations)
    assert base_value(axis, *planted("clean").declared_as()[:2]) == mm(1.0)
    settings, _, _ = planted("clean").declared_as()
    held = declared(settings, declarations, {axis: mm(1.1)}, "a test")
    assert base_value(axis, held.settings, held.declarations) == mm(1.1) and held.outside == ()
    outside = declared(settings, declarations, {axis: mm(1.2)}, "a test")
    assert outside.outside == (RADIUS,)
    assert not any(isinstance(d, RangeDeclaration) for d in outside.declarations), "built for its own value"


def test_a_vector_setting_of_a_kind_is_an_axis_built_per_value_and_has_no_range():
    from mrs_cases import free_space

    from mime.interactions import Drag

    experiment = free_space("egg", {"egg": Ellipsoid(semi_axes=mm((0.5, 0.7, 1.1)))},
                            Drag.MRS(spacing=mm(1.0), wall_kind="free_space"))
    axis = resolve("egg.parts.shell.semi_axes", experiment.declared_as()[1])
    assert axis.kind is AxisKind.GEOMETRY and axis.built and axis.component is None
    for path, low, high in (("egg.parts.shell.semi_axes", mm((0.5, 0.7, 1.0)), mm((0.5, 0.7, 1.2))),
                            ("egg.parts.shell.semi_axes[2]", mm(1.0), mm(1.2))):
        ranged = free_space("egg", {"egg": Ellipsoid(semi_axes=mm((0.5, 0.7, 1.1)))},
                            Drag.MRS(spacing=mm(1.0), wall_kind="free_space"))
        ranged.declare_range(path, low, high, reason=REASON)
        with pytest.raises(BuildRefused) as refused:
            ranged.build()
        assert codes(refused.value) == ["range.units"] and "built" in refused.value.refusals[0].message


@pytest.mark.parametrize(("path", "low", "high", "code"), [
    (DT, ms(0.5), ms(2.0), "range.kind"),
    (PRECISION, mm(1.0), mm(2.0), "range.kind"),
    (RADIUS, Hz(1.0), Hz(2.0), "range.units"),
    (RADIUS, mm(1.0), Hz(2.0), "range.units"),
    (RADIUS, mm(1.15), mm(1.0), "range.order"),
    (RADIUS, mm(1.05), mm(1.15), "range.outside"),
    ("ball.parts.shell.nothing", mm(1.0), mm(1.15), "range.path"),
    ("water.viscosity", mPa_s(2.0), mPa_s(3.0), "range.outside"),
], ids=["dt", "precision", "units", "mixed-units", "order", "outside", "unknown-path", "declared-value-outside"])
def test_a_range_that_cannot_hold_is_refused_with_its_code(path, low, high, code):
    experiment = planted("clean", ranged=None)
    experiment.declare_range(path, low, high, reason=REASON)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert codes(refused.value) == [code]


def test_a_numerical_setting_has_no_range():
    from ranged_rigs import pulled

    experiment = pulled(ranged=None)
    experiment.declare_range("drag:ball.spacing", mm(0.5), mm(0.7), reason=REASON)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert codes(refused.value) == ["range.kind"]
    assert "a numerical setting" in refused.value.refusals[0].message


def test_a_path_ranged_twice_in_a_script_is_refused_and_a_range_is_removed_by_its_path():
    experiment = planted("clean")
    experiment.declare_range(RADIUS, mm(1.0), mm(1.1), reason=REASON)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert codes(refused.value) == ["range.duplicate"]
    experiment.remove(RADIUS, kind="range")
    assert not any(isinstance(d, RangeDeclaration) for d in experiment.declared_as()[1])
    assert experiment.plan().ranges == {}


def test_more_than_the_most_ranges_is_refused_before_any_is_resolved():
    experiment = planted("clean", ranged=None)
    for index in range(MAX_RANGES + 1):
        experiment.declare_range(f"ball.parts.shell.radius{index}", mm(1.0), mm(1.15), reason=REASON)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert codes(refused.value) == ["range.too-many"]
    assert f"{2 ** (MAX_RANGES + 1)} plans" in refused.value.refusals[0].message


def test_the_research_profile_asks_a_range_for_its_reason():
    experiment = planted("clean", ranged=None)
    experiment.declare_range(RADIUS, mm(LOW), mm(HIGH))
    plans = experiment.plan()
    assert set(plans.ranges) == {RADIUS}, "exploratory takes it"
    settings, declarations, _ = experiment.declared_as()
    from dataclasses import replace

    with pytest.raises(BuildRefused) as refused:
        Experiment.declared_from(replace(settings, profile="research"), declarations).plan()
    assert "range.reason-pending" in codes(refused.value)


def test_a_range_whose_corner_the_plan_refuses_is_refused_at_build_naming_the_corner():
    experiment = planted("clean")
    experiment.declare_range("water.viscosity", mPa_s(-1.0), mPa_s(2.0), reason="a range past zero")
    experiment.plan()  # the experiment's own value is inside, and plans
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    assert set(codes(refused.value)) == {"range.corner"}
    assert all("water.viscosity -0.001" in r.message for r in refused.value.refusals)


def test_a_range_is_saved_in_the_document_and_read_back(tmp_path):
    experiment = planted("clean")
    document = experiment.document()
    assert document["ranges"] == {RADIUS: {"low": 1e-3, "high": 1.15e-3, "unit": "m", "reason": REASON}}
    assert list(document).index("ranges") == list(document).index("declared") + 1
    assert "ranges" not in planted("clean", ranged=None).document(), "written only where one is declared"
    path = tmp_path / "planted.mime.yaml"
    path.write_text(dump_document(document), encoding="utf-8")
    again = Experiment.from_document(path)
    (held,) = [d for d in again.declared_as()[1] if isinstance(d, RangeDeclaration)]
    assert (str(held.path), held.low, held.high, str(held.reason)) == (RADIUS, mm(LOW), mm(HIGH), REASON)
    assert again.document() == document


def test_a_part_whose_kind_is_written_again_keeps_its_range_by_the_path():
    experiment = planted("clean")
    axis = resolve(RADIUS, experiment.declared_as()[1])
    settings, declarations, _ = experiment.declared_as()
    held = declared(settings, declarations, {axis: mm(1.1)}, "a test")
    (body,) = [d.thing for d in held.declarations if getattr(getattr(d, "thing", None), "name", None) == "ball"]
    assert body.parts["shell"].geometry == Sphere(radius=mm(1.1))
    assert any(isinstance(d, RangeDeclaration) for d in held.declarations)
