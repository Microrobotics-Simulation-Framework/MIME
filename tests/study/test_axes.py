"""Axes through handles: an apparatus's commands in a namespace of their own, a body's initial state, the handle
``attach`` returns (a ``str``, as the id always was, carrying the law's settings), ``Experiment.handle`` for an
experiment with no handles, the run's ``DT`` and ``PRECISION``, and strings, with did-you-mean. Points from ``grid``,
``along`` and a table."""

from __future__ import annotations

import json
import pickle

import pytest
from script_a import reproduction_arm

from mime import Study
from mime.build import BuildRefused
from mime.build._refusals import ArgumentRefused
from mime.study import DT, PRECISION, AxisKind, along, grid
from mime.study._axes import resolve
from mime.things import CommandPath, InitialPath, LawHandle, SettingPath
from mime.units import Hz, kg, mm, s


def test_handles_name_commands_initial_values_and_law_settings_as_typed_paths():
    exp = reproduction_arm()
    field, umr, drag = exp.handle("field"), exp.handle("umr"), exp.handle("drag:umr")
    assert isinstance(field.commands.frequency, CommandPath)  # type: ignore[union-attr]
    assert str(field.commands.frequency) == "field.commands.frequency"  # type: ignore[union-attr]
    assert str(umr.initial.position) == "umr.initial.position" and isinstance(umr.initial.position, InitialPath)  # type: ignore[union-attr]
    assert isinstance(drag, LawHandle) and drag == "drag:umr" and isinstance(drag, str)
    assert isinstance(drag.lubrication.epsilon, SettingPath)  # type: ignore[union-attr]
    assert str(drag.lubrication.epsilon) == "drag:umr.lubrication.epsilon"  # type: ignore[union-attr]
    assert json.dumps({drag: 1}) == '{"drag:umr": 1}' and pickle.loads(pickle.dumps(drag)) == "drag:umr"
    assert str(exp.handle("swim_speed:umr").speed.path) == "swim_speed:umr.speed"  # type: ignore[union-attr]


def test_a_misspelt_name_is_refused_on_its_line_with_the_names_there_are():
    exp = reproduction_arm()
    with pytest.raises(AttributeError, match="frequency"):
        _ = exp.handle("field").commands.frequncy  # type: ignore[union-attr]
    with pytest.raises(AttributeError, match="epsilon"):
        _ = exp.handle("drag:umr").lubrication.epsilom  # type: ignore[union-attr]
    with pytest.raises(ArgumentRefused, match="field"):
        exp.handle("feild")


def test_strings_resolve_as_the_typed_paths_save_and_an_unknown_one_is_refused_with_its_candidates():
    _, declarations, _ = reproduction_arm().declared_as()
    kinds = {path: resolve(path, declarations).kind for path in (
        "field.commands.frequency", "umr.initial.position", "drag:umr.lubrication.epsilon", "umr.mass", "dt",
        "precision")}
    assert kinds == {"field.commands.frequency": AxisKind.COMMAND, "umr.initial.position": AxisKind.INITIAL,
                     "drag:umr.lubrication.epsilon": AxisKind.SETTING, "umr.mass": AxisKind.VALUE, "dt": AxisKind.DT,
                     "precision": AxisKind.PRECISION}
    assert resolve(DT, declarations).kind is AxisKind.DT and resolve(PRECISION, declarations).kind is AxisKind.PRECISION
    with pytest.raises(ArgumentRefused) as refused:
        resolve("field.commands.frequncy", declarations)
    assert "field.commands.frequency" in refused.value.refusals[0].action


def test_a_settings_quantity_is_its_value_and_its_own_fields_are_no_axis():
    """An axis stops at a quantity: ``drag.lubrication.epsilon`` is a setting, and the quantity's own fields under it
    are refused: ``value`` and ``dimension`` as strings, and ``dimension`` as a typed path (``value`` is the path's own
    attribute, the quantity the law declares). The control: the setting itself resolves, typed and as a string."""
    exp = reproduction_arm()
    _, declarations, _ = exp.declared_as()
    epsilon = exp.handle("drag:umr").lubrication.epsilon  # type: ignore[union-attr]
    assert resolve(epsilon, declarations).kind is AxisKind.SETTING
    assert resolve("drag:umr.lubrication.epsilon", declarations).kind is AxisKind.SETTING
    with pytest.raises(AttributeError, match="no setting 'dimension'"):
        _ = epsilon.dimension  # type: ignore[attr-defined]
    for field in ("value", "dimension"):
        with pytest.raises(ArgumentRefused) as refused:
            resolve(f"drag:umr.lubrication.epsilon.{field}", declarations)
        assert refused.value.codes() == ["study.unknown-path"], field


def test_a_field_named_commands_is_refused():
    from mime import Apparatus
    from mime.fields import UniformRotating

    exp = reproduction_arm()
    exp.add_apparatus(Apparatus("other", fields={"commands": UniformRotating(rotation_axis=(1.0, 0.0, 0.0),
                                                                             initial_direction=(0.0, 1.0, 0.0))},
                                commands={"frequency": Hz(1.0), "amplitude": mm(1.0)}))
    with pytest.raises(BuildRefused) as refused:
        exp.plan()
    assert "apparatus.reserved-field" in refused.value.codes()


def test_grid_along_and_a_table_give_the_points_in_order():
    a, b = [Hz(1.0), Hz(2.0)], [kg(1.0), kg(2.0)]
    assert grid({"x": a, "y": b}).rows == ((a[0], b[0]), (a[0], b[1]), (a[1], b[0]), (a[1], b[1]))
    assert along({"x": a, "y": b}).rows == ((a[0], b[0]), (a[1], b[1]))
    with pytest.raises(ArgumentRefused):
        along({"x": a, "y": b[:1]})
    with pytest.raises(ArgumentRefused):
        grid({"x": Hz(1.0)})  # type: ignore[dict-item]


def test_the_plan_says_what_split_a_dt_axis_and_a_precision_axis_into_groups():
    """Four groups, each split from the first by its step or its precision, and each member run in its own precision:
    one study planned and run, so each group is built once."""
    study = Study(reproduction_arm()).sweep(grid({DT: [s(5e-4), s(2.5e-4)], PRECISION: ["float32", "float64"]}),
                                            duration=s(0.01))
    plan = study.plan()
    assert len(plan.groups) == 4, plan.text()
    reasons = " ".join(" ".join(group.reasons) for group in plan.groups)
    assert "rate (dt)" in reasons and "precision" in reasons, reasons
    result = study.run()
    assert [result.member(i).final["umr"]["pose"].dtype.name for i in range(4)] == ["float32", "float64"] * 2
