"""The seam's verdicts: what the build does with each answer OpenMaterica's reader gives, until profiles land.

- ``met`` builds, with the record's value.
- ``not_met`` (water at 37 degC +- 0.5 K for a 20 degC experiment) is refused, with OpenMaterica's reason as it gave
  it.
- ``unknown`` (water's viscosity: its shear rate unstated) is refused without an acceptance, and builds with one,
  recorded in the saved document; an acceptance that does not cover every reason given still refuses.
- ``no_pick`` is refused even when accepted: no one value exists.
- An interval (N45's remanence): the declared value inside its bounds builds and both are recorded; one outside
  them is refused; a nominal on a field that holds no interval is refused.
- A table at a temperature (OpenMaterica's own fixture) gives the reader's value; outside its range it is refused,
  unless the experiment declares an extrapolation, whose value MIME computes and whose basis it records.

Each refusal has its control: the same experiment, changed only in what the refusal names, builds."""

from dataclasses import replace

import pytest
from records import density_table, water_at_37, water_experiment
from script_a import CONDITIONS, N45_RECORD, WATER_RECORD, record_arm

from mime import Medium
from mime.build import BuildRefused
from mime.materials import Acceptance, Extrapolation, Nominal, Record, packaged_record, record_file
from mime.units import K, T, Condition, degC, kg, kg_per_m3


def refused(experiment) -> list:
    with pytest.raises(BuildRefused) as caught:
        experiment.plan()
    return list(caught.value.refusals)


def values(experiment) -> dict:
    return {line.path: line for line in experiment.plan().values}


def test_a_met_read_builds_with_the_records_value():
    built = water_experiment(WATER_RECORD, CONDITIONS).build()
    line = {v.path: v for v in built.report.values}["water.density"]
    assert line.value == 998.207 and line.source.startswith("record of material 'water'")
    assert "met at temperature 293.15 K +- 0.5 K" in line.reason
    (read,) = built.plans.materials.reads
    assert (read.status, read.reasons, read.value) == ("met", (), 998.207)


def test_a_not_met_read_is_refused_with_openmatericas_reason(tmp_path):
    hot = Record("water", record_file(water_at_37(tmp_path)), require={"test_method": ("reference_formulation",)})
    found = [r for r in refused(water_experiment(hot, CONDITIONS)) if r.code == "record.not-met"]
    assert len(found) == 1 and "condition_outside temperature" in found[0].message
    assert "temperature is outside the tolerance" in found[0].message  # OpenMaterica's own detail, passed through
    at_37 = {"temperature": degC(37.0, plus_minus=0.5)}
    assert values(water_experiment(hot, at_37))["water.density"].value == 993.33  # the control: at 37 degC it is met


def test_an_unknown_read_is_refused_without_an_acceptance_and_builds_with_one_recorded_in_the_document():
    unaccepted = replace(WATER_RECORD, accept={})
    found = [r for r in refused(water_experiment(unaccepted, CONDITIONS, viscosity=True)) if r.code == "record.unknown"]
    assert len(found) == 1 and "condition_unstated shear_rate" in found[0].message
    assert "Acceptance(covers=('condition_unstated:shear_rate',)" in found[0].action
    accepted = water_experiment(WATER_RECORD, CONDITIONS, viscosity=True)
    line = values(accepted)["water.viscosity"]
    assert line.value == 0.0010016 and "accepted by the experiment: IAPWS R12-08" in line.reason
    document = accepted.document()
    assert document["materials"]["water"]["accept"]["fluid.dynamic_viscosity"] == {
        "covers": ["condition_unstated:shear_rate"], "reason": WATER_RECORD.accept["fluid.dynamic_viscosity"].reason}


def test_an_acceptance_covers_only_the_reasons_it_names():
    elsewhere = replace(WATER_RECORD, accept={"fluid.dynamic_viscosity": Acceptance(
        ("condition_unstated:temperature",), reason="covers another condition")})
    assert [r.code for r in refused(water_experiment(elsewhere, CONDITIONS, viscosity=True))] == ["record.unknown"]
    by_code = replace(WATER_RECORD, accept={"fluid.dynamic_viscosity": Acceptance(
        ("condition_unstated",), reason="every unstated condition")})
    assert values(water_experiment(by_code, CONDITIONS, viscosity=True))["water.viscosity"].value == 0.0010016


def test_no_pick_is_refused_even_when_accepted():
    wide = {"temperature": Condition(K(300.0), K(20.0))}
    accepting = replace(WATER_RECORD, accept={"bulk.density": Acceptance(("no_pick",), reason="either will do")})
    found = refused(water_experiment(accepting, wide))
    assert [r.code for r in found] == ["record.no-pick"]
    assert "no_pick" in found[0].message


def test_an_intervals_bounds_are_read_and_a_value_inside_them_is_used_and_recorded():
    plans = record_arm().plan()
    line = {v.path: v for v in plans.values}["umr.parts.magnet_a.remanence"]
    assert line.value == 1.345
    assert "bounds [1.32, 1.37] T (specification_limits); 1.345 T declared within them" in line.reason
    (read,) = [r for r in plans.materials.reads if r.field == "magnetic.remanence"]
    assert read.bounds == (1.32, 1.37) and read.request["requirement"]["want"] == "bounds"


@pytest.mark.parametrize("value", [1.31, 1.3701])
def test_a_value_outside_an_intervals_bounds_is_refused(value, monkeypatch):
    import script_a

    outside = replace(N45_RECORD, nominal={"magnetic.remanence": Nominal(T(value), reason="outside")})
    parts = {key: replace(part, material=outside) if part.material is not None else part
             for key, part in script_a.PARTS.items()}
    monkeypatch.setattr(script_a, "PARTS", parts)
    found = refused(script_a.record_arm())
    assert {r.code for r in found} == {"record.outside-bounds"}
    assert f"{value} T" in found[0].message and "[1.32, 1.37] T" in found[0].message


def test_a_remanence_with_no_nominal_is_refused_with_openmatericas_reason(monkeypatch):
    import script_a

    bare = replace(N45_RECORD, nominal={})
    monkeypatch.setattr(script_a, "PARTS", {key: replace(part, material=bare) if part.material is not None else part
                                            for key, part in script_a.PARTS.items()})
    found = refused(script_a.record_arm())
    assert {r.code for r in found} == {"record.not-met"}
    assert "want_mismatch" in found[0].message


def test_a_nominal_on_a_field_that_holds_no_interval_is_refused():
    nominal = replace(WATER_RECORD, nominal={"bulk.density": Nominal(T(1.0), reason="not an interval")})
    found = refused(water_experiment(nominal, CONDITIONS))
    assert [r.code for r in found] == ["record.not-met"] and "want_mismatch" in found[0].message


def table_material(tmp_path, **terms) -> Record:
    return Record("table", record_file(density_table(tmp_path)), **terms)


def test_a_table_is_read_at_the_experiments_temperature(tmp_path):
    line = values(water_experiment(table_material(tmp_path), CONDITIONS))["water.density"]
    assert line.value == 997.37  # 1000 - 4 x 13.15 / 20, OpenMaterica's evaluation of its table
    assert "from the record's table over temperature, read at temperature 293.15 K" in line.reason


def test_a_table_read_outside_its_range_is_refused_and_an_extrapolation_is_computed_with_its_basis(tmp_path):
    hot = {"temperature": Condition(K(350.0), K(0.5))}
    found = refused(water_experiment(table_material(tmp_path), hot))
    assert [r.code for r in found] == ["record.not-met"] and "outside_range temperature" in found[0].message
    constant = table_material(tmp_path, extrapolation={"bulk.density": Extrapolation(
        "constant", basis="the end value", reason="a test")})
    line = values(water_experiment(constant, hot))["water.density"]
    assert line.value == 986.0 and "extrapolated (constant)" in line.reason and "basis: the end value" in line.reason
    linear = table_material(tmp_path, extrapolation={"bulk.density": Extrapolation(
        "linear", basis="the last segment's slope", reason="a test")})
    line = values(water_experiment(linear, hot))["water.density"]
    assert line.value == pytest.approx(986.0 - 4.0 / 20.0 * 10.0, rel=1e-15)
    assert "computed by MIME" in line.reason and "basis: the last segment's slope" in line.reason


@pytest.mark.parametrize(("conditions", "code"), [
    ({"temperature": K(293.15)}, "conditions.tolerance"),
    ({"temperature": Condition(K(293.15), K(-0.5))}, "conditions.value"),
    ({"temperature": Condition(kg(1.0), kg(0.1))}, "conditions.units"),
    ("warm", "conditions.form"),
])
def test_malformed_conditions_are_refused(conditions, code):
    assert code in [r.code for r in refused(water_experiment(WATER_RECORD, conditions))]


def test_a_medium_value_declared_in_the_experiment_takes_the_place_of_its_records_and_says_so():
    experiment = water_experiment(WATER_RECORD, CONDITIONS)
    experiment.declare("water.density", kg_per_m3(1000.0), reason="the reference's")
    line = values(experiment)["water.density"]
    assert line.value == 1000.0
    assert line.source == "declared in the experiment, in place of material 'water''s bulk.density (not read)"


def test_two_materials_under_one_name_are_refused():
    other = Record("water", packaged_record("openmaterica_example_water"))
    experiment = water_experiment(WATER_RECORD, CONDITIONS)
    experiment.add_medium(Medium("brine", material=other))
    assert "material.name" in [r.code for r in refused(experiment)]
