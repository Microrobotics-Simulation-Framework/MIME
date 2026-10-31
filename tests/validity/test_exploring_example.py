"""``examples/script_a.py``'s ``exploring_arm``: the record arm as a first run writes it, in exploratory, with no
acceptance for what its records leave unstated. It builds the record arm's graph, hash for hash (a profile changes no
value and no graph), leaves pending exactly what the record arm states (its acceptances and the resin's reason), and is
refused in research for exactly those."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from mime.build import BuildRefused

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def example():
    spec = importlib.util.spec_from_file_location("example_script_a", ROOT / "examples" / "script_a.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_exploring_arm_builds_the_record_arm_s_graph_and_leaves_its_paperwork_pending(example):
    exploring, sealed = example.exploring_arm().build(), example.record_arm().build()
    assert exploring.record.hashes()["derived"] == sealed.record.hashes()["derived"]
    assert sealed.report.profile.pending == ()
    items = exploring.report.profile.pending
    assert len(items) == 3, "one item per read, however many reasons it waives (D10)"
    pending = {item.target: item.kind for item in items}
    assert pending == {"umr.parts.shell.density": "its reason pending",
                       "material 'water' fluid.dynamic_viscosity": "waived, its reason pending",
                       "material 'n45' magnetic.remanence": "waived, its reason pending"}
    n45 = [code for w in exploring.report.profile.waivers if w.target == "material 'n45' magnetic.remanence"
           for code in w.reasons]
    assert set(n45) == {"stated_without_origin", "condition_unstated:temperature",
                        "condition_unstated:specimen_shape", "condition_unstated:test_method"}
    (n45_item,) = [item for item in items if item.target == "material 'n45' magnetic.remanence"]
    assert sorted(n45_item.codes) == sorted(n45) and n45_item.python.count("accept=") == 1
    assert exploring.report.profile.stamp().text().count("material 'n45' magnetic.remanence") == 1


def test_the_exploring_arm_in_research_is_refused_for_exactly_its_paperwork(example):
    experiment = example.exploring_arm()
    experiment._settings = replace(experiment._settings, profile="research")
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    assert sorted(refused.value.codes()) == ["record.unknown", "record.unknown", "value.reason-pending"]
