"""Loading: entries resolved among installed packages and refused with the line to install, the three load checks,
the derived hash's cover of static arrays' contents, ``verify`` and ``mime verify``."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from maddening.core.node import SimulationNode

import mime.testing.toys._nodes as toy_nodes
from mime.build import BuildRefused, RefusalClass
from mime.build._report import graph_config
from mime.cli import main as mime_main
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import Delivery, Interaction, RoleKind, register_interaction
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused
from mime.testing.toys import LOAD, coupled_toy, full_toy
from mime.vocabulary import register_namespace

register_namespace("loading", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Unsavable(Interaction):
    """A test's own entry: no installed package provides its namespace, and one setting is a mapping."""

    name = "loading.unsavable"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    delivers = {"load": Delivery("body", LOAD.name)}
    table: Any = None

    def build(self, ctx, b):
        from mime.testing.toys import ToySpringNode

        node = b.add_response_node(ToySpringNode(ctx.node_name(), ctx.timestep))
        b.connect(node["load"], into=b.target("load"))


def _edit(folder, name, change):
    path = folder / name
    path.write_text(change(path.read_text()))


def _edit_lock(folder, change):
    lock = json.loads((folder / LOCK_FILE).read_text())
    change(lock)
    (folder / LOCK_FILE).write_text(json.dumps(lock))


@pytest.fixture
def saved(tmp_path):
    full_toy().save(tmp_path)
    return tmp_path


def test_a_saved_experiment_loads_and_verifies_bit_for_bit(saved):
    experiment = Experiment.load(saved)
    assert experiment.load_report is not None and experiment.load_report.same_environment
    assert "input hash: reproduced exactly" in experiment.load_report.text()
    report = experiment.build().verify()
    assert report.reproduced and report.same_environment and "bit for bit" in report.text()


def test_an_entry_no_installed_package_registers_is_refused_with_the_line_to_install(saved):
    _edit(saved, DOCUMENT_FILE, lambda t: t.replace("name: toy.spring", "name: labz.spring"))

    def name_its_package(lock):
        for record in lock["entries"]:
            if record["name"] == "toy.spring":
                record.update(name="labz.spring", distribution="mime-lab-z", distribution_version="1.2.0")

    _edit_lock(saved, name_its_package)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.unknown-entry" and refusal.category is RefusalClass.ENVIRONMENT
    assert "interaction labz.spring 0.1.0, which no installed package registers" in refusal.message
    assert refusal.action == "install the package it came from: pip install 'mime-lab-z==1.2.0'"


def test_another_version_is_refused_with_the_line_to_install(saved):
    _edit(saved, DOCUMENT_FILE, lambda t: t.replace("name: toy.spring\n  version: 0.1.0",
                                                     "name: toy.spring\n  version: 0.2.0"))

    def pin_another_release(lock):
        for record in lock["entries"]:
            if record["name"] == "toy.spring":
                record.update(version="0.2.0", distribution_version="0.3.0.dev9")

    _edit_lock(saved, pin_another_release)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.version"
    assert "names interaction toy.spring 0.2.0, and the installed packages register toy.spring at 0.1.0 only" \
        in refusal.message
    assert refusal.action == "install the package it came from: pip install 'mime-engine==0.3.0.dev9'"


def test_a_document_edited_after_its_lock_is_refused_naming_what_changed(saved):
    _edit(saved, DOCUMENT_FILE, lambda t: t.replace("  binds:\n    body: probe\nobservations",
                                                     "  binds:\n    body: probe\n  settings:\n    k:\n"
                                                     "      value: 2.5\n      unit: kg s^-2\nobservations"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.input-hash" and refusal.category is RefusalClass.USER
    assert "resolved.entries.toy.spring:probe.settings.k.value: 1.7 -> 2.5" in refusal.message


def _static_phase(built):
    return built.graph.get_node("source.push").static_data["phase"]


def test_a_static_arrays_contents_change_the_derived_hash_where_maddenings_description_does_not(monkeypatch):
    """The same shape and dtype with other contents: MADDENING's config and its static-data hash cannot see the
    change, and MIME's derived hash does. The control: the unchanged table gives the same hash."""
    first, again = full_toy().build(), full_toy().build()
    monkeypatch.setattr(toy_nodes, "PHASE", (0.0, 0.25, -0.75))
    changed = full_toy().build()
    a, b = _static_phase(first), _static_phase(changed)
    assert (a.value.shape, a.value.dtype) == (b.value.shape, b.value.dtype) and not np.array_equal(a.value, b.value)
    assert graph_config(first.graph) == graph_config(changed.graph)
    node = SimulationNode.static_data_hash
    assert node(first.graph.get_node("source.push")) == node(changed.graph.get_node("source.push"))
    assert first.report.derived_hash == again.report.derived_hash
    assert changed.report.derived_hash != first.report.derived_hash


def test_a_factory_that_builds_another_graph_under_the_same_version_is_refused(saved, monkeypatch):
    """The factory guard: the installed toy field now builds other contents under the same name and version, and a
    load in the environment the lock was written in refuses it."""
    monkeypatch.setattr(toy_nodes, "PHASE", (0.0, 0.25, -0.75))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(saved)
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.derived-hash"
    assert "physics.static.source.push.phase.sha256" in refusal.message


def test_in_another_environment_differences_are_reported_and_outputs_compared_within_tolerance(saved, monkeypatch):
    _edit_lock(saved, lambda lock: lock["environment"].update(jax="0.0.1", device="tpu"))
    experiment = Experiment.load(saved)
    report = experiment.load_report
    assert report is not None and not report.same_environment
    assert "jax: built under '0.0.1'" in report.text() and "device: built under 'tpu'" in report.text()
    assert experiment.build().verify().reproduced
    monkeypatch.setattr(toy_nodes, "PHASE", (0.0, 0.25, -0.75))
    moved = Experiment.load(saved)
    assert moved.load_report is not None and "static array source.push/phase" in moved.load_report.text()
    verified = moved.build().verify()
    print(f"\n{verified.text()}")
    assert not verified.reproduced and any(line.startswith("source.push.push_toy_hold_probe")
                                           for line in verified.moved)


def test_mime_verify_exits_with_what_it_found(saved, capsys, monkeypatch):
    assert mime_main(["verify", str(saved)]) == 0
    assert "verify: reproduced 8 steps bit for bit" in capsys.readouterr().out
    _edit_lock(saved, lambda lock: lock["environment"].update(jax="0.0.1"))
    monkeypatch.setattr(toy_nodes, "PHASE", (0.0, 0.25, -0.75))
    assert mime_main(["verify", str(saved)]) == 1
    _edit(saved, DOCUMENT_FILE, lambda t: t.replace("schema: mime.document/1", "schema: mime.document/0"))
    assert mime_main(["verify", str(saved)]) == 2
    assert "load.document-shape" in capsys.readouterr().err


def test_an_experiment_a_load_could_not_find_cannot_be_saved(tmp_path):
    experiment = coupled_toy()
    experiment.attach(Unsavable(table={"a": 1}), body="probe")
    built = experiment.build()
    with pytest.raises(BuildRefused) as refused:
        built.save(tmp_path)
    print(f"\n{refused.value}")
    assert refused.value.codes() == ["save.unsavable", "save.not-loadable"]
    assert "a dict; a document holds numbers" in refused.value.refusals[0].message
    assert "no installed package declares a 'mime.entries' entry point for namespace 'loading'" in \
        refused.value.refusals[1].message
    assert not (tmp_path / LOCK_FILE).exists()


def test_a_document_saved_before_a_version_changed_is_refused_naming_the_entry(tmp_path):
    """A document is reproduced by the versions it names or refused: one saved with rigid_inertial 0.1.0 (before 4c3
    solved its step for the change in velocity and held its contact inside the solve) names a version no installed
    package registers, and the load says which model changed."""
    from script_a import script_a

    script_a(drag=False).save(tmp_path)
    document = (tmp_path / DOCUMENT_FILE).read_text()
    assert "name: rigid_inertial\n        version: 0.2.0" in document
    _edit(tmp_path, DOCUMENT_FILE, lambda t: t.replace("name: rigid_inertial\n        version: 0.2.0",
                                                        "name: rigid_inertial\n        version: 0.1.0"))
    _edit_lock(tmp_path, lambda lock: [record.update(version="0.1.0") for record in lock["entries"]
                                       if record["name"] == "rigid_inertial"])
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(tmp_path)
    print(f"\n{refused.value}")
    (refusal,) = refused.value.refusals
    assert refusal.code == "load.version"
    assert "dynamics rigid_inertial 0.1.0, and the installed packages register rigid_inertial at 0.2.0 only" \
        in refusal.message
