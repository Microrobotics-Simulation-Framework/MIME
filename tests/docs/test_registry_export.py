"""The registry export, ``mime registry --json``, taken in a fresh process: every installed entry, each with every
``ClassVar`` and dataclass field it declares. Its completeness test reads each entry class itself and fails on a
value held at class level that the export does not carry, a ``ClassVar`` not declared (B3) or one it cannot encode.

The controls: an entry class planted with an undeclared class attribute, and one with a ``ClassVar`` of a type the
export has no rule for, each fail the completeness test by name; one declaring its ``ClassVar`` passes, carried with
its type, default and value. A namespace whose import fails for any reason but the missing ``verify`` extra fails the
export."""

import importlib
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import ClassVar

from mime.docs import completeness_problems, encode, entry_record, registry_export
from mime.gradients import GradientStatus
from mime.interactions import Drag
from mime.registry import NOT_ESTABLISHED, EntryKind, Interaction, Registered, RoleKind, dimensioned_settings
from mime.testing import installed_entries


def _class(defined_in: str) -> type:
    module, _, name = defined_in.rpartition(".")
    return getattr(importlib.import_module(module), name)


def test_the_export_lists_every_installed_entry_with_its_fields():
    exported = registry_export()
    keys = {(e["kind"], e["name"], e["version"]) for e in exported["entries"]}
    assert keys == set(installed_entries()) and ("interaction", "drag", "0.3.0") in keys
    assert exported["not_loaded"] == {}
    drag = next(e for e in exported["entries"] if e["name"] == "drag")
    assert drag["class_vars"]["builds"]["value"] == {
        "surrogate": ["mime.nodes.hydro._surrogate_drag.SurrogateDragNode"],
        "bem": ["mime.nodes.hydro.bem._node.BEMDragNode"]}
    assert drag["class_vars"]["family"]["value"] == "hydrodynamic"
    assert drag["class_vars"]["references"]["value"] == list(Drag.references)
    assert Drag.gradient is not None and drag["class_vars"]["gradient"]["value"]["route"] == Drag.gradient.route
    assert encode(NOT_ESTABLISHED) == {"not_established": True}
    assert drag["fields"]["lubrication"]["group"] == "lubrication"
    assert drag["settings"]["lubrication.epsilon"]["dimension"]["unit"] == "m"
    carried = set(drag["class_vars"])
    assert {"moving", "id_roles", "values", "rungs", "reserved_rungs", "support", "surrogates",
            "rung_gradients"} <= carried
    by_rung = drag["class_vars"]["rung_gradients"]["value"]
    assert {rung: held["differentiability"] for rung, held in by_rung.items()} == {"surrogate": "piecewise_jump",
                                                                                   "bem": "exact"}
    assert [d["owner"] for d in by_rung["surrogate"]["gradient"]["switches"]] == [
        d.owner for d in Drag.rung_gradients["surrogate"].gradient.switches] and by_rung["bem"]["gradient"] is None


def test_a_controllers_status_in_each_mode_is_carried_by_the_export():
    """``arm_tracking`` is ``piecewise_jump`` inside the compiled step (its law's, with its switches) and ``cut`` as a
    callback (the binding's, with its route): both are in the export."""
    arm = next(e for e in registry_export()["entries"] if e["name"] == "arm_tracking")
    assert arm["class_vars"]["differentiability"]["value"] == "piecewise_jump"
    callback = arm["class_vars"]["callback_gradient"]["value"]
    assert callback["differentiability"] == "cut" and "inside the compiled step" in callback["gradient"]["route"]


def test_every_entry_class_var_is_carried_by_the_export():
    exported = registry_export()
    checked = 0
    for record in exported["entries"]:
        assert completeness_problems(_class(record["defined_in"]), record) == []
        checked += 1
    assert checked == len(installed_entries()) > 20


# -- B3 -------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True)
class Undeclared(Interaction):
    name = "docs.undeclared"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    table = {"a": 1}


class _Opaque:
    pass


@dataclass(frozen=True, kw_only=True)
class Opaque(Interaction):
    name = "docs.opaque"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    held: ClassVar[object] = _Opaque()


@dataclass(frozen=True, kw_only=True)
class Declared(Interaction):
    name = "docs.declared"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    table: ClassVar[dict[str, int]] = {"a": 1}


def record_of(cls: type[Interaction]) -> dict:
    return entry_record(Registered(EntryKind.INTERACTION, cls.name, cls.version, cls, dimensioned_settings(cls)[0]))


def test_b3_a_class_attribute_not_declared_a_classvar_fails_the_completeness_test():
    (problem,) = completeness_problems(Undeclared, record_of(Undeclared))
    assert "Undeclared.table is held at class level and is not a declared ClassVar" in problem


def test_b3_a_classvar_the_export_cannot_encode_fails_the_completeness_test():
    (problem,) = completeness_problems(Opaque, record_of(Opaque))
    assert "ClassVar held holds a test_registry_export._Opaque, which the registry export has no rule for" in problem


def test_b3_a_declared_classvar_is_carried_with_its_type_default_and_value():
    record = record_of(Declared)
    assert completeness_problems(Declared, record) == []
    assert record["class_vars"]["table"] == {"type": "ClassVar[dict[str, int]]", "declared_by":
                                             "test_registry_export.Declared", "default": {"a": 1}, "value": {"a": 1}}


def test_a_namespace_failing_to_import_for_another_reason_fails_the_export():
    script = textwrap.dedent("""
        import importlib.metadata, sys
        real = importlib.metadata.entry_points
        def points(**kwargs):
            found = list(real(**kwargs))
            return found + [importlib.metadata.EntryPoint("broken", "no_such_module_of_a_lab", "mime.entries")]
        importlib.metadata.entry_points = points
        from mime.cli import main
        sys.exit(main(["registry", "--json"]))
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert done.returncode != 0 and "no_such_module_of_a_lab" in done.stderr
