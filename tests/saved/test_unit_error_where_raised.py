"""A unit error's refusal follows where it is raised (usability bundle 2, the review's second major).

``UnitError`` is a ``MimeError`` and a ``ValueError``. The handlers that run code on MIME's behalf and wrap what it raises
(an entry's ``setting_problems``, a controller's ``law()``, a reader of a results manifest) used to see a unit bug inside
that code as a plain ``ValueError`` and gave it their defect or format code; once it was a ``MimeError`` they passed it on
as ``call.units``, the user's mistake. Each plant below is one of the three the review made: a unit bug in a lab entry's
``setting_problems``, in a controller's ``law()``, and a manifest that names a unit MIME cannot read. Each gives at this
head the code it gave at the base (``entry.defect``, ``entry.defect``, ``result.format``). **The controls:** the same plant
raising a plain ``ValueError`` gives the same codes (the path is the handler's), and a real MIME refusal raised there still
keeps its own code (the handler still passes one on). Raised from a script's own expression the error keeps ``call.units``."""

import pytest
from cases import entry_assets_rig
from control_cases import Counter, spun_magnet

import mime.results._result as result_module
from mime.build import ArgumentRefused, MimeError, Refusal, RefusalClass
from mime.results._result import MANIFEST_FORMAT, READ_VERSIONS, RunResult
from mime.units import UnitError, mm, mm_per_s, s

lab = pytest.importorskip("mime_lab_fixture")


def unit_bug(*_: object) -> None:
    s(1.0).to(mm)  # a time read as a length: the entry's own unit bug


def plain_bug(*_: object) -> None:
    raise ValueError("a bug of the entry's that is no unit error")


def a_refusal_of_the_users(*_: object) -> None:
    raise ArgumentRefused([Refusal("setting.value", RefusalClass.USER, "a refusal the entry met below it", "change it")])


def refused_codes(build) -> list[str]:
    with pytest.raises(MimeError) as caught:
        build()
    return caught.value.codes()


# -- an entry's setting_problems ---------------------------------------------------------------------------------------


@pytest.fixture
def planted_setting_problems(monkeypatch):
    def plant(function):
        monkeypatch.setattr(lab.FinishedGantry, "setting_problems", function)
        return refused_codes(entry_assets_rig().build)

    return plant


def test_a_unit_bug_in_an_entrys_setting_problems_is_the_entrys_defect_as_at_the_base(planted_setting_problems):
    codes = planted_setting_problems(unit_bug)
    assert "entry.defect" in codes and "call.units" not in codes


def test_the_control_a_plain_error_there_gives_the_same_codes(planted_setting_problems):
    """The entry's failed gantry brings refusals of its own beside the defect (its mount, its command): the unit bug and
    the plain error give the same list, which is what the base gave."""
    codes = planted_setting_problems(plain_bug)
    assert "entry.defect" in codes and codes == planted_setting_problems(unit_bug)


def test_the_control_a_real_refusal_there_keeps_its_own_code(planted_setting_problems):
    assert "setting.value" in planted_setting_problems(a_refusal_of_the_users)


# -- a controller's law() ----------------------------------------------------------------------------------------------


@pytest.fixture
def planted_law(monkeypatch):
    def plant(function):
        monkeypatch.setattr(Counter, "law", lambda self, ctx: function())
        return refused_codes(spun_magnet(controller=Counter()).build)

    return plant


def test_a_unit_bug_in_a_controllers_law_is_the_entrys_defect_as_at_the_base(planted_law):
    assert planted_law(unit_bug) == ["entry.defect"]


def test_the_control_a_plain_error_in_a_law_gives_the_same_code(planted_law):
    assert planted_law(plain_bug) == ["entry.defect"]


def test_the_control_a_real_refusal_in_a_law_keeps_its_own_code(planted_law):
    assert planted_law(a_refusal_of_the_users) == ["setting.value"]


# -- a results manifest ------------------------------------------------------------------------------------------------


def manifest_naming(unit: str) -> dict:
    return {"format": MANIFEST_FORMAT, "version": max(READ_VERSIONS), "flags": [], "observations": [
        {"id": "o", "entry": "e", "outputs": [{"name": "x", "node": "n", "field": "f", "kind": "per_step", "unit": unit,
                                               "shape": [1], "description": "d"}]}]}


def read_codes(manifest: dict) -> list[str]:
    with pytest.raises(MimeError) as caught:
        RunResult.from_manifest(manifest, {})
    return caught.value.codes()


def test_a_manifest_naming_a_unit_mime_cannot_read_is_a_format_refusal_as_at_the_base():
    assert read_codes(manifest_naming("furlong")) == ["result.format"]


def test_the_control_a_plain_error_reading_the_manifest_gives_the_same_code(monkeypatch):
    monkeypatch.setattr(result_module, "parse_unit", plain_bug)
    assert read_codes(manifest_naming("m")) == ["result.format"]


# -- from a script's own expression ---------------------------------------------------------------------------------


def test_raised_from_a_scripts_own_expression_it_keeps_its_own_code():
    with pytest.raises(UnitError) as caught:
        mm_per_s(3.0).to(mm)
    assert caught.value.codes() == ["call.units"]
