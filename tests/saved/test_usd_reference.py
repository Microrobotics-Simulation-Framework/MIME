"""The ``mime:`` USD schema's reference (``docs/reference/mime_usd.md``) against what the code writes on a stage and
reads from one, in both directions (``mime.docs._usd_reference``).

The fixtures' stages are the round-trip cases' (``cases.py``, which use every registered entry and every kind of
thing), each planned and its stage written, no graph built. Every name they author and every constant the readers of a
stage read with must be documented, and every documented name authored or listed as read only. Each control doctors
one input of the real check (a stage, a reader module's copy, the reference's text) and fails on every run: the
inputs are deterministic.
"""

from pathlib import Path

import pytest
from cases import CASES

from mime.build._stage import stage_spec
from mime.docs._check import LISTS
from mime.docs._lists import read_list
from mime.docs._usd_reference import (
    READERS,
    SETTING_PATTERN,
    SOURCE,
    Authored,
    authored_on,
    documented,
    kind_settings,
    read_constants,
    reference_problems,
)
from mime.scene import stage_text

ROOT = Path(__file__).resolve().parents[2]
READ_ONLY = ROOT / "docs" / "ratchets" / "usd_read_only.txt"


@pytest.fixture(scope="module")
def stages() -> dict[str, str]:
    """Each round-trip case's stage, as a save would write it, by the case's name."""
    return {case.name: stage_text(stage_spec(case.make().plan())) for case in CASES if not case.refused}


@pytest.fixture(scope="module")
def authored(stages) -> set[Authored]:
    return {item for text in stages.values() for item in authored_on(text)}


@pytest.fixture(scope="module")
def names() -> dict[str, str]:
    held, problems = documented((ROOT / SOURCE).read_text(encoding="utf-8"))
    assert problems == []
    return held


def read() -> dict[str, dict[str, str]]:
    return {path.as_posix(): read_constants(ROOT / path) for path in READERS}


def read_only(path: Path = READ_ONLY):
    return read_list(path, LISTS["usd_read_only"])


def problems_of(names, authored, constants=None, listed=None):
    return reference_problems(names, read() if constants is None else constants, listed or read_only(),
                              kind_settings(), authored)


def test_the_reference_holds_in_both_directions_on_the_fixtures_stages(names, authored):
    assert problems_of(names, authored) == []
    assert {item.name for item in authored} >= {"MimeGeometryAPI", "MimeMagnetAPI", "mime:magnet:direction"}


def test_control_a_name_authored_on_a_stage_and_not_documented_fails(names, authored):
    planted = {*authored, Authored("mime:planted")}
    assert any(p.startswith("mime:planted is authored on a fixture's stage") for p in problems_of(names, planted))


def test_control_a_setting_its_prims_kind_does_not_write_is_not_documented_by_the_pattern(names, authored):
    planted = {*authored, Authored("mime:geometry:colour", ("screw", "0.3.0"))}
    problems = problems_of(names, planted)
    assert any("mime:geometry:colour is authored on a fixture's stage on a prim of the geometry kind screw 0.3.0" in p
               for p in problems), problems


def test_control_a_constant_a_reader_reads_and_the_reference_does_not_document_fails(names, authored, tmp_path):
    """A doctored copy of scene/_compose.py with one more constant read from a stage."""
    copy = tmp_path / "_compose.py"
    copy.write_text((ROOT / READERS[0]).read_text(encoding="utf-8")
                    + '\n#: planted\nPLANTED_AXIS = "mime:magnet:axis"\n', encoding="utf-8")
    constants = {**read(), "planted": read_constants(copy)}
    problems = problems_of(names, authored, constants)
    assert any("PLANTED_AXIS = 'mime:magnet:axis' is read from a stage" in p for p in problems), problems


def test_control_a_documented_name_no_stage_authors_fails_unless_listed_read_only(names, authored, tmp_path):
    planted = {**names, "mime:magnet:strength": "planted"}
    problems = problems_of(planted, authored)
    assert any("mime:magnet:strength: documented in the mime: reference and authored on no fixture's stage" in p
               for p in problems), problems
    listed = tmp_path / "usd_read_only.txt"
    listed.write_text(READ_ONLY.read_text(encoding="utf-8") + "mime:magnet:strength  # planted\n", encoding="utf-8")
    assert problems_of(planted, authored, listed=read_only(listed)) == []


def test_control_a_listed_name_a_stage_authors_fails_naming_its_line(names, authored, tmp_path):
    listed = tmp_path / "usd_read_only.txt"
    listed.write_text(READ_ONLY.read_text(encoding="utf-8") + "mime:thing  # planted\n", encoding="utf-8")
    problems = problems_of(names, authored, listed=read_only(listed))
    assert any("'mime:thing' now passes; delete this line" in p for p in problems), problems


def test_the_setting_pattern_is_documented_and_each_of_mimes_kinds_has_a_native_form_or_settings(names):
    assert SETTING_PATTERN in names
    mimes = {key: held for key, held in kind_settings().items() if "." not in key[0]}
    assert mimes and all(native or settings for native, settings in mimes.values())
