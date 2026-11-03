"""``mime push <folder> --setup``, a walk through ``sharing.yaml`` a field at a time, and a person's sharing defaults:
kept once in MIME's config directory, never in an experiment's folder; ``--init`` fills from them, ``mime push
--defaults`` shows them and where they live, and an ORCID iD in them is held to its check digit.

The walk is driven here by a person's answers given in order (``Person``), as the command line's display asks them;
``test_push_setup_terminal.py`` drives it through a real pseudo-terminal.

Controls, each caught on every run:

- **Enter read as no**: with ``use <value>? [Y/n]`` planted to take Enter for n, the walk that keeps every value by
  Enter is caught asking for new ones;
- **an invalid value taken**: with the ORCID iD's check planted away, a mistyped iD is caught written into the file,
  and caught in the defaults;
- **defaults kept in the folder**: with the defaults' place planted into the experiment's folder, the folder is caught
  holding them;
- **an existing file replaced**: with the edit planted to write the file anew, a comment of the person's is caught lost;
- **an edit that loses the file's own form** (the review of #112, F5): with the file read with its line endings
  translated, a CRLF file is caught written back in LF; with the old replace, a linked ``sharing.yaml`` is caught
  replaced by a file of its own, and a read-only one caught replaced;
- **a rewritten key's own comment** (F5's last item): with the comment planted away, a retitled line is caught losing
  it; with the file's reading planted away, a ``#`` inside a quoted title is caught kept as a comment, and a comment
  caught put inside a value written over several lines;
- **a question into the file** (F6): with ``--setup`` asking where every other question asks, standard output and
  standard error both redirected to files, its questions are caught in the file and none on the terminal;
- **an ORCID iD in another script's digits**: with the iD's pattern planted back to ``\\d``, an iD in full-width digits
  (which a CJK input method types, and a terminal shows as digits) is caught taken by the walk, the defaults and
  ``sharing.yaml`` alike (the review of #112, F4);
- **a ROR identifier in another script's digits** (F4's class): with the identifier's one pattern planted back to
  ``\\d``, a full-width one is caught taken by the walk, the defaults and ``sharing.yaml``, for an affiliation and a
  funder alike.
"""

from __future__ import annotations

import builtins
import io
import os
import re
import stat
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml
from hypothesis import given as drawn
from hypothesis import settings
from hypothesis import strategies as st
from push_cases import copy_of, saved_toy, write_sharing

from mime import push
from mime.build import BuildRefused
from mime import cli
from mime.cli import main
from mime.push import _defaults, _edits, _fields, _setup, _sharing

ORCID = "0000-0002-1825-0097"
#: The same iD with its check digit mistyped.
MISTYPED = "0000-0002-1825-0098"
#: The same iD in full-width digits but its check digit, as a CJK input method types it: its check digit sums.
FULL_WIDTH = "".join(chr(0xFF10 + int(each)) if each.isdigit() else each for each in ORCID[:-1]) + ORCID[-1]


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A saved toy with no sharing.yaml yet."""
    return saved_toy(tmp_path_factory.mktemp("toy"))


class Person:
    """A person at the terminal: each question answered in turn from ``answers``; what was asked and said, kept."""

    def __init__(self, *answers: str, save: bool = False) -> None:
        self.answers = list(answers)
        self.save = save
        self.asked: list[str] = []
        self.said: list[str] = []

    def typed(self, prompt: str) -> str:
        self.asked.append(prompt)
        assert self.answers, f"asked more than was answered: {prompt!r}"
        return self.answers.pop(0)

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        return self.save

    def say(self, line: str) -> None:
        self.said.append(line)


def walk(folder: Path, person: Person) -> dict[str, object]:
    push.setup(folder, typed=person.typed, ask=person.ask, say=person.say, service="zenodo-sandbox")
    assert person.answers == [], f"answers left unasked: {person.answers}"
    return yaml.safe_load((folder / "sharing.yaml").read_text(encoding="utf-8"))


#: Every field of a new file skipped by Enter but the creators (one person, an ORCID iD mistyped first) and the
#: uploader: title and licence kept by Enter.
FIRST = ("", "", "", "Carberry", "Josiah", MISTYPED, ORCID, "Brown University", "", "", "", "", "", "Carberry, Josiah",
         "Brown University", "", "", "", "")


def test_a_new_file_is_walked_field_by_field_in_the_scaffolds_order_each_with_its_line(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST)
    held = walk(folder, person)
    heads = [prompt.split("\n")[0].split(":")[0] for prompt in person.asked if "\n" in prompt
             and prompt.split("\n")[0].split(":")[0] in _setup.FIELDS]
    assert heads == [key for key in _setup.FIELDS if key != "fields"], "one field at a time, in the scaffold's order"
    for key, why in _setup.FIELDS.items():
        assert any(f"{key}: {why}" in each for each in [*person.asked, *person.said]), key
    assert person.asked[0].endswith('use "full_toy"? [Y/n] '), "a value already held is offered to keep"
    assert held["title"] == "full_toy" and held["licence"] == "CC-BY-4.0"
    assert held["creators"] == [{"family": "Carberry", "given": "Josiah", "orcid": ORCID,
                                 "affiliation": "Brown University"}]
    assert held["uploader"] == {"name": "Carberry, Josiah", "institution": "Brown University"}
    assert held["contributors"] == [] and held["keywords"] == [] and "access" not in held, "Enter skips"
    assert any(prompt.startswith("the ORCID iD's check digit is 8") for prompt in person.asked), \
        "an invalid value is said, and asked again"
    assert push.read_sharing(folder).creators[0].orcid == ORCID
    assert (folder / "sharing.yaml").read_text(encoding="utf-8").startswith("# The record's metadata for `mime push`")


def test_enter_or_y_keeps_a_value_and_n_asks_for_another(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    person = Person("y", "", "", "n", "Doe", "Jane", "", "", "", "", "", "", "", "", "", "", "")
    held = walk(folder, person)
    assert held["creators"] == [{"family": "Doe", "given": "Jane"}], "n asked for a new one"
    assert held["uploader"] == {"name": "Carberry, Josiah", "institution": "Brown University"}, "Enter kept it"
    assert sum("? [Y/n] " in prompt for prompt in person.asked) == 4, "title, licence, creators, uploader"


def test_an_existing_file_is_edited_key_by_key_and_said_never_replaced_without_a_word(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# a note of my own, kept\n", encoding="utf-8")
    person = Person("n", "A better title", *([""] * 11))
    walk(folder, person)
    text = path.read_text(encoding="utf-8")
    assert "# a note of my own, kept" in text and "# Who funded the work" in text
    assert yaml.safe_load(text)["title"] == "A better title"
    assert any(line.startswith(f"edited {path}: title,") for line in person.said), person.said
    unchanged = Person(*([""] * 12))
    walk(folder, unchanged)
    assert any(line.startswith(f"kept {path} as it was") for line in unchanged.said)


def test_a_licence_other_than_the_experiments_is_said_and_asked_again(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    person = Person("", "", "n", "MIT", "not a licence", "CC-BY-4.0", *FIRST[3:])
    assert walk(folder, person)["licence"] == "CC-BY-4.0"
    assert any(prompt.startswith("the experiment declares CC-BY-4.0") for prompt in person.asked)


def test_a_contributor_needs_its_type_and_a_related_work_its_relation(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:8], "", "", "Hamilton", "", "", "", "", "Boss", "Supervisor", "", "", "10.1234/x", "",
                    "Cites", "", "Carberry, Josiah", "", "", "", "", "")
    held = walk(folder, person)
    assert held["contributors"] == [{"family": "Hamilton", "type": "Supervisor"}]
    assert held["related"] == [{"identifier": "10.1234/x", "relation": "Cites"}]
    assert any(prompt.startswith("a contributor is credited with a DataCite contributor type") for prompt in
               person.asked) and any(prompt.startswith("Boss is not one of DataCite's") for prompt in person.asked)


def test_what_push_would_refuse_as_it_stands_is_said_and_not_refused(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    person = Person(*([""] * 12))
    held = walk(folder, person)
    assert held["creators"] == []
    assert any(line.startswith("! push would refuse it as it stands: sharing.yaml names no creator") for line in
               person.said)


def test_without_a_terminal_setup_is_refused_naming_init_and_writes_nothing(saved: Path, tmp_path: Path,
                                                                          capsys: pytest.CaptureFixture[str]):
    folder = copy_of(saved, tmp_path)
    with pytest.raises(BuildRefused) as refused:
        push.setup(folder, typed=None, ask=None, say=lambda line: None)
    assert refused.value.codes() == ["push.setup"] and "--init" in refused.value.refusals[0].action
    assert main(["push", str(folder), "--setup"]) == 2  # pytest's standard input is no terminal
    assert "push.setup" in capsys.readouterr().err and not (folder / "sharing.yaml").exists()


# --- the defaults ----------------------------------------------------------------------------------------------------


def test_the_walks_values_saved_as_defaults_live_in_the_config_directory_and_fill_init(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], private_home: Path):
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST, save=True)
    walk(folder, person)
    assert person.asked[-1].startswith("Save these as your defaults (creators, uploader)")
    path = _defaults.defaults_path()
    assert path.is_file() and private_home in path.parents and folder not in path.parents
    assert _defaults.read_defaults() == {"creators": [{"family": "Carberry", "given": "Josiah", "orcid": ORCID,
                                                       "affiliation": "Brown University"}],
                                         "uploader": {"name": "Carberry, Josiah", "institution": "Brown University"}}
    assert not list(folder.rglob(_defaults.DEFAULTS_FILE)), "the defaults are never in an experiment's folder"
    other = copy_of(saved, tmp_path / "other")
    assert main(["push", str(other), "--to", "zenodo-sandbox", "--init"]) == 0
    out = capsys.readouterr().out
    assert f"filled from your sharing defaults ({path}): creators, uploader" in out
    assert push.read_sharing(other).creators[0].orcid == ORCID
    assert main(["push", "--defaults"]) == 0
    shown = capsys.readouterr().out
    assert str(path) in shown and ORCID in shown
    again = Person(*([""] * 12), save=True)
    walk(copy_of(saved, tmp_path / "third"), again)
    assert not any(each.startswith("Save these") for each in again.asked), "the defaults themselves are not asked again"


def test_with_no_defaults_init_writes_the_scaffold_as_it_always_was(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    push.init(folder)
    assert (folder / "sharing.yaml").read_text(encoding="utf-8") == _sharing.scaffold("full_toy", "CC-BY-4.0")


def test_an_orcid_in_the_defaults_is_held_to_its_check_digit(tmp_path: Path):
    path = _defaults.defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"format: {_defaults.DEFAULTS_FORMAT}\ncreators: [{{family: Carberry, orcid: {MISTYPED}}}]\n",
                    encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _defaults.read_defaults()
    assert refused.value.codes() == ["push.creators"] and "check digit" in refused.value.refusals[0].message
    with pytest.raises(BuildRefused):
        _defaults.write_defaults({"creators": [{"family": "Carberry", "orcid": MISTYPED}]})
    path.write_text(f"format: {_defaults.DEFAULTS_FORMAT}\ntitle: not a default\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _defaults.read_defaults()
    assert refused.value.codes() == ["push.sharing"]
    del tmp_path


def test_a_stored_token_is_never_saved_into_the_defaults(monkeypatch: pytest.MonkeyPatch):
    token = "a-token-typed-as-a-name-" + "t" * 20
    monkeypatch.setenv("ZENODO_SANDBOX_TOKEN", token)
    with pytest.raises(BuildRefused) as refused:
        _defaults.write_defaults({"uploader": {"name": token}})
    assert refused.value.codes() == ["push.sharing"] and token not in str(refused.value)
    assert not _defaults.defaults_path().exists()


# --- an ORCID iD's digits are ASCII ones, wherever an iD is read (the review of #112, F4) ---------------------------

#: The zero of each run of ten decimal digits Unicode holds (its category Nd), ASCII's among them.
_ZEROS = [chr(code) for code in range(0x110000) if unicodedata.category(chr(code)) == "Nd" and
          unicodedata.digit(chr(code)) == 0 and all(unicodedata.digit(chr(code + step), None) == step
                                                     for step in range(10))]


def _check_character(digits: list[int]) -> str:
    total = 0
    for each in digits:
        total = (total + each) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)


@st.composite
def orcids(draw: st.DrawFn) -> tuple[str, bool]:
    """An ORCID iD whose check character sums: its digits all ASCII ones, or each drawn in any script's digits with at
    least one not ASCII. The iD, and whether every digit is ASCII."""
    values = draw(st.lists(st.integers(0, 9), min_size=15, max_size=15))
    check = _check_character(values)
    digits = [*values, *([int(check)] if check != "X" else [])]
    zeros = ["0"] * len(digits)
    if not draw(st.booleans()):
        zeros = draw(st.lists(st.sampled_from(_ZEROS), min_size=len(digits), max_size=len(digits)))
        zeros[draw(st.integers(0, len(digits) - 1))] = draw(st.sampled_from([each for each in _ZEROS if each != "0"]))
    characters = [chr(ord(zero) + value) for zero, value in zip(zeros, digits, strict=True)]
    text = "".join(characters) + ("X" if check == "X" else "")
    grouped = "-".join(text[at:at + 4] for at in range(0, 16, 4))
    return (f"https://orcid.org/{grouped}" if draw(st.booleans()) else grouped), all(zero == "0" for zero in zeros)


@settings(deadline=None, max_examples=300)
@drawn(drawn_orcid=orcids())
def test_an_orcid_is_taken_in_ascii_digits_alone_whatever_its_check_digit_sums_to(drawn_orcid: tuple[str, bool]):
    orcid, ascii_only = drawn_orcid
    problem = _sharing.orcid_problem(orcid)
    assert (problem is None) == ascii_only, (orcid, problem)


def test_a_full_width_orcid_is_said_and_asked_again_in_the_walk_and_never_saved(saved: Path, tmp_path: Path):
    assert _sharing.orcid_problem(FULL_WIDTH.replace("\uff10", "0")) is not None
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:5], FULL_WIDTH, *FIRST[6:], save=True)
    held = walk(folder, person)
    assert held["creators"][0]["orcid"] == ORCID  # type: ignore[index]
    assert any(prompt.startswith("an ORCID iD is four groups of four") and "ASCII" in prompt
               for prompt in person.asked), "the full-width iD is said, and asked again"
    assert FULL_WIDTH not in _defaults.defaults_path().read_text(encoding="utf-8")


def test_a_full_width_orcid_is_refused_in_the_defaults_and_in_sharing_yaml(saved: Path, tmp_path: Path):
    path = _defaults.defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"format: {_defaults.DEFAULTS_FORMAT}\ncreators: [{{family: Carberry, orcid: {FULL_WIDTH}}}]\n",
                    encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _defaults.read_defaults()
    assert refused.value.codes() == ["push.creators"]
    with pytest.raises(BuildRefused):
        _defaults.write_defaults({"creators": [{"family": "Carberry", "orcid": FULL_WIDTH}]})
    path.unlink()
    folder = copy_of(saved, tmp_path)
    write_sharing(folder, creators=f"{{family: Carberry, orcid: {FULL_WIDTH}}}")
    with pytest.raises(BuildRefused) as refused:
        push.read_sharing(folder)
    assert refused.value.codes() == ["push.creators"]


# --- a ROR identifier's digits are ASCII ones, wherever one is read (F4's class) -------------------------------------

#: Brown University's ROR identifier, an affiliation's, and the National Science Foundation's, a funder's.
BROWN, NSF = "05gq02987", "021nxhr62"
#: Each with its two check digits full-width, as a CJK input method types them.
FULL_WIDTH_BROWN, FULL_WIDTH_NSF = (ror[:-2] + "".join(chr(0xFF10 + int(each)) for each in ror[-2:])
                                    for ror in (BROWN, NSF))
#: What a ROR identifier's six middle characters are drawn from: Crockford's base 32, as ROR writes it.
_ROR_MIDDLE = "0123456789abcdefghjkmnpqrstvwxyz"


@st.composite
def rors(draw: st.DrawFn) -> tuple[str, bool]:
    """A ROR identifier's form (0, six of Crockford's base 32, two check digits): every digit an ASCII one, or each
    digit an ASCII one or another script's, at least one not ASCII. The identifier, and whether every digit is ASCII.
    Each digit's script is drawn on its own, so the check digits alone are often the ones not ASCII."""
    characters = ["0", *draw(st.lists(st.sampled_from(_ROR_MIDDLE), min_size=6, max_size=6)),
                  *(str(each) for each in draw(st.lists(st.integers(0, 9), min_size=2, max_size=2)))]
    digits = [at for at, each in enumerate(characters) if each.isdigit()]
    if not draw(st.booleans()):
        others = [each for each in _ZEROS if each != "0"]
        foreign = [at for at in digits if draw(st.booleans())] or [draw(st.sampled_from(digits))]
        for at in foreign:
            characters[at] = chr(ord(draw(st.sampled_from(others))) + int(characters[at]))
    ascii_only = all(each.isascii() for each in characters)
    text = "".join(characters)
    return (f"https://ror.org/{text}" if draw(st.booleans()) else text), ascii_only


@settings(deadline=None, max_examples=300)
@drawn(drawn_ror=rors())
def test_a_ror_identifier_is_taken_in_ascii_digits_alone_by_every_reader(drawn_ror: tuple[str, bool]):
    ror, ascii_only = drawn_ror
    walked = _sharing.ror_problem(ror)
    funded = _fields.funding_problems([{"funder": "A funder", "ror": ror}])[1]
    credited = _sharing.defaults_refusals({"format": _defaults.DEFAULTS_FORMAT, "creators": [
        {"family": "Carberry", "affiliation": "Brown University", "ror": ror}]}, Path("defaults.yaml"),
        _defaults.DEFAULTS_FORMAT)
    assert (walked is None) == (funded == []) == (credited == []) == ascii_only, (ror, walked, funded, credited)


def test_a_full_width_ror_is_said_and_asked_again_in_the_walk_and_never_saved(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:8], FULL_WIDTH_BROWN, BROWN, *FIRST[9:15], "National Science Foundation", FULL_WIDTH_NSF,
                    NSF, "", "", *FIRST[16:], save=True)
    held = walk(folder, person)
    assert held["creators"][0]["ror"] == BROWN and held["funding"] == [  # type: ignore[index]
        {"funder": "National Science Foundation", "ror": NSF}]
    said = [prompt for prompt in person.asked if "is not a ROR identifier" in prompt and "ASCII" in prompt]
    assert [prompt.split()[0] for prompt in said] == [FULL_WIDTH_BROWN, FULL_WIDTH_NSF], "each is said, and asked again"
    kept = _defaults.defaults_path().read_text(encoding="utf-8")
    assert BROWN in kept and NSF in kept and FULL_WIDTH_BROWN not in kept and FULL_WIDTH_NSF not in kept


def full_width_rors(saved: Path, tmp_path: Path) -> Path:
    """A ``sharing.yaml`` whose creator's affiliation and funder each carry a full-width ROR identifier."""
    folder = copy_of(saved, tmp_path)
    path = write_sharing(folder, creators=f"{{family: Carberry, affiliation: Brown University, "
                                          f"ror: {FULL_WIDTH_BROWN}}}")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("funding: []", f"funding: [{{funder: National Science Foundation, ror: "
                                                f"{FULL_WIDTH_NSF}}}]"), encoding="utf-8")
    return folder


def wheres(refused: BuildRefused) -> list[str]:
    """Where in its file each refusal points (``<file>, <where>: <problem>``)."""
    return [each.message.split(", ", 1)[1].split(": ", 1)[0] for each in refused.refusals]


def test_a_full_width_ror_is_refused_in_the_defaults_and_in_sharing_yaml(saved: Path, tmp_path: Path):
    values = {"creators": [{"family": "Carberry", "affiliation": "Brown University", "ror": FULL_WIDTH_BROWN}],
              "funding": [{"funder": "National Science Foundation", "ror": FULL_WIDTH_NSF}]}
    path = _defaults.defaults_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_defaults.defaults_text(values), encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _defaults.read_defaults()
    assert wheres(refused.value) == ["creators[0].ror", "funding[0].ror"]
    with pytest.raises(BuildRefused):
        _defaults.write_defaults(values)
    path.unlink()
    with pytest.raises(BuildRefused) as refused:
        push.read_sharing(full_width_rors(saved, tmp_path))
    assert wheres(refused.value) == ["creators[0].ror", "funding[0].ror"]


# --- an existing file edited in its own form: its line endings, a link, a read-only file (F5) -------------------------

#: A title changed on an existing file, every other field kept by Enter.
RETITLED = ("n", "A better title", *([""] * 11))


def in_crlf(folder: Path) -> bytes:
    """``sharing.yaml`` walked once, then turned to CRLF line endings, as an editor on Windows keeps it: its bytes."""
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    return path.read_bytes()


def test_a_crlf_file_is_edited_in_crlf_every_other_line_byte_for_byte(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    before = in_crlf(folder)
    walk(folder, Person(*RETITLED))
    after = (folder / "sharing.yaml").read_bytes()
    assert after.count(b"\n") == after.count(b"\r\n"), "every line ends as the file's did"
    changed = [line for line in after.split(b"\r\n") if line not in before.split(b"\r\n")]
    assert changed == [b'title: "A better title"'] or changed == [b"title: A better title"], changed
    assert yaml.safe_load(after)["title"] == "A better title"


def in_a_link(folder: Path, tmp_path: Path) -> Path:
    """``sharing.yaml`` kept elsewhere, the folder's a link to it: the file it links to."""
    walk(folder, Person(*FIRST))
    kept = tmp_path / "kept" / "sharing.yaml"
    kept.parent.mkdir()
    (folder / "sharing.yaml").replace(kept)
    (folder / "sharing.yaml").symlink_to(kept)
    return kept


def test_a_linked_file_is_edited_through_its_link_which_stays_a_link(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    kept = in_a_link(folder, tmp_path)
    person = Person(*RETITLED)
    walk(folder, person)
    assert (folder / "sharing.yaml").is_symlink() and (folder / "sharing.yaml").resolve() == kept.resolve()
    assert yaml.safe_load(kept.read_text(encoding="utf-8"))["title"] == "A better title"
    assert any(f"through the link, in {kept.resolve()}" in line for line in person.said), person.said


needs_permissions = pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0,
                                       reason="a read-only file is writable by root, and modes are POSIX's")


@needs_permissions
def test_a_read_only_file_is_refused_with_its_reason_before_anything_is_asked(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    before = path.read_bytes()
    path.chmod(0o444)
    with pytest.raises(BuildRefused) as refused:
        walk(folder, Person())
    assert refused.value.codes() == ["push.sharing"] and "is read-only" in refused.value.refusals[0].message
    assert path.read_bytes() == before and stat.S_IMODE(path.stat().st_mode) == 0o444


def test_control_a_file_read_with_its_endings_translated_is_caught_written_back_in_lf(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real = _edits.written_text
    monkeypatch.setattr(_edits, "written_text", lambda text, changed: real(text.replace("\r\n", "\n"), changed))
    folder = copy_of(saved, tmp_path)
    in_crlf(folder)
    walk(folder, Person(*RETITLED))
    assert b"\r\n" not in (folder / "sharing.yaml").read_bytes(), "the planted defect is not seen"


def _replacing(path: Path, text: str) -> None:
    """The edit as it was before the fix: the folder's own path replaced, whatever it is."""
    temporary = path.with_name(f".{path.name}.setup")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, path)


def test_control_with_the_path_replaced_a_link_is_caught_made_a_file_and_a_read_only_file_replaced(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_setup, "_replaced", _replacing)
    monkeypatch.setattr(_setup, "_unwritable", lambda path, root: None)
    folder = copy_of(saved, tmp_path)
    kept = in_a_link(folder, tmp_path)
    walk(folder, Person(*RETITLED))
    assert not (folder / "sharing.yaml").is_symlink(), "the planted defect is not seen"
    assert yaml.safe_load(kept.read_text(encoding="utf-8"))["title"] == "full_toy"
    if os.name == "posix" and os.geteuid() != 0:
        (folder / "sharing.yaml").chmod(0o444)
        walk(folder, Person("n", "Another title", *([""] * 11)))
        assert yaml.safe_load((folder / "sharing.yaml").read_text(encoding="utf-8"))["title"] == "Another title"


# --- the comment that ends a rewritten key's own line is kept (F5's last item) ----------------------------------------


def with_title_line(folder: Path, line: str, *, crlf: bool = False) -> bytes:
    """``sharing.yaml`` walked once, its title's line then written as ``line`` (in CRLF where ``crlf``): its bytes."""
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    lines = path.read_text(encoding="utf-8").split("\n")
    lines[next(at for at, each in enumerate(lines) if each.startswith("title:"))] = line
    path.write_bytes("\n".join(lines).replace("\n", "\r\n" if crlf else "\n").encode("utf-8"))
    return path.read_bytes()


def lines_changed(before: bytes, after: bytes) -> list[bytes]:
    return [line for line in after.split(b"\n") if line not in before.split(b"\n")]


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
@pytest.mark.parametrize(("line", "comment"), [
    ('title: "full_toy"  # kept by hand', "  # kept by hand"),
    ('title: "a # in the title"  # and a comment', "  # and a comment"),
    ("title: 'a '' # in the title'", ""),
    ("title: full_toy#no-comment", ""),
], ids=["comment", "hash-quoted-and-comment", "hash-quoted", "hash-unspaced"])
def test_the_comment_ending_a_rewritten_keys_line_is_kept_on_its_new_line(saved: Path, tmp_path: Path, line: str,
                                                                          comment: str, crlf: bool):
    folder = copy_of(saved, tmp_path)
    before = with_title_line(folder, line, crlf=crlf)
    walk(folder, Person(*RETITLED))
    after = (folder / "sharing.yaml").read_bytes()
    ending = b"\r" if crlf else b""
    assert lines_changed(before, after) == [f"title: A better title{comment}".encode() + ending]
    assert yaml.safe_load(after)["title"] == "A better title"


def test_a_comment_on_a_key_rewritten_over_several_lines_is_kept_on_its_first(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("keywords: []\n", "keywords: [microrobots]  # mine\n"),
                    encoding="utf-8")
    walk(folder, Person("", "", "", "", "", "n", "helices, rods", *([""] * 6)))
    text = path.read_text(encoding="utf-8")
    assert "keywords:  # mine\n- helices\n- rods\n" in text and yaml.safe_load(text)["keywords"] == ["helices", "rods"]


def test_a_comment_is_never_put_inside_a_value_written_over_several_lines():
    before = "format: mime.sharing/1\ntitle: x  # mine\n"
    after = _setup._edited(before, {"title": "one\ntwo"})  # pyright: ignore[reportPrivateUsage]
    assert yaml.safe_load(after)["title"] == "one\ntwo" and "# mine" not in after
    assert after == _edits.written_text(before, {"title": "one\ntwo"})


def test_p2b_write_back_writes_a_commented_line_as_it_did():
    assert _edits.written_text('format: mime.sharing/1\ntitle: "full_toy"  # kept by hand\n', {"title": "A"}) == \
        "format: mime.sharing/1\ntitle: A\n", "the write-back's text is #100's, unchanged"


def test_control_with_the_comment_planted_away_a_rewritten_title_is_caught_losing_it(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_setup, "_trailing_comment", lambda text, lines, at: None)
    folder = copy_of(saved, tmp_path)
    before = with_title_line(folder, 'title: "full_toy"  # kept by hand')
    walk(folder, Person(*RETITLED))
    assert lines_changed(before, (folder / "sharing.yaml").read_bytes()) == [b"title: A better title"], \
        "the planted defect is not seen"


def test_control_with_the_reading_planted_away_a_hash_in_a_value_is_caught_kept_as_a_comment(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_setup, "_reads", lambda text: None)
    folder = copy_of(saved, tmp_path)
    before = with_title_line(folder, 'title: "a # in the title"')
    walk(folder, Person(*RETITLED))
    assert lines_changed(before, (folder / "sharing.yaml").read_bytes()) == [
        b'title: A better title # in the title"'], "the planted defect is not seen"
    after = _setup._edited("format: mime.sharing/1\ntitle: x  # mine\n",  # pyright: ignore[reportPrivateUsage]
                           {"title": "one\ntwo"})
    assert yaml.safe_load(after)["title"] != "one\ntwo", "the planted defect is not seen"


# --- its questions, with standard output and standard error both in files (F6) ----------------------------------------


class Screen(io.StringIO):
    """The controlling terminal (``/dev/tty``) as ``--setup`` writes its questions to it."""

    def close(self) -> None:
        self.closed_by_walk = True


@contextmanager
def both_redirected(monkeypatch: pytest.MonkeyPatch, answers: list[str],
                    screen: Screen | None) -> object:
    """A person at a terminal (standard input), standard output and standard error both files (pytest's capture),
    ``screen`` the controlling terminal (``None``: there is none), each line ``input`` reads one of ``answers``, and
    whatever a prompt ``input`` is given written where ``input`` writes it, standard output."""
    def typed(prompt: object = "") -> str:
        sys.stdout.write(str(prompt))
        if not answers:
            raise EOFError
        return answers.pop(0)

    monkeypatch.setattr(cli, "_a_terminal", lambda: True)
    monkeypatch.setattr(cli, "_controlling_terminal", lambda: screen)
    monkeypatch.setattr(builtins, "input", typed)
    yield


def test_with_both_outputs_in_files_setup_asks_on_the_terminal_and_puts_no_question_in_the_file(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(saved, tmp_path)
    screen = Screen()
    with both_redirected(monkeypatch, list(FIRST) + ["n"], screen):
        assert main(["push", str(folder), "--setup", "--to", "zenodo-sandbox"]) == 0
    written = capsys.readouterr()
    asked = screen.getvalue()
    assert 'use "full_toy"? [Y/n] ' in asked and "creator 1, ORCID iD" in asked and "[y/N] " in asked
    assert "[Y/n]" not in written.out + written.err and "family name" not in written.out + written.err
    assert f"wrote {folder / 'sharing.yaml'}" in written.out and getattr(screen, "closed_by_walk", False)
    assert yaml.safe_load((folder / "sharing.yaml").read_text(encoding="utf-8"))["creators"][0]["orcid"] == ORCID


def test_with_both_outputs_in_files_and_no_terminal_to_show_its_questions_setup_is_refused(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(saved, tmp_path)
    with both_redirected(monkeypatch, list(FIRST), None):
        assert main(["push", str(folder), "--setup"]) == 2
    written = capsys.readouterr()
    assert "push.setup" in written.err and "[Y/n]" not in written.out and not (folder / "sharing.yaml").exists()


def test_control_with_setup_asking_where_every_question_asks_its_questions_are_caught_in_the_file(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    @contextmanager
    def as_before(voice: object) -> object:
        yield voice.typed, voice.ask  # type: ignore[attr-defined]

    monkeypatch.setattr(cli, "_setup_questions", as_before)
    folder = copy_of(saved, tmp_path)
    screen = Screen()
    with both_redirected(monkeypatch, list(FIRST) + ["n"], screen):
        main(["push", str(folder), "--setup", "--to", "zenodo-sandbox"])
    assert "[Y/n]" in capsys.readouterr().out and screen.getvalue() == "", "the planted defect is not seen"


# --- --setup and --defaults take no flag of a push or a publish (the review of #112, I4) -------------------------------


@pytest.mark.parametrize("flag", [["--confirm", "10.5072/zenodo.1"], ["--yes"], ["--keep-site-edits"],
                                  ["--overwrite-site-edits"], ["--full-plan"]])
def test_setup_and_defaults_refuse_a_flag_they_would_ignore_naming_it(flag: list[str], saved: Path, tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]):
    folder = copy_of(saved, tmp_path)
    for words in (["push", str(folder), "--setup", *flag], ["push", "--defaults", *flag]):
        with pytest.raises(SystemExit) as stopped:
            main(words)
        assert stopped.value.code == 2
        error = capsys.readouterr().err
        assert f"takes no {flag[0]}" in error and "pushes and publishes nothing" in error, error
    assert not (folder / "sharing.yaml").exists()


def test_control_with_the_check_planted_away_setup_is_caught_taking_yes_without_a_word(
        saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    real = cli._push_arguments  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(cli, "_push_arguments", lambda arguments: real(
        __import__("argparse").Namespace(**{**vars(arguments), "yes": False})))
    folder = copy_of(saved, tmp_path)
    assert main(["push", str(folder), "--setup", "--yes"]) == 2
    assert "takes no --yes" not in capsys.readouterr().err, "the planted defect is not seen"


# --- the controls --------------------------------------------------------------------------------------------------


def test_control_enter_read_as_no_is_caught_asking_for_new_values(saved: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    real = _setup._Walk.keep  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_setup._Walk, "keep",  # pyright: ignore[reportPrivateUsage]
                        lambda self, head, key, value: False if self.answer(f"{head}\nuse? [Y/n] ") == "" else
                        real(self, head, key, value))
    with pytest.raises(AssertionError, match="asked more than was answered"):
        walk(folder, Person("", "", "", "", ""))


def test_control_an_orcid_unchecked_is_caught_in_the_file_and_in_the_defaults(saved: Path, tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_setup, "orcid_problem", lambda text: None)
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:5], MISTYPED, *FIRST[7:])
    assert walk(folder, person)["creators"][0]["orcid"] == MISTYPED  # type: ignore[index]
    assert any("check digit" in line for line in person.said), "the file's own check is the one left to see it"
    monkeypatch.setattr(_sharing, "orcid_problem", lambda text: None)
    _defaults.write_defaults({"creators": [{"family": "Carberry", "orcid": MISTYPED}]})
    assert MISTYPED in _defaults.defaults_path().read_text(encoding="utf-8"), "the planted defect is not seen"


def test_control_defaults_planted_into_the_folder_are_caught_there(saved: Path, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(saved, tmp_path)
    monkeypatch.setattr(_defaults, "defaults_path", lambda: folder / _defaults.DEFAULTS_FILE)
    walk(folder, Person(*FIRST, save=True))
    assert list(folder.rglob(_defaults.DEFAULTS_FILE)), "the planted defect is not seen"


def test_control_an_edit_that_writes_the_file_anew_is_caught_losing_a_comment(saved: Path, tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(saved, tmp_path)
    walk(folder, Person(*FIRST))
    path = folder / "sharing.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "# a note of my own, kept\n", encoding="utf-8")
    from mime.push import _edits

    monkeypatch.setattr(_edits, "written_text", lambda text, updates: _sharing.scaffold(
        "full_toy", "CC-BY-4.0", {**yaml.safe_load(text), **updates}))
    walk(folder, Person("n", "A better title", *([""] * 11)))
    assert "# a note of my own, kept" not in path.read_text(encoding="utf-8"), "the planted defect is not seen"


def test_control_with_the_orcid_pattern_planted_back_to_any_digit_a_full_width_orcid_is_caught_taken(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_sharing, "_ORCID", re.compile(r"(?:https://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])"))
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:5], FULL_WIDTH, *FIRST[7:], save=True)
    assert walk(folder, person)["creators"][0]["orcid"] == FULL_WIDTH, "the planted defect is not seen"  # type: ignore[index]
    assert FULL_WIDTH in _defaults.defaults_path().read_text(encoding="utf-8"), "the planted defect is not seen"
    assert push.read_sharing(folder).creators[0].orcid == FULL_WIDTH, "the planted defect is not seen"


def test_control_with_the_ror_pattern_planted_back_to_any_digit_a_full_width_ror_is_caught_taken_by_every_reader(
        saved: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_fields, "_ROR", re.compile(r"(?:https://ror\.org/)?(0[a-z0-9]{6}\d{2})"))
    folder = copy_of(saved, tmp_path)
    person = Person(*FIRST[:8], FULL_WIDTH_BROWN, *FIRST[9:15], "National Science Foundation", FULL_WIDTH_NSF, "", "",
                    *FIRST[16:], save=True)
    held = walk(folder, person)
    assert held["creators"][0]["ror"] == FULL_WIDTH_BROWN, "the planted defect is not seen"  # type: ignore[index]
    assert held["funding"][0]["ror"] == FULL_WIDTH_NSF, "the planted defect is not seen"  # type: ignore[index]
    kept = _defaults.defaults_path().read_text(encoding="utf-8")
    assert FULL_WIDTH_BROWN in kept and FULL_WIDTH_NSF in kept, "the planted defect is not seen"
    sharing = push.read_sharing(full_width_rors(saved, tmp_path / "file"))
    assert (sharing.creators[0].ror, sharing.funding[0].ror) == (FULL_WIDTH_BROWN, FULL_WIDTH_NSF), \
        "the planted defect is not seen"
