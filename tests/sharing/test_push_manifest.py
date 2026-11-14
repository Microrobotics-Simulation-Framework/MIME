"""``sharing.yaml`` and ``mime push <folder> --to <service> --init``: the record's credit, beside the saved form and
outside it (design #199 question 1).

``--init`` writes it from what the folder knows (its name; its licence, ``CC-BY-4.0`` pre-filled where it declares
none, question 2) and never overwrites one. It is read with the safe loader against a closed schema: every key missing,
unknown or of another type refused by its path; an ORCID iD held to its form and its check digit (ISO 7064 MOD 11-2,
which catches every single mistyped digit and every swap of two neighbours, the property below checks both on drawn
iDs). ``pack`` neither carries nor refuses the file; the card names its creators.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from push_cases import copy_of, saved_toy, write_sharing

from mime import push
from mime.build import BuildRefused
from mime.cli import main
from mime.push._sharing import orcid_problem
from mime.saved import card_text, pack
from mime.saved._card import CREDIT_HOME


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("toy"))


@pytest.fixture(scope="module")
def unlicensed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("bare"), "bare_toy", licence=None)


def refused(folder: Path) -> list[str]:
    with pytest.raises(BuildRefused) as held:
        push.read_sharing(folder)
    return [f"{r.code} {r.message}" for r in held.value.refusals]


def check_digit(first: str) -> str:
    """The check digit of fifteen digits, written apart from ``orcid_problem``: ORCID's documented algorithm."""
    total = 0
    for digit in first:
        total = (total + int(digit)) * 2
    result = (12 - total % 11) % 11
    return "X" if result == 10 else str(result)


def grouped(digits: str) -> str:
    return "-".join(digits[i:i + 4] for i in range(0, 16, 4))


def test_init_writes_the_title_and_the_licence_the_folder_declares(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    path, written, licence = push.init(folder)
    assert written and licence == "CC-BY-4.0" and path == folder / "sharing.yaml"
    text = path.read_text(encoding="utf-8")
    assert 'title: "full_toy"' in text and "licence: CC-BY-4.0" in text and "creators: []" in text
    assert refused(folder) == ["push.creators sharing.yaml names no creator: a record names the people who made it"]


def test_init_prefills_cc_by_4_where_the_experiment_declares_none_and_never_overwrites(unlicensed: Path,
                                                                                      tmp_path: Path):
    folder = copy_of(unlicensed, tmp_path)
    path, written, licence = push.init(folder)
    assert written and licence is None and f"licence: {push.DEFAULT_LICENCE}" in path.read_text(encoding="utf-8")
    path.write_text("edited\n", encoding="utf-8")
    assert push.init(folder)[1] is False and path.read_text(encoding="utf-8") == "edited\n"


def test_a_written_file_reads_back_with_its_creator(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_sharing(folder)
    sharing = push.read_sharing(folder)
    assert sharing.title == "full_toy" and sharing.licence == "CC-BY-4.0"
    assert sharing.creators[0].words() == "Carberry, Josiah (ORCID 0000-0002-1825-0097; Brown University)"


def test_no_sharing_yaml_is_refused_with_init_as_the_line_to_run(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    with pytest.raises(BuildRefused) as held:
        push.read_sharing(folder)
    assert held.value.codes() == ["push.sharing-missing"] and "--init" in held.value.refusals[0].action


@pytest.mark.parametrize(("lines", "where"), [
    ("unknown: 1\n", "sharing.yaml, unknown: is not a key it holds"),
    ("creators: 3\n", "sharing.yaml, creators: is int, not a list"),
    ("creators: [{family: A, colour: red}]\n", "sharing.yaml, creators[0].colour: is not a key it holds"),
    ("creators: [{given: A}]\n", "sharing.yaml, creators[0]: names a person"),
    ("creators: [{family: A, organisation: B}]\n", "sharing.yaml, creators[0]: names a person"),
    ("contributors: [{family: A, type: Boss}]\n", "sharing.yaml, contributors[0].type: Boss is not one of DataCite"),
    ("contributors: [{family: A}]\n", "sharing.yaml, contributors[0].type: is missing"),
    ("related: [{identifier: 'not a doi', relation: Cites}]\n", "related[0].identifier: not a doi is neither"),
    ("related: [{identifier: 10.1234/x, relation: Likes}]\n", "related[0].relation: Likes is not one of"),
    ("licence: Creative Commons\n", "sharing.yaml, licence:"),
    ("format: other.sharing/9\n", "sharing.yaml, format: is 'other.sharing/9'"),
    ("creators: [{family: A, ror: x}]\n", "creators[0].ror: x is not a ROR identifier"),
    ("creators: [{organisation: A, orcid: 0000-0002-1825-0097}]\n", "creators[0].orcid: is an ORCID iD, which a"),
])
def test_each_key_missing_unknown_or_of_another_type_is_refused_by_its_path(toy: Path, tmp_path: Path, lines: str,
                                                                            where: str):
    folder = copy_of(toy, tmp_path)
    write_sharing(folder)
    text = (folder / "sharing.yaml").read_text(encoding="utf-8")
    for line in lines.splitlines():
        key = line.split(":")[0]
        text = "\n".join(each for each in text.splitlines() if not each.startswith(f"{key}:")) + "\n"
    (folder / "sharing.yaml").write_text(text + lines, encoding="utf-8")
    found = refused(folder)
    assert any(where in each for each in found), found
    assert all(each.startswith(("push.sharing ", "push.creators ")) for each in found), found


def test_not_yaml_and_not_a_mapping_are_refused(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    for text in ("{unclosed: [\n", "- a list\n"):
        (folder / "sharing.yaml").write_text(text, encoding="utf-8")
        assert [each.split()[0] for each in refused(folder)][:1] == ["push.sharing"]


@given(st.text("0123456789", min_size=15, max_size=15))
def test_an_orcid_with_its_check_digit_passes_in_both_its_forms(first: str):
    iD = grouped(first + check_digit(first))
    assert orcid_problem(iD) is None and orcid_problem(f"https://orcid.org/{iD}") is None


@given(st.text("0123456789", min_size=15, max_size=15), st.integers(0, 15), st.integers(1, 9))
def test_every_single_mistyped_digit_of_an_orcid_is_caught(first: str, at: int, by: int):
    digits = list(first + check_digit(first))
    if digits[at] == "X":
        digits[at] = str(by)
    else:
        digits[at] = str((int(digits[at]) + by) % 10)
    assert orcid_problem(grouped("".join(digits))) is not None


@given(st.text("0123456789", min_size=15, max_size=15), st.integers(0, 13))
def test_every_swap_of_two_neighbouring_digits_of_an_orcid_is_caught(first: str, at: int):
    digits = list(first + check_digit(first))
    if digits[at] == digits[at + 1]:
        return
    digits[at], digits[at + 1] = digits[at + 1], digits[at]
    assert orcid_problem(grouped("".join(digits))) is not None


def test_orcids_documented_example_passes_and_a_mistyped_one_is_refused_as_creators(toy: Path, tmp_path: Path):
    assert orcid_problem("0000-0002-1825-0097") is None
    folder = copy_of(toy, tmp_path)
    write_sharing(folder, creators="{family: Carberry, orcid: 0000-0002-1825-0098}")
    found = refused(folder)
    assert found and found[0].startswith("push.creators") and "check digit is 8" in found[0], found


def test_pack_neither_carries_nor_refuses_sharing_yaml(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    before = pack(folder, out=tmp_path / "before.mimebundle").read_bytes()
    write_sharing(folder)
    after = pack(folder, out=tmp_path / "after.mimebundle")
    assert after.read_bytes() == before
    with zipfile.ZipFile(after) as archive:
        assert not any("sharing" in name for name in archive.namelist())


def test_control_any_other_extra_file_is_still_refused_as_a_stray(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yml").write_text("title: x\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as held:
        pack(folder, out=tmp_path / "x.mimebundle")
    assert held.value.codes() == ["bundle.stray-file"]


def test_the_card_names_the_creators_where_sharing_yaml_holds_them(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    assert f"- **Creators:** {CREDIT_HOME}" in card_text(folder)
    path = write_sharing(folder)
    path.write_text(path.read_text(encoding="utf-8").replace(
        "contributors: []", "contributors: [{organisation: A Lab, type: HostingInstitution}]"), encoding="utf-8")
    card = card_text(folder)
    assert "- **Creators:** Carberry, Josiah (ORCID 0000-0002-1825-0097; Brown University)" in card
    assert "- **Contributors:** A Lab (HostingInstitution)" in card and CREDIT_HOME not in card


def test_mime_push_init_on_the_command_line(toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--init"]) == 0
    out = capsys.readouterr().out
    assert f"wrote {folder / 'sharing.yaml'}" in out and "licence: CC-BY-4.0, which the experiment declares" in out
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--init"]) == 0
    assert "kept" in capsys.readouterr().out


def test_mime_push_requires_to(toy: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as stopped:
        main(["push", str(toy), "--init"])
    assert stopped.value.code == 2 and "--to" in capsys.readouterr().err
