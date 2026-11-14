"""``mime.toml`` and ``mime push <folder> --to <service> --init``: the record's manifest, beside the saved form and
outside it (design #199 question 1; sharing design v4 section 3).

``--init`` writes it from what the folder knows (its name; its licence, ``CC-BY-4.0`` pre-filled where it declares none,
question 2, or the person's ``[licences] content``) and never overwrites one. It is read against a closed schema: every
key missing, unknown or of another type refused by its path, in its table; an ORCID iD held to its form and its check
digit (ISO 7064 MOD 11-2, which catches every single mistyped digit and every swap of two neighbours, the property below
checks both on drawn iDs). ``sharing.yaml`` is no longer read: alone it is refused with the route, beside ``mime.toml``
it is refused naming the one to delete. ``pack`` neither carries nor refuses either file, and the bundle's bytes are the
same with and without them; the card names the creators and ends with the works to cite as well.

Each plant is paired with its control: the same folder without the plant reads.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from push_cases import CREATOR, copy_of, saved_toy, write_manifest

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
        push.read_manifest(folder)
    return [f"{r.code} {r.message}" for r in held.value.refusals]


#: A manifest that reads, table by table, each key's value as TOML text: what a plant changes one key of.
BASE: Mapping[str, Mapping[str, str]] = {
    "": {"format": '"mime.toml/1"'},
    "bundle": {"licence": '"CC-BY-4.0"'},
    "record": {"title": '"full_toy"', "creators": f"[{CREATOR}]"},
}


def written(folder: Path, table: str = "", key: str = "", value: str | None = None) -> Path:
    """``BASE`` written as the folder's ``mime.toml``, with ``key`` of ``table`` set to ``value`` (``None``: left out)."""
    lines: list[str] = []
    for name, keys in BASE.items():
        held = dict(keys)
        if name == table:
            held.pop(key, None)
            if value is not None:
                held[key] = value
        lines += ([f"[{name}]"] if name else []) + [f"{each} = {text}" for each, text in held.items()]
    path = folder / "mime.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
    path, made, licence = push.init(folder)
    assert made and licence == "CC-BY-4.0" and path == folder / "mime.toml"
    text = path.read_text(encoding="utf-8")
    assert 'title = "full_toy"' in text and '[bundle]\nlicence = "CC-BY-4.0"' in text and "creators = []" in text
    assert refused(folder) == ["push.creators mime.toml names no creator: a record names the people who made it"]


def test_init_prefills_cc_by_4_where_the_experiment_declares_none_and_never_overwrites(unlicensed: Path,
                                                                                      tmp_path: Path):
    folder = copy_of(unlicensed, tmp_path)
    path, made, licence = push.init(folder)
    assert made and licence is None and f'licence = "{push.DEFAULT_LICENCE}"' in path.read_text(encoding="utf-8")
    path.write_text("edited\n", encoding="utf-8")
    assert push.init(folder)[1] is False and path.read_text(encoding="utf-8") == "edited\n"


def test_init_takes_the_persons_content_licence_where_the_experiment_declares_none(unlicensed: Path, tmp_path: Path):
    push.write_defaults(licences={"content": "CC0-1.0"})
    try:
        path, _, _ = push.init(copy_of(unlicensed, tmp_path))
        assert '[bundle]\nlicence = "CC0-1.0"' in path.read_text(encoding="utf-8")
    finally:
        push.defaults_path().unlink()


def test_a_written_file_reads_back_with_its_creator(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder)
    manifest = push.read_manifest(folder)
    assert manifest.kind == "bundle" and manifest.path == folder / "mime.toml"
    assert manifest.record.title == "full_toy" and manifest.record.licence == "CC-BY-4.0"
    assert manifest.record.creators[0].words() == "Carberry, Josiah (ORCID 0000-0002-1825-0097; Brown University)"


def test_the_base_manifest_reads_the_control_of_every_plant_below(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    written(folder)
    assert push.read_manifest(folder).record.title == "full_toy"


def test_no_mime_toml_is_refused_with_init_as_the_line_to_run(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    with pytest.raises(BuildRefused) as held:
        push.read_manifest(folder)
    assert held.value.codes() == ["manifest.missing"] and "--init" in held.value.refusals[0].action


def test_sharing_yaml_alone_is_refused_with_the_route_and_never_read(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yaml").write_text("format: mime.sharing/1\ntitle: x\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as held:
        push.read_manifest(folder)
    (refusal,) = held.value.refusals
    assert refusal.code == "manifest.missing" and "sharing.yaml, which this MIME no longer reads" in refusal.message
    assert "--setup" in refusal.action and "--init" in refusal.action and "never deletes your file" in refusal.action
    assert (folder / "sharing.yaml").is_file()


def test_both_files_are_refused_naming_the_one_to_delete_and_mime_toml_alone_reads(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder)
    (folder / "sharing.yaml").write_text("title: x\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as held:
        push.read_manifest(folder)
    assert held.value.codes() == ["manifest.both"]
    assert f"delete {folder / 'sharing.yaml'}" in held.value.refusals[0].action
    (folder / "sharing.yaml").unlink()
    assert push.read_manifest(folder).record.creators


@pytest.mark.parametrize(("table", "key", "value", "where"), [
    ("record", "unknown", "1", "mime.toml, [record] unknown: is not a key it holds"),
    ("record", "creators", "3", "[record] creators: is int, not a list"),
    ("record", "creators", '[{ family = "A", colour = "red" }]', "[record] creators[0].colour: is not a key it holds"),
    ("record", "creators", '[{ given = "A" }]', "[record] creators[0]: names a person"),
    ("record", "creators", '[{ family = "A", organisation = "B" }]', "[record] creators[0]: names a person"),
    ("record", "creators", '[{ family = "A", ror = "x" }]', "creators[0].ror: x is not a ROR identifier"),
    ("record", "creators", '[{ organisation = "A", orcid = "0000-0002-1825-0097" }]',
     "creators[0].orcid: is an ORCID iD, which a"),
    ("record", "contributors", '[{ family = "A", type = "Boss" }]',
     "[record] contributors[0].type: Boss is not one of DataCite"),
    ("record", "contributors", '[{ family = "A" }]', "[record] contributors[0].type: is missing"),
    ("record", "related", '[{ identifier = "not a doi", relation = "Cites" }]',
     "related[0].identifier: not a doi is neither"),
    ("record", "related", '[{ identifier = "10.1234/x", relation = "Likes" }]', "related[0].relation: Likes is not one of"),
    ("record", "also_cite", '[{ identifier = "not a doi" }]', "[record] also_cite[0].identifier: not a doi is neither"),
    ("record", "also_cite", '[{ identifier = "10.1234/x", page = 3 }]', "[record] also_cite[0].page: is not a key"),
    ("record", "also_cite", '[{ text = "a paper" }]', "[record] also_cite[0].identifier: is missing"),
    ("record", "also_cite", '[{ identifier = "10.1234/x", text = 3 }]', "[record] also_cite[0].text: is int"),
    ("record", "title", "3", "[record] title: is int"),
    ("record", "title", None, "[record] title: is missing"),
    ("record", "keywords", '["a", 3]', "[record] keywords[1].keyword: is int"),
    ("record", "uploader", '{ institution = "X" }', "[record] uploader.name: is missing"),
    ("record", "access", '{ files = "secret" }', "[record] access.files"),
    ("record", "fields", '{ title = "x" }', "[record] fields.title: is written as [record] `title`"),
    ("", "format", '"other.toml/9"', "mime.toml, format: is 'other.toml/9'"),
    ("", "format", None, "mime.toml, format: is None"),
    ("", "colour", '"red"', "mime.toml, colour: is not a key it holds"),
    ("bundle", "licence", '"Creative Commons"', "mime.toml, [bundle] licence:"),
    ("bundle", "licence", None, "mime.toml, [bundle] licence: is missing"),
    ("bundle", "colour", "1", "mime.toml, [bundle] colour: is not a key it holds"),
])
def test_each_key_missing_unknown_or_of_another_type_is_refused_by_its_table_and_path(
        toy: Path, tmp_path: Path, table: str, key: str, value: str | None, where: str):
    folder = copy_of(toy, tmp_path)
    written(folder, table, key, value)
    found = refused(folder)
    assert any(where in each for each in found), found
    assert all(each.startswith(("manifest.key ", "push.creators ")) for each in found), found


def test_a_record_table_that_is_missing_is_refused(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "mime.toml").write_text('format = "mime.toml/1"\n[bundle]\nlicence = "CC-BY-4.0"\n', encoding="utf-8")
    assert any("[record] the table: is missing" in each for each in refused(folder))


@pytest.mark.parametrize("text", ["[record\n", "title = \n", "a = 1\na = 2\n"])
def test_text_that_is_not_toml_is_refused_as_manifest_read(toy: Path, tmp_path: Path, text: str):
    folder = copy_of(toy, tmp_path)
    (folder / "mime.toml").write_text(text, encoding="utf-8")
    assert [each.split()[0] for each in refused(folder)] == ["manifest.read"]


def test_bytes_that_are_not_utf8_are_refused_as_manifest_read(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "mime.toml").write_bytes(b'title = "\xff"\n')
    assert [each.split()[0] for each in refused(folder)] == ["manifest.read"]


@pytest.mark.parametrize(("text", "words"), [
    ('format = "mime.toml/1"\n[record]\ntitle = "x"\n', "names no kind"),
    ('format = "mime.toml/1"\n[bundle]\nlicence = "CC-BY-4.0"\n[study]\nlicence = "CC-BY-4.0"\n',
     "names more than one kind, bundle, study"),
    ('format = "mime.toml/1"\n[study]\nlicence = "CC-BY-4.0"\n', "names [study], and full_toy is a saved folder"),
    ('format = "mime.toml/1"\n[block]\nid = "labx.demo"\n', "holds a block's table ([block])"),
    ('format = "mime.toml/1"\n[bundle]\nlicence = "CC-BY-4.0"\n[requires]\nmime-engine = ">=0.3"\n',
     "holds a block's table ([requires])"),
])
def test_a_kind_table_none_more_than_one_another_folders_or_a_blocks_is_manifest_kind(toy: Path, tmp_path: Path,
                                                                                       text: str, words: str):
    folder = copy_of(toy, tmp_path)
    (folder / "mime.toml").write_text(text, encoding="utf-8")
    ((code, message),) = [each.split(" ", 1) for each in refused(folder)]
    assert code == "manifest.kind" and words in message, message


def test_signals_is_no_table_of_mime_toml(toy: Path, tmp_path: Path):
    """An apparatus keeps its ports in its own declarations, never in ``mime.toml``: ``[signals]`` is an unknown key."""
    folder = copy_of(toy, tmp_path)
    path = write_manifest(folder)
    path.write_text(path.read_text(encoding="utf-8").replace(
        "\n[record]\n", '\n[signals]\nspeed = { units = "m/s" }\n\n[record]\n'), encoding="utf-8")
    assert any("mime.toml, signals: is not a key it holds" in each for each in refused(folder))


def test_a_newer_format_is_refused_as_a_newer_form(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    written(folder, "", "format", '"mime.toml/2"')
    with pytest.raises(BuildRefused) as held:
        push.read_manifest(folder)
    assert held.value.codes() == ["load.newer-form"] and "mime.toml/2" in held.value.refusals[0].message


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
    write_manifest(folder, creators='{ family = "Carberry", orcid = "0000-0002-1825-0098" }')
    found = refused(folder)
    assert found and found[0].startswith("push.creators") and "check digit is 8" in found[0], found


def test_a_bare_toml_day_is_read_as_its_text(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, extra='access = { files = "restricted", embargo = { until = 2030-01-01 } }\n')
    assert push.read_manifest(folder).record.access == {"files": "restricted", "embargo": {"until": "2030-01-01"}}


def test_also_cite_is_stated_as_supplement_to_unless_related_names_it(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, extra='also_cite = [{ identifier = "10.1234/paper", text = "A. Author, 2027" }, '
                                 '{ identifier = "https://example.org/b" }]\n')
    record = push.read_manifest(folder).record
    assert [(each.identifier, each.relation) for each in record.related_with_citations()] == [
        ("10.1234/paper", "IsSupplementTo"), ("https://example.org/b", "IsSupplementTo")]
    named = replace(record, related=(push.Related("10.1234/paper", "Cites"),))
    assert [(each.identifier, each.relation) for each in named.related_with_citations()] == [
        ("10.1234/paper", "Cites"), ("https://example.org/b", "IsSupplementTo")]


@pytest.mark.parametrize("files", [(), ("mime.toml",), ("sharing.yaml",), ("mime.toml", "sharing.yaml")])
def test_the_folder_loads_and_packs_to_the_same_bytes_whichever_sharing_files_it_holds(
        toy: Path, tmp_path: Path, files: tuple[str, ...]):
    """No form changes: the saved folder packs (and so loads, which pack does first) to the bundle it always made."""
    before = pack(toy, out=tmp_path / "before.mimebundle").read_bytes()
    folder = copy_of(toy, tmp_path)
    for name in files:
        (folder / name).write_text("title = 'x'\n", encoding="utf-8")
    after = pack(folder, out=tmp_path / "after.mimebundle")
    assert after.read_bytes() == before
    with zipfile.ZipFile(after) as archive:
        assert not any("sharing" in name or "mime.toml" in name for name in archive.namelist())


def test_control_any_other_extra_file_is_still_refused_as_a_stray(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "mime.yaml").write_text("title: x\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as held:
        pack(folder, out=tmp_path / "x.mimebundle")
    assert held.value.codes() == ["bundle.stray-file"]


def test_the_card_names_the_creators_and_ends_with_the_works_to_cite(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    assert f"- **Creators:** {CREDIT_HOME}" in card_text(folder) and "please also cite" not in card_text(folder)
    path = write_manifest(folder, extra='also_cite = [{ identifier = "10.1234/paper", text = "A. Author, 2027" }]\n')
    path.write_text(path.read_text(encoding="utf-8").replace(
        "contributors = []", 'contributors = [{ organisation = "A Lab", type = "HostingInstitution" }]'),
        encoding="utf-8")
    card = card_text(folder)
    assert "- **Creators:** Carberry, Josiah (ORCID 0000-0002-1825-0097; Brown University)" in card
    assert "- **Contributors:** A Lab (HostingInstitution)" in card and CREDIT_HOME not in card
    assert card.rstrip().endswith("If you use this, please also cite: A. Author, 2027 (10.1234/paper).")


def test_the_card_refuses_sharing_yaml_alone_or_beside_mime_toml(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yaml").write_text("title: x\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as alone:
        card_text(folder)
    write_manifest(folder)
    with pytest.raises(BuildRefused) as both:
        card_text(folder)
    assert alone.value.codes() == ["manifest.missing"] and both.value.codes() == ["manifest.both"]


def test_mime_push_init_on_the_command_line(toy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--init"]) == 0
    out = capsys.readouterr().out
    assert f"wrote {folder / 'mime.toml'}" in out and "licence: CC-BY-4.0, which the experiment declares" in out
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--init"]) == 0
    assert "kept" in capsys.readouterr().out


def test_mime_push_requires_to(toy: Path, capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as stopped:
        main(["push", str(toy), "--init"])
    assert stopped.value.code == 2 and "--to" in capsys.readouterr().err
