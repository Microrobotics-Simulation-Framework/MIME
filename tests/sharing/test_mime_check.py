"""``mime check <folder>``: a shared folder's ``mime.toml`` checked offline, on the profile workflow a build follows.

Each item has a class and the profile says what each class is (``mime.push._check.RULES``): a defect is a finding in
every profile; what push still needs is a to-do in exploratory and a finding in research; what a reviewer would ask is
listed by research alone and never a finding; ``strict`` is refused by name. The profile is the experiment's own unless
one is given. A finding exits 1, to-dos and listed items alone exit 0, a folder that cannot be checked exits 2. Each item
comes with the line to write.

Controls, each caught on every run: with research's rule for what push still needs planted to a to-do, the research
check of a folder naming no creator is caught exiting 0; with the asked class's rule planted into exploratory, its items
are caught listed there (and a row added to the table is all it takes, no other edit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy, with_token, write_manifest

from mime import push
from mime.build import BuildRefused
from mime.cli import main
from mime.push import _check
from mime.validity import Profile

ORCID = "0000-0002-1825-0097"
COMPLETE = (f'{{ family = "Carberry", given = "Josiah", orcid = "{ORCID}" }}',
            'description = "A toy pushed by its tests."\nkeywords = ["helices"]\n')


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("toy"))


def verdicts(checked: _check.Checked) -> dict[str, list[str]]:
    return {verdict: [item.what for item in checked.of(verdict)] for verdict in (_check.FINDING, _check.TO_DO,
                                                                                    _check.LISTED)}


def test_a_new_manifest_is_a_to_do_in_exploratory_and_a_finding_in_research(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    explored = push.check(folder)
    assert explored.profile is Profile.EXPLORATORY and not explored.given, "the experiment's own profile"
    assert verdicts(explored) == {_check.FINDING: [], _check.LISTED: [],
                                  _check.TO_DO: ["mime.toml names no creator: a record names the people who made it"]}
    researched = push.check(folder, "research")
    assert researched.given and verdicts(researched)[_check.FINDING] == verdicts(explored)[_check.TO_DO]
    assert verdicts(researched)[_check.LISTED] == [
        "mime.toml, [record] description: is empty, so the record opens with the card alone",
        "mime.toml, [record] keywords: names none, so the record is found by mime-experiment alone"]


def test_a_person_with_no_orcid_is_asked_about_in_research_and_never_a_finding(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, creators='{ family = "Carberry", given = "Josiah" }', extra=COMPLETE[1])
    assert verdicts(push.check(folder, "research")) == {_check.FINDING: [], _check.TO_DO: [], _check.LISTED: [
        "mime.toml, [record] creators[0] (Carberry): names a person with no ORCID iD, by which an index tells two "
        "people of one name apart"]}
    assert push.check(folder).items == (), "exploratory asks nothing of it"


def test_a_complete_manifest_checks_clean_in_every_built_profile_and_says_so(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, creators=COMPLETE[0], extra=COMPLETE[1])
    for profile in ("exploratory", "research"):
        checked = push.check(folder, profile)
        assert checked.items == () and checked.text().endswith(
            "nothing to do: push would take the manifest as it stands (the folder's own files are `mime pack`'s to "
            "check).\n")


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_defect_is_a_finding_in_every_profile_with_its_line_to_write(toy: Path, tmp_path: Path, profile: str):
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, creators='{ family = "Carberry", orcid = "0000-0002-1825-0098" }', extra=COMPLETE[1])
    (finding,) = push.check(folder, profile).findings
    assert "check digit is 8" in finding.what and finding.line == 'orcid = "0000-0000-0000-0000"'
    assert finding.place == "in mime.toml, under [record]"


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_licence_unlike_the_experiments_is_a_finding(toy: Path, tmp_path: Path, profile: str):
    folder = copy_of(toy, tmp_path)
    path = write_manifest(folder, creators=COMPLETE[0], extra=COMPLETE[1])
    path.write_text(path.read_text(encoding="utf-8").replace('licence = "CC-BY-4.0"', 'licence = "MIT"'),
                    encoding="utf-8")
    (finding,) = push.check(folder, profile).findings
    assert finding.what == ("mime.toml, [bundle] licence: is MIT, and the experiment declares CC-BY-4.0, which its "
                            "files carry")
    assert (finding.line, finding.place) == ('licence = "CC-BY-4.0"', "in mime.toml, under [bundle]")


def test_an_experiment_with_no_licence_is_a_to_do_then_a_finding_naming_the_line_where_it_is_built(tmp_path: Path):
    folder = saved_toy(tmp_path, "bare_toy", licence=None)
    write_manifest(folder, creators=COMPLETE[0], extra=COMPLETE[1])
    (todo,) = push.check(folder).of(_check.TO_DO)
    assert todo.what == "the experiment declares no licence, and a record is never published without one"
    assert (todo.line, todo.place) == ("exp.licence('CC-BY-4.0')", "where the experiment is built, then save it again")
    assert [item.what for item in push.check(folder, "research").findings] == [todo.what]


def test_the_default_profile_is_the_experiments_own(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    real = _check._document  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_check, "_document", lambda root, kind: {**real(root, kind), "profile": "research"})
    checked = push.check(folder)
    assert checked.profile is Profile.RESEARCH and not checked.given and checked.findings
    assert "checked under the research profile (the experiment's)" in checked.text()


def test_strict_is_refused_by_name_and_a_name_that_is_no_profile_too(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    for given, code in (("strict", "profile.reserved"), ("careful", "profile.unknown")):
        with pytest.raises(BuildRefused) as refused:
            push.check(folder, given)
        assert refused.value.codes() == [code]


@pytest.mark.parametrize(("setup", "code"), [
    (lambda folder: None, "manifest.missing"),
    (lambda folder: (folder / "sharing.yaml").write_text("title: x\n", encoding="utf-8"), "manifest.missing"),
    (lambda folder: (push.init(folder), (folder / "sharing.yaml").write_text("x: 1\n", encoding="utf-8")),
     "manifest.both"),
    (lambda folder: (folder / "mime.toml").write_text("[record\n", encoding="utf-8"), "manifest.read"),
    (lambda folder: (folder / "mime.toml").write_text('[study]\nlicence = "MIT"\n', encoding="utf-8"),
     "manifest.kind"),
    (lambda folder: (folder / "mime.toml").write_text('format = "mime.toml/9"\n', encoding="utf-8"),
     "load.newer-form"),
])
def test_a_folder_that_cannot_be_checked_is_refused_and_exits_two(toy: Path, tmp_path: Path, setup, code: str,
                                                                    capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    setup(folder)
    with pytest.raises(BuildRefused) as refused:
        push.check(folder)
    assert refused.value.codes() == [code]
    assert main(["check", str(folder)]) == 2 and code in capsys.readouterr().err


def test_on_the_command_line_a_finding_exits_one_and_the_text_is_the_checks(toy: Path, tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    assert main(["check", str(folder)]) == 0
    assert capsys.readouterr().out == push.check(folder).text()
    assert main(["check", str(folder), "--profile", "research"]) == 1
    out = capsys.readouterr().out
    assert out == push.check(folder, "research").text()
    assert out.splitlines()[1:5] == [
        "findings: what must change, which push refuses as it stands",
        "1. mime.toml names no creator: a record names the people who made it",
        "   in mime.toml, under [record]:",
        '       creators = [{ family = "<family name>", given = "<given name>", orcid = "0000-0000-0000-0000" }]']


def test_a_study_folder_is_checked_by_its_base_experiments_licence_and_profile(toy: Path, tmp_path: Path):
    folder = tmp_path / "wall_scan"
    copy_of(toy, folder).rename(folder / "experiment")
    (folder / "study.json").write_text(json.dumps({"format": "mime-study-outputs"}), encoding="utf-8")
    push.init(folder)
    checked = push.check(folder)
    assert checked.kind == "study" and checked.profile is Profile.EXPLORATORY
    assert [item.what for item in checked.of(_check.TO_DO)] == [
        "mime.toml names no creator: a record names the people who made it"]
    path = folder / "mime.toml"
    path.write_text(path.read_text(encoding="utf-8").replace('licence = "CC-BY-4.0"', 'licence = "MIT"'),
                    encoding="utf-8")
    assert [item.what for item in push.check(folder).findings] == [
        "mime.toml, [study] licence: is MIT, and the study's base experiment declares CC-BY-4.0, which its files carry"]


def test_a_rule_is_a_row_of_the_table_and_nothing_else(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The control of the table: a row added for the asked class in exploratory lists its items there, no other edit."""
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    assert push.check(folder).of(_check.LISTED) == []
    monkeypatch.setattr(_check, "RULES", {**_check.RULES, (Profile.EXPLORATORY, _check.ASKED): _check.LISTED})
    assert len(push.check(folder).of(_check.LISTED)) == 2, "the planted row is not seen"


def test_control_what_push_needs_planted_a_to_do_in_research_is_caught_exiting_zero(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(toy, tmp_path)
    push.init(folder)
    monkeypatch.setattr(_check, "RULES", {**_check.RULES, (Profile.RESEARCH, _check.NEEDED): _check.TO_DO})
    assert main(["check", str(folder), "--profile", "research"]) == 0, "the planted rule is not seen"


# --- what push refuses on the folder and [record]: each has its counterpart in the check ---------------------------


def permitted(tmp_path: Path) -> Path:
    """A saved folder whose experiment declares a permission granted to ``Carberry, Josiah``, with ``--init``'s manifest
    naming a creator and no uploader."""
    from terms_cases import layered, same_reference_run, sha256_of, write_mesh

    from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

    shell = write_mesh(tmp_path / "made", "shell", 900)
    own = CustomLicence("LicenseRef-Maker-1.0", "Maker Licence", "https://example.org/maker")
    permission = Permission("Maker Ltd", "Carberry, Josiah", (sha256_of(shell),), Scope(True, True, True),
                            "2026-11-01", Evidence("public_statement", "https://example.org/permission"))
    with same_reference_run():
        layered("granted", [(shell, Terms(own, "Maker Ltd", "presentation"))], licence="CC-BY-4.0",
                permissions=[permission]).build().save(tmp_path / "granted")
    write_manifest(tmp_path / "granted", creators=COMPLETE[0], extra=COMPLETE[1])
    return tmp_path / "granted"


def uploader_line(folder: Path, name: str) -> None:
    path = folder / "mime.toml"
    path.write_text(path.read_text(encoding="utf-8") + f'uploader = {{ name = "{name}" }}\n', encoding="utf-8")


def test_a_declared_permission_with_no_uploader_named_is_what_push_still_needs(tmp_path: Path,
                                                                                capsys: pytest.CaptureFixture[str]):
    """Push refuses it (``push.permission``), so the check must not say push would take the folder as it stands."""
    folder = permitted(tmp_path)
    explored = push.check(folder)
    (todo,) = explored.of(_check.TO_DO)
    assert "declares 1 permission" in todo.what and "names no uploader" in todo.what
    assert todo.line is not None and todo.line.startswith("uploader = {")
    assert explored.findings == [] and main(["check", str(folder)]) == 0
    capsys.readouterr()
    assert [item.what for item in push.check(folder, "research").findings] == [todo.what]
    assert main(["check", str(folder), "--profile", "research"]) == 1
    assert "push.permission" not in capsys.readouterr().out, "the text is the item's, not a code"


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_permission_granted_to_another_than_the_uploader_is_a_finding_in_every_profile(tmp_path: Path, profile: str):
    folder = permitted(tmp_path)
    uploader_line(folder, "Somebody, Else")
    (finding,) = push.check(folder, profile).findings
    assert "names the grantee Carberry, Josiah, who is neither the uploader (Somebody, Else)" in finding.what


def test_a_permission_granted_to_the_uploader_checks_clean(tmp_path: Path):
    folder = permitted(tmp_path)
    uploader_line(folder, "Carberry, Josiah")
    assert push.check(folder, "research").items == ()


@pytest.mark.parametrize("secret", ["HOME", "TOKEN"])
def test_a_path_of_this_machine_or_a_token_in_the_record_is_a_finding_in_every_profile(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secret: str):
    """Push's metadata scan refuses both (``bundle.machine-path``, ``bundle.credential``)."""
    text = with_token(monkeypatch) if secret == "TOKEN" else str(Path.home() / "data" / "run.npz")
    folder = copy_of(toy, tmp_path)
    write_manifest(folder, creators=COMPLETE[0], extra=f'description = "see {text}"\nkeywords = ["helices"]\n')
    for profile in ("exploratory", "research"):
        (finding,) = push.check(folder, profile).findings
        assert finding.what.startswith("the record's metadata holds ")
        assert text not in finding.what, "the path or token is never quoted"
