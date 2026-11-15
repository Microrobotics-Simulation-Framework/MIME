"""What ``mime scaffold`` guarantees, each with a control that fails on today's code (without scaffold there is nothing to
refuse) and a plant that shows the check can fail:

- **never overwrites** (``scaffold.exists``): an existing path, or a folder holding ``mime.toml`` or ``sharing.yaml``, is
  refused naming ``mime push <folder> --setup`` where a manifest is held, and every byte stays; planted with the guard
  away and the writer replacing, the same check sees a file changed;
- **``--yes`` asks nothing and accepts no licence**: no question reaches the person, no acceptance is written; planted with the
  walk forced on, a question is caught; planted with a scaffold that records an acceptance, the file is caught;
- **no terminal and no ``--yes``** (``scaffold.no-terminal``) names ``--yes``, writes nothing, and a terminal is not refused;
- **an unknown ``--set``** (``scaffold.set``) names every known key of the kind, and nothing is written; so do a value its
  check refuses, a key given twice, and ``--from`` or ``--socket`` on a kind that takes none;
- **a token in a value is never written** (``scaffold.set``, naming the key); planted with the token search emptied, the token
  is caught in the file;
- **``--dry-run`` writes nothing**, draws the tree and where each value came from; planted as a real run, the writing is caught;
- **the alias**: ``mime push --init`` and ``mime scaffold bundle|study --yes`` write the same bytes, and differ in what an
  existing ``mime.toml`` meets (kept softly, refused) and in a ``sharing.yaml`` (migrated, refused); planted with one
  writer's text changed, the bytes are caught apart;
- **a plugin's namespace is never one that is taken** (``scaffold.set``): each first word of MIME's refusal codes and entries,
  read from the registry, is refused at ``--set`` and where the folder's name or the defaults give it, and the core's own
  ``register_namespace`` refuses the same words (the two are held together); a walk asks again;
- **a kind that is not the folder's** is ``manifest.kind``, naming both; the ``scaffold.exists`` action of a bundle or a study
  names ``--setup``, never a new name;
- **a link is a path that exists**: a dangling symlink where a kind is scaffolded, or as the ``mime.toml`` of a saved folder, is
  ``scaffold.exists`` and nothing is written; planted with ``is_symlink`` answering no, the same check sees the error."""

import tomllib
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy
from scaffold_rig import KINDS, NAMESPACE, Person, made, tree_of
from study_rigs import write_study

from mime import credentials, push, scaffold
from mime.assets._acceptances import acceptances_path
from mime.build import REFUSAL_TABLE, BuildRefused
from mime.cli import main as mime_main
from mime.push import _defaults, _draft
from mime.scaffold import _scaffold, _values
from mime.vocabulary import register_namespace


def quiet(line: str) -> None:
    return None


def refusal_of(call) -> BuildRefused:
    with pytest.raises(BuildRefused) as refused:
        call()
    return refused.value


def codes(refused: BuildRefused) -> list[str]:
    return [each.code for each in refused.refusals]


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("saved"))


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_study(tmp_path_factory.mktemp("study") / "toy_study", figure=False)


def target(kind: str, tmp_path: Path, saved: Path, study: Path) -> Path:
    """A path to scaffold ``kind`` at: a copy of the saved or study folder for the two kinds that add ``mime.toml``."""
    if kind == "bundle":
        return copy_of(saved, tmp_path)
    if kind == "study":
        return copy_of(study, tmp_path)
    return made(kind, tmp_path)


# -- never overwrites ----------------------------------------------------------------------------------------------------


def overwritten(kind: str, path: Path) -> list[str]:
    """What a second scaffold at ``path`` changed: empty when it was refused (``scaffold.exists``) and nothing moved."""
    before = tree_of(path)
    try:
        scaffold.scaffold(kind, path, yes=True, say=quiet)
    except BuildRefused as refused:
        if codes(refused) != ["scaffold.exists"]:
            return [f"refused as {codes(refused)}, not scaffold.exists"]
    after = tree_of(path)
    return [name for name in sorted({*before, *after}) if before.get(name) != after.get(name)]


@pytest.mark.parametrize("kind", KINDS)
def test_a_second_scaffold_is_refused_and_changes_nothing(kind: str, tmp_path: Path, saved: Path, study: Path):
    path = target(kind, tmp_path, saved, study)
    scaffold.scaffold(kind, path, yes=True, say=quiet)
    assert overwritten(kind, path) == []
    refused = refusal_of(lambda: scaffold.scaffold(kind, path, yes=True, say=quiet))
    assert codes(refused) == ["scaffold.exists"]
    assert "scaffold never overwrites" in refused.refusals[0].message
    if kind in ("bundle", "study", "block"):
        assert f"mime push {path} --setup" in refused.refusals[0].action
    assert ("another name" in refused.refusals[0].action) == (kind not in ("bundle", "study")), \
        "a bundle or a study adds to an existing folder: scaffolding into another name is no route for it"


@pytest.mark.parametrize("kind", ["experiment", "block", "plugin"])
def test_a_path_that_exists_is_refused_though_it_holds_nothing_of_ours(kind: str, tmp_path: Path):
    path = tmp_path / "taken"
    path.mkdir()
    (path / "mine.txt").write_text("mine", encoding="utf-8")
    refused = refusal_of(lambda: scaffold.scaffold(kind, path, yes=True, say=quiet))
    assert codes(refused) == ["scaffold.exists"] and "--setup" not in refused.refusals[0].action
    assert tree_of(path) == {"mine.txt": b"mine"}


def test_a_folder_holding_sharing_yaml_is_refused_and_push_init_still_migrates_it(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    (folder / "sharing.yaml").write_text("format: mime.sharing/1\n", encoding="utf-8")
    refused = refusal_of(lambda: scaffold.scaffold("bundle", folder, yes=True, say=quiet))
    assert codes(refused) == ["scaffold.exists"] and "sharing.yaml" in refused.refusals[0].message
    assert not (folder / "mime.toml").exists()


def test_the_overwrite_check_fails_when_the_guard_is_planted_away(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "experiment_a"
    scaffold.scaffold("experiment", path, yes=True, say=quiet)
    monkeypatch.setattr(_scaffold, "_exists", lambda *a: None)

    def replacing(kind, root, plan):
        for name, content in plan.files.items():
            (root / name).write_text(content + "\n# replaced\n", encoding="utf-8") if isinstance(content, str) else None

    monkeypatch.setattr(_scaffold, "_written", replacing)
    assert overwritten("experiment", path) == ["experiment.py"]


# -- --yes asks nothing and accepts no licence -------------------------------------------------------------------------


def asked_under_yes(kind: str, path: Path) -> list[str]:
    person = Person()
    scaffold.scaffold(kind, path, yes=True, typed=person.typed, ask=person.ask, say=person.say)
    return person.asked


@pytest.mark.parametrize("kind", KINDS)
def test_yes_asks_nothing_and_accepts_no_licence(kind: str, tmp_path: Path, saved: Path, study: Path):
    assert asked_under_yes(kind, target(kind, tmp_path, saved, study)) == []
    assert not acceptances_path().exists(), "scaffold accepted a licence"


def test_the_yes_check_fails_for_a_scaffold_that_asks_or_accepts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_scaffold, "_walks", lambda yes, dry_run: True)
    person = Person("")
    with pytest.raises(AssertionError):
        scaffold.scaffold("block", tmp_path / "asking", yes=True, typed=person.typed, ask=person.ask, say=quiet)
    assert person.asked, "the planted walk asked a question, and the check sees it"
    acceptances_path().parent.mkdir(parents=True, exist_ok=True)
    acceptances_path().write_text("{}", encoding="utf-8")
    assert acceptances_path().exists(), "the check on a planted acceptance sees the file"


# -- no terminal ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_no_terminal_and_no_yes_is_refused_naming_yes_and_writes_nothing(kind: str, tmp_path: Path, saved: Path, study: Path):
    path = target(kind, tmp_path, saved, study)
    before = tree_of(tmp_path)
    refused = refusal_of(lambda: scaffold.scaffold(kind, path, say=quiet))
    assert codes(refused) == ["scaffold.no-terminal"] and "--yes" in refused.refusals[0].action
    assert tree_of(tmp_path) == before


def test_a_terminal_is_not_refused(tmp_path: Path):
    person = Person(*([""] * 20))
    scaffold.scaffold("block", tmp_path / "walked", typed=person.typed, ask=person.ask, say=person.say)
    assert (tmp_path / "walked" / "mime.toml").is_file()


def test_the_command_line_refuses_with_exit_two_when_nothing_answers(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert mime_main(["scaffold", "block", str(tmp_path / "b")]) == 2
    err = capsys.readouterr().err
    assert "scaffold.no-terminal" in err and "--yes" in err and not (tmp_path / "b").exists()


# -- --set ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_an_unknown_set_key_names_the_known_keys_and_writes_nothing(kind: str, tmp_path: Path, saved: Path, study: Path):
    path = target(kind, tmp_path, saved, study)
    before = tree_of(tmp_path)
    refused = refusal_of(lambda: scaffold.scaffold(kind, path, yes=True, sets=["record.titel=x"], say=quiet))
    assert codes(refused) == ["scaffold.set"]
    for key in scaffold.KEYS[kind]:
        assert key in refused.refusals[0].action, key
    assert tree_of(tmp_path) == before


@pytest.mark.parametrize(("sets", "words"), [
    (["block.id"], "no `=`"),
    (["block.id=a.b", "block.id=c.d"], "twice"),
    (["block.id=NOT_AN_ID"], "namespace"),
    (["block.version=one"], "PEP 440"),
    (["block.licence=Not-A-Licence"], "licence"),
    (["record.creators=Carberry, Josiah, 0000-0002-1825-0098"], "ORCID"),
])
def test_a_set_that_cannot_be_taken_is_refused_naming_the_key(sets: list[str], words: str, tmp_path: Path):
    refused = refusal_of(lambda: scaffold.scaffold("block", tmp_path / "b", yes=True, sets=sets, say=quiet))
    assert codes(refused) == ["scaffold.set"] and words in refused.refusals[0].message
    assert not (tmp_path / "b").exists()


def test_set_is_taken_over_the_defaults_and_scaffolds_own(tmp_path: Path):
    _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah"}], "keywords": ["microrobot"]},
                             licences={"content": "CC0-1.0"}, namespace="labx")
    held = scaffold.scaffold("block", tmp_path / "probe", yes=True, sets=["block.version=2.0.0",
                                                                          "record.keywords=a, b"], say=quiet).values
    assert {key: (value.value, value.source) for key, value in held.items() if key != "record.title"} == {
        "block.id": ("labx.probe", "defaults.toml"), "block.version": ("2.0.0", "--set"),
        "block.licence": ("CC0-1.0", "defaults.toml"), "record.description": ("", "scaffold's own"),
        "record.creators": ([{"family": "Carberry", "given": "Josiah"}], "defaults.toml"),
        "record.keywords": (["a", "b"], "--set")}
    record = tomllib.loads((tmp_path / "probe" / "mime.toml").read_text(encoding="utf-8"))["record"]
    assert record["keywords"] == ["a", "b"] and record["creators"][0]["family"] == "Carberry"


@pytest.mark.parametrize(("kind", "flags", "words"), [
    ("experiment", {"source": "somewhere"}, "--from"),
    ("plugin", {"sockets": ["medium=bath"]}, "--socket"),
    ("block", {"sockets": ["medium=bath"]}, "no --from"),
])
def test_from_and_socket_on_a_kind_that_takes_none_are_refused(kind: str, flags: dict, words: str, tmp_path: Path):
    refused = refusal_of(lambda: scaffold.scaffold(kind, tmp_path / "x", yes=True, say=quiet, **flags))
    assert codes(refused) == ["scaffold.set"] and words in refused.refusals[0].message
    assert not (tmp_path / "x").exists()


# -- tokens --------------------------------------------------------------------------------------------------------------

TOKEN = "t0ken_" + "Abc123XyZ" * 4


def token_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv(next(iter(credentials.SERVICES.values())), TOKEN)
    path = tmp_path / "tokened"
    try:
        scaffold.scaffold("experiment", path, yes=True, sets=[f"experiment.name=run_{TOKEN}"], say=quiet)
    except BuildRefused as refused:
        assert codes(refused) == ["scaffold.set"] and "experiment.name" in refused.refusals[0].message
        assert TOKEN not in refused.refusals[0].message
    return [str(each) for each in path.rglob("*") if each.is_file() and TOKEN.encode() in each.read_bytes()]


def test_a_token_in_a_value_is_never_written_and_the_refusal_names_the_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert token_written(tmp_path, monkeypatch) == []
    assert not (tmp_path / "tokened").exists()


def test_the_token_check_fails_when_the_search_is_planted_away(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_scaffold, "held_tokens", lambda: [])
    assert token_written(tmp_path, monkeypatch) != []


# -- --dry-run -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_dry_run_writes_nothing_draws_the_tree_and_says_where_each_value_came_from(kind: str, tmp_path: Path, saved: Path,
                                                                                    study: Path):
    path = target(kind, tmp_path, saved, study)
    before = tree_of(tmp_path)
    said: list[str] = []
    result = scaffold.scaffold(kind, path, yes=True, dry_run=True, sets=[], say=said.append)
    assert tree_of(tmp_path) == before and not result.written
    assert said[0].startswith("Would scaffold") and "(nothing written)" in said[0]
    assert any(line.startswith("  from scaffold's own:") for line in said)
    assert all(name.rstrip("/").rpartition("/")[2] in "\n".join(said) for name, _ in result.files), "every file is in the tree"


def test_dry_run_asks_nothing_though_a_person_is_there(tmp_path: Path):
    person = Person()
    scaffold.scaffold("block", tmp_path / "b", dry_run=True, typed=person.typed, ask=person.ask, say=person.say)
    assert person.asked == [] and not (tmp_path / "b").exists()


def test_the_dry_run_check_fails_for_a_run_that_writes(tmp_path: Path):
    scaffold.scaffold("block", tmp_path / "b", yes=True, say=quiet)
    assert tree_of(tmp_path), "the snapshot the dry-run checks compare sees a written block"


# -- the alias: mime push --init -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bundle", "study"])
@pytest.mark.parametrize("defaulted", [False, True])
def test_push_init_and_scaffold_yes_write_the_same_bytes(kind: str, defaulted: bool, tmp_path: Path, saved: Path, study: Path):
    if defaulted:
        _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah", "orcid": "0000-0002-1825-0097"}],
                                  "keywords": ["microrobot"]}, licences={"content": "CC0-1.0"})
    source = saved if kind == "bundle" else study
    by_init, by_scaffold = copy_of(source, tmp_path / "init"), copy_of(source, tmp_path / "scaffold")
    push.init(by_init)
    scaffold.scaffold(kind, by_scaffold, yes=True, say=quiet)
    assert (by_init / "mime.toml").read_bytes() == (by_scaffold / "mime.toml").read_bytes()


def test_the_alias_check_fails_when_one_writer_is_planted_apart(tmp_path: Path, saved: Path, monkeypatch: pytest.MonkeyPatch):
    by_init, by_scaffold = copy_of(saved, tmp_path / "init"), copy_of(saved, tmp_path / "scaffold")
    real = _draft.new_manifest
    monkeypatch.setattr(_draft, "new_manifest", lambda *a, **k: real(*a, **k) + "# one writer's own line\n")
    push.init(by_init)
    monkeypatch.setattr(_draft, "new_manifest", real)
    scaffold.scaffold("bundle", by_scaffold, yes=True, say=quiet)
    assert (by_init / "mime.toml").read_bytes() != (by_scaffold / "mime.toml").read_bytes()


def test_an_existing_mime_toml_is_kept_softly_by_init_and_refused_by_scaffold(saved: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = copy_of(saved, tmp_path)
    push.init(folder)
    held = (folder / "mime.toml").read_bytes()
    assert mime_main(["push", str(folder), "--to", "zenodo-sandbox", "--init"]) == 0
    out = capsys.readouterr().out
    assert "kept" in out and "which exists" in out
    assert mime_main(["scaffold", "bundle", str(folder), "--yes"]) == 2
    assert "scaffold.exists" in capsys.readouterr().err
    assert (folder / "mime.toml").read_bytes() == held


# -- a plugin's namespace -----------------------------------------------------------------------------------------------


def mime_words() -> list[str]:
    """The words MIME's own refusal codes and entries begin with, as the core's ``register_namespace`` reads them."""
    return sorted(word for word, held in _values.taken_namespaces().items() if held.startswith("the first word of MIME's"))


def test_each_first_word_of_mimes_codes_and_entries_is_refused_as_a_namespace_and_the_core_refuses_the_same():
    words = mime_words()
    assert {"helix", "camera", "tube", "sphere"} <= set(words), "the words come from the registry and the table"
    assert {code.split(".", 1)[0] for code in REFUSAL_TABLE} <= set(words)
    for word in words:
        assert _values.namespace_check(word) is not None, word
        with pytest.raises(ValueError):
            register_namespace(word, "0.0.1")
    assert _values.namespace_check("a_free_lab_word") is None


def test_a_namespace_an_installed_distribution_declares_is_refused_read_from_the_environment(tmp_path: Path):
    """``labx`` is MIME's lab fixture's, installed in the environment the suite runs in (``lab_fixture/``)."""
    held = _values.taken_namespaces()
    assert "labx" in held and "mime-lab-fixture" in held["labx"]
    refused = refusal_of(lambda: scaffold.scaffold("plugin", tmp_path / "p", yes=True, sets=["plugin.namespace=labx"], say=quiet))
    assert codes(refused) == ["scaffold.set"] and "mime-lab-fixture" in refused.refusals[0].message
    assert not (tmp_path / "p").exists()


def test_set_refuses_a_taken_namespace_naming_it_and_a_free_one(tmp_path: Path):
    for word in ("helix", "camera"):
        refused = refusal_of(lambda word=word: scaffold.scaffold("plugin", tmp_path / "p", yes=True,
                                                                  sets=[f"plugin.namespace={word}"], say=quiet))
        refusal = refused.refusals[0]
        assert codes(refused) == ["scaffold.set"] and word in refusal.message and f"lab_{word}" in refusal.message
        assert not (tmp_path / "p").exists()
    scaffold.scaffold("plugin", tmp_path / "p", yes=True, sets=["plugin.namespace=lab_helix"], say=quiet)
    assert (tmp_path / "p" / "pyproject.toml").is_file()


def test_a_namespace_derived_from_the_folders_name_is_refused_too(tmp_path: Path):
    refused = refusal_of(lambda: scaffold.scaffold("plugin", tmp_path / "helix-robot", yes=True, say=quiet))
    refusal = refused.refusals[0]
    assert codes(refused) == ["scaffold.set"] and "helix" in refusal.message and "folder's name" in refusal.message
    assert "--set plugin.namespace=" in refusal.action and not (tmp_path / "helix-robot").exists()
    dry = refusal_of(lambda: scaffold.scaffold("plugin", tmp_path / "helix-robot", yes=True, dry_run=True, say=quiet))
    assert codes(dry) == ["scaffold.set"], "a dry run shows what --yes would do, which is the refusal"


def test_a_namespace_the_defaults_hold_is_refused_naming_the_defaults(tmp_path: Path):
    _defaults.write_defaults({}, namespace="helix")
    refused = refusal_of(lambda: scaffold.scaffold("plugin", tmp_path / "robot", yes=True, say=quiet))
    assert codes(refused) == ["scaffold.set"] and "defaults" in refused.refusals[0].message
    assert "mime defaults" in refused.refusals[0].action
    scaffold.scaffold("plugin", tmp_path / "robot", yes=True, sets=["plugin.namespace=lab_robot"], say=quiet)


def test_a_walk_does_not_offer_a_taken_namespace_and_asks_until_a_free_one_is_typed(tmp_path: Path):
    """The distribution kept, then at the namespace: Enter (refused, asked again), a taken word (said, asked again), a free one."""
    person = Person("", "", "helix", "lab_robot", "", "", "", "", "")
    scaffold.scaffold("plugin", tmp_path / "helix-robot", typed=person.typed, ask=person.ask, say=person.say)
    assert not any("use helix?" in each or 'use "helix"?' in each for each in person.asked), "never offered to keep"
    assert any("cannot be used" in each for each in person.asked), "the reason is said at the question"
    assert sum("first word of MIME's" in each for each in person.asked) >= 3, "Enter and the typed word are each told why"
    project = tomllib.loads((tmp_path / "helix-robot" / "pyproject.toml").read_text(encoding="utf-8"))
    assert list(project["project"]["entry-points"]["mime.entries"]) == ["lab_robot"]


# -- a kind that is not the folder's ---------------------------------------------------------------------------------------


def test_a_kind_that_is_not_the_folders_is_manifest_kind_naming_both(tmp_path: Path, saved: Path, study: Path):
    for asked, folder, found in (("study", copy_of(saved, tmp_path / "a"), "bundle"),
                                 ("bundle", copy_of(study, tmp_path / "b"), "study")):
        before = tree_of(folder)
        refused = refusal_of(lambda asked=asked, folder=folder: scaffold.scaffold(asked, folder, yes=True, say=quiet))
        refusal = refused.refusals[0]
        assert codes(refused) == ["manifest.kind"]
        assert f"a {asked} was asked for" in refusal.message and f"is a {found} folder" in refusal.message
        assert f"mime scaffold {found} " in refusal.action
        assert tree_of(folder) == before


def test_an_existing_mime_toml_of_another_kind_is_scaffold_exists_with_the_route_to_edit_it(tmp_path: Path, saved: Path):
    folder = copy_of(saved, tmp_path)
    scaffold.scaffold("bundle", folder, yes=True, say=quiet)
    refused = refusal_of(lambda: scaffold.scaffold("study", folder, yes=True, say=quiet))
    assert codes(refused) == ["scaffold.exists"] and f"mime push {folder} --setup" in refused.refusals[0].action


# -- a link is a path that exists ----------------------------------------------------------------------------------------


def linked(kind: str, tmp_path: Path, saved: Path, study: Path) -> tuple[Path, Path]:
    """A path where ``kind`` would be scaffolded, with a dangling symlink in the way: at the path itself, or for a bundle and a
    study as the folder's ``mime.toml``; and the path to scaffold at."""
    path = target(kind, tmp_path, saved, study)
    link = path / "mime.toml" if kind in ("bundle", "study") else path
    link.symlink_to(tmp_path / "nowhere")
    return path, link


def link_problems(kind: str, tmp_path: Path, saved: Path, study: Path) -> list[str]:
    """Empty where a scaffold meets the dangling link with ``scaffold.exists`` and leaves it be."""
    path, link = linked(kind, tmp_path, saved, study)
    try:
        scaffold.scaffold(kind, path, yes=True, say=quiet)
    except BuildRefused as refused:
        found = codes(refused)
        problems = [] if found == ["scaffold.exists"] else [f"refused as {found}"]
    except Exception as error:  # noqa: BLE001 - what the planted guard lets through is the finding
        return [f"raised {type(error).__name__}"]
    else:
        return ["scaffolded over a dangling link"]
    return problems if link.is_symlink() and not link.exists() else [*problems, "the link was changed"]


@pytest.mark.parametrize("kind", KINDS)
def test_a_dangling_symlink_is_refused_as_a_path_that_exists(kind: str, tmp_path: Path, saved: Path, study: Path):
    assert link_problems(kind, tmp_path, saved, study) == []


@pytest.mark.parametrize("kind", KINDS)
def test_the_symlink_check_fails_when_the_guard_does_not_see_a_link(kind: str, tmp_path: Path, saved: Path, study: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    """The planted guard: ``is_symlink`` answering no, as a guard that asked only ``exists()`` would."""
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    assert link_problems(kind, tmp_path, saved, study) != []
