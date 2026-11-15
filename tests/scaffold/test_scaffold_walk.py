"""The walk, ``--from`` and the command line of ``mime scaffold``.

- **The walk** is ``mime push --setup``'s: each field with its line on what it is for, ``use <value>? [Y/n]`` where a value
  is held (a default, or scaffold's own), Enter keeping it or skipping a field, an invalid value said and asked again; a
  ``--set`` value is taken as given and not asked. At its end "Save these as your defaults?" is asked only where the
  walk was told a value ``defaults.toml`` holds, and a yes writes it (merged with the defaults already saved).
  Controls: with Enter planted to read as "no", the walk that keeps every value is caught asking for new ones; with the
  offer planted to save without asking, a "no" is caught saved.
- **``--from``** writes a block's declarations from a saved experiment, the things named its sockets, through
  ``fragment_from``; what it cannot write is refused, naming it, before the folder is made.
- **The command line**: the exit codes (0, 2 when refused), ``mime new`` retired, scaffold's help group and its example."""

import tomllib
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy
from scaffold_rig import Person, check_of, tree_of

from mime import scaffold
from mime.blocks import read_definition
from mime.build import BuildRefused
from mime.cli import main as mime_main
from mime.cli import _guide
from mime.push import _check, _defaults
from mime.scaffold import _scaffold
from mime.testing.toys import coupled_toy

#: A block's walk with no default held: the id, version, licence and title kept; a description; one creator (family and
#: given name, no ORCID iD, no affiliation, then no more); keywords.
ANSWERS = ("", "", "", "", "A probe", "Carberry", "Josiah", "", "", "", "microrobot, demo")


def walked(path: Path, person: Person) -> scaffold.Scaffolded:
    result = scaffold.scaffold("block", path, typed=person.typed, ask=person.ask, say=person.say)
    assert person.answers == [], f"answers left unasked: {person.answers}"
    return result


def test_a_new_block_is_walked_field_by_field_with_its_line_and_values_held_are_offered(tmp_path: Path):
    person = Person(*ANSWERS)
    result = walked(tmp_path / "probe", person)
    heads = [prompt.split("\n")[0].split(":")[0] for prompt in person.asked if prompt.split("\n")[0].split(":")[0]
             in scaffold.KEYS["block"]]
    assert heads == ["block.id", "block.version", "block.licence", "record.title", "record.description",
                     "record.creators", "record.keywords"], "each key asked once, in the walk's order"
    assert 'use "mylab.probe"? [Y/n] ' in person.asked[0] and "block.id: the block's id" in person.asked[0]
    held = {key: (value.value, value.source) for key, value in result.values.items()}
    assert held["block.id"] == ("mylab.probe", "scaffold's own")
    assert held["record.description"] == ("A probe", "asked")
    assert held["record.creators"] == ([{"family": "Carberry", "given": "Josiah"}], "asked")
    assert held["record.keywords"] == (["microrobot", "demo"], "asked")
    assert read_definition(tmp_path / "probe", record=False)[0] is not None


def test_a_value_that_is_not_valid_is_said_and_asked_again(tmp_path: Path):
    person = Person("n", "NOT AN ID", "labx.probe", *([""] * 6))
    result = walked(tmp_path / "probe", person)
    assert any("is not <namespace>.<name>" in prompt for prompt in person.asked)
    assert result.values["block.id"].value == "labx.probe"


def test_a_set_value_is_taken_as_given_and_not_asked(tmp_path: Path):
    person = Person(*([""] * 14))
    result = scaffold.scaffold("block", tmp_path / "probe", sets=["block.id=labx.given"], typed=person.typed,
                               ask=person.ask, say=person.say)
    assert (result.values["block.id"].value, result.values["block.id"].source) == ("labx.given", "--set")
    assert not any(prompt.startswith("block.id") for prompt in person.asked)


def test_the_walk_offers_to_save_what_it_was_told_and_never_under_yes(tmp_path: Path):
    person = Person(*ANSWERS, save=True)
    walked(tmp_path / "probe", person)
    assert any("Save these as your defaults (creators, keywords)" in prompt for prompt in person.asked)
    held = _defaults.read_all()
    assert held.values["creators"] == [{"family": "Carberry", "given": "Josiah"}] and held.values["keywords"] == [
        "microrobot", "demo"]
    again = Person()
    scaffold.scaffold("block", tmp_path / "probe2", yes=True, typed=again.typed, ask=again.ask, say=again.say)
    assert again.asked == []
    second = scaffold.scaffold("block", tmp_path / "probe3", yes=True, say=lambda line: None)
    assert second.values["record.creators"].source == "defaults.toml"


def test_a_no_to_the_offer_saves_nothing_and_a_walk_with_nothing_new_does_not_ask(tmp_path: Path):
    person = Person("", "", "", "", "", "", "microrobot", save=False)
    walked(tmp_path / "probe", person)
    assert any("Save these as your defaults" in prompt for prompt in person.asked)
    assert not _defaults.defaults_path().exists()
    _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah"}]})
    kept = Person(*([""] * 7))
    scaffold.scaffold("block", tmp_path / "probe2", typed=kept.typed, ask=kept.ask, say=kept.say)
    assert not any("Save these" in prompt for prompt in kept.asked)


def test_the_save_offer_adds_to_the_defaults_already_saved_and_names_a_blocks_namespace(tmp_path: Path):
    _defaults.write_defaults({"keywords": ["kept"]}, licences={"content": "CC0-1.0"})
    person = Person("n", "labx.probe", *([""] * 6), save=True)
    walked(tmp_path / "probe", person)
    held = _defaults.read_all()
    assert held.namespace == "labx" and held.licences == {"content": "CC0-1.0"} and held.values["keywords"] == ["kept"]


def test_enter_read_as_no_is_caught_by_the_walk_that_keeps_every_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah"}], "keywords": ["microrobot"]})
    keeping = Person("", "", "", "", "", "", "")
    scaffold.scaffold("block", tmp_path / "kept", typed=keeping.typed, ask=keeping.ask, say=keeping.say)
    assert keeping.answers == []
    original = _scaffold.Walk.keep

    def no_for_enter(self, head, value):
        return False if self.answer(f"{head}\nuse {value}? [Y/n] ") == "" else original(self, head, value)

    monkeypatch.setattr(_scaffold.Walk, "keep", no_for_enter)
    planted = Person(*([""] * 7))
    with pytest.raises(AssertionError):
        scaffold.scaffold("block", tmp_path / "planted", typed=planted.typed, ask=planted.ask, say=planted.say)


# -- the other kinds' walks -----------------------------------------------------------------------------------------------


def test_a_bundle_is_walked_over_the_experiments_declared_licence(tmp_path: Path):
    folder = saved_toy(tmp_path)
    person = Person("", "", "", "Carberry", "Josiah", "", "", "", "")
    scaffold.scaffold("bundle", folder, typed=person.typed, ask=person.ask, say=person.say)
    held = tomllib.loads((folder / "mime.toml").read_text(encoding="utf-8"))
    assert held["bundle"]["licence"] == "CC-BY-4.0" and held["record"]["creators"][0]["family"] == "Carberry"
    assert 'use "CC-BY-4.0"? [Y/n] ' in person.asked[0]
    found = _check.check(folder)
    assert found.findings == [] and found.of(_check.TO_DO) == [], "a creator named, nothing is left to do"


def test_a_licence_the_experiment_does_not_declare_is_refused_for_a_bundle(tmp_path: Path):
    folder = saved_toy(tmp_path)
    with pytest.raises(BuildRefused) as refused:
        scaffold.scaffold("bundle", folder, yes=True, sets=["bundle.licence=MIT"], say=lambda line: None)
    assert refused.value.refusals[0].code == "scaffold.set" and "declares CC-BY-4.0" in refused.value.refusals[0].message
    assert not (folder / "mime.toml").exists()


def test_a_plugins_names_come_from_its_folder_and_the_defaults_and_set(tmp_path: Path):
    _defaults.write_defaults({"creators": [{"family": "Carberry", "given": "Josiah"}]}, licences={"code": "MIT"},
                             namespace="labx")
    result = scaffold.scaffold("plugin", tmp_path / "labx-probes", yes=True, sets=["plugin.package=probes"],
                               say=lambda line: None)
    held = {key: (value.value, value.source) for key, value in result.values.items()}
    assert held["plugin.distribution"] == ("labx-probes", "scaffold's own")
    assert held["plugin.package"] == ("probes", "--set") and held["plugin.namespace"] == ("labx", "defaults.toml")
    assert held["plugin.licence"] == ("MIT", "defaults.toml")
    project = tomllib.loads((tmp_path / "labx-probes" / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["authors"] == [{"name": "Josiah Carberry"}] and project["project"]["license"] == "MIT"
    assert project["project"]["entry-points"]["mime.entries"] == {"labx": "probes.entries"}
    assert (tmp_path / "labx-probes" / "src" / "probes" / "entries" / "example_load.py").is_file()


def test_the_namespace_a_plugin_takes_is_the_registrys_form(tmp_path: Path):
    with pytest.raises(BuildRefused) as refused:
        scaffold.scaffold("plugin", tmp_path / "p", yes=True, sets=["plugin.namespace=mime"], say=lambda line: None)
    assert refused.value.refusals[0].code == "scaffold.set"


# -- --from ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    experiment = coupled_toy()
    experiment.licence("CC-BY-4.0")
    folder = tmp_path_factory.mktemp("toy") / "coupled"
    experiment.save(folder)
    return folder


def test_from_writes_the_declarations_of_a_saved_experiment_with_the_named_thing_as_a_socket(toy: Path, tmp_path: Path):
    result = scaffold.scaffold("block", tmp_path / "from_toy", yes=True, source=toy, sockets=["water=bath"],
                               say=lambda line: None)
    folder = tmp_path / "from_toy"
    assert result.written and (folder / "cases").is_dir()
    fragment = (folder / "block.mime.yaml").read_text(encoding="utf-8")
    assert "socket: water" in fragment and "bath" not in fragment.replace("toy.couple", "")
    held = tomllib.loads((folder / "mime.toml").read_text(encoding="utf-8"))
    assert held["sockets"] == {"water": {"kind": "medium"}}
    assert check_of("block", folder) == []


def test_from_refuses_what_it_cannot_write_before_the_folder_is_made(toy: Path, tmp_path: Path):
    for sockets, words in ((["water=ocean"], "ocean"), (["name=bath"], "name")):
        with pytest.raises(BuildRefused) as refused:
            scaffold.scaffold("block", tmp_path / "bad", yes=True, source=toy, sockets=sockets, say=lambda line: None)
        assert refused.value.refusals[0].code == "block.socket" and words in refused.value.refusals[0].message
        assert not (tmp_path / "bad").exists()
    with pytest.raises(BuildRefused) as refused:
        scaffold.scaffold("block", tmp_path / "bad", yes=True, source=toy, sockets=["bath"], say=lambda line: None)
    assert refused.value.refusals[0].code == "scaffold.set" and not (tmp_path / "bad").exists()


def test_the_folder_is_removed_again_when_a_step_after_it_was_made_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_scaffold, "_save_experiment", lambda root, text: (_ for _ in ()).throw(RuntimeError("full disk")))
    with pytest.raises(RuntimeError):
        scaffold.scaffold("experiment", tmp_path / "half", yes=True, say=lambda line: None)
    assert not (tmp_path / "half").exists()


# -- the command line -----------------------------------------------------------------------------------------------------


def test_the_command_line_scaffolds_with_yes_and_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / "probe"
    assert mime_main(["scaffold", "block", str(path), "--yes", "--set", "block.id=labx.probe", "-y"]) == 0
    out = capsys.readouterr().out
    assert "Scaffolded a block labx.probe" in out and "Next:" in out and f"mime check {path}" in out
    assert tomllib.loads((path / "mime.toml").read_text(encoding="utf-8"))["block"]["id"] == "labx.probe"
    assert mime_main(["check", str(path)]) == 0


def test_the_command_line_refuses_with_exit_two_and_a_catalogued_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert mime_main(["scaffold", "block", str(tmp_path / "p"), "--yes", "--set", "block.idd=a.b"]) == 2
    err = capsys.readouterr().err
    assert "scaffold.set" in err and "block.id" in err
    (tmp_path / "taken").mkdir()
    assert mime_main(["scaffold", "plugin", str(tmp_path / "taken"), "--yes"]) == 2
    assert "scaffold.exists" in capsys.readouterr().err


def test_dry_run_on_the_command_line_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    before = tree_of(tmp_path)
    assert mime_main(["scaffold", "plugin", str(tmp_path / "p"), "--yes", "--dry-run"]) == 0
    assert "(nothing written)" in capsys.readouterr().out and tree_of(tmp_path) == before


def test_mime_new_is_retired_and_scaffold_is_the_one_start(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as stopped:
        mime_main(["new", "experiment"])
    assert stopped.value.code == 2
    assert "scaffold" in capsys.readouterr().err, "the nearest command is named"
    start = dict(_guide.GROUPS)["Start"]
    assert start == ("scaffold",) and ("scaffold",) in _guide.EXAMPLES
    readme = (Path(__file__).resolve().parents[2] / "src" / "mime" / "cli" / "README.md").read_text(encoding="utf-8")
    assert "- `mime new`" not in readme and "`mime scaffold <kind> <path>" in readme
