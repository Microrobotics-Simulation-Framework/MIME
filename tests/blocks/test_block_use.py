"""``exp.use``: a block used in an experiment, saved, loaded, and refused by its codes.

**The tier's guarantees**, each with the control that must fail it on every run (``MIME_PLANT_P4A``):

- *A copied block's load never needs the block.* The block folder is deleted after the save and the load reproduces the
  input hash. Control ``dropped``: the document's writer leaves out one declaration the block provides (drawn from
  ``MIME_SEED``), as a writer that left the expansion to the load would; the load without the block then cannot
  reproduce the experiment.
- *A copied block's licence reaches the document's verdict.* The block's licence is drawn from MIME's conditional list,
  and every file of the author's own carries its conditions. Control ``unjoined``: the verdict computed without the
  document's blocks.

**Each refusal**, with the control that removes its check: ``block.requires`` (``unrequired``: every requirement read as
met), ``block.entry`` (``unlisted``), ``block.socket`` at the binding (``unbound``), at the definition
(``undeclared``) and in the fragment (``unwritten``), ``block.fragment`` (``unshaped``), ``block.format``
(``unformatted``) and ``block.name`` (``unnamed``). Each draws the defect it plants from ``MIME_SEED`` where there is
more than one."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from blocks import CONDITIONAL, OPEN, block_copy, edited, host, own_cache, plant, rng

from mime.blocks import _check, _definition, _fragment, _use
from mime.blocks._use import MET, BlockRefused, RequirementRow
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, _record, _terms
from mime.saved._terms import folder_terms
from mime.terms import licence_verdict
from mime.things import Body, State
from mime.testing.toys import ToyBody
from mime.units import kg, m


def saved_use(root: Path, folder: Path, *, licence: str = "CC-BY-4.0") -> tuple[Path, dict[str, Any]]:
    """An experiment of the user's using the block at ``folder``, saved below ``root``, and its document."""
    experiment, water = host()
    experiment.use(folder, medium=water)
    experiment.licence(licence)
    out = root / "saved"
    with own_cache(root):
        experiment.save(out, steps=3)
    return out, yaml.safe_load((out / DOCUMENT_FILE).read_text(encoding="utf-8"))


def codes(error: pytest.ExceptionInfo[BuildRefused]) -> list[str]:
    return error.value.codes()


# --- the copied tier's guarantees ---------------------------------------------------------------------------------


def _dropping(write: Any) -> Any:
    """The document's writer, leaving out one declaration the block provides, drawn from the seed."""
    def planted(settings: Any, declarations: Any) -> Any:
        document = dict(write(settings, declarations))
        (use,) = settings.blocks.values()
        provided = [(section, name) for section in ("things", "interactions", "declared")
                    for name in use["names"].get(section, [])]
        section, name = rng().choice(provided)
        if section == "interactions":
            document[section] = [entry for entry in document[section] if entry["id"] != name]
        else:
            document[section] = {key: value for key, value in document[section].items() if key != name}
        return document

    return planted


def test_a_copied_block_loads_without_the_block_and_reproduces_its_input_hash(tmp_path: Path):
    folder = block_copy(tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "dropped":
            patch.setattr(_record, "document_of", _dropping(_record.document_of))
        saved, document = saved_use(tmp_path, folder)
    shutil.rmtree(folder)
    use = document["blocks"]["nudged_probe"]
    assert use["tier"] == "copied" and use["sockets"] == {"medium": "water"}
    assert set(use["names"]["things"]) <= set(document["things"])
    with own_cache(tmp_path):
        loaded = Experiment.load(saved)
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()


def test_a_copied_blocks_licence_reaches_the_verdict_of_every_file_of_the_authors_own(tmp_path: Path):
    draw = rng()
    block, own = draw.choice(CONDITIONAL), draw.choice(OPEN)
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'licence = "CC-BY-4.0"', f'licence = "{block}"')
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unjoined":
            patch.setattr(_terms, "copied_blocks", lambda document: [])
        saved, document = saved_use(tmp_path, folder, licence=own)
        from mime.saved import read_folder
        from mime.saved._folder import named_files

        form = read_folder(saved)
        judged = folder_terms(saved, form.document, form.lock, named_files(form), form.version)
    expected = licence_verdict(block)
    yours = [file for file in judged.files if file.owner == "yours"]
    assert yours and document["blocks"]["nudged_probe"]["licence"] == block
    for file in yours:
        assert (file.verdict.tier, file.verdict.conditions) == ("conditional", expected.conditions), file
        assert f"block labx.nudged_probe 0.1.0 ({block}) copied in" in file.verdict.by


def test_two_copied_blocks_under_different_conditional_licences_leave_the_authors_files_unknown_and_carried():
    from mime.saved._terms import own_verdict
    from mime.terms import DIFFERING_CONDITIONS_BY

    verdict = own_verdict("CC-BY-4.0", {}, [("block a 1", "CC-BY-NC-4.0"), ("block b 1", "CC-BY-SA-4.0")])
    assert (verdict.status, verdict.tier) == ("unknown", "carried") and DIFFERING_CONDITIONS_BY in verdict.by
    assert own_verdict("CC-BY-4.0", {}, [("block a 1", "MIT")]).tier == "open"
    assert own_verdict(None, {}, [("block a 1", "CC-BY-NC-4.0")]).status == "unknown"


def test_the_notice_and_the_card_name_each_block_and_its_licence(tmp_path: Path):
    from mime.saved import card_text, notice

    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'licence = "CC-BY-4.0"', 'licence = "CC-BY-NC-4.0"')
    saved, document = saved_use(tmp_path, folder)
    content = document["blocks"]["nudged_probe"]["content"]
    text = notice(saved)
    assert "Blocks it uses" in text and f"block labx.nudged_probe 0.1.0 (used from a block folder, no DOI), licensed " \
                                       f"CC-BY-NC-4.0, copied" in text and content in text
    assert "CC-BY-NC-4.0: https://spdx.org/licenses/CC-BY-NC-4.0.html" in text
    assert "- **Uses block labx.nudged_probe 0.1.0** (copied in under CC-BY-NC-4.0" in card_text(saved)


def test_a_document_using_no_block_holds_no_blocks_key_and_no_block_words(tmp_path: Path):
    from mime.saved import notice

    experiment, _ = host()
    experiment.licence("CC-BY-4.0")
    with own_cache(tmp_path):
        experiment.save(tmp_path / "plain", steps=2)
    document = yaml.safe_load((tmp_path / "plain" / DOCUMENT_FILE).read_text(encoding="utf-8"))
    assert "blocks" not in document and "Blocks it uses" not in notice(tmp_path / "plain")


# --- what the use records ----------------------------------------------------------------------------------------


def test_a_declaration_over_a_provided_one_replaces_it_in_a_script_and_is_recorded_as_an_override(tmp_path: Path):
    folder = block_copy(tmp_path)
    experiment, water = host()
    probe = experiment.use(folder, medium=water)
    assert probe.probe.name == "probe" and probe.handle("toy.drag:probe") == "toy.drag:probe"
    with pytest.warns(UserWarning, match=r"it replaces the one block labx.nudged_probe 0.1.0 provides \(the use "
                                         r"'nudged_probe'\), recorded among the use's overrides"):
        experiment.declare("probe.mass", kg(2.5), reason="our unit, weighed")
    experiment.remove("toy.spring:probe")
    use = experiment.document()["blocks"]["nudged_probe"]
    assert use["overrides"] == ["probe.mass", "toy.spring:probe"]
    assert "toy.spring:probe" in use["names"]["interactions"]
    assert experiment.document()["declared"]["probe.mass"]["value"] == 2.5


def test_the_same_use_made_again_is_kept_once_and_a_named_use_renames_the_blocks_things_and_ids(tmp_path: Path):
    folder = block_copy(tmp_path)
    experiment, water = host()
    experiment.use(folder, medium=water)
    with pytest.warns(UserWarning, match="the same as before: it is kept once"):
        experiment.use(folder, medium=water)
    second = experiment.use(folder, name="spare", medium="water")
    assert second.probe.name == "spare_probe" and second.handle("labx.nudge:probe") == "labx.nudge:spare_probe"
    document = experiment.document()
    assert sorted(document["blocks"]) == ["nudged_probe", "spare"]
    assert document["blocks"]["spare"]["names"]["things"] == ["spare_probe"]
    assert {"toy.couple:probe", "toy.couple:spare_probe"} <= {entry["id"] for entry in document["interactions"]}
    assert {entry["binds"]["medium"] for entry in document["interactions"] if entry["name"] == "toy.couple"} == {"water"}
    assert sorted(path for path in document["declared"]) == ["probe.mass", "spare_probe.mass"]
    experiment.build()


def test_an_experiment_based_on_one_that_uses_a_block_keeps_it_and_records_a_later_override(tmp_path: Path):
    folder = block_copy(tmp_path)
    saved, _ = saved_use(tmp_path, folder)
    with own_cache(tmp_path):
        child = Experiment.based_on(saved, name="child")
    child.declare("probe.mass", kg(4.0), reason="ours")
    assert child.document()["blocks"]["nudged_probe"]["overrides"] == ["probe.mass"]


def test_fragment_from_writes_back_the_saved_experiment_its_named_things_as_sockets(tmp_path: Path):
    """The round trip: a block written from a saved experiment, used by an experiment that declares the named thing, gives
    back the same declarations."""
    from mime.blocks import fragment_from

    folder = block_copy(tmp_path)
    saved, original = saved_use(tmp_path, folder)
    drafted = fragment_from(saved, medium="water")
    assert drafted.sockets == {"medium": "medium"} and drafted.files == ()
    (folder / "block.mime.yaml").write_text(drafted.text, encoding="utf-8")
    experiment, water = host()
    experiment.use(folder, medium=water)
    experiment.licence("CC-BY-4.0")
    again = experiment.document()
    for key in ("things", "interactions", "observations", "declared", "sources", "controller"):
        assert again[key] == original[key], key


# --- refusals ----------------------------------------------------------------------------------------------------


def _all_met(rows: Any) -> Any:
    def planted(definition: Any) -> list[RequirementRow]:
        return [RequirementRow(row.name, row.wanted, row.found, row.pinned_by, MET) for row in rows(definition)]

    return planted


def test_a_requirement_not_met_here_is_refused_with_its_pin_row_and_one_install_line(tmp_path: Path):
    import importlib.metadata

    from packaging.version import Version

    draw = rng()
    distribution = draw.choice(("mime-engine", "mime-lab-fixture"))
    found = Version(importlib.metadata.version(distribution))
    part = draw.randrange(3)
    floor = ".".join(str(number + (draw.randint(1, 3) if index == part else 0))
                     for index, number in enumerate(found.release[:3] + (0,) * (3 - len(found.release[:3]))))
    folder = block_copy(tmp_path)
    toml = (folder / "mime.toml").read_text(encoding="utf-8")
    line = next(each for each in toml.splitlines() if each.startswith(f"{distribution} ="))
    (folder / "mime.toml").write_text(toml.replace(line, f'{distribution} = ">={floor}"'), encoding="utf-8")
    experiment, water = host()
    before = experiment.document()
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unrequired":
            patch.setattr(_use, "requirement_rows", _all_met(_use.requirement_rows))
        with pytest.raises(BlockRefused) as refused:
            experiment.use(folder, medium=water)
    assert codes(refused) == ["block.requires"]
    (row,) = refused.value.pin_rows
    assert (row.name, row.wanted, row.verdict) == (distribution, f">={floor}", "outside its range")
    assert f"pip install '{distribution}>={floor}'" in refused.value.refusals[0].action
    assert experiment.document() == before, "nothing is added by a refused use"


def test_an_entry_from_a_distribution_requires_does_not_name_is_the_blocks_defect(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'mime-lab-fixture = ">=0.1,<0.2"\n', "")
    experiment, water = host()
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unlisted":
            patch.setattr(_use, "entry_refusals", lambda definition, sections: [])
        with pytest.raises(BlockRefused) as refused:
            experiment.use(folder, medium=water)
    assert codes(refused) == ["block.entry"]
    assert "labx.nudge 0.1.0, from the distribution mime-lab-fixture" in refused.value.refusals[0].message


def test_a_socket_bound_wrongly_is_refused_as_block_socket(tmp_path: Path):
    folder = block_copy(tmp_path)
    experiment, water = host()
    experiment.add_body(Body("ball", dynamics=ToyBody(), initial=State(position=m((0.0, 0.0, 0.0)))))
    defect, given = rng().choice((("unbound", {}), ("unknown socket", {"medium": water, "extra": water}),
                                  ("another kind", {"medium": "ball"}), ("not declared", {"medium": "ghost"})))

    def bound_by_name(definition: Any, sockets: Any, things: Any) -> Any:
        return {name: getattr(value, "name", value) for name, value in sockets.items()}, []

    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unbound":
            patch.setattr(_use, "bound_sockets", bound_by_name)
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, **given)
    assert "block.socket" in codes(refused), defect


def test_a_socket_the_definition_cannot_declare_is_refused_and_a_frame_names_its_route(tmp_path: Path):
    folder = block_copy(tmp_path)
    defect, line, words = rng().choice((("a frame", 'base = { kind = "frame", of = "world" }', "is a frame socket"),
                                        ("a reserved name", 'name = { kind = "body" }', "is not a socket's name"),
                                        ("an unknown kind", 'pump = { kind = "pump" }', "its kind is 'pump'")))
    edited(folder, "mime.toml", 'medium = { kind = "medium" }', f'medium = {{ kind = "medium" }}\n{line}')
    experiment, water = host()
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "undeclared":
            read = _definition._sockets  # noqa: SLF001

            def accepting(reader: Any, value: Any, path: Any) -> Any:
                found, _ = read(reader, value, path)
                return {**found, **{key: str(held.get("kind")) for key, held in value.items() if key not in found}}, []

            patch.setattr(_definition, "_sockets", accepting)
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, medium=water)
    assert any(each.code == "block.socket" and words in each.message for each in refused.value.refusals), defect
    if defect == "a frame":
        assert "the apparatus lane's AP6" in str(refused.value)


def test_a_socket_the_fragment_writes_wrongly_is_refused_as_block_socket(tmp_path: Path):
    folder = block_copy(tmp_path)
    defect = rng().choice(("bare", "unknown", "unused"))
    words = {"bare": "written as a bare name", "unknown": "names the socket 'bath', which [sockets] does not declare",
             "unused": "never uses the socket spare"}[defect]
    if defect == "bare":
        edited(folder, "block.mime.yaml", "medium: {socket: medium}", "medium: medium")
    elif defect == "unknown":
        edited(folder, "block.mime.yaml", "medium: {socket: medium}", "medium: {socket: bath}")
    else:
        edited(folder, "mime.toml", 'medium = { kind = "medium" }', 'medium = { kind = "medium" }\nspare = { kind = "body" }')
    experiment, water = host()
    experiment.add_body(Body("ball", dynamics=ToyBody(), initial=State(position=m((0.0, 0.0, 0.0)))))
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unwritten":
            written = _fragment._Sockets.written  # noqa: SLF001

            def quiet(self: Any, value: object, where: str) -> object:
                held = len(self.refusals)
                result = written(self, value, where)
                del self.refusals[held:]
                if isinstance(value, dict) and "socket" in value:
                    return value["socket"]
                return result

            patch.setattr(_fragment._Sockets, "written", quiet)  # noqa: SLF001
            patch.setattr(_fragment._Sockets, "unused_refusals", lambda self, things: [])  # noqa: SLF001
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, medium=water, **({"spare": "ball"} if defect == "unused" else {}))
    assert any(each.code == "block.socket" and words in each.message for each in refused.value.refusals), defect


def test_a_fragment_unlike_the_documents_shapes_is_refused_by_path(tmp_path: Path):
    folder = block_copy(tmp_path)
    defect, old, new = rng().choice((
        ("a host's key", "schema: mime.fragment/1\n", "schema: mime.fragment/1\nprofile: research\n"),
        ("an unknown key", "    kind: body\n", "    kind: body\n    colour: red\n"),
        ("a prim elsewhere", "prim: /World/probe", "prim: /World/elsewhere"),
        ("a value of another type", "    body: probe\n- id: toy.couple", "    body: 3\n- id: toy.couple")))
    edited(folder, "block.mime.yaml", old, new)
    experiment, water = host()
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unshaped":
            patch.setattr(_fragment, "shape_refusals", lambda path, held, written: [])
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, medium=water)
    assert "block.fragment" in codes(refused), defect


def test_a_fragment_of_another_form_is_refused_as_block_format_and_a_later_one_as_newer(tmp_path: Path):
    folder = block_copy(tmp_path)
    schema = rng().choice(("mime.document/14", "mime.fragment/0", "fragment", "3"))
    edited(folder, "block.mime.yaml", "schema: mime.fragment/1", f"schema: {schema}")
    experiment, water = host()
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unformatted":
            patch.setattr(_fragment, "schema_refusal", lambda path, held: None)
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, medium=water)
    assert codes(refused) == ["block.format"], schema
    edited(folder, "block.mime.yaml", f"schema: {schema}", "schema: mime.fragment/2")
    with pytest.raises(BuildRefused) as refused:
        experiment.use(folder, medium=water)
    assert codes(refused) == ["load.newer-form"]


def test_a_name_the_experiment_already_declares_is_refused_naming_name(tmp_path: Path):
    folder = block_copy(tmp_path)
    experiment, water = host()
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=State(position=m((0.0, 0.0, 0.0)))))
    if rng().random() < 0.5:
        experiment.declare("probe.mass", kg(1.0), reason="ours")
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unnamed":
            patch.setattr(_use, "name_refusals", lambda *args: [])
        with pytest.raises(BuildRefused) as refused:
            experiment.use(folder, medium=water)
    assert set(codes(refused)) == {"block.name"}
    assert 'exp.use(..., name="<name>")' in refused.value.refusals[0].action
    assert experiment.use(folder, name="lab", medium=water).probe.name == "lab_probe"


def test_a_block_its_licence_makes_referenced_is_refused_until_its_container_is_read(tmp_path: Path):
    folder = block_copy(tmp_path)
    edited(folder, "mime.toml", 'licence = "CC-BY-4.0"', 'licence = "LicenseRef-labx-terms"\n\n'
                                                          '[licences.LicenseRef-labx-terms]\ntitle = "Demo terms"\n'
                                                          'url = "https://example.org/terms"')
    experiment, water = host()
    with pytest.raises(BuildRefused) as refused:
        experiment.use(folder, medium=water)
    assert codes(refused) == ["load.referenced-block"]
    assert "lets no copy of its content into your saved document" in refused.value.refusals[0].message


def test_the_content_hash_names_the_definition_and_the_fragment_and_not_the_record_or_a_cases_bytes(tmp_path: Path):
    folder = block_copy(tmp_path)
    content = _definition.definition_of(folder).content_hash()
    edited(folder, "mime.toml", 'title = "A nudged probe', 'title = "A probe, nudged')
    (folder / "cases").mkdir()
    (folder / "cases" / "note.txt").write_text("a case's file", encoding="utf-8")
    assert _definition.definition_of(folder).content_hash() == content
    edited(folder, "block.mime.yaml", "value: 2.0", "value: 2.5")
    assert _definition.definition_of(folder).content_hash() != content
    assert hashlib.sha256(b"").hexdigest() != content


def test_a_case_saved_against_other_content_is_found_stale(tmp_path: Path):
    from mime.push import check

    folder = block_copy(tmp_path)
    experiment, water = host("case")
    experiment.use(folder, medium=water)
    experiment.licence("CC-BY-4.0")
    with own_cache(tmp_path):
        experiment.save(folder / "cases" / "in_water", steps=2)
    assert check(folder).findings == [] and check(folder).of("to do") == []
    old, new = rng().choice((("value: 2.0", "value: 2.25"), ("rung: quadratic", "rung: linear"),
                             ("value: [0.3, -0.2, 0.1]", "value: [0.3, -0.2, 0.2]")))
    edited(folder, "block.mime.yaml", old, new)
    with pytest.MonkeyPatch.context() as patch:
        if plant() == "unstale":
            patch.setattr(_check, "stale_uses", lambda uses, content: [])
        found = check(folder).findings
        definition = _definition.definition_of(folder)
        refusals = _check.case_refusals(definition, definition.content_hash())
    assert [refusal.code for _, refusal in refusals] == ["block.case-stale"]
    assert any("cases/in_water was saved against other content of block labx.nudged_probe 0.1.0" in item.what
               for item in found), found
