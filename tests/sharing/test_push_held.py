"""What is published is what was seen, and what is recorded is true of the record, against the offline fake: the record's
description names the record; an access setting on the site is a site edit; site edits survive a new version; push's
record of a published version is the record's own listing; a script publishes what the draft holds; and the draft is read
again immediately before the request that cannot be undone.

Controls, each caught on every run:

- **a card not given the publication**: the published description is caught saying it is not published;
- **access not read as a field**: a draft embargoed on its page is caught published with no question;
- **a new version's marks saved before its edits are settled**: the same command run twice is caught writing over the
  edit it refused; **every field read as one the service resets**: a paragraph on the record's page is caught dropped;
- **the published files taken from the plan**: a re-saved folder is caught reading "unchanged since" and no version made;
- **no hold on what a script publishes**: a new creator is caught published, never on the draft;
- **no read before the request**: an edit, an added file, a changed DOI and an access setting are each caught published.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from push_cases import copy_of, saved_toy, with_token, write_sharing
from terms_cases import NO_REDISTRIBUTION, layered, same_reference_run, write_mesh
from zenodo_records import Entry, FakeRecords, FakeTransport, Killed

from mime import assets, push
from mime.build import BuildRefused
from mime.cli import main
from mime.push import _client, _draft, _edits, _plan, _publish, _sharing, _stage, _state
from mime.push._state import Uploaded, metadata_marks, read_state, write_state
from mime.saved import card_text, pack
from mime.saved._bundle import _own_location
from mime.saved._folder import lock_text
from mime.saved._pull import parse_source
from mime.terms import Terms

NUMBER = "629525"
DOI = f"10.5072/zenodo.{NUMBER}"
BUNDLE = "full_toy.mimebundle"
EMBARGO = {"record": "public", "files": "restricted",
           "embargo": {"active": True, "until": "2030-01-01", "reason": "under review"}}


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


@pytest.fixture(scope="module")
def changed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The toy saved with one reference step more: the same experiment, other content."""
    from mime.testing.toys import full_toy

    experiment = full_toy()
    experiment.licence("CC-BY-4.0")
    folder = tmp_path_factory.mktemp("changed") / "full_toy"
    experiment.save(folder, steps=9)
    return folder


class Terminal:
    """A person at a terminal: each y/N question answered from ``answers``, the record number typed from ``typed``, and
    ``on_typed`` run while the prompt waits; every line and question kept, in order, in ``events``."""

    def __init__(self, answers: tuple[bool, ...] = (), typed: tuple[str, ...] = (),
                 on_typed: Callable[[], None] | None = None) -> None:
        self.answers = list(answers)
        self.typing = list(typed)
        self.on_typed = on_typed
        self.said: list[str] = []
        self.asked: list[str] = []
        self.events: list[str] = []

    def say(self, line: str) -> None:
        self.said.append(line)
        self.events.append(line)

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        self.events.append(question)
        return self.answers.pop(0) if self.answers else False

    def typed(self, prompt: str) -> str:
        self.asked.append(prompt)
        self.events.append(prompt)
        if self.on_typed is not None:
            self.on_typed()
        return self.typing.pop(0) if self.typing else ""

    def text(self) -> str:
        return "\n".join(self.said)


def pushed(folder: Path, fake: FakeRecords, service: push.Service, person: Terminal | None = None,
           **extra: object) -> push.Drafted:
    """A push: in a terminal (``person``), or from a script."""
    return push.push(folder, service, confirm=person.ask if person else None, say=person.say if person else _quiet,
                     transport=FakeTransport(fake), pause=0, sleep=lambda seconds: None, **extra)  # type: ignore[arg-type]


def published(folder: Path, fake: FakeRecords, service: push.Service, person: Terminal | None = None,
              confirm: str | None = None, **extra: object) -> push.Published | None:
    """A publish: in a terminal (``person``), or from a script (``person`` None, with ``confirm``)."""
    return push.publish(folder, service, confirm=confirm, ask=person.ask if person else None,
                        typed=person.typed if person else None, say=person.say if person else _quiet,
                        transport=FakeTransport(fake), pause=0, sleep=lambda seconds: None, **extra)  # type: ignore[arg-type]


def _quiet(line: str) -> None:
    pass


def publishes(fake: FakeRecords) -> int:
    return sum(1 for method, path, _, _ in fake.requests if method == "POST" and path.endswith("/actions/publish"))


def codes(action: Callable[[], object]) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        action()
    return refused.value.codes()


def metadata(fake: FakeRecords, record: str = NUMBER) -> dict[str, object]:
    return fake.records[record].body["metadata"]  # type: ignore[return-value]


def sharing_text(folder: Path) -> str:
    return (folder / "sharing.yaml").read_text(encoding="utf-8")


def rewritten(folder: Path, old: str, new: str) -> None:
    text = sharing_text(folder)
    assert old in text
    (folder / "sharing.yaml").write_text(text.replace(old, new), encoding="utf-8")


def replaced_content(folder: Path, source: Path) -> None:
    """The folder's saved form replaced by ``source``'s, its sharing.yaml kept."""
    for path in folder.iterdir():
        if path.name != "sharing.yaml":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in source.iterdir():
        target = folder / path.name
        shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)


def production(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRecords, push.Service]:
    fake = FakeRecords(with_token(monkeypatch, "zenodo"), prefix="10.5281")
    return fake, fake.service("zenodo", production=True)


def sandbox(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRecords, push.Service]:
    fake = FakeRecords(with_token(monkeypatch))
    return fake, fake.service()


# --- F1: the record's description names the record -----------------------------------------------------------------


def card_problems(description: object, doi: str, version: int) -> list[str]:
    """What is false of a published record in its description."""
    text = str(description)
    found = [f"says {words!r}" for words in ("not published", "none yet", "No DOI yet", "the record's DOI")
             if words in text]
    if f"<strong>DOI:</strong> {doi}<" not in text:
        found.append("names no DOI")
    if f"<strong>Version:</strong> {version}<" not in text:
        found.append("names no version")
    return found


def test_a_published_records_description_names_its_version_doi_and_how_to_pull_and_cite_it(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    doi = f"10.5281/zenodo.{NUMBER}"
    pushed(folder, fake, service)
    assert card_problems(metadata(fake)["description"], doi, 1) == [], "the draft shows what the record will say"
    assert published(folder, fake, service, confirm=doi) is not None
    description = str(metadata(fake)["description"])
    assert card_problems(description, doi, 1) == []
    assert f"mime pull {doi}" in description and f"Cite this version by its DOI, {doi};" in description
    assert "Concept DOI" not in description, "a first version's concept DOI is minted at publish, and not yet known"
    card = card_text(folder)
    assert f"- **DOI:** {doi}" in card and "- **Version:** 1" in card, "mime card reads the lock that names the record"
    replaced_content(folder, changed)
    drafted = pushed(folder, fake, service)
    assert published(folder, fake, service, confirm=drafted.doi) is not None
    second = str(metadata(fake, drafted.record)["description"])
    assert card_problems(second, drafted.doi, 2) == []
    assert f"<strong>Concept DOI (every version):</strong> {fake.concept_doi(NUMBER)}<" in second
    assert f"every version by its concept DOI, {fake.concept_doi(NUMBER)};" in second


def test_a_sandbox_records_description_says_how_the_sandbox_pulls_it(toy: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=DOI)
    assert card_problems(metadata(fake)["description"], DOI, 1) == []
    assert f"mime pull --from zenodo-sandbox {NUMBER}" in str(metadata(fake)["description"])


def test_control_a_card_not_given_the_publication_is_caught_saying_it_is_not_published(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real = _plan.card_of
    monkeypatch.setattr(_plan, "card_of", lambda facts, publication: real(facts, None))
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=DOI)
    assert "says 'not published'" in card_problems(metadata(fake)["description"], DOI, 1), \
        "the planted card is not seen"


# --- F2: access set on the draft's page ----------------------------------------------------------------------------


def embargoed(fake: FakeRecords) -> None:
    fake.records[NUMBER].body["access"] = json.loads(json.dumps(EMBARGO))


def test_an_embargo_set_on_the_page_is_shown_and_kept_by_default_and_published_as_the_plan_says(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    person = Terminal(answers=(False,), typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None and publishes(fake) == 1
    assert fake.records[NUMBER].body["access"] == EMBARGO
    assert "! access was edited on the site since push last wrote it:" in person.said
    assert person.asked[0].startswith("Write sharing.yaml's over the site's?")
    assert _sharing.read_sharing(folder).access == {"files": "restricted", "embargo": {"until": "2030-01-01",
                                                                                       "reason": "under review"}}
    warning = "! access once published is not open: the record public, its files restricted, embargoed until 2030-01-01"
    assert any(line.startswith(warning) for line in person.said), person.text()
    assert person.events.index(next(line for line in person.said if line.startswith(warning))) < \
        person.events.index(f"Type the record number to publish ({NUMBER}): ")
    assert "checked against the record as published: its DOI, each file's size and md5, its metadata" in person.text()


def test_an_embargo_set_on_the_page_is_written_over_only_when_answered_yes_and_says_so(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    person = Terminal(answers=(True,), typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None
    assert fake.records[NUMBER].body["access"] == {"record": "public", "files": "public"}
    assert "! wrote over what was edited on the site: access" in person.said


def test_a_script_is_stopped_by_an_embargo_set_on_the_page_on_the_same_day_and_a_later_one(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    assert codes(lambda: published(folder, fake, service, confirm=DOI)) == ["push.metadata-edited"]
    metadata(fake)["publication_date"] = "2026-01-01"  # a publish on a later day writes the draft
    state = read_state(service, "full_toy")
    assert state is not None
    state.written["publication_date"] = metadata_marks({"metadata": metadata(fake)})["publication_date"]
    write_state(service, state)
    assert codes(lambda: published(folder, fake, service, confirm=DOI)) == ["push.metadata-edited"]
    assert publishes(fake) == 0 and fake.records[NUMBER].body["access"] == EMBARGO
    assert published(folder, fake, service, confirm=DOI, keep_site_edits=True) is not None
    assert fake.records[NUMBER].published and fake.records[NUMBER].body["access"] == EMBARGO


def test_an_access_the_service_changed_is_seen_by_the_after_publish_check(toy: Path, tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    fake.publish(NUMBER)  # the site's button: nothing asked
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, Terminal())
    assert refused.value.codes() == ["push.metadata-read-back"]
    assert "access" in refused.value.refusals[0].message and "restricted" in refused.value.refusals[0].message


def test_control_access_not_read_as_a_field_is_caught_published_with_no_question(toy: Path, tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_state, "access_of", lambda held: {})
    monkeypatch.setattr(_edits, "access_of", lambda held: {})
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    person = Terminal(typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None
    assert publishes(fake) == 1 and person.asked == [f"Type the record number to publish ({NUMBER}): "], \
        "the planted blindness to access is not seen"


def test_a_preview_chosen_on_the_page_survives_a_write(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    fake.records[NUMBER].body["files"] = {"enabled": True, "default_preview": "NOTICE"}
    rewritten(folder, 'description: ""', 'description: "A paragraph."')
    pushed(folder, fake, service)
    assert "A paragraph." in str(metadata(fake)["description"])
    assert fake.records[NUMBER].body["files"] == {"enabled": True, "default_preview": "NOTICE"}


@pytest.mark.parametrize(("lines", "where"), [
    ("access: public", "access"), ("access: {files: open}", "access.files"), ("access: {colour: red}", "access.colour"),
    ("access: {files: public, embargo: {until: 2030-01-01}}", "access.embargo"),
    ("access: {files: restricted, embargo: {until: soon}}", "access.embargo.until"),
    ("access: {files: restricted, embargo: {reason: x}}", "access.embargo.until")])
def test_each_problem_of_access_is_refused_by_its_path(toy: Path, tmp_path: Path, lines: str, where: str):
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yaml").write_text(sharing_text(folder) + lines + "\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _sharing.read_sharing(folder)
    assert [each.code for each in refused.value.refusals] == ["push.sharing"]
    assert f"sharing.yaml, {where}: " in refused.value.refusals[0].message, refused.value.refusals[0].message


def test_an_unquoted_day_of_an_embargo_is_read_as_the_day(toy: Path, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yaml").write_text(sharing_text(folder) + "access: {files: restricted, embargo: {until: "
                                                                "2030-01-01}}\n", encoding="utf-8")
    assert _sharing.read_sharing(folder).access == {"files": "restricted", "embargo": {"until": "2030-01-01"}}


# --- F3: site edits survive a new version --------------------------------------------------------------------------


def published_then_changed(folder: Path, changed: Path, fake: FakeRecords, service: push.Service,
                           edit: Callable[[dict[str, object]], None]) -> None:
    """Version 1 published, ``edit`` made on the record's page, and the folder changed."""
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=DOI)
    edit(metadata(fake))
    replaced_content(folder, changed)


def test_the_same_command_run_twice_stops_at_the_same_edit_of_the_published_record(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: held.update(title="Corrected on the page"))
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
    draft = fake.drafts()[0].id
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"], \
        "the same command, run again, stops at what it refused"
    assert metadata(fake, draft)["title"] == "Corrected on the page"
    person = Terminal(answers=(False,))
    drafted = pushed(folder, fake, service, person)
    assert drafted.record == draft and drafted.version == 2
    assert metadata(fake, draft)["title"] == "Corrected on the page"
    assert _sharing.read_sharing(folder).title == "Corrected on the page"


def test_control_marks_saved_before_the_edit_is_settled_are_caught_writing_over_it_on_the_second_run(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real = _draft.Drafting.settle

    def planted(self: _draft.Drafting, found: dict[str, object], record: str) -> list[str]:
        state = self.state
        assert state is not None
        if state.published is not None and not state.sent:
            held = state.written
            state.written = metadata_marks(found)
            self.save()  # on the disk, the draft's marks, the edit among them; in this run, as they were
            state.written = held
        return real(self, found, record)

    monkeypatch.setattr(_draft.Drafting, "settle", planted)
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: held.update(title="Corrected on the page"))
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
    drafted = pushed(folder, fake, service)
    assert metadata(fake, drafted.record)["title"] == "full_toy", "the planted early save is not seen"


def test_a_paragraph_written_on_the_published_page_is_asked_about_and_kept_above_the_new_card(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)

    def paragraph(held: dict[str, object]) -> None:
        held["description"] = f"<p>A correction on the page.</p>\n{held['description']}"

    published_then_changed(folder, changed, fake, service, paragraph)
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
    person = Terminal(answers=(False,))
    drafted = pushed(folder, fake, service, person)
    assert len(person.asked) == 1 and _sharing.read_sharing(folder).description == "A correction on the page."
    held = str(metadata(fake, drafted.record)["description"])
    assert held.startswith("<p>A correction on the page.</p>\n<h1>full_toy</h1>")
    assert card_problems(held, drafted.doi, 2) == [], "the card below the paragraph is the new version's"


def test_a_word_changed_inside_the_published_card_is_asked_about(toy: Path, changed: Path, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)

    def word(held: dict[str, object]) -> None:
        held["description"] = str(held["description"]).replace("<h1>", "<h1>Edited ", 1)

    published_then_changed(folder, changed, fake, service, word)
    person = Terminal(answers=(False,))
    assert codes(lambda: pushed(folder, fake, service, person)) == ["push.metadata-edited"]
    assert person.asked[0].startswith("sharing.yaml cannot hold it")
    assert "Edited" in str(metadata(fake, fake.drafts()[0].id)["description"]), "a stopped push writes nothing"


@pytest.mark.parametrize(("field", "value"), [("locations", {"features": [{"place": "Delft"}]}),
                                              ("publisher", "A press")])
def test_a_field_sharing_yaml_cannot_hold_edited_on_the_published_page_is_never_dropped_unasked(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: held.update({field: value}))
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
    declined = Terminal(answers=(False,))
    assert codes(lambda: pushed(folder, fake, service, declined)) == ["push.metadata-edited"]
    assert declined.asked[0].startswith("sharing.yaml cannot hold it")
    assert metadata(fake, fake.drafts()[0].id)[field] == value
    person = Terminal(answers=(True,))
    drafted = pushed(folder, fake, service, person)
    assert f"! wrote over what was edited on the site: metadata.{field}" in person.said
    assert metadata(fake, drafted.record).get(field) != value


def test_what_the_service_resets_on_a_new_version_is_no_edit(toy: Path, changed: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: None)
    real = fake._draft_answer

    def reset(record: object, rdm: bool, **given_: object) -> tuple[int, str, bytes]:
        status, kind, data = real(record, rdm, **given_)  # type: ignore[arg-type]
        held = json.loads(data)
        if record.id != NUMBER and record.revision == 1:  # type: ignore[attr-defined]
            for name in _draft.VERSION_RESETS:
                held["metadata"].pop(name, None)
        return status, kind, json.dumps(held).encode()

    monkeypatch.setattr(fake, "_draft_answer", reset)
    person = Terminal()
    assert pushed(folder, fake, service, person).version == 2 and person.asked == []


class _Everything:
    def __contains__(self, name: object) -> bool:
        return True


def test_control_every_field_read_as_one_the_service_resets_is_caught_dropping_a_paragraph(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft, "VERSION_RESETS", _Everything())
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)

    def paragraph(held: dict[str, object]) -> None:
        held["description"] = f"<p>A correction on the page.</p>\n{held['description']}"

    published_then_changed(folder, changed, fake, service, paragraph)
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    assert person.asked == [] and "A correction" not in str(metadata(fake, drafted.record)["description"]), \
        "the planted filter is not seen"


# --- F4: push's record of a published version is the record's own ---------------------------------------------------


def lost_publish_then_resave(folder: Path, changed: Path, fake: FakeRecords, service: push.Service, how: str) -> None:
    pushed(folder, fake, service)
    if how == "the site's button":
        fake.publish(NUMBER)
    else:
        fake.plant("kill-after", lambda method, path: path.endswith("/actions/publish"))
        with pytest.raises(Killed):
            published(folder, fake, service, confirm=f"{fake.prefix}/zenodo.{NUMBER}")
    replaced_content(folder, changed)


@pytest.mark.parametrize("how", ["the site's button", "a killed publish"])
def test_a_publish_lost_and_a_resave_leave_the_next_version_to_be_made(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    lost_publish_then_resave(folder, changed, fake, service, how)
    assert codes(lambda: pushed(folder, fake, service)) == ["push.published"]
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, Terminal())
    assert refused.value.codes() == ["push.after-publish"]
    assert "makes its next version" in refused.value.refusals[0].action
    state = read_state(service, "full_toy")
    assert state is not None and state.published is not None
    assert f"md5:{state.published.files[BUNDLE].md5}" == fake.records[NUMBER].files[BUNDLE].checksum, \
        "push's record states the record's files, not the folder's"
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    assert drafted.changed and drafted.version == 2 and drafted.record != NUMBER
    assert published(folder, fake, service, confirm=drafted.doi) is not None
    assert fake.records[drafted.record].published and not fake.drafts()


def test_control_the_files_taken_from_the_plan_are_caught_reading_unchanged_and_no_version_made(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_publish, "listed", lambda files, plan: {
        file.key: Uploaded(file.size, file.md5, file.sha256) for file in plan.files})
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    lost_publish_then_resave(folder, changed, fake, service, "the site's button")
    assert codes(lambda: published(folder, fake, service, Terminal())) == ["push.after-publish"]
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    assert not drafted.changed and not fake.drafts() and "is unchanged since" in person.text(), \
        "the planted record of the plan's files is not seen"


def test_a_record_naming_anothers_published_record_is_not_adopted(toy: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
    from push_cases import saved_named

    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    root = tmp_path / "other"
    root.mkdir()
    other = saved_named(root, "another")
    made = pushed(other, fake, service)
    published(other, fake, service, confirm=made.doi)
    state = read_state(service, "full_toy")
    assert state is not None
    write_state(service, replace(state, record=made.record, doi=made.doi))
    assert codes(lambda: published(folder, fake, service, Terminal())) == ["push.draft-foreign"]
    state = read_state(service, "full_toy")
    assert state is not None and state.published is None


# --- F5: what a publish writes ---------------------------------------------------------------------------------------


SHARING_CHANGES = [
    ('title: "full_toy"', 'title: "A different title"', "A different title"),
    ("creators: [{family: Carberry", "creators: [{family: Stranger, given: Never Asked}, {family: Carberry", "Stranger"),
    ("contributors: []", "contributors: [{family: Unasked, given: Colleague, type: Supervisor}]", "Unasked"),
    ('description: ""', 'description: "A claim I typed and forgot."', "A claim I typed"),
]


@pytest.mark.parametrize(("old", "new", "word"), SHARING_CHANGES)
def test_a_script_publishes_only_what_the_draft_holds(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      old: str, new: str, word: str):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    revision = fake.records[NUMBER].revision
    rewritten(folder, old, new)
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, confirm=DOI)
    assert refused.value.codes() == ["push.metadata-changed"] and word in refused.value.refusals[0].message
    assert publishes(fake) == 0 and fake.records[NUMBER].revision == revision, "nothing written, nothing published"
    pushed(folder, fake, service)
    assert published(folder, fake, service, confirm=DOI) is not None and fake.records[NUMBER].published
    assert word in json.dumps(metadata(fake))


def test_a_script_is_refused_a_field_removed_from_sharing_yaml_too(toy: Path, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    rewritten(folder, "funding: []", "funding: [{funder: A small foundation}]")
    pushed(folder, fake, service)
    rewritten(folder, "funding: [{funder: A small foundation}]", "funding: []")
    assert codes(lambda: published(folder, fake, service, confirm=DOI)) == ["push.metadata-changed"]


def test_a_script_is_refused_writing_over_a_site_edit_but_may_keep_one(toy: Path, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    metadata(fake)["title"] = "Edited on the site"
    assert codes(lambda: published(folder, fake, service, confirm=DOI, overwrite_site_edits=True)) == \
        ["push.metadata-changed"]
    assert metadata(fake)["title"] == "Edited on the site" and publishes(fake) == 0
    assert published(folder, fake, service, confirm=DOI, keep_site_edits=True) is not None
    assert metadata(fake)["title"] == "Edited on the site" and fake.records[NUMBER].published


def test_control_no_hold_on_what_a_script_publishes_is_caught_naming_a_new_creator(toy: Path, tmp_path: Path,
                                                                                   monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_publish, "changed_since_draft", lambda found, body, state: [])
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    rewritten(folder, "creators: [{family: Carberry", "creators: [{family: Stranger, given: Never Asked}, {family: Carberry")
    published(folder, fake, service, confirm=DOI)
    assert "Stranger" in json.dumps(metadata(fake)) and fake.records[NUMBER].published, \
        "the planted publish of a name no draft held is not seen"


def test_in_a_terminal_each_change_is_shown_with_its_new_value_before_the_number(toy: Path, tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    for old, new, _ in SHARING_CHANGES:
        rewritten(folder, old, new)
    person = Terminal(typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None
    number = person.events.index(f"Type the record number to publish ({NUMBER}): ")
    before = "\n".join(person.events[:number])
    assert "! changed in sharing.yaml since the draft was written, and written to it now:" in before
    assert "metadata.title on the draft: full_toy" in before and "metadata.title to publish: A different title" in before
    assert "Stranger" in before and "Unasked" in before and "A claim I typed" in before
    assert before.index("metadata.title on the draft") < before.index("You state that everyone named")


def test_the_short_plan_names_the_title_and_the_contributors_always(toy: Path, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    rewritten(folder, "contributors: []", "contributors: [{family: Named, given: Colleague, type: Supervisor}]")
    person = Terminal()
    pushed(folder, fake, service, person)
    assert "  title: full_toy" in person.said and any(line.endswith("contributors Named, Colleague")
                                                      for line in person.said), person.text()
    person = Terminal(typed=(NUMBER,))
    published(folder, fake, service, person)
    assert "  title: full_toy" in person.said and any(line.endswith("contributors Named, Colleague")
                                                      for line in person.said), person.text()
    assert person.text().index("contributors Named") < person.text().index("You state that everyone named")


# --- F6: read again immediately before the request --------------------------------------------------------------------


def _title(fake: FakeRecords) -> None:
    metadata(fake)["title"] = "Edited while the prompt waited"


def _file(fake: FakeRecords) -> None:
    data = b"added on the site"
    fake.records[NUMBER].files["added.pdf"] = Entry("added.pdf", data, "completed",
                                                    "md5:" + hashlib.md5(data, usedforsecurity=False).hexdigest())


def _doi(fake: FakeRecords) -> None:
    fake.records[NUMBER].pids["doi"] = {"identifier": "10.1234/someone.elses", "provider": "external"}


def _access(fake: FakeRecords) -> None:
    embargoed(fake)


MOVES = {"title": _title, "added.pdf": _file, "10.1234/someone.elses": _doi, "restricted": _access}


@pytest.mark.parametrize("word", list(MOVES))
def test_a_draft_that_moved_while_the_number_waited_is_not_published(toy: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch, word: str):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,), on_typed=lambda: MOVES[word](fake))
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, person)
    assert refused.value.codes() == ["push.draft-moved"] and word in refused.value.refusals[0].message
    assert publishes(fake) == 0 and not fake.records[NUMBER].published


@pytest.mark.parametrize("word", list(MOVES))
def test_control_without_the_read_before_the_request_each_move_is_caught_published(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, word: str):
    monkeypatch.setattr(_publish, "_held_at_post", lambda plan, state, client: True)
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,), on_typed=lambda: MOVES[word](fake))
    try:
        published(folder, fake, service, person)
    except BuildRefused:
        pass  # found afterwards, when it cannot be undone
    assert publishes(fake) == 1 and fake.records[NUMBER].published, "the planted missing read is not seen"


def test_a_doi_changed_on_the_site_is_refused_at_the_first_read_and_a_push_takes_it(toy: Path, tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    _doi(fake)
    person = Terminal(typed=(NUMBER,))
    assert codes(lambda: published(folder, fake, service, person)) == ["push.draft-moved"]
    assert person.asked == [] and publishes(fake) == 0, "refused before it asks"
    assert codes(lambda: published(folder, fake, service, confirm=DOI)) == ["push.draft-moved"]
    again = Terminal()
    pushed(folder, fake, service, again)
    assert any(line.startswith("! the draft holds the DOI 10.1234/someone.elses") for line in again.said), again.text()
    state = read_state(service, "full_toy")
    assert state is not None and state.doi == "10.1234/someone.elses"


def test_a_draft_published_on_the_site_while_the_number_waited_is_recorded_and_not_sent(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,), on_typed=lambda: fake.publish(NUMBER))
    assert published(folder, fake, service, person) is not None
    assert publishes(fake) == 0 and "was published while the question waited" in person.text()


def test_a_draft_discarded_while_the_number_waited_is_not_published(toy: Path, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,), on_typed=lambda: fake.records.pop(NUMBER))
    assert codes(lambda: published(folder, fake, service, person)) == ["push.no-draft"]
    assert publishes(fake) == 0


# --- F7: a second concept, said or refused ---------------------------------------------------------------------------


def test_a_folder_whose_lock_names_its_record_is_told_and_is_never_published_as_a_second_concept(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    doi = f"10.5281/zenodo.{NUMBER}"
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=doi)
    _state.clear_state(service, "full_toy")  # a new machine, a cleared state
    full = Terminal()
    pushed(folder, fake, service, full, full=True)
    assert "! the folder's lock names the published record" in full.text()
    _state.clear_state(service, "full_toy")
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    assert any(line.startswith(f"! the folder's lock names the published record {doi} on zenodo") and
               "--publish refuses it" in line for line in person.said), person.text()
    assert codes(lambda: published(folder, fake, service, confirm=drafted.doi)) == ["push.second-concept"]
    assert publishes(fake) == 1 and not fake.records[drafted.record].published


def test_a_lock_naming_a_sandbox_record_going_to_production_is_said_and_not_refused(toy: Path, tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    lock = json.loads((folder / "mime.lock").read_text(encoding="utf-8"))
    lock["publication"] = {"service": "zenodo-sandbox", "record": "11", "doi": "10.5072/zenodo.11", "concept": "10",
                           "version": 1}
    (folder / "mime.lock").write_text(lock_text(lock), encoding="utf-8")
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    words = next(line for line in person.said if line.startswith("! the folder's lock names"))
    assert "10.5072/zenodo.11 on zenodo-sandbox" in words and "--publish refuses" not in words
    assert published(folder, fake, service, confirm=drafted.doi) is not None


# --- F8, F9, F22: a new version's files and number --------------------------------------------------------------------


def test_a_file_added_on_the_site_to_the_concepts_open_version_is_never_deleted_unasked(
        toy: Path, changed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    token = with_token(monkeypatch)
    fake = FakeRecords(token)
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: None)
    headers = {"Authorization": f"Bearer {token}", "Accept": push.RDM_TYPE}
    opened = str(json.loads(fake.handle("POST", f"{fake.api}/records/{NUMBER}/versions", headers, None).body)["id"])
    fake.handle("POST", f"{fake.api}/records/{opened}/draft/actions/files-import", headers, None)
    data = b"the author's own supplement"
    fake.records[opened].files["supplement.pdf"] = Entry("supplement.pdf", data, "completed", "md5:" + hashlib.md5(
        data, usedforsecurity=False).hexdigest())
    assert codes(lambda: pushed(folder, fake, service)) == ["push.draft-changed"]
    assert "supplement.pdf" in fake.records[opened].files


def test_a_version_published_on_the_site_between_pushs_two_is_counted(toy: Path, changed: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    token = with_token(monkeypatch)
    fake = FakeRecords(token)
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: None)
    headers = {"Authorization": f"Bearer {token}", "Accept": push.RDM_TYPE}
    opened = str(json.loads(fake.handle("POST", f"{fake.api}/records/{NUMBER}/versions", headers, None).body)["id"])
    fake.handle("POST", f"{fake.api}/records/{opened}/draft/actions/files-import", headers, None)
    fake.publish(opened)
    person = Terminal()
    drafted = pushed(folder, fake, service, person)
    assert drafted.version == 3 == fake.records[drafted.record].index
    assert metadata(fake, drafted.record)["version"] == "3"
    assert any(line.startswith(f"! {service.name} counts this as version 3") for line in person.said), person.text()
    assert card_problems(metadata(fake, drafted.record)["description"], drafted.doi, 3) == []
    held = published(folder, fake, service, confirm=drafted.doi)
    assert held is not None and held.version == 3


TABLE = b"a wall table's bytes, the same in every version" * 40


@pytest.fixture
def with_asset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The folder's lock names an open asset with no Zenodo location, in the cache: a file of the record."""
    from mime.assets import AssetRef

    cached = tmp_path / "cache" / "table.npz"
    cached.parent.mkdir()
    cached.write_bytes(TABLE)
    ref = AssetRef("table", hashlib.sha256(TABLE).hexdigest(), len(TABLE), "CC0-1.0",
                   ("https://example.org/tables/table.npz",), unlisted=True)
    monkeypatch.setattr(_plan, "folder_assets", lambda root: (ref,))
    monkeypatch.setattr(_plan, "locate", lambda held, folder=None: cached)
    return "table.npz"


@pytest.mark.parametrize("stop", ["an edit", "a killed import"])
def test_a_new_version_stopped_before_its_import_still_links_the_unchanged_file(
        toy: Path, changed: Path, tmp_path: Path, with_asset: str, monkeypatch: pytest.MonkeyPatch, stop: str):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    if stop == "an edit":
        published_then_changed(folder, changed, fake, service, lambda held: held.update(title="Corrected"))
        assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
        drafted = pushed(folder, fake, service, keep_site_edits=True)
    else:
        published_then_changed(folder, changed, fake, service, lambda held: None)
        fake.plant("kill-before", lambda method, path: path.endswith("/files-import"))
        with pytest.raises(Killed):
            pushed(folder, fake, service)
        drafted = pushed(folder, fake, service)
    assert fake.puts(with_asset) == 1 and fake.records[drafted.record].files[with_asset].linked == NUMBER


# --- F10 (D14): only the record's own location passes ---------------------------------------------------------------


ON_ZENODO = {"service": "zenodo", "record": "629525", "doi": "10.5281/zenodo.629525", "concept": "629524", "version": 1}
ON_SANDBOX = {**ON_ZENODO, "service": "zenodo-sandbox", "doi": "10.5072/zenodo.629525"}


@pytest.mark.parametrize(("location", "publication", "own"), [
    ("doi:10.5281/zenodo.629525/Arm.usda", ON_ZENODO, True),
    ("https://sandbox.zenodo.org/api/records/629525/files/Arm.usda/content", ON_SANDBOX, True),
    ("https://sandbox.zenodo.org/api/records/629525/files/Arm.usda/content", ON_ZENODO, False),
    ("doi:10.5072/zenodo.629525/Arm.usda", ON_SANDBOX, False),
    ("https://evil.example/records/629525/files/Arm.usda/content", ON_ZENODO, False),
    ("https://evil.example/records/629525/files/Arm.usda/content", ON_SANDBOX, False),
    ("https://evil.example/api/records/629525/files/Arm.usda/content", ON_SANDBOX, False),
    ("http://sandbox.zenodo.org/api/records/629525/files/Arm.usda/content", ON_SANDBOX, False),
    ("https://user:pw@sandbox.zenodo.org:8443/api/records/629525/files/x/content", ON_SANDBOX, False),
    ("https://sandbox.zenodo.org/api/records/629526/files/Arm.usda/content", ON_SANDBOX, False),
    ("doi:10.5281/zenodo.629525/../../elsewhere", ON_ZENODO, False),
    ("doi:10.5281/zenodo.629525/..", ON_ZENODO, False),
    ("doi:10.5281/zenodo.629526/Arm.usda", ON_ZENODO, False),
    ("doi:10.9999/evil/x", {"service": "somewhere", "doi": "10.9999/evil", "record": "1"}, False),
])
def test_only_the_records_own_location_is_its_own(location: str, publication: dict[str, object], own: bool):
    assert _own_location(location, publication) is own


def test_a_location_push_wrote_is_passed_in_and_no_other(tmp_path: Path):
    mesh = write_mesh(tmp_path / "sources", "Arm", 5000)
    terms = Terms(NO_REDISTRIBUTION, "Robot Maker Ltd", "presentation", ("https://example.org/maker/Arm.usda",))
    with same_reference_run():
        layered("referenced", [(mesh, terms)], licence="CC-BY-4.0").build().save(tmp_path / "referenced")
    folder = tmp_path / "referenced"
    lock = json.loads((folder / "mime.lock").read_text(encoding="utf-8"))
    lock["publication"] = ON_ZENODO
    origin = lock["assets"][0]["locations"]
    fake_host = "https://zenodo-records.invalid/api/records/629525/files/Arm.usda/content"
    for first, packs, given_ in (("doi:10.5281/zenodo.629525/Arm.usda", True, frozenset()),
                                 ("https://evil.example/records/629525/files/Arm.usda/content", False, frozenset()),
                                 (fake_host, False, frozenset()), (fake_host, True, frozenset({fake_host}))):
        lock["assets"][0]["locations"] = [first, *origin]
        (folder / "mime.lock").write_text(lock_text(lock), encoding="utf-8")
        if packs:
            assert pack(folder, out=tmp_path / "out.mimebundle", own_locations=given_).is_file()
        else:
            assert codes(lambda f=given_: pack(folder, out=tmp_path / "out.mimebundle", own_locations=f)) == \
                ["bundle.lock-terms"]


# --- F11: the metadata never holds the lock back ---------------------------------------------------------------------


def normalising(fake: FakeRecords, monkeypatch: pytest.MonkeyPatch) -> None:
    """The service keeps a field otherwise as it publishes."""
    real = fake._published

    def normalised(record: object) -> None:
        real(record)  # type: ignore[arg-type]
        record.body["metadata"]["title"] = "full_toy, as the service keeps it"  # type: ignore[attr-defined]

    monkeypatch.setattr(fake, "_published", normalised)


def test_a_field_the_service_normalises_at_publish_is_named_with_the_version_recorded_and_the_lock_written(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    doi = f"10.5281/zenodo.{NUMBER}"
    pushed(folder, fake, service)
    normalising(fake, monkeypatch)
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, confirm=doi)
    assert refused.value.codes() == ["push.metadata-read-back"]
    message = refused.value.refusals[0].message
    assert "is published and recorded" in message and "the folder names it" in message and "as the service keeps" in message
    assert pack(folder, out=tmp_path / "again.mimebundle").read_bytes() == fake.records[NUMBER].files[BUNDLE].data, \
        "mime pack makes the published bundle: the lock was not held back"
    state = read_state(service, "full_toy")
    assert state is not None and state.published is not None and state.published.differed == ["title"]
    person = Terminal()
    assert published(folder, fake, service, person) is not None, "a later command refuses nothing, and owes nothing"
    assert "! read back when it was published, its metadata differed from what the draft held: metadata.title" in \
        person.said and publishes(fake) == 1
    again = Terminal()
    assert not pushed(folder, fake, service, again).changed and "is unchanged since" in again.said[-1]


def test_a_lock_owed_after_a_kill_is_said_and_a_later_command_writes_it_whatever_the_metadata_says(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    doi = f"10.5281/zenodo.{NUMBER}"
    pushed(folder, fake, service)
    normalising(fake, monkeypatch)
    real = _publish.rewrite_lock

    def killed(folder_: Path, before: bytes, staged: bytes) -> None:
        raise Killed("between the record and the rewrite")

    monkeypatch.setattr(_publish, "rewrite_lock", killed)
    with pytest.raises(Killed):
        published(folder, fake, service, confirm=doi)
    monkeypatch.setattr(_publish, "rewrite_lock", real)
    assert _stage.publication_of((folder / "mime.lock").read_bytes()) is None
    person = Terminal()
    assert not pushed(folder, fake, service, person).changed
    assert any(line.startswith("! the folder's lock does not name the record yet") for line in person.said), person.text()
    assert published(folder, fake, service, Terminal()) is not None
    assert _stage.publication_of((folder / "mime.lock").read_bytes()) is not None


def test_what_the_service_adds_as_it_stores_a_draft_is_no_difference_after_publishing(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    real = fake._draft_answer

    def filled_in(record: object, rdm: bool, **given_: object) -> tuple[int, str, bytes]:
        status, kind, data = real(record, rdm, **given_)  # type: ignore[arg-type]
        held = json.loads(data)
        for right in held["metadata"]["rights"]:
            right["link"] = "https://spdx.org/licenses/added-by-the-service"
        held["metadata"]["creators"][0]["affiliations"][0]["id"] = "05gq02987"
        return status, kind, json.dumps(held).encode()

    monkeypatch.setattr(fake, "_draft_answer", filled_in)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None
    assert "its metadata, and the bundle's sha256" in person.text()


def test_sharing_yaml_edited_after_publishing_is_not_reported_as_what_the_record_differs_in(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=f"10.5281/zenodo.{NUMBER}")
    rewritten(folder, 'title: "full_toy"', 'title: "A better title"')
    person = Terminal()
    assert published(folder, fake, service, person) is not None
    assert "not what" not in person.text() and "edited on the record's page" not in person.text()


def test_an_edit_on_the_published_page_is_said_by_a_later_check_and_refuses_nothing(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, confirm=f"10.5281/zenodo.{NUMBER}")
    metadata(fake)["title"] = "Corrected on the page"
    person = Terminal()
    assert published(folder, fake, service, person) is not None
    assert "! edited on the record's page since it was published: metadata.title" in person.said


# --- F12, F13, F14 ---------------------------------------------------------------------------------------------------


def test_an_edit_made_during_an_upload_is_not_taken_for_pushs_own(toy: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    real = fake._file

    def editing(record: object, key: str, action: object, method: str, body: bytes, rdm: bool, kind: object
                ) -> tuple[int, str, bytes]:
        if action == "/content" and method == "PUT":
            metadata(fake)["title"] = "Edited during the upload"
        return real(record, key, action, method, body, rdm, kind)  # type: ignore[arg-type]

    monkeypatch.setattr(fake, "_file", editing)
    pushed(folder, fake, service)
    monkeypatch.setattr(fake, "_file", real)
    assert codes(lambda: pushed(folder, fake, service)) == ["push.metadata-edited"]
    assert metadata(fake)["title"] == "Edited during the upload"


def _modes_of(folder: Path, change: Callable[[int], int]) -> None:
    for path in [folder, *folder.rglob("*")]:
        os.chmod(path, change(path.stat().st_mode))


skip_read_only = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                    reason="a read-only folder does not hold back root, nor Windows' directories")


@skip_read_only
def test_a_read_only_folder_pushes_and_leaves_nothing_wedged(toy: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    _modes_of(folder, lambda mode: mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    try:
        drafted = pushed(folder, fake, service)
        assert drafted.changed and not pushed(folder, fake, service).changed
        before = len(fake.requests)
        assert codes(lambda: published(folder, fake, service, confirm=drafted.doi)) == ["push.lock-unwritten"]
        assert publishes(fake) == 0 and len(fake.requests) == before + 1, "refused before anything is written"
    finally:
        _modes_of(folder, lambda mode: mode | stat.S_IWUSR)
    assert published(folder, fake, service, confirm=drafted.doi) is not None


def test_a_lock_that_cannot_be_replaced_is_a_refusal_with_the_records_link_and_a_later_command_finishes(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = production(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    real = os.replace

    def failing(source: object, target: object) -> None:
        if str(target).endswith("mime.lock"):
            raise PermissionError("planted")
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", failing)
    person = Terminal(typed=(NUMBER,))
    assert codes(lambda: published(folder, fake, service, person)) == ["push.lock-unwritten"]
    assert person.said[-1] == f"{fake.site}/records/{NUMBER}"
    monkeypatch.setattr(os, "replace", real)
    assert published(folder, fake, service, Terminal()) is not None
    assert _stage.publication_of((folder / "mime.lock").read_bytes()) is not None


@given(st.text(min_size=1, max_size=400), st.text(min_size=1, max_size=400))
def test_a_question_never_shows_two_values_that_differ_as_two_lines_that_do_not(site: str, ours: str):
    shown_site, shown_ours = _edits.apart(site, ours)
    assert (shown_site == shown_ours) == (site == ours)


def test_a_seventh_creators_edit_is_shown_on_the_two_lines(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    people = ", ".join(f"{{family: Family{index}, given: Given{index}, affiliation: A rather long affiliation {index}}}"
                       for index in range(6))
    rewritten(folder, "creators: [{family: Carberry", f"creators: [{people}, {{family: Carberry")
    pushed(folder, fake, service)
    metadata(fake)["creators"][-1]["person_or_org"]["given_name"] = "Changed on the site"  # type: ignore[index]
    person = Terminal(answers=(True,))
    pushed(folder, fake, service, person)
    site = next(line for line in person.said if line.startswith("    on the site:"))
    file = next(line for line in person.said if line.startswith("    sharing.yaml:"))
    assert "Changed on the site" in site and "Josiah" in file


# --- the rest: answers, words and the write-back -----------------------------------------------------------------------


def test_a_publish_answered_2xx_listing_errors_is_a_refusal_naming_them(toy: Path, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    fake.plant("status", lambda method, path: path.endswith("/actions/publish"), status=202,
               headers={"content-type": push.RDM_TYPE},
               body={"errors": [{"field": "metadata.title", "messages": ["Too plain."]}]})
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, confirm=DOI)
    assert refused.value.codes() == ["push.rejected"] and "Too plain." in refused.value.refusals[0].message


def test_a_new_version_answered_2xx_listing_errors_is_a_refusal_naming_them(toy: Path, changed: Path, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: None)
    fake.plant("status", lambda method, path: path.endswith("/versions"), status=201,
               headers={"content-type": push.RDM_TYPE},
               body={"errors": [{"field": "metadata.title", "messages": ["Too plain."]}]})
    with pytest.raises(BuildRefused) as refused:
        pushed(folder, fake, service)
    assert "Too plain." in refused.value.refusals[0].message


def test_a_publish_refused_for_its_scope_names_deposit_actions(toy: Path, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    fake.plant("status", lambda method, path: path.endswith("/actions/publish"), status=403)
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, confirm=DOI)
    assert refused.value.codes() == ["push.token-scope"]
    assert "deposit:actions" in refused.value.refusals[0].message
    assert "a draft needs" not in refused.value.refusals[0].message


def test_keeping_and_overwriting_site_edits_cannot_be_asked_together(toy: Path, tmp_path: Path):
    with pytest.raises(SystemExit) as stopped:
        main(["push", str(toy), "--to", "zenodo-sandbox", "--keep-site-edits", "--overwrite-site-edits"])
    assert stopped.value.code == 2


@pytest.mark.parametrize("value", ["first second", "first second", "first\x85second", "first\x0csecond",
                                   "tabs\tand\rreturns"])
def test_keeping_a_title_leaves_another_keys_line_breaks_as_they_were(toy: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch, value: str):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    rewritten(folder, 'description: ""', f"description: {json.dumps(value, ensure_ascii=False)}")
    before = _sharing.read_sharing(folder).description
    pushed(folder, fake, service)
    held = str(metadata(fake)["description"])
    metadata(fake)["title"] = "Edited on the site"
    pushed(folder, fake, service, keep_site_edits=True)
    assert _sharing.read_sharing(folder).description == before and metadata(fake)["description"] == held


def test_a_crlf_sharing_yaml_keeps_its_line_endings_and_is_put_back_byte_for_byte(toy: Path, tmp_path: Path,
                                                                                  monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    path = folder / "sharing.yaml"
    pushed(folder, fake, service)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    metadata(fake)["title"] = "Edited on the site"
    pushed(folder, fake, service, keep_site_edits=True)
    written = path.read_bytes()
    assert b"title: Edited on the site\r\n" in written and written.count(b"\n") == written.count(b"\r\n")
    raw = path.read_bytes()
    metadata(fake)["title"] = "Edited again"
    metadata(fake)["creators"][0]["affiliations"] = [{"name": "A"}, {"name": "B"}]  # type: ignore[index]
    monkeypatch.setattr(_edits, "_others", lambda text, updates: {"a planted": text})
    assert codes(lambda: pushed(folder, fake, service, Terminal(answers=(False, False)), keep_site_edits=True)) != []
    assert path.read_bytes() == raw


@skip_read_only
def test_a_read_only_sharing_yaml_stops_the_keep_with_the_lines_to_copy(toy: Path, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    metadata(fake)["title"] = "Edited on the site"
    os.chmod(folder / "sharing.yaml", 0o444)
    with pytest.raises(BuildRefused) as refused:
        pushed(folder, fake, service, keep_site_edits=True)
    os.chmod(folder / "sharing.yaml", 0o644)
    assert refused.value.codes() == ["push.sharing"] and "title: Edited on the site" in refused.value.refusals[0].action
    assert metadata(fake)["title"] == "Edited on the site"


COMMENT = st.text(st.characters(exclude_categories=("Cs", "Cc", "Zl", "Zp")), max_size=24)


@given(st.lists(st.text(max_size=24), max_size=5), st.text(min_size=1, max_size=24), st.text(max_size=24), COMMENT)
def test_written_back_keys_read_back_and_another_keys_text_and_a_comment_stay(words: list[str], title: str, other: str,
                                                                              comment: str):
    scaffold = _sharing.scaffold("full_toy", "CC-BY-4.0").replace("creators: []", "creators:\n  - {family: A}\n")
    scaffold = scaffold.replace('description: ""\n', f"description: {json.dumps(other)}\n# {comment}\n")
    before = yaml.safe_load(scaffold)
    after = _edits.written_text(scaffold, {"keywords": words, "title": title})
    read = yaml.safe_load(after)
    assert read["keywords"] == words and read["title"] == title
    assert {key: value for key, value in read.items() if key not in ("keywords", "title")} == \
        {key: value for key, value in before.items() if key not in ("keywords", "title")}
    assert f"# {comment}\n" in after, "a comment outside the replaced lines stays"


@pytest.mark.parametrize("location", [
    "doi:10.5072/zenodo.1/a", "DOI:10.5072/zenodo.1/a", "https://doi.org/10.5072/zenodo.1/a",
    "https://www.doi.org/10.5072/zenodo.1/a", "https://doi.org:443/10.5072/zenodo.1/a",
    "https://doi.org/10.5072%2Fzenodo.1", "https://dx.doi.org/10.5072/zenodo.1", "http://doi.org/10.5072/zenodo.1"])
def test_the_test_prefix_is_read_in_every_spelling_of_its_address(location: str):
    assert assets.under_test_prefix(location)


@pytest.mark.parametrize("location", ["https://doi.org/10.5281/zenodo.1", "https://example.org/10.5072/zenodo.1",
                                      "doi:10.5281/zenodo.1/a", "https://sandbox.zenodo.org/api/records/1"])
def test_nothing_else_is_read_as_under_the_test_prefix(location: str):
    assert not assets.under_test_prefix(location)


@pytest.mark.parametrize("source", ["https://www.doi.org/10.5072/zenodo.629529", "https://doi.org:443/10.5072/zenodo.629529",
                                    "https://doi.org/10.5072%2Fzenodo.629529"])
def test_pull_refuses_a_sandbox_doi_in_every_spelling(source: str):
    with pytest.raises(BuildRefused) as refused:
        parse_source(source)
    assert refused.value.codes() == ["pull.sandbox-doi"] and "--from zenodo-sandbox 629529" in \
        refused.value.refusals[0].action


@pytest.mark.parametrize("status", [404, 410])
def test_a_published_record_gone_says_what_a_new_version_cannot_be_made_of(toy: Path, changed: Path, tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch, status: int):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    published_then_changed(folder, changed, fake, service, lambda held: None)
    fake.plant("status", lambda method, path: method == "POST" and path.endswith("/versions"), status=status)
    with pytest.raises(BuildRefused) as refused:
        pushed(folder, fake, service)
    assert refused.value.codes() == ["push.service"]
    assert "cannot be made" in refused.value.refusals[0].message and "delete push's record" in \
        refused.value.refusals[0].action


def test_a_publish_answered_503_after_it_acted_is_never_sent_again(toy: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    fake.plant("status-after", lambda method, path: path.endswith("/actions/publish"), status=503,
               headers={"retry-after": "1"})
    person = Terminal(typed=(NUMBER,))
    assert published(folder, fake, service, person) is not None
    assert publishes(fake) == 1 and "the answer to the publish was lost" in person.text()


def test_a_publish_answered_429_is_sent_again_since_the_service_never_read_it(toy: Path, tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    fake.plant("status", lambda method, path: path.endswith("/actions/publish"), status=429,
               headers={"retry-after": "0"})
    assert published(folder, fake, service, confirm=DOI) is not None
    assert publishes(fake) == 2 and fake.records[NUMBER].published


def test_status_says_the_access_the_record_would_have(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    embargoed(fake)
    said: list[str] = []
    push.status(folder, service, say=said.append, transport=FakeTransport(fake), pause=0)
    assert any("access once published: the record public, its files restricted, embargoed until 2030-01-01" in line
               for line in said), said


def test_stdin_closed_at_a_publish_is_a_refusal_not_a_crash(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", service)  # type: ignore[index]
    monkeypatch.setattr(_client, "HttpTransport", lambda *a, **k: FakeTransport(fake))
    monkeypatch.setattr("sys.stdin", None)
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--publish"]) == 2
    assert "push.unconfirmed" in capsys.readouterr().err
    assert publishes(fake) == 0


def test_an_interrupt_at_the_number_says_so_and_publishes_nothing(toy: Path, tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch,
                                                                  capsys: pytest.CaptureFixture[str]):
    fake, service = sandbox(monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", service)  # type: ignore[index]
    monkeypatch.setattr(_client, "HttpTransport", lambda *a, **k: FakeTransport(fake))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def interrupted(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--publish"]) == 130
    assert "interrupted" in capsys.readouterr().err and publishes(fake) == 0
