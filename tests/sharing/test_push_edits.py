"""Edits made on the site, never lost, and the metadata real users need, against the offline fake.

Each field edited on the site since push last wrote it is shown against ``sharing.yaml`` and, field by field, kept
(written back into ``sharing.yaml``, its other lines kept) or written over; a field ``sharing.yaml`` cannot hold is
written over or stops the push. A write-back that does not read back is undone. Funding, a conference and any other
field of the record (``fields:``) reach the record as the records API holds them, each checked by its path first.

The control (a site edit written over without being shown) is ``test_push_draft.py``'s; here, a write-back whose
reading is planted away is caught leaving ``sharing.yaml`` written wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from push_cases import copy_of, saved_toy, with_token, write_sharing
from zenodo_records import FakeRecords, FakeTransport

from mime import push
from mime.build import BuildRefused
from mime.push import _edits, _sharing

NUMBER = "629525"
EXTRA = """funding:
  - {funder: European Commission, ror: 00k4n6c32, award: "101001234", title: A microrobotics award, url: "https://example.org/award"}
  - {funder: A small foundation}
conference: {title: Conference on Microrobots, acronym: CMR, dates: 3-7 May 2027, place: Delft, part: "2"}
fields:
  languages: [{id: eng}]
  dates: [{date: 2026-10, type: {id: collected}, description: the reference run}]
  journal:journal: {title: A Journal, volume: "4"}
  thesis:university: A University
"""


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


class Terminal:
    def __init__(self, *answers: bool) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []
        self.said: list[str] = []

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        return self.answers.pop(0)


def pushed(folder: Path, fake: FakeRecords, person: Terminal | None = None, **extra: object) -> push.Drafted:
    held = person or Terminal()
    return push.push(folder, fake.service(), confirm=held.ask if person else None, say=held.said.append,
                     transport=FakeTransport(fake), pause=0, **extra)  # type: ignore[arg-type]


def codes(action: Callable[[], object]) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        action()
    return refused.value.codes()


def metadata(fake: FakeRecords, record: str = NUMBER) -> dict[str, object]:
    return fake.records[record].body["metadata"]  # type: ignore[return-value]


def sharing_text(folder: Path) -> str:
    return (folder / "sharing.yaml").read_text(encoding="utf-8")


# --- edits on the site ---------------------------------------------------------------------------------------------


def test_an_edit_on_the_site_is_shown_and_kept_into_sharing_yaml_when_asked(toy: Path, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    creators = metadata(fake)["creators"]
    creators[0]["person_or_org"]["given_name"] = "J."  # type: ignore[index]
    creators.append({"person_or_org": {"type": "personal", "family_name": "Second", "given_name": "Ann"}})  # type: ignore[attr-defined]
    person = Terminal(False)
    drafted = pushed(folder, fake, person)
    assert len(person.asked) == 1 and person.asked[0].startswith("Write sharing.yaml's over the site's?")
    assert "! metadata.creators was edited on the site since push last wrote it:" in person.said
    held = _sharing.read_sharing(folder)
    assert [(each.family, each.given, each.orcid) for each in held.creators] == \
        [("Carberry", "J.", "0000-0002-1825-0097"), ("Second", "Ann", None)]
    assert "# The record's metadata for `mime push`" in sharing_text(folder), "sharing.yaml's other lines are kept"
    assert drafted.changed and [each["person_or_org"]["given_name"] for each in metadata(fake)["creators"]] == \
        ["J.", "Ann"]  # type: ignore[index,union-attr]
    again = Terminal()
    assert not pushed(folder, fake, again).changed and again.asked == [], "kept, the edit is no longer at stake"


def test_an_edit_on_the_site_is_written_over_when_answered_yes(toy: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    before = sharing_text(folder)
    metadata(fake)["title"] = "Edited on the site"
    pushed(folder, fake, Terminal(True))
    assert metadata(fake)["title"] == "full_toy" and sharing_text(folder) == before


def test_a_field_mime_builds_edited_on_the_site_is_written_over_or_stops_the_push(toy: Path, tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    metadata(fake)["publisher"] = "Someone else"
    person = Terminal(False)
    assert codes(lambda: pushed(folder, fake, person)) == ["push.metadata-edited"]
    assert person.asked[0].startswith("sharing.yaml cannot hold it (MIME writes it)")
    assert metadata(fake)["publisher"] == "Someone else", "a stopped push writes nothing"
    assert codes(lambda: pushed(folder, fake, keep_site_edits=True)) == ["push.metadata-edited"]
    pushed(folder, fake, Terminal(True))
    assert metadata(fake)["publisher"] == "Zenodo"


def test_keywords_funding_a_conference_and_other_fields_edited_on_the_site_are_kept(toy: Path, tmp_path: Path,
                                                                                    monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    held = metadata(fake)
    held["subjects"] = [*held["subjects"], {"subject": "helical swimmer"}]  # type: ignore[misc]
    held["funding"] = [{"funder": {"id": "00k4n6c32"}, "award": {"title": {"en": "An award"}, "number": "42"}}]
    held["languages"] = [{"id": "eng", "title": {"en": "English"}}]
    fake.records[NUMBER].body["custom_fields"] = {"meeting:meeting": {"title": "A meeting", "place": "Delft"}}
    person = Terminal()
    pushed(folder, fake, person, keep_site_edits=True)
    assert person.asked == []
    kept = _sharing.read_sharing(folder)
    assert kept.keywords == ("helical swimmer",) and kept.conference == {"title": "A meeting", "place": "Delft"}
    assert [(each.funder, each.ror, each.award, each.title) for each in kept.funding] == \
        [("00k4n6c32", "00k4n6c32", "42", "An award")]
    assert kept.fields == {"languages": [{"id": "eng"}]}
    assert not pushed(folder, fake).changed, "every edit kept, the draft is what push would write"


def test_a_write_back_that_does_not_read_back_puts_sharing_yaml_back(toy: Path, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    before, revision = sharing_text(folder), fake.records[NUMBER].revision
    metadata(fake)["title"] = "Edited on the site"
    monkeypatch.setattr(_edits, "written_text", lambda text, updates: text.replace("full_toy", "something else"))
    with pytest.raises(BuildRefused) as refused:
        pushed(folder, fake, keep_site_edits=True)
    assert refused.value.codes() == ["push.sharing"] and "title: Edited on the site" in refused.value.refusals[0].action
    assert sharing_text(folder) == before and fake.records[NUMBER].revision == revision


def test_control_a_write_back_whose_reading_is_planted_away_is_caught_writing_sharing_yaml_wrong(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    metadata(fake)["title"] = "Edited on the site"
    monkeypatch.setattr(_edits, "written_text", lambda text, updates: text.replace("full_toy", "something else"))
    monkeypatch.setattr(_edits, "differs", lambda name, site, ours: False)
    try:
        pushed(folder, fake, keep_site_edits=True)
    except BuildRefused:
        pass
    assert "something else" in sharing_text(folder), "the planted write-back is not seen"


TEXT = st.text(alphabet=st.characters(categories=("L", "N", "P", "Zs")), min_size=1, max_size=24)


@given(st.lists(TEXT, max_size=5), TEXT)
def test_written_back_keys_read_back_as_written_and_every_other_key_as_it_was(words: list[str], title: str):
    scaffold = _sharing.scaffold("full_toy", "CC-BY-4.0").replace("creators: []", "creators:\n  - {family: A}\n")
    after = _edits.written_text(scaffold, {"keywords": words, "title": title})
    before, read = yaml.safe_load(scaffold), yaml.safe_load(after)
    assert read["keywords"] == words and read["title"] == title
    assert {key: value for key, value in read.items() if key not in ("keywords", "title")} == \
        {key: value for key, value in before.items() if key not in ("keywords", "title")}


def test_an_edit_on_the_site_is_settled_before_a_publish_publishes_it(toy: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    metadata(fake)["title"] = "Edited on the site"
    events: list[str] = []

    def answer(question: str) -> bool:
        events.append(f"asked {question}")
        return False

    def typed(prompt: str) -> str:
        events.append(f"typed {prompt}")
        return NUMBER

    held = push.publish(folder, fake.service(), ask=answer, typed=typed, say=lambda line: events.append(line),
                        transport=FakeTransport(fake), pause=0)
    assert held is not None and fake.records[NUMBER].published
    assert metadata(fake)["title"] == "Edited on the site" and _sharing.read_sharing(folder).title == \
        "Edited on the site"
    asked = next(index for index, event in enumerate(events) if event.startswith("asked Write sharing.yaml's"))
    plans = [index for index, event in enumerate(events) if event.startswith("Plan for 'full_toy'")]
    assert asked < plans[-1] < events.index(f"typed Type the record number to publish ({NUMBER}): "), \
        "the edit is settled, then the plan published is shown, then the number is asked"


def test_an_edit_of_the_published_record_is_asked_about_in_its_next_version(toy: Path, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    from mime.testing.toys import full_toy

    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake)
    push.publish(folder, fake.service(), confirm=f"10.5072/zenodo.{NUMBER}", transport=FakeTransport(fake), pause=0)
    metadata(fake)["title"] = "Corrected on the record's page"
    experiment = full_toy()
    experiment.licence("CC-BY-4.0")
    experiment.save(folder, steps=9)
    person = Terminal(False)
    drafted = pushed(folder, fake, person)
    assert drafted.version == 2 and len(person.asked) == 1
    assert metadata(fake, drafted.record)["title"] == "Corrected on the record's page"


# --- the metadata real users need ----------------------------------------------------------------------------------


def test_funding_a_conference_and_other_fields_reach_the_record_as_the_records_api_holds_them(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    (folder / "sharing.yaml").write_text(sharing_text(folder).replace("funding: []\n", "").replace(
        "fields: {}\n", "") + EXTRA, encoding="utf-8")
    pushed(folder, fake, full=True)
    held = metadata(fake)
    assert held["funding"] == [
        {"funder": {"id": "00k4n6c32"}, "award": {"title": {"en": "A microrobotics award"}, "number": "101001234",
                                                   "identifiers": [{"scheme": "url",
                                                                    "identifier": "https://example.org/award"}]}},
        {"funder": {"name": "A small foundation"}}]
    assert held["languages"] == [{"id": "eng"}] and held["dates"][0]["type"] == {"id": "collected"}  # type: ignore[index]
    custom = fake.records[NUMBER].body["custom_fields"]
    assert custom == {"meeting:meeting": {"title": "Conference on Microrobots", "acronym": "CMR",
                                          "dates": "3-7 May 2027", "place": "Delft", "session_part": "2"},
                      "journal:journal": {"title": "A Journal", "volume": "4"}, "thesis:university": "A University"}


@pytest.mark.parametrize(("lines", "where"), [
    ("funding: [{funder: X, ror: not-a-ror}]", "funding[0].ror"),
    ("funding: [{ror: 00k4n6c32}]", "funding[0].funder"),
    ("funding: [{funder: X, title: An award}]", "funding[0]"),
    ("conference: {acronym: CMR}", "conference.title"),
    ("conference: {title: A, room: 3}", "conference.room"),
    ("fields: {languages: [{id: english}]}", "fields.languages[0].id"),
    ("fields: {dates: [{date: 2026-10, type: {id: when}}]}", "fields.dates[0].type.id"),
    ("fields: {dates: [{date: last week, type: {id: collected}}]}", "fields.dates[0].date"),
    ("fields: {title: Another title}", "fields.title"),
    ("fields: {publisher: Me}", "fields.publisher"),
    ("fields: {rights: []}", "fields.rights"),
    ("fields: {colour: blue}", "fields.colour"),
    ("fields: {journal:journal: {volume: '4'}}", "fields.journal:journal.title"),
    ("fields: {references: []}", "fields.references"),
])
def test_each_problem_of_the_metadata_is_refused_by_its_path(toy: Path, tmp_path: Path, lines: str, where: str):
    folder = copy_of(toy, tmp_path)
    text = sharing_text(folder).replace("funding: []\n", "").replace("fields: {}\n", "")
    (folder / "sharing.yaml").write_text(text + lines + "\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        _sharing.read_sharing(folder)
    assert [each.code for each in refused.value.refusals] == ["push.sharing"]
    assert f"sharing.yaml, {where}: " in refused.value.refusals[0].message, refused.value.refusals[0].message


@pytest.mark.parametrize(("field", "owner"), [
    ("metadata.funding.0.funder", "sharing"), ("metadata.languages.0.id", "sharing"),
    ("custom_fields.journal:journal.title", "sharing"), ("custom_fields.meeting:meeting", "sharing"),
    ("metadata.version", "mime"), ("metadata.dates.0.type", "sharing")])
def test_every_field_the_author_writes_is_theirs_when_the_service_refuses_it(field: str, owner: str):
    assert push.field_owner(field) == owner
