"""Publishing against the offline fake: the confirmation (each declared permission asked on its own, the record number
typed or ``--confirm <DOI>``), the checks afterwards, the folder's lock rewritten so ``pack`` makes the published
bundle again, versions under one concept, and a push with nothing changed said in one line.

Controls, each caught on every run:

- **a permission confirmed by ``--yes``**: ``--yes`` reaches the publish, and with no terminal the publish is refused
  before its request; with the refusal planted away, a declaration nobody confirmed is caught published;
- **a publish without the typed number**: ``y``, an empty answer and a wrong number publish nothing; with the number's
  check planted away, an answer of ``y`` is caught publishing;
- **a published bundle that ``mime pack <folder>`` does not reproduce**: after a production publish the folder packs to
  the bytes the fake holds; with the lock's rewrite planted away, it is caught packing other bytes;
- **a new version that re-uploads an unchanged file where linking was possible**: an asset the record holds, unchanged,
  is linked from the version before; with the import planted away, it is caught sent again.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy, with_token, write_sharing
from terms_cases import layered, same_reference_run, sha256_of, write_mesh
from zenodo_records import FakeRecords, FakeTransport

from mime import push
from mime.build import BuildRefused
from mime.push import _client, _publish, _stage
from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

NUMBER = "629525"


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


@pytest.fixture(scope="module")
def granted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder with a layer under a licence of its own and the permission that lets it travel."""
    root = tmp_path_factory.mktemp("granted")
    shell = write_mesh(root / "made", "shell", 900)
    own = CustomLicence("LicenseRef-Maker-1.0", "Maker Licence", "https://example.org/maker")
    permission = Permission("Maker Ltd", "Carberry, Josiah", (sha256_of(shell),), Scope(True, True, True),
                            "2026-11-01", Evidence("public_statement", "https://example.org/permission"))
    with same_reference_run():
        layered("granted", [(shell, Terms(own, "Maker Ltd", "presentation"))], licence="CC-BY-4.0",
                permissions=[permission]).build().save(root / "granted")
    write_sharing(root / "granted", extra='uploader: {name: "Carberry, Josiah"}\n')
    return root / "granted"


class Terminal:
    """A person at a terminal: each y/N question answered from ``answers`` (``y`` or not), the record number typed
    from ``typed``; every line said and every question kept, in order."""

    def __init__(self, answers: tuple[bool, ...] = (), typed: tuple[str, ...] = ()) -> None:
        self.answers = list(answers)
        self.typing = list(typed)
        self.said: list[str] = []
        self.asked: list[str] = []

    def say(self, line: str) -> None:
        self.said.append(line)

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        return self.answers.pop(0) if self.answers else False

    def typed(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.typing.pop(0) if self.typing else ""

    def text(self) -> str:
        return "\n".join(self.said)


def given(fake: FakeRecords, service: push.Service | None = None) -> dict[str, object]:
    return {"transport": FakeTransport(fake), "pause": 0, "sleep": lambda seconds: None}


def pushed(folder: Path, fake: FakeRecords, service: push.Service, person: Terminal | None = None,
           **extra: object) -> push.Drafted:
    held = person or Terminal()
    return push.push(folder, service, confirm=held.ask, say=held.say, **given(fake), **extra)  # type: ignore[arg-type]


def published(folder: Path, fake: FakeRecords, service: push.Service, person: Terminal | None,
              confirm: str | None = None) -> push.Published | None:
    """A publish: in a terminal (``person``), or from a script (``person`` None, with ``confirm``)."""
    held = person or Terminal()
    return push.publish(folder, service, confirm=confirm, ask=held.ask if person else None,
                        typed=held.typed if person else None, say=held.say, **given(fake))  # type: ignore[arg-type]


def publishes(fake: FakeRecords) -> int:
    return sum(1 for method, path, _, _ in fake.requests if method == "POST" and path.endswith("/actions/publish"))


def codes(action: Callable[[], object]) -> list[str]:
    with pytest.raises(BuildRefused) as refused:
        action()
    return refused.value.codes()


# --- the confirmation --------------------------------------------------------------------------------------------


def test_a_draft_asks_nothing_and_a_publish_asks_each_permission_then_the_typed_number(
        granted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(granted, tmp_path)
    person = Terminal()
    pushed(folder, fake, service, person)
    assert person.asked == [], "a private draft is made without a question"
    person = Terminal(answers=(True,), typed=(NUMBER,))
    held = published(folder, fake, service, person)
    assert held is not None and publishes(fake) == 1 and fake.records[NUMBER].published
    assert person.asked == ["Declare permission 1 publicly, under your name?",
                            f"Type the record number to publish ({NUMBER}): "]
    text = person.text()
    assert "You are declaring publicly, under your name, that Maker Ltd permitted you to redistribute this file" in text
    assert text.index("You are declaring publicly") < text.index("published 'granted'")
    assert person.said[-1] == f"{fake.site}/records/{NUMBER}", "a publish ends with the record's link"


def test_a_permission_declined_publishes_nothing(granted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(granted, tmp_path)
    pushed(folder, fake, fake.service())
    person = Terminal(answers=(False,), typed=(NUMBER,))
    assert published(folder, fake, fake.service(), person) is None
    assert publishes(fake) == 0 and person.asked == ["Declare permission 1 publicly, under your name?"]


def test_yes_and_confirm_never_declare_a_permission(granted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mime.cli import main

    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(granted, tmp_path)
    pushed(folder, fake, service)
    assert codes(lambda: published(folder, fake, service, None, confirm=f"10.5072/zenodo.{NUMBER}")) == \
        ["push.permission-unconfirmed"]
    monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", service)  # type: ignore[index]
    monkeypatch.setattr(_client, "HttpTransport", lambda *a, **k: FakeTransport(fake))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["push", str(folder), "--to", "zenodo-sandbox", "--publish", "--yes", "--confirm",
                 f"10.5072/zenodo.{NUMBER}"]) == 2
    assert publishes(fake) == 0, "neither --yes nor --confirm declared the permission"


def test_control_a_permission_confirmed_without_a_terminal_is_caught_published(
        granted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_publish, "_permissions", lambda plan, state, ask, say: True)
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(granted, tmp_path)
    pushed(folder, fake, fake.service())
    published(folder, fake, fake.service(), None, confirm=f"10.5072/zenodo.{NUMBER}")
    assert publishes(fake) == 1, "the planted publish of an unconfirmed declaration is not seen"


def test_a_permission_confirmed_in_a_terminal_lets_a_script_publish_exactly_those_bytes(
        granted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(granted, tmp_path)
    pushed(folder, fake, fake.service())
    assert published(folder, fake, fake.service(), Terminal(answers=(True,), typed=("",))) is None
    assert published(folder, fake, fake.service(), None, confirm=f"10.5072/zenodo.{NUMBER}") is not None
    assert publishes(fake) == 1


@pytest.mark.parametrize("typed", ["y", "", "629526", " 629525x"])
def test_a_publish_without_the_typed_number_publishes_nothing(toy: Path, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch, typed: str):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    person = Terminal(typed=(typed,))
    assert published(folder, fake, fake.service(), person) is None and publishes(fake) == 0
    assert person.said[-2].startswith("Not published") and person.said[-1] == f"{fake.site}/uploads/{NUMBER}"


def test_control_a_publish_whose_number_check_is_planted_away_is_caught(toy: Path, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_publish, "_number", lambda state, confirm, typed, say, service: True)
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    published(folder, fake, fake.service(), Terminal(typed=("y",)))
    assert publishes(fake) == 1, "the planted publish on an answer of y is not seen"


def test_confirm_must_be_the_drafts_doi_and_no_terminal_without_it_is_refused(toy: Path, tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    sent = len(fake.requests)
    assert codes(lambda: published(folder, fake, fake.service(), None, confirm="10.5072/zenodo.629527")) == \
        ["push.not-confirmed"]
    assert len(fake.requests) == sent, "a wrong --confirm is refused before any request"
    assert codes(lambda: published(folder, fake, fake.service(), None)) == ["push.unconfirmed"]
    assert publishes(fake) == 0
    assert published(folder, fake, fake.service(), None, confirm=f"10.5072/zenodo.{NUMBER}") is not None


def test_a_folder_changed_since_its_draft_is_not_published(toy: Path, changed: Path, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    _replace_content(folder, changed)
    assert codes(lambda: published(folder, fake, fake.service(), Terminal(typed=(NUMBER,)))) == \
        ["push.folder-changed"]
    assert publishes(fake) == 0


def test_without_a_draft_there_is_nothing_to_publish(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    assert codes(lambda: published(copy_of(toy, tmp_path), fake, fake.service(), Terminal())) == ["push.no-draft"]
    assert fake.requests == []


# --- after publishing ----------------------------------------------------------------------------------------------


def _replace_content(folder: Path, source: Path) -> None:
    """The folder's saved form replaced by ``source``'s, its sharing.yaml kept."""
    for path in folder.iterdir():
        if path.name != "sharing.yaml":
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    for path in source.iterdir():
        target = folder / path.name
        shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)


def production(fake: FakeRecords, monkeypatch: pytest.MonkeyPatch) -> push.Service:
    """The fake as a production service; its token is the one ``with_token(monkeypatch, "zenodo")`` made it with."""
    del monkeypatch
    return fake.service("zenodo", production=True)


def held_bundle(fake: FakeRecords, record: str = NUMBER) -> bytes:
    data = fake.records[record].files["full_toy.mimebundle"].data
    assert data is not None
    return data


def test_a_production_publish_rewrites_the_lock_so_pack_makes_the_published_bundle(
        toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mime.experiment import Experiment
    from mime.saved import pack

    fake = FakeRecords(with_token(monkeypatch, "zenodo"), prefix="10.5281")
    service = production(fake, monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    person = Terminal(typed=(NUMBER,))
    held = published(folder, fake, service, person)
    assert held is not None and held.doi == f"10.5281/zenodo.{NUMBER}" and held.concept_doi == "10.5281/zenodo.629524"
    assert pack(folder, out=tmp_path / "again.mimebundle").read_bytes() == held_bundle(fake), \
        "mime pack <folder> makes the published bundle again, byte for byte"
    lock = _stage.publication_of((folder / "mime.lock").read_bytes())
    assert lock is not None and lock["doi"] == held.doi and lock["record"] == NUMBER
    Experiment.load(folder)  # the folder still loads: the publication is outside the input hash
    assert "the folder's lock now names the record" in person.text()


def test_control_without_the_lock_rewrite_pack_is_caught_making_other_bytes(toy: Path, tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch):
    from mime.saved import pack

    monkeypatch.setattr(_publish, "rewrite_lock", lambda folder, before, staged: None)
    fake = FakeRecords(with_token(monkeypatch, "zenodo"), prefix="10.5281")
    service = production(fake, monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, Terminal(typed=(NUMBER,)))
    assert pack(folder, out=tmp_path / "again.mimebundle").read_bytes() != held_bundle(fake), \
        "the planted skip of the rewrite is not seen"


def test_a_lock_changed_between_staging_and_publish_is_left_alone(toy: Path, tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch, "zenodo"), prefix="10.5281")
    service = production(fake, monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    real = _publish.finish

    def edited_meanwhile(plan, state, client, say, **keywords):  # noqa: ANN001, ANN202
        (folder / "mime.lock").write_bytes(plan.lock + b" ")
        return real(plan, state, client, say, **keywords)

    monkeypatch.setattr(_publish, "finish", edited_meanwhile)
    assert codes(lambda: published(folder, fake, service, Terminal(typed=(NUMBER,)))) == ["push.lock-changed"]
    assert (folder / "mime.lock").read_bytes().endswith(b" "), "push left the changed lock as it was"
    assert fake.records[NUMBER].published


def test_a_sandbox_publish_never_writes_into_the_folder(toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    pushed(folder, fake, fake.service())
    assert published(folder, fake, fake.service(), Terminal(typed=(NUMBER,))) is not None
    assert {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before
    staged = _stage.publication_of(_lock_in(held_bundle(fake)))
    assert staged is not None and staged["doi"] == f"10.5072/zenodo.{NUMBER}", "the bundle's lock names its record"


def _lock_in(bundle: bytes) -> bytes:
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        return archive.read("experiment/mime.lock")


def test_a_publish_whose_answer_was_lost_is_never_sent_again(toy: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    fake.plant("drop-after", lambda method, path: path.endswith("/actions/publish"))
    person = Terminal(typed=(NUMBER,))
    assert published(folder, fake, fake.service(), person) is not None
    assert publishes(fake) == 1 and "the answer to the publish was lost" in person.text()
    fake.plant("kill-after", lambda method, path: False)
    again = Terminal(typed=(NUMBER,))
    assert published(folder, fake, fake.service(), again) is not None and publishes(fake) == 1
    assert again.asked == [], "a publish found done is checked again, never asked or sent again"


def test_a_publish_the_site_completed_is_recorded_without_a_publish_request(toy: Path, tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    fake.publish(NUMBER)
    assert codes(lambda: pushed(folder, fake, fake.service())) == ["push.published"]
    assert published(folder, fake, fake.service(), Terminal()) is not None and publishes(fake) == 0


def test_the_checks_after_publishing_find_a_record_that_differs(toy: Path, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch, "zenodo"), prefix="10.5281")
    service = production(fake, monkeypatch)
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    before = (folder / "mime.lock").read_bytes()
    real = fake._published

    def tampered(record) -> None:  # noqa: ANN001
        real(record)
        record.body["metadata"]["title"] = "Changed in publishing"
        record.files["NOTICE"].data = b"other bytes"

    monkeypatch.setattr(fake, "_published", tampered)
    with pytest.raises(BuildRefused) as refused:
        published(folder, fake, service, Terminal(typed=(NUMBER,)))
    assert refused.value.codes() == ["push.after-publish"]
    message = refused.value.refusals[0].message
    assert "NOTICE" in message and "title" in message
    assert (folder / "mime.lock").read_bytes() == before, "a record that differs never has its lock written"


# --- versions ------------------------------------------------------------------------------------------------------


TABLE = b"a wall table's bytes, the same in every version" * 40


@pytest.fixture
def with_asset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """The folder's lock names an open asset with no Zenodo location, in the cache: a file of the record (as MIME's
    wall tables are, once they are referenced), whose name in the record is ``table.npz``."""
    import hashlib

    from mime.assets import AssetRef
    from mime.push import _plan

    cached = tmp_path / "cache" / "table.npz"
    cached.parent.mkdir()
    cached.write_bytes(TABLE)
    ref = AssetRef("table", hashlib.sha256(TABLE).hexdigest(), len(TABLE), "CC0-1.0",
                   ("https://example.org/tables/table.npz",), unlisted=True)
    monkeypatch.setattr(_plan, "folder_assets", lambda root: (ref,))
    monkeypatch.setattr(_plan, "locate", lambda held, folder=None: cached)
    return "table.npz"


def test_a_changed_folder_is_a_new_version_whose_unchanged_files_are_linked(toy: Path, changed: Path,
                                                                           tmp_path: Path, with_asset: str,
                                                                           monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    assert set(fake.records[NUMBER].files) == {"full_toy.mimebundle", "NOTICE", with_asset}
    published(folder, fake, service, Terminal(typed=(NUMBER,)))
    person = Terminal()
    same = pushed(folder, fake, service, person)
    assert not same.changed and person.said == [
        f"'full_toy' is unchanged since 10.5072/zenodo.{NUMBER} (version 1 on {service.name}), so no version is "
        f"made: {fake.site}/records/{NUMBER}"]
    _replace_content(folder, changed)
    drafted = pushed(folder, fake, service)
    version = fake.records[drafted.record]
    assert drafted.version == 2 and version.parent == fake.records[NUMBER].parent and version.index == 2
    assert fake.puts(with_asset) == 1 and version.files[with_asset].data == TABLE, \
        "the unchanged asset is linked from version 1, never uploaded again"
    assert version.files[with_asset].linked == NUMBER and fake.puts("NOTICE") == 2, "the changed NOTICE is sent"
    assert published(folder, fake, service, Terminal(typed=(drafted.record,))) is not None
    lock = _stage.publication_of(_lock_in(held_bundle(fake, drafted.record)))
    assert lock is not None and lock["version"] == 2 and lock["concept_doi"] == fake.concept_doi(NUMBER)


def test_control_a_new_version_without_the_import_is_caught_uploading_the_unchanged_file(
        toy: Path, changed: Path, tmp_path: Path, with_asset: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(push.Client, "import_files", lambda self, record: {})
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, Terminal(typed=(NUMBER,)))
    _replace_content(folder, changed)
    pushed(folder, fake, service)
    assert fake.puts(with_asset) == 2, "the planted upload of the unchanged asset is not seen"


def test_discarding_a_new_versions_draft_keeps_the_published_version(toy: Path, changed: Path, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    service = fake.service()
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, service)
    published(folder, fake, service, Terminal(typed=(NUMBER,)))
    _replace_content(folder, changed)
    drafted = pushed(folder, fake, service)
    assert push.discard(folder, service, yes=True, transport=FakeTransport(fake), pause=0)
    assert drafted.record not in fake.records and fake.records[NUMBER].published
    again = pushed(folder, fake, service)
    assert again.version == 2 and again.record != drafted.record


# --- one push's words ----------------------------------------------------------------------------------------------


def test_the_default_output_is_short_and_the_full_plan_is_behind_a_flag(toy: Path, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    short, full = Terminal(), Terminal()
    pushed(copy_of(toy, tmp_path / "a"), fake, fake.service(), short)
    pushed(copy_of(toy, tmp_path / "b"), FakeRecords(with_token(monkeypatch)), fake.service(), full, full=True)
    assert len(short.said) <= 8 and len(" ".join(short.said).split()) <= 100, short.said
    assert "Inside the bundle (4 files):" in full.text() and "Inside the bundle" not in short.text()
    assert short.said[-1] == f"{fake.site}/uploads/{NUMBER}"


def test_a_push_with_nothing_changed_writes_nothing_and_says_one_line(toy: Path, tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    fake = FakeRecords(with_token(monkeypatch))
    folder = copy_of(toy, tmp_path)
    pushed(folder, fake, fake.service())
    sent = len(fake.requests)
    person = Terminal()
    drafted = pushed(folder, fake, fake.service(), person)
    assert not drafted.changed and len(person.said) == 1 and person.said[0].endswith(f"{fake.site}/uploads/{NUMBER}")
    assert all(method == "GET" for method, _, _, _ in fake.requests[sent:]), "nothing is written"
