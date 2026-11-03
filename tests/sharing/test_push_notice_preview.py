"""The NOTICE a record holds beside its bundle, under a name Zenodo previews: ``NOTICE.txt`` (Zenodo picks a previewer by
a file's extension, its plain text one taking ``.txt``; the NOTICE is plain text whose indented lines a Markdown renderer
would run together), byte for byte the NOTICE the bundle carries as its member ``NOTICE``, which ``mime card --notice``
writes of the folder the bundle unpacks to (its lock the staged lock that names the record). A publish makes it the record's preview (``files.default_preview``), once the draft holds it (InvenioRDM refuses
a preview naming a file the draft does not hold, as its source reads); a preview chosen on the draft's page is kept.
``mime pack``'s bundle is as it was, and ``mime pull`` downloads the bundle alone: the NOTICE comes inside it.

Controls, each caught on every run:

- **the NOTICE under its old name**: with the record's key planted back to ``NOTICE``, the record is caught holding a
  file Zenodo previews with nothing;
- **a preview set before the draft holds the file**: with the preview planted into every write, the first push is
  caught refused by the fake, as InvenioRDM refuses it;
- **a preview chosen on the page written over**: with the draft's own choice planted away, the published record is
  caught previewing the NOTICE in place of the person's choice;
- **a publish that cannot be confirmed writing the preview** (the review of #112, I7): with the preview written on every
  publish, one refused for want of a terminal or ``--confirm`` is caught writing the draft first;
- **a NOTICE replaced unsaid** (I5): with the deletion left out of what a push says, a draft #100's push made (its
  NOTICE under the old name), pushed again, is caught saying only what it uploaded.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy, with_token, write_sharing
from zenodo_records import FakeRecords, served

from mime import push
from mime.build import BuildRefused
from mime.cli import main
from mime.push import _draft, _plan
from mime.saved import pull

NUMBER = "629525"
DOI = f"10.5072/zenodo.{NUMBER}"
#: The file extensions Zenodo's previewers take (invenio-previewer's entry points, read in its source): the NOTICE's
#: must be one of them.
PREVIEWED = (".txt", ".md")


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    token = with_token(monkeypatch)
    monkeypatch.setattr(push._client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with served(token) as fake:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", fake.service())  # type: ignore[index]
        yield fake


def published(toy: Path, fake: FakeRecords, tmp_path: Path, *, chosen: str | None = None) -> Path:
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    push.push(folder, service, pause=0)
    if chosen is not None:  # chosen on the draft's page
        fake.records[NUMBER].body["files"] = {"enabled": True, "default_preview": chosen}
    push.publish(folder, service, confirm=DOI, pause=0)
    return folder


def notice_on_the_record(fake: FakeRecords) -> bytes:
    entries = [entry for key, entry in fake.records[NUMBER].files.items() if key.startswith("NOTICE")]
    assert len(entries) == 1 and entries[0].data is not None
    assert entries[0].key.endswith(PREVIEWED), f"Zenodo previews {entries[0].key} with nothing"
    return entries[0].data


def preview(fake: FakeRecords) -> object:
    files = fake.records[NUMBER].body.get("files")
    return files.get("default_preview") if isinstance(files, dict) else None


def test_the_record_holds_the_bundles_notice_as_notice_txt_and_previews_it(toy: Path, sandbox: FakeRecords,
                                                                         tmp_path: Path,
                                                                         capsys: pytest.CaptureFixture[str]):
    folder = published(toy, sandbox, tmp_path)
    held = notice_on_the_record(sandbox)
    assert set(sandbox.records[NUMBER].files) == {"full_toy.mimebundle", "NOTICE.txt"}
    bundle = sandbox.records[NUMBER].files["full_toy.mimebundle"].data
    assert bundle is not None
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert "NOTICE" in archive.namelist() and archive.read("NOTICE") == held, "the bundle's member keeps its name"
    assert preview(sandbox) == "NOTICE.txt"
    del folder, capsys


def test_a_preview_chosen_on_the_drafts_page_is_kept(toy: Path, sandbox: FakeRecords, tmp_path: Path):
    published(toy, sandbox, tmp_path, chosen="full_toy.mimebundle")
    assert preview(sandbox) == "full_toy.mimebundle"


def test_a_pull_downloads_the_bundle_alone_and_its_notice_is_the_records(toy: Path, sandbox: FakeRecords,
                                                                         tmp_path: Path,
                                                                         capsys: pytest.CaptureFixture[str]):
    published(toy, sandbox, tmp_path)
    held = notice_on_the_record(sandbox)
    sandbox.seen.clear()
    into = pull(NUMBER, into=tmp_path / "pulled", from_service="zenodo-sandbox")
    assert not any("NOTICE" in path for _, path, _, _ in sandbox.seen), "the NOTICE comes inside the bundle"
    assert not (into / "NOTICE").exists() and not (into / "NOTICE.txt").exists()
    capsys.readouterr()
    assert main(["card", str(into), "--notice", "--out", str(tmp_path / "notice")]) == 0
    assert (tmp_path / "notice").read_bytes() == held, "the pulled folder's NOTICE is the record's, byte for byte"


def test_control_the_notice_under_its_old_name_is_caught_unpreviewed(toy: Path, sandbox: FakeRecords, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_plan, "NOTICE_KEY", "NOTICE")
    monkeypatch.setattr(_draft, "NOTICE_KEY", "NOTICE")
    published(toy, sandbox, tmp_path)
    with pytest.raises(AssertionError, match="previews NOTICE with nothing"):
        notice_on_the_record(sandbox)


def test_control_a_preview_set_before_the_draft_holds_the_file_is_refused_as_invenio_refuses_it(
        toy: Path, sandbox: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real = _draft.Drafting._write  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_draft.Drafting, "_write", lambda self, record, found, over, preview=False: real(
        self, record, found, over, preview=True))
    with pytest.raises(BuildRefused) as refused:
        published(toy, sandbox, tmp_path)
    assert "files.default_preview" in str(refused.value), "the planted defect is not seen"


def test_control_a_choice_on_the_page_written_over_is_caught(toy: Path, sandbox: FakeRecords, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft, "chosen_preview", lambda draft: None)
    real = _draft.Drafting._write  # pyright: ignore[reportPrivateUsage]

    def written(self: _draft.Drafting, record: str, found: dict[str, object], over: list[str],
                preview: bool = False) -> object:
        return real(self, record, {**found, "files": {}}, over, preview=preview)

    monkeypatch.setattr(_draft.Drafting, "_write", written)
    published(toy, sandbox, tmp_path, chosen="full_toy.mimebundle")
    assert preview(sandbox) == "NOTICE.txt", "the planted defect is not seen"


# --- a publish that cannot be confirmed writes nothing new (I7); a NOTICE replaced is said (I5) ------------------------


def puts(fake: FakeRecords) -> int:
    return sum(1 for method, _, _, _ in fake.seen if method == "PUT")


def refused_unconfirmed(toy: Path, fake: FakeRecords, tmp_path: Path) -> int:
    """A draft pushed, then a publish with no terminal and no ``--confirm``: how many writes it sent before its refusal."""
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    push.push(folder, service, pause=0)
    before = puts(fake)
    with pytest.raises(BuildRefused) as refused:
        push.publish(folder, service, pause=0)
    assert refused.value.codes() == ["push.unconfirmed"]
    return puts(fake) - before


def test_a_publish_that_cannot_be_confirmed_writes_no_preview_before_its_refusal(toy: Path, sandbox: FakeRecords,
                                                                                 tmp_path: Path):
    assert refused_unconfirmed(toy, sandbox, tmp_path) == 0 and preview(sandbox) is None


def test_control_with_the_preview_written_on_every_publish_the_unconfirmed_one_is_caught_writing(
        toy: Path, sandbox: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft.Drafting, "previewing", property(lambda self: True, lambda self, value: None),
                        raising=False)
    assert refused_unconfirmed(toy, sandbox, tmp_path) == 1, "the planted defect is not seen"


def pushed_with_the_old_notice(toy: Path, fake: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A draft as #100's push left it, its NOTICE under the name ``NOTICE``, then pushed by this one: what it said."""
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    with monkeypatch.context() as old:
        old.setattr(_plan, "NOTICE_KEY", "NOTICE")
        old.setattr(_draft, "NOTICE_KEY", "NOTICE")
        push.push(folder, service, pause=0)
    said: list[str] = []
    push.push(folder, service, pause=0, say=lambda line: said.append(str(line)))
    assert set(fake.records[NUMBER].files) == {"full_toy.mimebundle", "NOTICE.txt"}
    return "\n".join(said)


def test_a_notice_under_its_old_name_replaced_by_a_push_is_said(toy: Path, sandbox: FakeRecords, tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch):
    said = pushed_with_the_old_notice(toy, sandbox, tmp_path, monkeypatch)
    assert "uploaded 1 file" in said and "deleted NOTICE, which push put in the draft earlier" in said, said


def test_control_with_the_deletion_unsaid_the_replaced_notice_is_caught_unsaid(
        toy: Path, sandbox: FakeRecords, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_draft.Drafting, "deleted", property(lambda self: [], lambda self, value: None), raising=False)
    said = pushed_with_the_old_notice(toy, sandbox, tmp_path, monkeypatch)
    assert "deleted NOTICE" not in said, "the planted defect is not seen"
