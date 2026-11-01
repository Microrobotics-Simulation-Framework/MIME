"""``mime push`` on the command line, through the real transport (the standard library's HTTP client) to the offline
fake behind a server on this machine: a round trip of ``--init``, the push, the push again, ``--status`` and
``--discard``, an upload cut part way that is sent again whole, and the refusals in 7l-b's shape (the command named
on the first line, each reason as ``[class] code: what happened`` with ``do:`` and ``see:``, exit 2). The fake is
injected through Python (the service's entry in ``mime.push.SERVICES``), never through the environment.

This file runs on macOS and Windows too (``tests/platforms.txt``): a push's one cheap round trip, there."""

from __future__ import annotations

from pathlib import Path

import pytest
from push_cases import copy_of, saved_toy, with_token, write_sharing
from zenodo_records import FakeRecords, served

from mime import push
from mime.cli import main


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("toy"))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    token = with_token(monkeypatch)
    from mime.push import _client

    monkeypatch.setattr(_client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    with served(token) as held:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", held.service())  # type: ignore[index]
        yield held


def run(capsys: pytest.CaptureFixture[str], *words: str) -> tuple[int, str, str]:
    code = main(list(words))
    held = capsys.readouterr()
    return code, held.out, held.err


def test_a_round_trip_through_the_real_transport(toy: Path, fake: FakeRecords, tmp_path: Path,
                                                 capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    assert run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--init")[0] == 0
    write_sharing(folder)
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes")
    assert code == 0 and "Plan for 'full_toy'" in out and "made the draft 629525" in out
    assert out.index("Next: ") < out.index("made the draft"), "the plan is shown before the draft is made"
    page = f"{fake.site}/uploads/629525"
    assert out.splitlines()[-1] == page, "a push ends with the draft's link on a line of its own"
    record = fake.records["629525"]
    assert set(record.files) == {"full_toy.mimebundle", "NOTICE"}
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes")
    assert code == 0 and out.count("not sent again") == 2 and out.splitlines()[-1] == page
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--status")
    assert code == 0 and "reserved DOI: 10.5072/zenodo.629525" in out and "as push uploaded and verified it" in out
    assert out.splitlines()[-1] == page
    record.body["metadata"]["title"] = "Edited on the site"  # type: ignore[index]
    code, out, err = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes")
    assert code == 2 and "[user] push.metadata-edited" in err and out.splitlines()[-1] == page
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes", "--overwrite-site-edits")
    assert code == 0 and record.body["metadata"]["title"] == "full_toy"  # type: ignore[index]
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--discard", "--yes")
    assert code == 0 and "deleted the draft 629525" in out and fake.drafts() == []
    assert out.splitlines()[-1] == f"{fake.site}/me/uploads"


def test_an_upload_cut_part_way_through_the_real_transport_is_sent_again_whole(toy: Path, fake: FakeRecords,
                                                                               tmp_path: Path,
                                                                               capsys: pytest.CaptureFixture[str]):
    folder = copy_of(toy, tmp_path)
    write_sharing(folder)
    fake.plant("cut", lambda method, path: method == "PUT" and path.endswith(".mimebundle/content"))
    code, out, _ = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes")
    assert code == 0 and "its connection dropped" in out
    staged = push.state_directory(push.SERVICES["zenodo-sandbox"], "full_toy") / "staged" / "full_toy.mimebundle"
    assert fake.records["629525"].files["full_toy.mimebundle"].data == staged.read_bytes()


def test_a_refusal_is_in_7l_bs_shape_and_exits_two(toy: Path, fake: FakeRecords, tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str],
                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIME_TRACEBACK", raising=False)
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path / "tracebacks"))
    folder = copy_of(toy, tmp_path)
    code, _, err = run(capsys, "push", str(folder), "--to", "zenodo-sandbox", "--yes")
    lines = err.splitlines()
    assert code == 2 and lines[0] == "mime push is refused:"
    assert lines[1].startswith("[user] push.sharing-missing: ")
    assert any(line.strip().startswith("do: write it from what the folder knows, `mime push") for line in lines)
    assert any(line.strip().startswith("see: ") and line.rstrip().endswith("/errors/push.sharing-missing")
               for line in lines)
    assert fake.requests == []


def test_without_a_terminal_and_without_yes_nothing_is_sent(toy: Path, fake: FakeRecords, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str],
                                                             monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(toy, tmp_path)
    write_sharing(folder)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, out, err = run(capsys, "push", str(folder), "--to", "zenodo-sandbox")
    assert code == 2 and "Plan for 'full_toy'" in out and "[user] push.unconfirmed" in err and fake.requests == []
