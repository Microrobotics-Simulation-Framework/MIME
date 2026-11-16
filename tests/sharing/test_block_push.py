"""``mime push`` of a block (lane P, P4b), through the offline fake of the records API (never Zenodo): draft, upload,
publish, then ``mime pull`` of the published number. The sandbox lane's ``--demo block`` is a dry run only; ``mime push``
itself is as real for a block as for a bundle (it has no ``--dry-run``), and this drives the whole path against the same
fake the bundle tests use."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from push_cases import with_token
from zenodo_records import FakeRecords, served

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blocks"))

from blocks import block_copy, own_cache  # noqa: E402
from test_block_container import with_case  # noqa: E402

from mime.push import _client, _draft, _publish, _service  # noqa: E402
from mime.cli import main  # noqa: E402


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRecords]:
    token = with_token(monkeypatch)
    monkeypatch.setattr(_client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with served(token) as fake:
        monkeypatch.setitem(_service.SERVICES, "zenodo-sandbox", fake.service())  # type: ignore[index]
        yield fake


def test_a_block_is_drafted_uploaded_published_and_pulled_back_whole(
        sandbox: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        monkeypatch.delenv("MIME_OFFLINE")  # push goes online, to the fake on this machine only
        service = _service.SERVICES["zenodo-sandbox"]
        drafted = _draft.push(folder, service, pause=0)
        record = sandbox.records[drafted.record]
        assert sorted(record.files) == ["NOTICE.txt", "labx.nudged_probe-0.1.0.mimeblock"]
        metadata = record.body["metadata"]
        assert isinstance(metadata, dict)
        subjects = [each["subject"] for each in metadata["subjects"]]  # type: ignore[index, union-attr]
        assert "mime-block" in subjects and "mime-block:labx.nudged_probe" in subjects
        assert "mime-study" not in subjects and "mime-experiment" not in subjects
        again = _draft.push(folder, service, pause=0)
        assert again.uploaded == (), "a second push finds every file uploaded and verified"
        _publish.publish(folder, service, confirm=drafted.doi, pause=0)
        into = tmp_path / "pulled"
        assert main(["pull", "--from", "zenodo-sandbox", drafted.record, "--into", str(into)]) == 0
        assert "Block labx.nudged_probe 0.1.0 unpacked to" in capsys.readouterr().out
        assert (into / "block.mime.yaml").is_file() and (into / "cases" / "in_water").is_dir()
        assert main(["verify", str(into)]) == 0
