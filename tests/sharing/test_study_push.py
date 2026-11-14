"""``mime push`` of studies (lane P, P2), against the offline fake of the records API: a publishing folder is one record
of one ``.mimestudy`` per study and one ``NOTICE.txt``, its metadata from the folder's ``mime.toml``, its keyword
``mime-study``; no asset is uploaded and no container is packed again once the DOI is reserved (a study container names
no DOI of its own: the record's description does). Published, it pulls back by its number into a folder each and
verifies. The real sandbox is the maintainer's step (``scripts/zenodo_sandbox_lane.py --demo study``)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from push_cases import with_token, write_manifest
from study_plants import SOURCE_VARIABLE
from study_rigs import write_study
from zenodo_records import FakeRecords, served

from mime import push
from mime.build import BuildRefused
from mime.cli import main


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = os.environ.get(SOURCE_VARIABLE)
    target = tmp_path_factory.mktemp("written") / "wall_scan"
    return Path(shutil.copytree(source, target)) if source else write_study(target)


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRecords]:
    """The fake behind a server on this machine, standing in for the sandbox in ``mime.push.SERVICES``."""
    token = with_token(monkeypatch)
    monkeypatch.setattr(push._client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with served(token) as fake:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", fake.service())  # type: ignore[index]
        yield fake


def publishing(written: Path, root: Path) -> Path:
    folder = root / "paper"
    folder.mkdir(parents=True)
    for name in ("stiffness_scan", "step_scan"):
        shutil.copytree(written, folder / name)
    (folder / "README.md").write_text("notes\n", encoding="utf-8")
    write_manifest(folder)
    return folder


def test_a_publishing_folder_is_one_record_of_its_studies_and_pulls_back_a_folder_each(
        written: Path, sandbox: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = publishing(written, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    drafted = push.push(folder, service, pause=0)
    record = sandbox.records[drafted.record]
    assert sorted(record.files) == ["NOTICE.txt", "step_scan.mimestudy", "stiffness_scan.mimestudy"]
    metadata = record.body["metadata"]
    assert isinstance(metadata, dict)
    subjects = [each["subject"] for each in metadata["subjects"]]  # type: ignore[index, union-attr]
    assert "mime-study" in subjects and "mime-experiment" not in subjects
    assert drafted.doi in str(metadata["description"]), "the description names the reserved DOI"
    assert "step_scan" in str(metadata["description"]) and "README" not in str(metadata["description"])
    again = push.push(folder, service, pause=0)
    assert again.uploaded == (), "a second push finds every file uploaded and verified"
    push.publish(folder, service, confirm=drafted.doi, pause=0)
    into = tmp_path / "pulled"
    assert main(["pull", "--from", "zenodo-sandbox", drafted.record, "--into", str(into)]) == 0
    assert sorted(path.name for path in into.iterdir()) == ["step_scan", "stiffness_scan"]
    capsys.readouterr()
    assert main(["verify", str(into)]) == 0
    assert capsys.readouterr().out.count("saved experiments reproduce") == 2


def test_the_record_licence_must_be_every_studys(written: Path, sandbox: FakeRecords, tmp_path: Path):
    folder = publishing(written, tmp_path)
    path = folder / "mime.toml"
    path.write_text(path.read_text(encoding="utf-8").replace('licence = "CC-BY-4.0"', 'licence = "MIT"'),
                    encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        push.push(folder, push.SERVICES["zenodo-sandbox"], pause=0)
    assert set(refused.value.codes()) == {"push.licence"} and len(refused.value.refusals) == 4
    assert sandbox.records == {}, "nothing was sent"


def test_a_single_study_folder_is_pushed_as_a_record_of_one(written: Path, sandbox: FakeRecords, tmp_path: Path):
    folder = Path(shutil.copytree(written, tmp_path / "wall_scan"))
    write_manifest(folder)
    drafted = push.push(folder, push.SERVICES["zenodo-sandbox"], pause=0)
    assert sorted(sandbox.records[drafted.record].files) == ["NOTICE.txt", "wall_scan.mimestudy"]
