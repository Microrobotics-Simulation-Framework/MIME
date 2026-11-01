"""Folders and fakes for the tests of ``mime push``: a saved toy under a licence with its ``sharing.yaml``, a token in
the environment (the conftest's home holds no other), the offline fake, and a plan made once and kept outside push's
staging directory, for the tests of the drafting engine that run push many times."""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path

import pytest
from zenodo_records import FakeRecords, FakeTransport, Killed

from mime import push
from mime.build import BuildRefused
from mime.push import Client, Plan, Service
from mime.push._plan import RecordFile, file_hashes
from mime.testing.toys import full_toy

#: The creator every test folder names: ORCID's own documented test identity.
CREATOR = "{family: Carberry, given: Josiah, orcid: 0000-0002-1825-0097, affiliation: Brown University}"


def a_token() -> str:
    """A token as Zenodo's look: forty characters of letters and digits, fresh each time."""
    return secrets.token_urlsafe(30)[:40]


def with_token(monkeypatch: pytest.MonkeyPatch, service: str = "zenodo-sandbox") -> str:
    """A fresh token for ``service``, in its environment variable (the conftest's home holds no credentials file)."""
    from mime import credentials

    token = a_token()
    monkeypatch.setenv(credentials.SERVICES[service], token)
    return token


def write_sharing(folder: Path, *, creators: str = CREATOR, extra: str = "") -> Path:
    """``--init``'s file with one creator named, and ``extra`` lines added."""
    path, _, _ = push.init(folder)
    text = path.read_text(encoding="utf-8").replace("creators: []", f"creators: [{creators}]")
    path.write_text(text + extra, encoding="utf-8")
    return path


def saved_toy(root: Path, name: str = "full_toy", *, licence: str | None = "CC-BY-4.0") -> Path:
    experiment = full_toy()
    if licence is not None:
        experiment.licence(licence)
    folder = root / name
    experiment.save(folder)
    return folder


def saved_named(root: Path, name: str) -> Path:
    """A toy experiment named ``name``, saved under the licence CC-BY-4.0 with its sharing.yaml."""
    from mime.testing.toys._experiments import declare_in_order, full_declarations, toy_experiment

    experiment = declare_in_order(lambda: toy_experiment(name), full_declarations(), None)
    experiment.licence("CC-BY-4.0")
    folder = root / name
    experiment.save(folder)
    write_sharing(folder)
    return folder


def copy_of(folder: Path, root: Path) -> Path:
    return Path(shutil.copytree(folder, root / folder.name))


def client(fake: FakeRecords, service: Service | None = None, **given: object) -> Client:
    from mime import credentials

    held = service or fake.service()
    token = credentials.token(held.credential)
    assert token is not None
    return Client(held, token, FakeTransport(fake), pause=0, sleep=lambda seconds: None, **given)  # type: ignore[arg-type]


def plan_of(fake: FakeRecords, files: dict[str, bytes], where: Path, service: Service | None = None) -> Plan:
    """A plan of exactly ``files`` (key to bytes), written under ``where``, with a payload the fake accepts: the
    drafting engine's input, without a folder to pack."""
    where.mkdir(parents=True, exist_ok=True)
    held: list[RecordFile] = []
    for key, data in files.items():
        path = where / key
        path.write_bytes(data)
        md5, sha256, size = file_hashes(path)
        held.append(RecordFile(key, path, size, md5, sha256, "CC0-1.0", "a test's file"))
    payload: dict[str, object] = {
        "metadata": {"resource_type": {"id": "dataset"}, "title": "a test", "publication_date": "2026-11-01",
                     "publisher": "Zenodo",
                     "creators": [{"person_or_org": {"type": "personal", "family_name": "Carberry"}}],
                     "rights": [{"id": "cc0-1.0"}]},
        "access": {"record": "public", "files": "public"}, "files": {"enabled": True}}
    from mime.push._sharing import Person, Sharing
    from mime.saved._terms import FolderTerms

    sharing = Sharing("a test", "", "CC0-1.0", (Person("Carberry", "Josiah"),))
    return Plan(service or fake.service(), "a_test", where, tuple(held), FolderTerms((), "CC0-1.0", {}, (), 4), (),
                "CC0-1.0", sharing, payload, "0.3.0")


def run_until_drafted(plan: Plan, fake: FakeRecords, said: list[str]) -> tuple[push.Drafted, int]:
    """Push again, as a person would after a stop, until a push completes: the push that completed, and how many
    stopped before it."""
    stops = 0
    while True:
        try:
            return push.send(plan, push.read_state(plan.service, plan.experiment), client(fake), said.append), stops
        except (Killed, BuildRefused):
            stops += 1
            assert stops < 6, said[-5:]
