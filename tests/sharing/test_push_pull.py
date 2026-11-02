"""``mime pull`` of what ``mime push`` published: a sandbox record by its number at the sandbox's records API (its
test-prefix DOIs do not resolve through doi.org), and a concept DOI resolved to its latest version, named.

Controls, each caught on every run:

- **the sandbox's DOI resolved through doi.org**: ``mime pull 10.5072/...`` and an asset location under the test prefix
  never reach ``requests``; with either guard planted away, the request to doi.org is caught.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import requests
from assets_rig import Rig
from fake_zenodo import DOI, RECORD, zenodo
from push_cases import copy_of, saved_toy, with_token, write_sharing
from zenodo_records import FakeRecords, served

from mime import assets, push
from mime.assets import _fetch
from mime.build import BuildRefused
from mime.cli import main
from mime.saved import _pull, pack, pull


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = saved_toy(tmp_path_factory.mktemp("toy"))
    write_sharing(folder)
    return folder


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRecords]:
    """The fake behind a server on this machine, standing in for the sandbox in ``mime.push.SERVICES``."""
    token = with_token(monkeypatch)
    monkeypatch.setattr(push._client, "PAUSE_SECONDS", 0.0)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with served(token) as fake:
        monkeypatch.setitem(push.SERVICES, "zenodo-sandbox", fake.service())  # type: ignore[index]
        yield fake


def doi_org_asked(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every address ``requests.get`` is asked for: none may be doi.org's for a test-prefix DOI."""
    asked: list[str] = []
    real = requests.get

    def watched(url: str, *given: object, **keywords: object) -> requests.Response:
        asked.append(url)
        if "doi.org" in url:
            raise requests.ConnectionError("doi.org does not resolve 10.5072")
        return real(url, *given, **keywords)  # type: ignore[arg-type]

    monkeypatch.setattr(requests, "get", watched)
    return asked


def test_a_published_sandbox_record_is_pulled_by_its_number_and_verifies(toy: Path, sandbox: FakeRecords,
                                                                       tmp_path: Path,
                                                                       capsys: pytest.CaptureFixture[str],
                                                                       monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    push.push(folder, service, pause=0)
    push.publish(folder, service, confirm="10.5072/zenodo.629525", pause=0)
    asked = doi_org_asked(monkeypatch)
    into = tmp_path / "pulled"
    assert main(["pull", "--from", "zenodo-sandbox", "629525", "--into", str(into)]) == 0
    out = capsys.readouterr().out
    assert "the record 629525 on zenodo-sandbox (version 1, 10.5072/zenodo.629525" in out
    assert not [url for url in asked if "doi.org" in url]
    assert main(["verify", str(into)]) == 0
    assert pack(into, out=tmp_path / "again.mimebundle").read_bytes() == \
        sandbox.records["629525"].files["full_toy.mimebundle"].data, "the pulled folder packs to the published bundle"


def test_a_newer_version_the_service_does_not_name_sends_the_person_to_the_records_page(
        toy: Path, sandbox: FakeRecords, tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch):
    folder = copy_of(toy, tmp_path)
    service = push.SERVICES["zenodo-sandbox"]
    push.push(folder, service, pause=0)
    push.publish(folder, service, confirm="10.5072/zenodo.629525", pause=0)
    monkeypatch.setattr(sandbox, "_latest", lambda parent: None)  # not the latest, and `versions/latest` names none
    assert main(["pull", "--from", "zenodo-sandbox", "629525", "--into", str(tmp_path / "pulled")]) == 0
    out = capsys.readouterr().out
    assert f"! it is not the latest version: the record's page, {sandbox.site}/records/629525, names the latest" in out
    assert "a newer one" not in out


def test_a_draft_or_an_unknown_record_is_not_pulled(toy: Path, sandbox: FakeRecords, tmp_path: Path):
    folder = copy_of(toy, tmp_path)
    push.push(folder, push.SERVICES["zenodo-sandbox"], pause=0)
    for record in ("629525", "999999"):
        with pytest.raises(BuildRefused) as refused:
            pull(record, into=tmp_path / record, from_service="zenodo-sandbox")
        assert refused.value.codes() == ["pull.record"]
    with pytest.raises(BuildRefused) as refused:
        pull("10.5072/zenodo.629525", into=tmp_path / "x", from_service="zenodo-sandbox")
    assert refused.value.codes() == ["pull.source"]


@pytest.mark.parametrize("given", ["10.5072/zenodo.629525", "doi:10.5072/zenodo.629525",
                                   "https://doi.org/10.5072/zenodo.629525/full_toy.mimebundle"])
def test_a_sandbox_doi_is_never_resolved_through_doi_org(given: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    asked = doi_org_asked(monkeypatch)
    with pytest.raises(BuildRefused) as refused:
        pull(given, into=tmp_path / "out")
    assert refused.value.codes() == ["pull.sandbox-doi"] and asked == []
    assert "mime pull --from zenodo-sandbox 629525" in refused.value.refusals[0].action


def test_control_without_the_guard_a_sandbox_doi_is_caught_asking_doi_org(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_pull, "TEST_PREFIX", "10.0000/")
    asked = doi_org_asked(monkeypatch)
    with pytest.raises(BuildRefused):
        pull("10.5072/zenodo.629525", into=tmp_path / "out")
    assert any("doi.org" in url for url in asked), "the planted resolution through doi.org is not seen"


def test_an_asset_location_under_the_test_prefix_is_never_resolved(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    asked = doi_org_asked(monkeypatch)
    ref = assets.AssetRef("sandboxed", "ab" * 32, 10, "CC0-1.0", ("doi:10.5072/zenodo.629525/t.npz",), unlisted=True)
    with pytest.raises(BuildRefused):
        assets.fetch(ref)
    assert asked == []
    monkeypatch.setattr(_fetch, "TEST_DOI", "doi:10.0000/")
    with pytest.raises(BuildRefused):
        assets.fetch(ref)
    assert any("doi.org" in url for url in asked), "the control: without the guard, doi.org is asked"


def test_a_concept_doi_pulls_the_latest_version_and_names_it(toy: Path, rig: Rig, tmp_path: Path):
    bundle = pack(toy, out=tmp_path / "full_toy.mimebundle")
    concept = f"10.5281/zenodo.{RECORD - 1}"
    said: list[str] = []
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}, concept=concept, version="3"):
        folder = pull(concept, into=tmp_path / "out", say=said.append)
    assert folder.is_dir()
    assert any(f"{concept} is the concept DOI, which names every version; it resolved to the latest, {DOI} "
               f"(version 3)" in line for line in said), said
