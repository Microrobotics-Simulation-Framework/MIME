"""The study container (``.mimestudy``, lane P's P2): one study folder as one file, packed, pulled, verified and carded as
a bundle is, with the study's own records held at every exit; publishing folders, one container per study; a record of
several studies pulled into a folder each.

The controls plant each defect at random (``study_plants.py``): ``test_a_clean_study_round_trips`` passes clean and
fails under any plant (``MIME_P2_PLANT``, the seed in ``DETECTION_SEED``), which ``detection_rate.py`` rates; the tests
below name the code each plant is refused with, over several seeds."""

from __future__ import annotations

import json
import os
import random
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from assets_rig import Rig
from fake_zenodo import DOI, zenodo
from push_cases import saved_toy, write_manifest
from study_plants import PLANT_VARIABLE, PLANTS, SEED_VARIABLE, SOURCE_VARIABLE, rewritten
from study_rigs import write_study

from mime import credentials, push
from mime.build import BuildRefused, MimeError
from mime.cli import main
from mime.results import StudyResult
from mime.saved import _bundle, pack, pack_studies, pull, unpack
from mime.saved._studies import STUDY_CONTAINER, container_kind, unpack_study


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A toy study (two groups, a figure), written once: by the run, or copied from a folder a probe wrote."""
    source = os.environ.get(SOURCE_VARIABLE)
    target = tmp_path_factory.mktemp("written") / "wall_scan"
    if source:
        return Path(shutil.copytree(source, target))
    return write_study(target)


@pytest.fixture
def study(written: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(written, tmp_path / "work" / written.name))


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except MimeError as refused:
        return [each.code for each in refused.refusals]
    raise AssertionError("not refused")


def tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def round_trip(folder: Path, work: Path, plant: str | None = None, seed: int = 0) -> Path:
    """The study packed, its container pulled into a new folder and read: with ``plant``, its defect planted first."""
    draw = random.Random(seed)
    held = PLANTS[plant] if plant is not None else None
    if held is not None and held.folder is not None:
        held.folder(folder, draw)
    container = pack(folder, out=work / f"{folder.name}.mimestudy")
    if held is not None and held.container is not None:
        container = held.container(container, draw)
    out = pull(container, into=work / "pulled", fetch_assets=False)
    StudyResult.read(out)
    return out


# --- the control, rated by detection_rate.py -----------------------------------------------------------------------------


def test_a_clean_study_round_trips(study: Path, tmp_path: Path):
    """Passes clean; fails under any plant (``MIME_P2_PLANT``), placed by ``DETECTION_SEED``."""
    out = round_trip(study, tmp_path, os.environ.get(PLANT_VARIABLE) or None, int(os.environ.get(SEED_VARIABLE, "0")))
    assert tree(out) == {name: data for name, data in tree(study).items() if name != "mime.toml"}


@pytest.mark.parametrize("plant", sorted(PLANTS))
@pytest.mark.parametrize("seed", range(4))
def test_each_planted_defect_is_refused_with_its_code(study: Path, tmp_path: Path, plant: str, seed: int):
    found = codes_of(lambda: round_trip(study, tmp_path, plant, seed))
    assert set(found) & PLANTS[plant].codes, (plant, seed, found)
    assert not (tmp_path / "pulled").exists(), "a refused pull leaves nothing"


def test_a_planted_token_is_refused_naming_the_study_container_and_never_the_token(
        study: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    secret = "zt" + os.urandom(19).hex()
    monkeypatch.setenv(credentials.SERVICES["zenodo"], secret)
    with (study / "study.mime.yaml").open("a", encoding="utf-8") as file:
        file.write(f"# token={secret}\n")
    with pytest.raises(BuildRefused) as refused:
        pack(study)
    assert refused.value.codes() == ["bundle.credential"]
    assert "study.mime.yaml (in the study container)" in str(refused.value) and secret not in str(refused.value)


# --- what a study container is ---------------------------------------------------------------------------------------------


def test_a_study_container_holds_its_folder_and_a_closed_manifest_and_is_the_same_bytes_every_time(
        study: Path, tmp_path: Path):
    container = pack(study)
    assert container == study.with_name("wall_scan.mimestudy")
    with zipfile.ZipFile(container) as archive:
        names = [info.filename for info in archive.infolist()]
        manifest = json.loads(archive.read("manifest.json"))
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
    assert names[:2] == ["manifest.json", "NOTICE"] and names[2:] == sorted(names[2:])
    assert all(name.startswith("study/") for name in names[2:])
    assert set(manifest) == {"format", "name", "files", "referenced", "notice", "assets", "container_hash"}
    assert (manifest["format"], manifest["name"]) == (STUDY_CONTAINER, "wall_scan")
    assert {"study.json", "study.mime.yaml", "groups/1/experiment/mime.lock", "figures/trace.json"} <= set(
        manifest["files"])
    again = shutil.copytree(study, tmp_path / "again" / "wall_scan")
    for path in again.rglob("*"):
        os.utime(path, (1, 1))
    assert pack(again, out=tmp_path / "again.mimestudy").read_bytes() == container.read_bytes()


def test_mime_toml_is_neither_carried_nor_refused(study: Path, tmp_path: Path):
    write_manifest(study)
    container = pack(study)
    with zipfile.ZipFile(container) as archive:
        assert not any(name.endswith("mime.toml") for name in archive.namelist())


def test_a_stopped_study_is_refused_naming_the_members_not_run(study: Path):
    path = study / "study.json"
    held = json.loads(path.read_text(encoding="utf-8"))
    held["members"][1]["status"] = "pending"
    path.write_text(json.dumps(held, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        pack(study)
    assert "study.incomplete" in refused.value.codes()
    assert "1 of its 4 members is not run (point 1)" in str(refused.value)


def test_a_container_named_as_another_kind_or_a_block_is_refused_and_never_unpacked(study: Path, tmp_path: Path):
    container = pack(study)
    assert container_kind(container).name == "study"
    assert codes_of(lambda: unpack(container, tmp_path / "a")) == ["container.kind"]
    toy = saved_toy(tmp_path / "saved")
    bundle = pack(toy)
    as_study = bundle.with_name("full_toy.mimestudy")
    shutil.copy(bundle, as_study)
    assert codes_of(lambda: pull(as_study, into=tmp_path / "b")) == ["container.kind"]
    block = rewritten(container, lambda members, manifest: manifest.update(format="mime.block/1"))
    named = block.with_name("probe.mimeblock")
    block.rename(named)
    assert codes_of(lambda: pull(named, into=tmp_path / "c")) == ["container.kind"]
    newer = rewritten(container, lambda members, manifest: manifest.update(format="mime.study/2"))
    assert codes_of(lambda: pull(newer, into=tmp_path / "d")) == ["load.newer-form"]
    assert not any((tmp_path / each).exists() for each in "abcd")


@pytest.mark.parametrize("name", ["study/../evil", "study/.env", "study/groups/01/outputs.npz", "elsewhere/x",
                                  "study/experiment/notes.txt", "study/groups/0/experiment/mime.lock"])
def test_a_member_the_study_layout_does_not_hold_is_refused(study: Path, tmp_path: Path, name: str):
    container = rewritten(pack(study), lambda members, manifest: members.update({name: b"x"}))
    assert "bundle.unsafe" in codes_of(lambda: unpack_study(container, tmp_path / "out"))
    assert not (tmp_path / "out").exists()


# --- mime pack, verify and card ---------------------------------------------------------------------------------------------


def test_the_cli_packs_verifies_and_cards_a_study(study: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["pack", str(study)]) == 0
    said = capsys.readouterr().out
    assert "packed the study wall_scan into" in said and "carried: " in said
    out = pull(study.with_name("wall_scan.mimestudy"), into=tmp_path / "pulled" / "wall_scan", fetch_assets=False)
    assert main(["verify", str(out)]) == 0
    said = capsys.readouterr().out
    assert "study wall_scan: its files are its run's, and its 2 saved experiments reproduce" in said
    assert "groups/1/experiment/: reproduced" in said and "It bounds nothing about physics" in said
    assert main(["card", str(out)]) == 0
    card = capsys.readouterr().out
    assert "### wall_scan" in card and "axis `toy.spring:probe.k`" in card and "mime study run" in card
    assert main(["card", str(out), "--notice"]) == 0
    with zipfile.ZipFile(study.with_name("wall_scan.mimestudy")) as archive:
        packed = archive.read("NOTICE").decode()
    assert capsys.readouterr().out == packed


def test_verify_refuses_a_study_its_records_refuse_and_runs_nothing(study: Path, capsys: pytest.CaptureFixture[str]):
    PLANTS["input-hash"].folder(study, random.Random(1))  # type: ignore[misc]
    assert main(["verify", str(study)]) == 2
    said = capsys.readouterr()
    assert "refused, nothing run" in said.out and "study.input-hash" in said.err


# --- publishing folders ------------------------------------------------------------------------------------------------------


def publishing(study: Path, root: Path) -> Path:
    folder = root / "paper"
    folder.mkdir(parents=True)
    shutil.copytree(study, folder / "stiffness_scan")
    shutil.copytree(study, folder / "step_scan")
    return folder


def test_a_publishing_folder_packs_one_container_per_study_and_carries_nothing_else(
        study: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = publishing(study, tmp_path)
    secret = os.urandom(24).hex().encode()
    (folder / ".env").write_bytes(b"TOKEN=" + secret)
    (folder / "scripts").mkdir()
    (folder / "scripts" / "plot.py").write_bytes(b"# " + secret)
    assert main(["pack", str(folder), "--out", str(tmp_path / "out")]) == 0
    said = capsys.readouterr().out
    assert "! left out, carried by no container (not study folders): .env, scripts/" in said
    written = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert written == ["step_scan.mimestudy", "stiffness_scan.mimestudy"]
    assert not any(secret in (tmp_path / "out" / name).read_bytes() for name in written)


def test_a_refused_study_writes_no_container_of_its_publishing_folder(study: Path, tmp_path: Path):
    folder = publishing(study, tmp_path)
    (folder / "step_scan" / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        pack_studies(folder, out=tmp_path / "out")
    assert refused.value.codes() == ["study.stray-file"] and "step_scan: " in str(refused.value)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_mime_check_holds_each_study_of_a_publishing_folder_to_the_record_licence(study: Path, tmp_path: Path):
    folder = publishing(study, tmp_path)
    write_manifest(folder)
    assert push.check(folder).findings == []
    path = folder / "mime.toml"
    path.write_text(path.read_text(encoding="utf-8").replace('licence = "CC-BY-4.0"', 'licence = "MIT"'),
                    encoding="utf-8")
    findings = [item.what for item in push.check(folder).findings]
    assert "mime.toml, [study] licence: is MIT, and stiffness_scan: the study's base experiment declares CC-BY-4.0, " \
           "which its files carry" in findings and len(findings) == 4


# --- pull of a record of several studies -------------------------------------------------------------------------------------


def test_a_record_of_two_studies_needs_into_before_anything_downloads_then_unpacks_a_folder_each(
        study: Path, rig: Rig, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    paths = pack_studies(publishing(study, tmp_path), out=tmp_path / "containers")
    files = {path.name: path.read_bytes() for path in paths}
    with zenodo(rig, {**files, "NOTICE.txt": b"notice"}):
        assert codes_of(lambda: pull(DOI)) == ["pull.destination"]
        assert not list(rig.cache.glob("[0-9a-f]" * 64)), "nothing is downloaded before --into is given"
        folder = pull(DOI, into=tmp_path / "studies")
        assert sorted(path.name for path in folder.iterdir()) == ["step_scan", "stiffness_scan"]
        assert pull(f"doi:{DOI}/step_scan.mimestudy", into=tmp_path / "one").is_dir()
    assert main(["verify", str(folder)]) == 0
    said = capsys.readouterr().out
    assert said.count("saved experiments reproduce") == 2


def test_a_record_holding_a_bundle_and_a_study_is_refused(study: Path, rig: Rig, tmp_path: Path):
    container = pack(study)
    bundle = pack(saved_toy(tmp_path / "saved"))
    with zenodo(rig, {"wall_scan.mimestudy": container.read_bytes(), "full_toy.mimebundle": bundle.read_bytes()}):
        assert codes_of(lambda: pull(DOI, into=tmp_path / "x")) == ["pull.record"]


def test_based_on_refuses_a_study_container(study: Path):
    from mime.experiment import Experiment

    container = pack(study)
    assert codes_of(lambda: Experiment.based_on(container, name="from_a_study")) == ["container.kind"]


def test_the_bundle_kind_is_unchanged_by_the_study_kind(tmp_path: Path):
    """A bundle packs, pulls and names its kind as before: the refactor's first commit holds this for every test of a
    bundle; this one holds it across the new reader."""
    toy = saved_toy(tmp_path / "saved")
    bundle = pack(toy)
    assert container_kind(bundle) is _bundle.BUNDLE
    assert pull(bundle, into=tmp_path / "out", fetch_assets=False).is_dir()
