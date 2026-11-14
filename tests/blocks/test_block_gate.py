"""The gate on a referenced block's own licence, and the container's other doors (lane P, P4b's fix round).

**The guarantee:** *a block's content is first used only where its own licence's conditions have been accepted, whichever
door it came in by.* The lock's word on the licence is never what is gated (a lock is outside the hashed record, and
anyone can edit it); the container's own licence, read after its content hash is verified, is. Doors: a load by the lock's
record, a load beside the block folder that holds the saved folder as a case, ``mime verify`` of a block's folder and of
its ``.mimeblock``, ``mime pull``. ``mime show`` is the one door that is not gated: it says the licence and the
conditions a person needs in order to decide, and expands nothing.

These tests are deterministic (no draw, no plant): each was run on the head before the fix and failed there, which the
fix round's commit message records."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest
from blocks import block_copy, host, own_cache
from test_block_container import referenced, saved_by_reference, with_case

from mime.assets._acceptances import remember
from mime.assets._gates import licences_of
from mime.blocks._container import pack_block, unpack_block
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import LOCK_FILE, LoadRefused


def forget_acceptances(root: Path) -> None:
    (root / "licences.json").unlink(missing_ok=True)


def codes_of(refused: pytest.ExceptionInfo[LoadRefused] | pytest.ExceptionInfo[BuildRefused]) -> list[str]:
    return refused.value.codes()


def test_a_lock_edited_to_state_another_licence_is_refused_not_trusted(tmp_path: Path):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        saved_by_reference(tmp_path, folder, tmp_path / "saved")
        lock_path = tmp_path / "saved" / LOCK_FILE
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        (record,) = [held for held in lock["assets"] if held["name"].endswith(".mimeblock")]
        assert record["licence"] == "NOASSERTION"
        record["licence"] = "CC0-1.0"  # an open licence: the gate on the lock's word would ask nothing
        lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
        shutil.rmtree(folder)
        forget_acceptances(tmp_path)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(tmp_path / "saved")
    assert codes_of(refused) == ["block.changed"]
    message = refused.value.refusals[0].message
    assert "CC0-1.0" in message and "NOASSERTION" in message and "the container's own is the one that binds" in message


def test_a_referenced_blocks_own_licence_is_asked_about_when_it_is_loaded_by_the_lock(tmp_path: Path):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        saved_by_reference(tmp_path, folder, tmp_path / "saved")
        shutil.rmtree(folder)
        forget_acceptances(tmp_path)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(tmp_path / "saved")
        assert codes_of(refused) == ["asset.licence-not-accepted"] and "NOASSERTION" in refused.value.refusals[0].message
        remember(licences_of(_recorded(tmp_path / "saved")))
        assert Experiment.load(tmp_path / "saved").load_report is not None


def _recorded(saved: Path):  # noqa: ANN202
    from mime.assets import AssetRef

    lock = json.loads((saved / LOCK_FILE).read_text(encoding="utf-8"))
    (record,) = [held for held in lock["assets"] if held["name"].endswith(".mimeblock")]
    return AssetRef.from_record(record)


def test_the_block_folder_that_holds_a_case_is_gated_as_the_lock_route_is(tmp_path: Path):
    """The enclosing-folder route found its container with no ``locate`` and so no gate: one function now finds a
    referenced container by either route and gates it on the block's own licence."""
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        forget_acceptances(tmp_path)
        with pytest.raises(LoadRefused) as refused:
            Experiment.load(folder / "cases" / "in_water")
        assert codes_of(refused) == ["asset.licence-not-accepted"]
        remember(licences_of(_recorded(folder / "cases" / "in_water")))
        assert Experiment.load(folder / "cases" / "in_water").load_report is not None


def test_verify_of_a_blocks_folder_and_of_its_container_asks_about_its_licence(tmp_path: Path,
                                                                             capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
        for target in (folder, container):
            forget_acceptances(tmp_path)
            assert main(["verify", str(target)]) == 2
            printed = capsys.readouterr()
            assert "asset.licence-not-accepted" in printed.out + printed.err, target
        remember(licences_of(_recorded(folder / "cases" / "in_water")))
        for target in (folder, container):
            assert main(["verify", str(target)]) == 0, target
            capsys.readouterr()


def test_show_is_not_gated_says_the_licence_and_expands_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                                monkeypatch: pytest.MonkeyPatch):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
        forget_acceptances(tmp_path)

        def never(*args: object, **kwargs: object) -> None:
            raise AssertionError("show expanded or ran something")

        from mime.blocks import _fragment

        monkeypatch.setattr(_fragment, "expand", never)
        monkeypatch.setattr(Experiment, "load", never)
        monkeypatch.setattr(Experiment, "build", never)
        for target in (folder, container):
            assert main(["show", str(target)]) == 0, target
            printed = capsys.readouterr().out
            assert "NOASSERTION" in printed and "used by reference" in printed, target
        assert not (tmp_path / "licences.json").exists(), "show asked and recorded nothing"


def test_pull_into_an_existing_empty_folder_puts_the_blocks_files_in_it(tmp_path: Path,
                                                                        capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
        existing = tmp_path / "empty"
        existing.mkdir()
        assert main(["pull", str(container), "--into", str(existing)]) == 0
        printed = capsys.readouterr().out
    assert (existing / "block.mime.yaml").is_file() and not (existing / "block").exists()
    assert f"unpacked to {existing}" in printed.replace("\\", "/") or f"unpacked to {existing.name}" in printed
    assert f"Next: mime verify" in printed and "/block" not in printed.split("Next:")[1]


def test_a_pull_of_a_gated_block_moves_nothing_out_of_scratch_before_the_gate(tmp_path: Path,
                                                                               capsys: pytest.CaptureFixture[str],
                                                                               monkeypatch: pytest.MonkeyPatch):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
        forget_acceptances(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)
        existing = tmp_path / "empty"
        existing.mkdir()
        for into in (None, existing):
            args = ["pull", str(container)] + (["--into", str(into)] if into is not None else [])
            assert main(args) == 2
            printed = capsys.readouterr()
            assert "asset.licence-not-accepted" in printed.out + printed.err
            assert list(work.iterdir()) == [] and list(existing.iterdir()) == [], "nothing left scratch"


def test_a_manifest_naming_a_traversal_among_its_referenced_files_is_refused_at_unpack(tmp_path: Path):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "ok.mimeblock")
    from mime.blocks._container import reference_problems
    from mime.saved._container import container_hash, write_container
    from mime.saved._studies import BLOCK

    with zipfile.ZipFile(container) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    names = ("cases/in_water/../../../evil", "/evil", "cases/in_water/.git/config", "block.mime.yaml", "outside/evil")
    for name in names:
        manifest = json.loads(members["manifest.json"])
        manifest["referenced"][name] = {"name": "x", "sha256": "0" * 64, "size": 1}
        manifest["container_hash"] = container_hash(manifest, "container_hash")
        text = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
        forged = write_container(tmp_path / "forged.mimeblock", text, members["NOTICE"],
                                 {each.removeprefix("block/"): data for each, data in members.items()
                                  if each.startswith("block/")}, BLOCK)
        assert [each.code for each in reference_problems(manifest)] == ["bundle.unsafe"], name
        with pytest.raises(BuildRefused) as refused:
            unpack_block(forged, tmp_path / f"unpacked_{abs(hash(name))}")
        assert refused.value.codes() == ["bundle.unsafe"], name
        assert not (tmp_path / "evil").exists()
        forged.unlink()
    assert [each.code for each in reference_problems(json.loads(members["manifest.json"]))] == []
