"""The referenced tier, the block's container (``.mimeblock``), pull, verify, card and push of a block.

**The referenced tier's guarantee**, with its control (``MIME_PLANT_P4A``): *a referenced block's changed content is
refused.* An experiment using a block by reference is saved; the block is edited (a declared value, a law's setting, a
thing's initial state or a comment of its fragment, or a key of its definition, drawn from ``MIME_SEED``) and packed again,
and the lock's record of the container is pointed at the new one, as another container under the same name would be. The
load refuses it as ``block.changed``. Control ``unheld``: the content hash not held to the document's.

**Verify's controls**, each a defect the run must report: a requirement above what is installed (refused, nothing run),
a case edited after it was saved (it does not reproduce), the fragment edited after its case was saved (refused as
``block.case-stale``)."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from blocks import block_copy, edited, host, own_cache, plant, rng

from mime.assets._acceptances import remember
from mime.assets._gates import licences_of
from mime.blocks import _referenced
from mime.blocks._card import ACCEPTANCE, block_card_text, verify_block
from mime.blocks._container import pack_block, unpack_block
from mime.build import BuildRefused
from mime.cli import main
from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused


def referenced(folder: Path) -> None:
    """The block made one its licence uses by reference: no licence asserted."""
    edited(folder, "mime.toml", 'licence = "CC-BY-4.0"', 'licence = "NOASSERTION"')


def saved_by_reference(root: Path, folder: Path, out: Path) -> Experiment:
    """An experiment using the block at ``folder`` by reference, its container's licence accepted here, saved."""
    experiment, water = host()
    experiment.use(folder, medium=water, accept_licences=["NOASSERTION"])
    (container,) = [ref for ref in experiment._settings.assets if ref.name.endswith(".mimeblock")]  # noqa: SLF001
    remember(licences_of(container))
    experiment.licence("CC-BY-4.0")
    experiment.save(out, steps=3)
    return experiment


def with_case(root: Path, folder: Path) -> Path:
    """The block at ``folder`` with its case ``cases/in_water`` saved."""
    experiment, water = host("case")
    experiment.use(folder, medium=water, accept_licences=["NOASSERTION"])
    for ref in experiment._settings.assets:  # noqa: SLF001
        remember(licences_of(ref))
    experiment.licence("CC-BY-4.0")
    experiment.save(folder / "cases" / "in_water", steps=3)
    return folder


def test_a_referenced_block_stays_out_of_the_document_and_loads_from_its_container(tmp_path: Path):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        experiment = saved_by_reference(tmp_path, folder, tmp_path / "saved")
        document = yaml.safe_load((tmp_path / "saved" / DOCUMENT_FILE).read_text(encoding="utf-8"))
        lock = json.loads((tmp_path / "saved" / LOCK_FILE).read_text(encoding="utf-8"))
        shutil.rmtree(folder)
        loaded = Experiment.load(tmp_path / "saved")
    assert sorted(document["things"]) == ["water"] and document["interactions"] == []
    assert document["blocks"]["nudged_probe"]["tier"] == "referenced"
    (record,) = [held for held in lock["assets"] if held["name"] == "labx.nudged_probe-0.1.0.mimeblock"]
    assert record["locations"] == [] and record["licence"] == "NOASSERTION"
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    assert loaded.document() == experiment.document()


def test_a_referenced_blocks_changed_content_is_refused_as_block_changed(tmp_path: Path):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        saved_by_reference(tmp_path, folder, tmp_path / "saved")
        file, old, new = rng().choice((
            ("block.mime.yaml", "value: 2.0", "value: 2.25"), ("block.mime.yaml", "rung: quadratic", "rung: linear"),
            ("block.mime.yaml", "value: [0.3, -0.2, 0.1]", "value: [0.3, -0.2, 0.2]"),
            ("block.mime.yaml", "# The labx lab's probe", "# The lab's probe"),
            ("mime.toml", 'mime-lab-fixture = ">=0.1,<0.2"', 'mime-lab-fixture = ">=0.1,<0.3"')))
        edited(folder, file, old, new)
        other = pack_block(folder, out=tmp_path / "other.mimeblock", cases=False)
        from mime.blocks._container import container_record, stored
        from mime.blocks._definition import definition_of

        ref = container_record(definition_of(folder), other)
        stored(ref, other)
        remember(licences_of(ref))
        lock_path = tmp_path / "saved" / LOCK_FILE
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["assets"] = [dict(ref.to_record()) if held["name"] == ref.name else held for held in lock["assets"]]
        lock_path.write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
        shutil.rmtree(folder)
        with pytest.MonkeyPatch.context() as patch:
            if plant() == "unheld":
                patch.setattr(_referenced, "content_problem", lambda definition, use, words: None)
            with pytest.raises(LoadRefused) as refused:
                Experiment.load(tmp_path / "saved")
    assert refused.value.codes() == ["block.changed"], (file, old)


def test_pack_unpack_and_pull_keep_the_block_and_its_case_whole(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "out" / "labx.nudged_probe-0.1.0.mimeblock")
        again = pack_block(folder, out=tmp_path / "again.mimeblock")
        assert container.read_bytes() == again.read_bytes(), "the same folder makes the same bytes"
        with zipfile.ZipFile(container) as archive:
            names = archive.namelist()
            definition = archive.read("block/mime.toml").decode("utf-8")
        assert names[:2] == ["manifest.json", "NOTICE"] and "block/cases/in_water/experiment.mime.yaml" in names
        assert "[record]" not in definition and "#" not in definition
        unpacked = unpack_block(container, tmp_path / "unpacked")
        assert unpacked.manifest["content"] == _referenced.definition_of(folder).content_hash()
        destination = tmp_path / "pulled"
        assert main(["pull", str(container), "--into", str(destination)]) == 0
        assert "Block labx.nudged_probe 0.1.0 unpacked to" in capsys.readouterr().out
        assert verify_block(destination).reproduced


def test_a_container_whose_file_and_manifest_were_both_changed_is_refused_by_its_content(tmp_path: Path):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "block.mimeblock")
    from mime.saved._container import container_hash, sha256_of

    with zipfile.ZipFile(container) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    fragment = members["block/block.mime.yaml"].replace(b"value: 2.0", b"value: 9.0")
    members["block/block.mime.yaml"] = fragment
    manifest = json.loads(members["manifest.json"])
    manifest["files"]["block.mime.yaml"] = {"sha256": sha256_of(fragment), "size": len(fragment)}
    manifest["container_hash"] = container_hash(manifest, "container_hash")
    members["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=1) + "\n").encode("utf-8")
    from mime.saved._container import write_container
    from mime.saved._studies import BLOCK

    forged = write_container(tmp_path / "forged.mimeblock", members["manifest.json"], members["NOTICE"],
                             {name.removeprefix("block/"): data for name, data in members.items()
                              if name.startswith("block/")}, BLOCK)
    with pytest.raises(BuildRefused) as refused:
        unpack_block(forged, tmp_path / "unpacked")
    assert refused.value.codes() == ["block.changed"]


def test_pack_refuses_a_block_with_no_case_and_an_asset_file_no_record_names(tmp_path: Path):
    folder = block_copy(tmp_path)
    with pytest.raises(BuildRefused) as refused:
        pack_block(folder, out=tmp_path / "block.mimeblock")
    assert refused.value.codes() == ["block.no-case"]
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
    (folder / "assets").mkdir()
    (folder / "assets" / "stray.bin").write_bytes(b"bytes no record names")
    with pytest.raises(BuildRefused) as refused:
        pack_block(folder, out=tmp_path / "block.mimeblock")
    assert refused.value.codes() == ["bundle.referenced-copy"]


def _verify_code(folder: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(["verify", str(folder)])
    held = capsys.readouterr()
    return code, held.out + held.err


def test_verify_reruns_the_cases_and_reports_each_defect(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        code, printed = _verify_code(folder, capsys)
        assert code == 0 and ACCEPTANCE in printed and "The author says: The probe builds" in printed
        case = folder / "cases" / "in_water" / DOCUMENT_FILE
        text = case.read_text(encoding="utf-8")
        case.write_text(text.replace("seed: 0", "seed: 1"), encoding="utf-8")
        code, printed = _verify_code(folder, capsys)
        assert code == 2 and "load.input-hash" in printed
        case.write_text(text, encoding="utf-8")
        edited(folder, "mime.toml", 'mime-lab-fixture = ">=0.1,<0.2"', 'mime-lab-fixture = ">=7"')
        code, printed = _verify_code(folder, capsys)
        assert code == 2 and "block.requires" in printed and "so none of its 1 case ran" in printed
        edited(folder, "mime.toml", 'mime-lab-fixture = ">=7"', 'mime-lab-fixture = ">=0.1,<0.2"')
        edited(folder, "block.mime.yaml", "value: 2.0", "value: 2.5")
        code, printed = _verify_code(folder, capsys)
        assert code == 2 and "block.case-stale" in printed


def test_a_cases_referenced_block_is_read_from_the_block_that_holds_it(tmp_path: Path):
    folder = block_copy(tmp_path)
    referenced(folder)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
    (tmp_path / "elsewhere").mkdir()
    with own_cache(tmp_path / "elsewhere"):  # another cache: the case's container is not in it, nor is an acceptance
        from mime.assets import licences_named

        remember(licences_named("NOASSERTION"))  # the block's own licence is gated, found by either route
        unpacked = unpack_block(container, tmp_path / "unpacked").folder
        assert verify_block(unpacked).reproduced


def test_exp_use_takes_a_blocks_file(tmp_path: Path):
    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        container = pack_block(folder, out=tmp_path / "labx.nudged_probe-0.1.0.mimeblock")
        experiment, water = host()
        probe = experiment.use(container, medium=water)
    assert probe.probe.name == "probe" and experiment.document()["blocks"]["nudged_probe"]["tier"] == "copied"


def test_the_card_and_the_push_plan_of_a_block(tmp_path: Path):
    from mime.push import _service
    from mime.push._manifest import read_record
    from mime.push._plan import make_block_plan

    folder = block_copy(tmp_path)
    with own_cache(tmp_path):
        with_case(tmp_path, folder)
        card = block_card_text(folder)
        assert card.startswith("# A nudged probe (a lab's block in miniature)") and "Sockets, which its user binds: " \
               "`medium` (medium)." in card
        plan = make_block_plan(folder, _service.SERVICES["zenodo-sandbox"], read_record(folder), tmp_path / "staged")
    assert [file.key for file in plan.files] == ["labx.nudged_probe-0.1.0.mimeblock", "NOTICE.txt"]
    keywords: Any = plan.payload["metadata"]["subjects"] if "subjects" in plan.payload["metadata"] else \
        plan.payload["metadata"].get("keywords")
    assert "mime-block" in json.dumps(keywords) and "mime-block:labx.nudged_probe" in json.dumps(keywords)
