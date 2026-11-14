"""A lab's own MPEM calibration (the calibration type ``mpem/1``): a record its package ships through the
``mime.assets`` entry point, named by the record's name and sha256, read from the asset cache through its licence gate.

The lab fixture ships two small synthetic three-coil calibrations: ``labx_emns.yaml`` under Apache-2.0 (open) and
``labx_emns_nc.yaml`` under CC-BY-NC-4.0 (conditional). Stored in a cache of the test's own:

- the open one builds; its numbers are the file's as the reader reads it; the lock records the record; the build
  report names its type, tier and the unit its file names;
- **the gate**: the conditional one, never accepted, is refused where nobody can be asked
  (``asset.licence-not-accepted``, its own code, not an entry's defect); its control, accepted, builds;
- a record not stored is refused (``asset.missing``); a sha256 other than the record's is refused, naming the record;
  a name that is no packaged calibration and no record is refused as before."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from believed_cases import believed_sphere, model_of

from mime import assets
from mime.build import BuildRefused
from mime.data import Asset
from mime.fields import CalibratedMultipole
from mime.nodes.actuation import read_calibration
from mime.saved import LOCK_FILE

lab = pytest.importorskip("mime_lab_fixture")
OPEN, CONDITIONAL = "labx_emns.yaml", "labx_emns_nc.yaml"
#: The plant a control's detection rate is measured with (``AP3_PLANT``): ``gate``, the calibration read around the
#: licence gate (the cached file taken with no acceptance asked for).
PLANT = os.environ.get("AP3_PLANT", "")


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    held = tmp_path / "cache"
    monkeypatch.setenv(assets.ASSETS_DIR_VARIABLE, str(held))
    monkeypatch.delenv(assets.OFFLINE_VARIABLE, raising=False)
    return held


def lab_coils(name: str, sha256: str | None = None) -> CalibratedMultipole:
    return CalibratedMultipole(calibration=Asset(name, sha256 or assets.reference(name).sha256))


def rig(name: str, sha256: str | None = None):
    """The believed sphere's rig on the lab's array: its declared region is about the world's origin."""
    return believed_sphere(truth=lab_coils(name, sha256))


def test_the_labs_open_calibration_builds_its_numbers_the_files_and_the_lock_records_it(cache: Path, tmp_path: Path):
    ref = assets.reference(OPEN)
    assets.fetch(ref)
    held = read_calibration(assets.locate(ref).read_bytes())
    model = model_of(rig(OPEN), "coils.magnetic")
    assert list(model.theta["coefficients"]) == list(held.coefficients.ravel())
    assert model.commands["currents"].shape == (3,)
    built = believed_sphere(truth=lab_coils(OPEN), controller=False).build()
    (line,) = built.report.calibrations
    assert line.startswith(f"coils.magnetic: calibration {OPEN} (mpem/1, sha256 {ref.sha256[:12]}, Apache-2.0, tier open)")
    assert "as the file names it: labx_emns" in line and "declared, never fitted: Workspace_Dimensions" in line
    folder = tmp_path / "saved"
    built.save(folder, steps=2)
    recorded = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8")).get("assets", [])
    assert [held["name"] for held in recorded] == [OPEN]


@pytest.mark.parametrize("accepted", [False, True], ids=["never_accepted", "control_accepted"])
def test_a_conditional_calibration_nothing_accepted_is_refused_by_the_gate(cache: Path, accepted: bool,
                                                                        monkeypatch: pytest.MonkeyPatch):
    from mime.assets import _acceptances, licences_of

    ref = assets.reference(CONDITIONAL)
    assets.fetch(ref, accept_licences=("CC-BY-NC-4.0",))  # stored, its acceptance holding for that call alone
    if accepted:
        _acceptances.remember(list(licences_of(ref)))  # as a person who fetched it at a prompt accepted it
        rig(CONDITIONAL).plan()
        return
    if PLANT == "gate":
        from mime.assets import _cache

        monkeypatch.setattr(assets, "locate", lambda held, **_: _cache._found(held, None, None))  # noqa: SLF001
    with pytest.raises(BuildRefused) as refused:
        rig(CONDITIONAL).plan()
    assert refused.value.codes() == ["asset.licence-not-accepted"]


def test_a_record_not_stored_is_refused_as_missing(cache: Path):
    with pytest.raises(BuildRefused) as refused:
        rig(OPEN).plan()
    assert refused.value.codes() == ["asset.missing"]


def test_another_sha256_than_the_records_is_refused_naming_the_record(cache: Path):
    assets.fetch(assets.reference(OPEN))
    with pytest.raises(BuildRefused) as refused:
        rig(OPEN, "0" * 64).plan()
    assert refused.value.codes() == ["setting.value"] and f"the record {OPEN!r} names the bytes" in str(refused.value)


def test_a_name_that_is_no_calibration_and_no_record_is_refused_as_before(cache: Path):
    with pytest.raises(BuildRefused) as refused:
        rig("labx_missing.yaml", "0" * 64).plan()
    assert refused.value.codes() == ["setting.value"] and "is no calibration's" in str(refused.value)
