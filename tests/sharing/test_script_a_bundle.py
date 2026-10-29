"""Script A, a real experiment, shared: saved, packed, pulled into a new folder and verified bit for bit in the
environment it was written in. The record arm adds the material records (OpenMaterica's water and N45) and the parts.

"Bit for bit" is the saved form's claim within one environment, and this is that claim across a bundle; on another
machine the same bundle verifies within the lock's tolerances and claims no bits."""

import zipfile
from pathlib import Path

import pytest
from script_a import ALL_RECORDS, record_arm, script_a

from mime.experiment import Experiment
from mime.saved import pack, pull


@pytest.mark.parametrize("make", [script_a, lambda: record_arm(ALL_RECORDS)], ids=["reproduction arm", "record arm"])
def test_script_a_round_trips_through_a_bundle_bit_for_bit(tmp_path: Path, make):
    saved = tmp_path / "saved"
    make().save(saved)
    bundle = pack(saved)
    folder = pull(bundle, into=tmp_path / "pulled", fetch_assets=False)
    for path in sorted(saved.rglob("*")):
        if path.is_file():
            assert (folder / path.relative_to(saved)).read_bytes() == path.read_bytes()
    experiment = Experiment.load(folder)
    report = experiment.build().verify()
    assert report.reproduced and report.same_environment, report.text()
    assert experiment.load_report is not None and experiment.load_report.same_environment
    names = zipfile.ZipFile(bundle).namelist()
    assert (any(name.startswith("experiment/records/") for name in names)) == (make is not script_a)
