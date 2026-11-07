"""The small warning a reference made on a GPU with XLA's deterministic mode off carries: once, where the save writes it,
and as the last line of ``verify``'s report on such a folder; never on the CPU, never in deterministic mode. The GPU is
the lock's word here: its environment is planted, since no GPU runs in these tests. The card's tolerance line states the
same levels, and the rule of the folder's form."""

import dataclasses
import shutil
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mime.saved import NondeterministicReference, verify_history
from mime.saved import _record
from mime.saved._card import card_facts, card_of
from mime.saved._folder import DETERMINISTIC_OFF, SAVED_DETERMINISTIC_OFF

GPU = {"device_kind": "Tesla T4", "driver": "580.82.07", "cuda_driver": 13000, "cuda_runtime": 13000, "cudnn": 92800}


@contextmanager
def built_under(device: str, deterministic: bool) -> Iterator[None]:
    """A build whose environment says ``device`` with the mode ``deterministic``, as a GPU's would."""
    real = _record.environment

    def planted():
        held = dict(real())
        if device == "gpu":
            held.update(device="gpu", gpu=GPU, deterministic=deterministic,
                        xla_flags="--xla_gpu_deterministic_ops=true" if deterministic else "")
        return held

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_record, "environment", planted)
        yield


def saved_warnings(tmp_path: Path, device: str, deterministic: bool) -> list[warnings.WarningMessage]:
    from mime.testing.toys import full_toy

    with built_under(device, deterministic), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        full_toy().save(tmp_path / "toy")
    return [held for held in caught if issubclass(held.category, NondeterministicReference)]


def test_a_save_on_a_gpu_with_deterministic_mode_off_warns_once(tmp_path: Path):
    (found,) = saved_warnings(tmp_path, "gpu", False)
    assert str(found.message) == SAVED_DETERMINISTIC_OFF
    assert "within tolerance, not bit for bit" in SAVED_DETERMINISTIC_OFF and "--deterministic" in SAVED_DETERMINISTIC_OFF


def test_a_save_on_the_cpu_never_warns(tmp_path: Path):
    assert saved_warnings(tmp_path, "cpu", False) == []


def test_a_save_on_a_gpu_in_deterministic_mode_never_warns(tmp_path: Path):
    assert saved_warnings(tmp_path, "gpu", True) == []


def report_of(device: str, deterministic: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: [])
    lock = {"reference": {"steps": 2, "rtol": 1e-5, "atol": 0.0},
            "environment": {"device": device, "deterministic": deterministic}}
    reference = {"history/n/x": np.arange(2.0)}
    return verify_history(SimpleNamespace(lock=lock, reference=reference, version=10), {"n": {"x": np.arange(2.0)}})


def test_verify_of_a_gpu_reference_made_with_deterministic_mode_off_ends_with_the_warning(
        monkeypatch: pytest.MonkeyPatch):
    report = report_of("gpu", False, monkeypatch)
    assert report.warning == DETERMINISTIC_OFF and report.text().splitlines()[-1] == DETERMINISTIC_OFF
    assert report.text().count(DETERMINISTIC_OFF) == 1


@pytest.mark.parametrize(("device", "deterministic"), [("cpu", False), ("cpu", True), ("gpu", True)],
                         ids=["cpu", "cpu_flagged", "gpu_deterministic"])
def test_verify_of_a_cpu_or_a_deterministic_reference_never_warns(device: str, deterministic: bool,
                                                                 monkeypatch: pytest.MonkeyPatch):
    report = report_of(device, deterministic, monkeypatch)
    assert report.warning is None and DETERMINISTIC_OFF not in report.text() and report.bit_for_bit


FREE9 = Path(__file__).parent / "form9" / "free_ball"


def reproduces(facts, **environment: object) -> str:
    """The card's "What reproduces" section, its lock's environment changed by ``environment``."""
    lock = {**facts.lock, "environment": {**facts.lock["environment"], **environment}}
    text = card_of(dataclasses.replace(facts, lock=lock), None)
    return text[text.index("## What reproduces"):text.index("## Validity lines")]


def test_the_cards_tolerance_line_states_each_level_and_the_rule_of_the_folders_form(tmp_path: Path):
    from mime.testing.toys import full_toy

    full_toy().save(tmp_path / "toy")
    tenth = card_facts(tmp_path / "toy")
    here = reproduces(tenth)
    assert "bit for bit in the environment the lock was written in" in here and "never bit for bit" not in here
    rtol = tenth.lock["reference"]["rtol"]
    assert f"within rtol {rtol!r} of the largest magnitude of each block of a field" in here
    assert "30.0 times the block's resolution" in here and "with no absolute part" in here
    assert "element by element" not in here
    off = reproduces(tenth, device="gpu", deterministic=False)
    assert "within tolerance everywhere, the GPU it was made on" in off and "never bit for bit" in off
    assert "deterministic mode off" in off and "bit for bit in the environment the lock was written in" not in off
    on = reproduces(tenth, device="gpu", deterministic=True)
    assert "bit for bit in the environment the lock was written in" in on and "never bit for bit" not in on, \
        "a reference made in deterministic mode is claimed bit for bit where it was written, as the CPU's"
    ninth = card_facts(Path(shutil.copytree(FREE9, tmp_path / "free_ball")))
    held = reproduces(ninth)
    assert "within rtol 1e-06 and atol 1e-09, element by element" in held and "resolution" not in held
