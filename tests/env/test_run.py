"""``mime run``, the command a runner's script calls after ``mime env check``: it runs a saved experiment in the
precision its lock records and writes the outputs, a record and the run's own saved form, whose lock records the
environment the run was made in, the runner's record included, so that a later run there reproduces it bit for bit.
And XLA's cache directory: one per jaxlib release, platform and device, a CPU's by the hash of its flags."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from mime.cli import main as mime_main
from mime.env import cpu_record, use_xla_cache, xla_cache_dir
from mime.env._xla_cache import point_xla_cache
from mime.saved import read_folder
from mime.testing.toys import full_toy
from _platform_support import needs_readable_cpu_flags


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


@needs_readable_cpu_flags
def test_a_run_writes_its_outputs_and_its_own_saved_form_in_the_locks_precision(saved, tmp_path, monkeypatch,
                                                                                capsys):
    monkeypatch.setenv("MIME_RUNNER", json.dumps({"runner": "a test"}))
    jax.config.update("jax_enable_x64", False)
    dt = read_folder(saved).lock["resolved"]["run"]["dt"]
    out = tmp_path / "run"
    assert mime_main(["run", str(saved), "--duration", repr(4 * dt), "--out", str(out), "--history"]) == 0
    assert jax.config.jax_enable_x64 is True
    record = json.loads((out / "run.json").read_text())
    assert record["steps"] == 4 and record["environment"]["runner"] == {"runner": "a test"}
    with np.load(out / "outputs.npz", allow_pickle=False) as outputs:
        finals = [key for key in outputs.files if key.startswith("final/")]
        histories = [key for key in outputs.files if key.startswith("history/")]
        assert finals and len(histories) == len(finals)
        assert all(outputs[key].shape[0] == 4 for key in histories)
    lock = read_folder(out / "experiment").lock
    assert lock["environment"]["runner"] == {"runner": "a test"} and lock["environment"]["x64"] is True
    capsys.readouterr()
    assert mime_main(["verify", str(out / "experiment")]) == 0
    assert "bit for bit" in capsys.readouterr().out


def test_a_duration_that_is_not_whole_steps_is_refused(saved, tmp_path, capsys):
    dt = read_folder(saved).lock["resolved"]["run"]["dt"]
    assert mime_main(["run", str(saved), "--duration", repr(1.5 * dt), "--out", str(tmp_path / "run")]) == 2
    assert "not a whole number of base steps" in capsys.readouterr().err and not (tmp_path / "run").exists()


def test_the_xla_cache_directory_is_one_per_jaxlib_platform_and_what_decides_the_compiled_code(tmp_path):
    kinds = ("NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe")
    gpus = {xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="gpu", device_kind=kind, cpu_flags_sha256=None,
                          compute_capability="8.0") for kind in kinds}
    assert gpus == {tmp_path / "jaxlib-0.12.0.1" / "gpu" / "sm_80"}
    hopper = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="gpu", device_kind="NVIDIA H100 80GB HBM3",
                           cpu_flags_sha256=None, compute_capability="9.0")
    ada = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="gpu", device_kind="NVIDIA L4", cpu_flags_sha256=None,
                        compute_capability="8.9")
    assert hopper.name == "sm_90" and ada.name == "sm_89" and len({hopper, ada, *gpus}) == 3
    later = xla_cache_dir(tmp_path, jaxlib="0.12.1", platform="gpu", device_kind=kinds[0], cpu_flags_sha256=None,
                          compute_capability="8.0")
    assert later != next(iter(gpus))
    unreported = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="gpu", device_kind="NVIDIA A100-SXM4-40GB",
                               cpu_flags_sha256=None)
    assert unreported == tmp_path / "jaxlib-0.12.0.1" / "gpu" / "NVIDIA_A100-SXM4-40GB"
    cpu = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="cpu", device_kind="cpu", cpu_flags_sha256="f" * 64)
    other = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="cpu", device_kind="cpu", cpu_flags_sha256="e" * 64)
    assert cpu == tmp_path / "jaxlib-0.12.0.1" / "cpu" / ("f" * 16) and other != cpu


def test_the_cache_is_pointed_at_this_processs_directory_and_turned_off_again(tmp_path):
    try:
        held = use_xla_cache(tmp_path)
        assert held.is_dir() and jax.config.jax_compilation_cache_dir == str(held)
        assert held.name == (cpu_record()["flags_sha256"] or "unknown")[:16]
    finally:
        point_xla_cache(None)
    assert jax.config.jax_compilation_cache_dir is None


@needs_readable_cpu_flags
def test_mime_verify_applies_the_locks_precision_so_a_float64_experiment_is_not_reported_moved(saved, capsys):
    """In a float32 process, as ``mime verify`` finds itself by default; the lock records float64."""
    jax.config.update("jax_enable_x64", False)
    assert mime_main(["verify", str(saved)]) == 0
    out = capsys.readouterr().out
    assert jax.config.jax_enable_x64 is True and "bit for bit" in out and "moved" not in out
    assert "x64: built under True, loaded under False" not in out


def test_mime_verify_without_applying_the_lock_reports_a_float64_experiment_moved(saved, capsys):
    """The control the test above rests on: the same folder, verified by the API in a float32 process, does move."""
    from mime.experiment import Experiment

    jax.config.update("jax_enable_x64", False)
    report = Experiment.load(saved).build().verify()
    assert not report.reproduced and not report.same_environment


def test_the_apis_run_to_writes_what_mime_run_does(saved, tmp_path):
    from mime.env import apply_lock_precision
    from mime.experiment import RUN_EXPERIMENT, RUN_OUTPUTS, RUN_RECORD, Experiment
    from mime.units import s

    dt = read_folder(saved).lock["resolved"]["run"]["dt"]
    apply_lock_precision(saved)
    experiment = Experiment.load(saved)
    written = experiment.build().run_to(tmp_path / "api", s(3 * dt), history=True, source=saved)
    assert written.steps == 3 and written.outputs.name == RUN_OUTPUTS and written.experiment.name == RUN_EXPERIMENT
    assert (tmp_path / "api" / RUN_RECORD).is_file() and json.loads(written.record_file.read_text()) == written.record
    assert mime_main(["run", str(saved), "--duration", repr(3 * dt), "--out", str(tmp_path / "cli"), "--history"]) == 0
    with np.load(tmp_path / "api" / RUN_OUTPUTS) as api, np.load(tmp_path / "cli" / RUN_OUTPUTS) as cli:
        assert sorted(api.files) == sorted(cli.files)
        assert all(np.array_equal(api[key], cli[key]) for key in api.files)
    assert (read_folder(tmp_path / "api" / RUN_EXPERIMENT).lock["environment"]
            == read_folder(tmp_path / "cli" / RUN_EXPERIMENT).lock["environment"])
