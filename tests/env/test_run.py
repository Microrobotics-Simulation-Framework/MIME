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


def test_the_xla_cache_directory_is_one_per_jaxlib_platform_and_device(tmp_path):
    gpu = xla_cache_dir(tmp_path, jaxlib="0.12.0.1", platform="gpu", device_kind="NVIDIA A100-SXM4-40GB",
                        cpu_flags_sha256=None)
    assert gpu == tmp_path / "jaxlib-0.12.0.1" / "gpu" / "NVIDIA_A100-SXM4-40GB"
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
