"""The precision a run computes in is stated (UX-1's F14): the build report and the result say ``float32`` or ``float64``, where
before neither did (the README quickstart runs in float32, JAX's default, and nothing printed it), and a build on a GPU whose
matmul precision is not ``highest`` warns (the 0.2 line's drag was 15 % off under reduced-precision TF32 matmuls).

**What this covers and what it does not.** The CPU path is tested for real: the report, the result, the run folder's
``outputs.json`` and a folder written before the precision was recorded. The GPU warning's *logic* is tested with the device and
the matmul precision read through a seam (``mime.env._host.device_and_matmul``), so that no GPU is needed: **the real GPU path
(``jax.default_backend()`` answering ``gpu``, and ``jax_default_matmul_precision`` as a GPU environment sets it) is not tested in
CI**, which has no GPU. The controls are the same builds with the highest precision and on a CPU, which warn of nothing."""

from __future__ import annotations

import json
import warnings

import jax
import pytest

from mime.experiment import Experiment
from mime.results import RUN_MANIFEST, RunResult
from mime.testing.toys import full_toy
from mime.units import s


@pytest.fixture(params=[False, True], ids=["float32", "float64"])
def x64(request):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param)
    yield "float64" if request.param else "float32"
    jax.config.update("jax_enable_x64", held)


def test_the_build_report_and_the_result_state_the_precision_in_both_dtypes(x64):
    built = full_toy().build()
    assert f"precision: {x64}" in built.report.summary() and built.report.precision == x64
    assert f"precision: {x64}" in built.report.text()
    result = built.run(s(0.04))
    assert f"precision: {x64}" in result.text().splitlines()
    assert ("float64" if jax.config.jax_enable_x64 else "float32") == x64


def test_the_precision_is_in_the_run_folder_and_a_folder_written_before_it_reads_without_it(x64, tmp_path):
    built = full_toy().build()
    out = tmp_path / "run"
    built.run_to(out, s(0.04))
    assert json.loads((out / RUN_MANIFEST).read_text(encoding="utf-8"))["precision"] == x64
    assert f"precision: {x64}" in RunResult.read(out).text()
    manifest = json.loads((out / RUN_MANIFEST).read_text(encoding="utf-8"))
    del manifest["precision"]
    (out / RUN_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    older = RunResult.read(out)
    assert older.layout.precision is None and "precision" not in older.text()


@pytest.mark.parametrize(("backend", "precision", "warned"), [
    ("gpu", None, True), ("gpu", "default", True), ("gpu", "tensorfloat32", True), ("gpu", "high", True),
    ("gpu", "highest", False), ("cpu", None, False), ("cpu", "default", False)])
def test_the_gpu_warning_is_made_for_a_gpu_whose_matmul_precision_is_not_the_highest(backend, precision, warned):
    from mime.env._host import matmul_precision_warning, precision_note

    text = matmul_precision_warning(backend, precision)
    assert (text is not None) is warned
    if warned:
        assert "'highest'" in text and "JAX_DEFAULT_MATMUL_PRECISION=highest" in text and "15 %" in text
    assert (precision_note(lambda: (backend, precision))[1] is not None) is warned


def test_a_build_on_a_gpu_warns_and_the_report_says_so_read_through_the_seam(monkeypatch):
    from mime.env import _host
    from mime.env._host import MatmulPrecisionReduced

    monkeypatch.setattr(_host, "device_and_matmul", lambda: ("gpu", None))
    with pytest.warns(MatmulPrecisionReduced, match="not 'highest'"):
        built = full_toy().build()
    assert any(line.startswith("WARNING: this run is on a GPU") for line in built.report.summary())
    assert built.report.matmul_warning is not None


@pytest.mark.parametrize("read", [lambda: ("gpu", "highest"), lambda: ("cpu", None)], ids=["gpu at the highest", "cpu"])
def test_control_the_highest_precision_and_a_cpu_build_without_a_warning(monkeypatch, read):
    from mime.env import _host
    from mime.env._host import MatmulPrecisionReduced

    monkeypatch.setattr(_host, "device_and_matmul", read)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatmulPrecisionReduced)
        built = full_toy().build()
    assert built.report.matmul_warning is None and not any("WARNING" in line for line in built.report.summary())
