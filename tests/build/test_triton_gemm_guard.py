"""The GPU build warns when XLA's Triton GEMM fusion is left on (MIME-ANO-157), beside the matmul-precision warning of
``test_precision_visible.py``: a build on a GPU whose ``XLA_FLAGS`` do not hold ``--xla_gpu_enable_triton_gemm=false``
warns (``TritonGemmUnguarded``) and the report says so; ``mime env check`` prints the same line (``tests/env/test_check.py``).

**What this covers and what it does not.** The warning's logic is tested with the device and ``XLA_FLAGS`` read through the
seam ``mime.env._host.gpu_environment``, and the seam's reading half is tested on its own (it answers the environment's
flags): so no GPU is needed. **The real GPU path, and the wrong float64 operator the flag prevents, are not tested in CI**,
which has no GPU; that the flag prevents it is one comparison on one A100, recorded in the anomaly. The controls are the same
builds with the flag, and on a CPU, which warn of nothing, and a flag whose name only starts like the Triton one.

MIME never sets the flag for a process it did not start: the last test reads the environment before and after a build."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

import pytest

from mime.testing.toys import full_toy

OFF = "--xla_gpu_enable_triton_gemm=false"


@pytest.mark.parametrize(("backend", "flags", "warned"), [
    ("gpu", "", True), ("gpu", OFF, False), ("gpu", f"--xla_dump_to=/x {OFF}", False), ("gpu", f"{OFF} --foo=1", False),
    ("gpu", "--xla_gpu_enable_triton_gemm=0", False), ("gpu", "--xla_gpu_enable_triton_gemm=False", False),
    ("gpu", "--xla_gpu_enable_triton_gemm=true", True), ("gpu", "--xla_gpu_enable_triton_gemm", True),
    ("gpu", f"{OFF} --xla_gpu_enable_triton_gemm=true", True), ("gpu", f"--xla_gpu_enable_triton_gemm=true {OFF}", False),
    ("gpu", "--xla_gpu_deterministic_ops=true", True),
    ("gpu", "--xla_gpu_enable_triton_gemm_any=false", True),
    ("cpu", "", False), ("cpu", "--xla_gpu_enable_triton_gemm=true", False), ("cpu", OFF, False)])
def test_the_warning_is_made_for_a_gpu_whose_flags_do_not_turn_the_triton_gemm_fusion_off(backend, flags, warned):
    from mime.env._host import precision_note, triton_gemm_warning

    text = triton_gemm_warning(backend, flags)
    assert (text is not None) is warned
    if warned:
        assert OFF in text and "XLA_FLAGS" in text and "MIME-ANO-157" in text and "CPU-equality" in text
    assert (precision_note(lambda: (backend, "highest", flags))[2] is not None) is warned


def test_the_seam_answers_the_environments_flags(monkeypatch):
    from mime.env import _host

    monkeypatch.setenv("XLA_FLAGS", OFF)
    assert _host.gpu_environment()[2] == OFF
    monkeypatch.delenv("XLA_FLAGS")
    assert _host.gpu_environment()[2] == ""


def test_a_build_on_a_gpu_without_the_flag_warns_and_the_report_says_so_read_through_the_seam(monkeypatch):
    from mime.env import _host
    from mime.env._host import TritonGemmUnguarded

    monkeypatch.setattr(_host, "gpu_environment", lambda: ("gpu", "highest", ""))
    with pytest.warns(TritonGemmUnguarded, match="xla_gpu_enable_triton_gemm=false"):
        built = full_toy().build()
    assert any(line.startswith("WARNING: this run is on a GPU without --xla_gpu_enable_triton_gemm=false")
               for line in built.report.summary())
    assert built.report.triton_warning is not None and built.report.matmul_warning is None


def test_a_gpu_build_without_either_guard_carries_both_warnings(monkeypatch):
    from mime.env import _host
    from mime.env._host import MatmulPrecisionReduced, TritonGemmUnguarded

    monkeypatch.setattr(_host, "gpu_environment", lambda: ("gpu", None, ""))
    with pytest.warns() as caught:
        built = full_toy().build()
    kinds = {type(w.message) for w in caught}
    assert {MatmulPrecisionReduced, TritonGemmUnguarded} <= kinds
    assert built.report.matmul_warning is not None and built.report.triton_warning is not None


@pytest.mark.parametrize("read", [lambda: ("gpu", "highest", OFF), lambda: ("cpu", None, ""),
                                  lambda: ("cpu", None, "--xla_gpu_enable_triton_gemm=true")],
                         ids=["gpu with the flag", "cpu", "cpu, the flag on"])
def test_control_the_flag_and_a_cpu_build_without_this_warning(monkeypatch, read):
    from mime.env import _host
    from mime.env._host import TritonGemmUnguarded

    monkeypatch.setattr(_host, "gpu_environment", read)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TritonGemmUnguarded)
        built = full_toy().build()
    assert built.report.triton_warning is None
    assert not any("triton" in line.lower() for line in built.report.summary())


def test_the_real_cpu_path_builds_without_the_warning(monkeypatch):
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        built = full_toy().build()
    assert built.report.triton_warning is None


READS_ONLY = """
import os, sys
import mime.env
from jax._src import xla_bridge
os.environ["XLA_FLAGS"] = sys.argv[1]
text = mime.env.triton_gemm_warning("gpu", mime.env.xla_flags())
none = mime.env.triton_gemm_warning("cpu", mime.env.xla_flags())
print(xla_bridge.backends_are_initialized(), text is not None, none is None)
"""


@pytest.mark.parametrize("flags", ["", OFF])
def test_reading_the_flags_for_the_warning_starts_no_backend(flags):
    """The seam's own ``jax.default_backend()`` starts the backend a build already computes on; what the warning adds
    (the variable, its parse) does not."""
    held = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
    done = subprocess.run([sys.executable, "-c", READS_ONLY, flags], capture_output=True, text=True, check=False,
                          env={**held, "JAX_PLATFORMS": "cpu"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["False", str(flags == ""), "True"]


def test_a_build_never_sets_the_flag_for_the_process(monkeypatch):
    from mime.env import _host

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    monkeypatch.setattr(_host, "gpu_environment", lambda: ("gpu", "highest", ""))
    with pytest.warns(_host.TritonGemmUnguarded):
        full_toy().build()
    assert "XLA_FLAGS" not in os.environ
