"""The lock's environment holds what no pin can: the GPU (its device kind, its driver, CUDA and cuDNN), the CPU (its
model and the hash of its flags), the matmul precision, XLA's flags and its deterministic mode, and the runner's own
record. They round-trip through a save and a load, and a difference in any one of them, in the lock or in the process
that loads it, makes the load another environment, so that ``verify`` compares within the lock's tolerances and claims
no bits."""

import json
from pathlib import Path

import jax
import pytest

from mime.build import BuildRefused
from mime.env import CpuRecord, GpuRecord, deterministic, runner_record, with_deterministic
from mime.experiment import Experiment
from mime.saved import LOCK_FILE, environment, read_folder
from mime.saved._schema import CpuRecordDoc, GpuRecordDoc, typed_dict_fields
from mime.testing.toys import full_toy

NEW_FIELDS = ("gpu", "cpu", "matmul_precision", "xla_flags", "deterministic", "runner")
#: For each new field, a value of its type that this process does not have.
ANOTHER = {
    "gpu": {"device_kind": "NVIDIA A100-SXM4-40GB", "driver": "610.57.04", "cuda_driver": 13030,
            "cuda_runtime": 12090, "cudnn": 92700},
    "cpu": {"model": "another CPU", "flags_sha256": "0" * 64},
    "matmul_precision": "highest",
    "xla_flags": "--xla_gpu_deterministic_ops=true",
    "deterministic": True,
    "runner": {"runner": "modal", "image_builder": "2026.10"},
}


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(autouse=True)
def no_runner(monkeypatch):
    monkeypatch.delenv("MIME_RUNNER", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("saved")
    full_toy().save(folder)
    return folder


def _copy_with(saved: Path, target: Path, key: str | None = None) -> Path:
    target.mkdir()
    for path in saved.rglob("*"):
        if path.is_file():
            (target / path.relative_to(saved)).parent.mkdir(parents=True, exist_ok=True)
            (target / path.relative_to(saved)).write_bytes(path.read_bytes())
    if key is not None:
        lock = json.loads((target / LOCK_FILE).read_text())
        lock["environment"][key] = ANOTHER[key]
        (target / LOCK_FILE).write_text(json.dumps(lock))
    return target


def test_the_new_fields_round_trip_through_a_save_and_a_load(saved):
    lock = read_folder(saved).lock
    here = environment()
    assert {key: lock["environment"][key] for key in NEW_FIELDS} == {key: here[key] for key in NEW_FIELDS}
    assert lock["environment"]["gpu"] is None and lock["environment"]["runner"] == {}
    assert lock["environment"]["cpu"]["flags_sha256"] is not None
    experiment = Experiment.load(saved)
    assert experiment.load_report is not None and experiment.load_report.same_environment
    assert experiment.build().verify().same_environment


@pytest.mark.parametrize("key", NEW_FIELDS)
def test_a_change_in_any_new_field_compares_within_tolerances_and_claims_no_bits(saved, tmp_path, key):
    folder = _copy_with(saved, tmp_path / "changed", key)
    experiment = Experiment.load(folder)
    report = experiment.load_report
    assert report is not None and not report.same_environment
    assert any(line.startswith(f"{key}: built under ") for line in report.environment), report.text()
    verified = experiment.build().verify()
    print(f"\n{verified.text()}")
    assert verified.reproduced and not verified.same_environment and "within rtol" in verified.text()


def test_a_runner_set_in_the_loading_process_is_another_environment(saved, monkeypatch):
    monkeypatch.setenv("MIME_RUNNER", json.dumps({"runner": "modal"}))
    experiment = Experiment.load(saved)
    assert experiment.load_report is not None and not experiment.load_report.same_environment
    assert experiment.load_report.environment == ("runner: built under {}, loaded under {'runner': 'modal'}",)


def test_a_runner_record_is_a_json_object_of_strings():
    assert runner_record({}) == {} and runner_record({"MIME_RUNNER": " "}) == {}
    assert runner_record({"MIME_RUNNER": '{"b": "2", "a": "1"}'}) == {"a": "1", "b": "2"}
    for raw in ("modal", "[]", '{"a": 1}', '{"a": null}', json.dumps({str(i): "x" for i in range(33)}),
                json.dumps({"a": "x" * 257})):
        with pytest.raises(BuildRefused) as refused:
            runner_record({"MIME_RUNNER": raw})
        assert refused.value.codes() == ["env.runner"]


@pytest.mark.parametrize(("flags", "on"), [
    ("", False), ("--xla_gpu_deterministic_ops", True), ("--xla_gpu_deterministic_ops=true", True),
    ("--xla_gpu_deterministic_ops=1", True), ("--xla_gpu_deterministic_ops=false", False),
    ("--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_ops=false", False),
    ("--xla_gpu_deterministic_ops_x=true", False), ("--xla_cpu_max_isa=AVX2", False),
])
def test_the_deterministic_mode_is_read_from_xla_flags(flags, on):
    assert deterministic(flags) is on
    assert deterministic(with_deterministic(flags, True)) and not deterministic(with_deterministic(flags, False))


def test_the_records_and_the_locks_forms_are_one():
    assert typed_dict_fields(GpuRecordDoc) == typed_dict_fields(GpuRecord)
    assert typed_dict_fields(CpuRecordDoc) == typed_dict_fields(CpuRecord)
