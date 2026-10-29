"""The lock's environment holds what no pin can: the GPU (its device kind, its driver, CUDA and cuDNN), the CPU (its
model and the hash of its flags), the matmul precision, XLA's flags and its deterministic mode, and the runner's own
record. They round-trip through a save and a load, and a difference in any one of them, in the lock or in the process
that loads it, makes the load another environment, so that ``verify`` compares within the lock's tolerances and claims
no bits. The runner's information (a workspace's image ID, a client's version) is stored and never compared: two runs
that differ only there are the same environment. A field of a device that could not be read is never equal to
anything, ``None`` included."""

import json
from pathlib import Path

import jax
import pytest

from mime.build import BuildRefused
import mime.saved._folder as folder_module
from mime.env import CpuRecord, GpuRecord, deterministic, runner_info, runner_record, with_deterministic
from mime.experiment import Experiment
from mime.saved import LOCK_FILE, environment, environment_differences, read_folder
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
    monkeypatch.delenv("MIME_RUNNER_INFO", raising=False)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    held = jax.config.jax_default_matmul_precision
    yield
    jax.config.update("jax_default_matmul_precision", held)


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("saved")
    full_toy().save(folder)
    return folder


def _copy_with(saved: Path, target: Path, key: str | None = None, value: object = None) -> Path:
    target.mkdir()
    for path in saved.rglob("*"):
        if path.is_file():
            (target / path.relative_to(saved)).parent.mkdir(parents=True, exist_ok=True)
            (target / path.relative_to(saved)).write_bytes(path.read_bytes())
    if key is not None:
        lock = json.loads((target / LOCK_FILE).read_text())
        lock["environment"][key] = ANOTHER[key] if value is None else value
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
    assert any(line.startswith((f"{key}: built under ", f"{key}.")) for line in report.environment), report.text()
    verified = experiment.build().verify()
    print(f"\n{verified.text()}")
    assert verified.reproduced and not verified.same_environment and "within rtol" in verified.text()


@pytest.mark.parametrize("flags", ["--xla_cpu_max_isa=AVX2", "--xla_gpu_deterministic_ops=true"])
def test_xla_flags_and_the_deterministic_mode_are_read_from_the_loading_process(saved, monkeypatch, flags):
    """A recorder that wrote a constant would pass the lock-side tests; this one changes the process, not the lock."""
    monkeypatch.setenv("XLA_FLAGS", flags)
    here = environment()
    assert here["xla_flags"] == flags and here["deterministic"] is deterministic(flags)
    differences = environment_differences(read_folder(saved).lock)
    assert any(line.startswith("xla_flags: built under ''") and flags in line for line in differences)
    assert any(line.startswith("deterministic:") for line in differences) is deterministic(flags)
    report = Experiment.load(saved).load_report
    assert report is not None and not report.same_environment


def test_the_matmul_precision_is_read_from_the_loading_process(saved):
    jax.config.update("jax_default_matmul_precision", "float32")
    assert environment()["matmul_precision"] == "float32"
    differences = environment_differences(read_folder(saved).lock)
    assert differences == ["matmul_precision: built under None, loaded under 'float32'"]
    experiment = Experiment.load(saved)
    assert experiment.load_report is not None and not experiment.load_report.same_environment
    assert "within rtol" in experiment.build().verify().text()


def test_a_runner_set_in_the_loading_process_is_another_environment(saved, monkeypatch):
    monkeypatch.setenv("MIME_RUNNER", json.dumps({"runner": "modal"}))
    experiment = Experiment.load(saved)
    assert experiment.load_report is not None and not experiment.load_report.same_environment
    assert experiment.load_report.environment == ("runner: built under {}, loaded under {'runner': 'modal'}",)


def test_the_runners_information_is_stored_and_never_compared(saved, tmp_path, monkeypatch):
    """Two workspaces build one image under two IDs and use two clients: that is not another environment."""
    folder = _copy_with(saved, tmp_path / "other_workspace", "runner_info",
                        {"image": "im-one", "modal_client": "1.6.3"})
    monkeypatch.setenv("MIME_RUNNER_INFO", json.dumps({"image": "im-two", "modal_client": "1.7.0"}))
    assert environment()["runner_info"] == {"image": "im-two", "modal_client": "1.7.0"}
    assert read_folder(folder).lock["environment"]["runner_info"] == {"image": "im-one", "modal_client": "1.6.3"}
    assert environment_differences(read_folder(folder).lock) == []
    experiment = Experiment.load(folder)
    assert experiment.load_report is not None and experiment.load_report.same_environment
    verified = experiment.build().verify()
    assert verified.same_environment and "bit for bit" in verified.text()


def test_two_workspaces_records_with_different_image_ids_are_the_same_environment_and_a_changed_pin_is_not(saved,
                                                                                                         tmp_path):
    record = {"runner": "modal", "image_builder": "2026.10", "base": "python:3.14@sha256:x",
              "requirements_sha256": "a" * 64, "uv": "0.13.0@sha256:y"}
    one = _copy_with(saved, tmp_path / "one", "runner", record)
    two = _copy_with(saved, tmp_path / "two", "runner", record)
    lock = json.loads((two / LOCK_FILE).read_text())
    lock["environment"]["runner_info"] = {"image": "im-other-workspace", "modal_client": "1.7.0"}
    (two / LOCK_FILE).write_text(json.dumps(lock))
    assert read_folder(one).lock["environment"]["runner"] == read_folder(two).lock["environment"]["runner"]
    assert read_folder(one).lock["environment"]["runner_info"] != read_folder(two).lock["environment"]["runner_info"]
    # The same two locks, compared by a loader that holds the first one's record, differ only in what is information.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MIME_RUNNER", json.dumps(record))
        for folder in (one, two):
            assert environment_differences(read_folder(folder).lock) == []
        patch.setenv("MIME_RUNNER", json.dumps({**record, "requirements_sha256": "b" * 64}))
        (difference,) = environment_differences(read_folder(two).lock)
        assert difference.startswith("runner: built under ") and "b" * 64 in difference


def _unreadable(monkeypatch, saved_environment, here_environment):
    monkeypatch.setattr(folder_module, "environment", lambda: here_environment)
    return environment_differences({"environment": saved_environment, "entries": []})


GPU = {"device_kind": "NVIDIA A100-SXM4-40GB", "driver": "610.57.04", "cuda_driver": 13030, "cuda_runtime": 12090,
       "cudnn": 92700}


def test_a_field_of_a_device_that_neither_run_could_read_is_not_the_same_environment(monkeypatch):
    with_driver = {"gpu": GPU, "cpu": {"model": "m", "flags_sha256": "f"}, "x64": True}
    assert _unreadable(monkeypatch, with_driver, dict(with_driver)) == []
    for name in ("driver", "cuda_driver", "cuda_runtime", "cudnn"):
        unread = {**with_driver, "gpu": {**GPU, name: None}}
        (difference,) = _unreadable(monkeypatch, unread, dict(unread))
        assert difference.startswith(f"gpu.{name}: built under None, loaded under None") and "unreadable" in difference
        assert _unreadable(monkeypatch, with_driver, unread) and _unreadable(monkeypatch, unread, with_driver)
    for name in ("model", "flags_sha256"):
        unread = {**with_driver, "cpu": {"model": "m", "flags_sha256": "f", name: None}}
        (difference,) = _unreadable(monkeypatch, unread, dict(unread))
        assert difference.startswith(f"cpu.{name}:") and "unreadable" in difference
    nothing = {"gpu": None, "cpu": {"model": "m", "flags_sha256": "f"}, "x64": True}
    assert _unreadable(monkeypatch, nothing, dict(nothing)) == []


def test_a_runner_record_is_a_json_object_of_strings():
    assert runner_record({}) == {} and runner_record({"MIME_RUNNER": " "}) == {}
    assert runner_record({"MIME_RUNNER": '{"b": "2", "a": "1"}'}) == {"a": "1", "b": "2"}
    for raw in ("modal", "[]", '{"a": 1}', '{"a": null}', json.dumps({str(i): "x" for i in range(33)}),
                json.dumps({"a": "x" * 257})):
        with pytest.raises(BuildRefused) as refused:
            runner_record({"MIME_RUNNER": raw})
        assert refused.value.codes() == ["env.runner"]
        with pytest.raises(BuildRefused) as refused:
            runner_info({"MIME_RUNNER_INFO": raw})
        assert refused.value.codes() == ["env.runner"] and "MIME_RUNNER_INFO" in refused.value.refusals[0].message


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
