"""``mime env check``: an environment that differs from the pins, the base image or the GPU requested is refused,
each difference named, and one that matches has the lock's precision applied.

The distributions, the devices and the system are given to the check, as a runner's would be read; the controls plant
one difference each into an environment the check accepts, so each refusal is shown to come from its plant alone."""

import importlib.metadata
import json
from dataclasses import dataclass, replace

import jax
import pytest
from packaging.markers import default_environment

from mime.build import BuildRefused
from mime.cli import main as mime_main
from mime.env import (
    EXACT_GPUS,
    LOOSE_GPUS,
    EnvRefused,
    System,
    applicable,
    bases,
    check,
    gpu_type,
    installed_distributions,
    is_kind_of,
    pins,
)
from mime.env._check import expected_distributions

SERIES = "3.14"
BASE = bases()[SERIES]
MIME_VERSION = importlib.metadata.version("mime-engine")


@dataclass(frozen=True)
class Device:
    platform: str
    device_kind: str


CPU = [Device("cpu", "cpu")]
A100_40GB = [Device("gpu", "NVIDIA A100-SXM4-40GB")]


def base_system(**changes) -> System:
    system = System(BASE["python"], BASE["debian"], BASE["debian_version"], BASE["glibc"], default_environment(),
                    f"Debian GNU/Linux {BASE['debian']}")
    return replace(system, **changes)


def pinned(variant: str, *, base: bool = True) -> dict[str, tuple[str, ...]]:
    """An environment holding exactly what the variant, MIME and (with ``base``) the base image pin."""
    held = {name: (pin.version,) for name, pin in applicable(pins(variant)).items()}
    held["mime-engine"] = (MIME_VERSION,)
    if base:
        held.update({name: (version,) for name, version in BASE["distributions"].items()})
    return held


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    monkeypatch.delenv("MIME_RUNNER", raising=False)
    config = jax.config
    held = (config.jax_enable_x64, config.jax_default_matmul_precision)
    yield
    config.update("jax_enable_x64", held[0])
    config.update("jax_default_matmul_precision", held[1])


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "mime.lock").write_text(json.dumps({"environment": {"x64": True}}))
    return tmp_path


def refused(**arguments) -> list[tuple[str, str]]:
    with pytest.raises(EnvRefused) as caught:
        check(**arguments)
    print(f"\n{caught.value}")
    return [(r.code, r.message) for r in caught.value.refusals]


def test_the_pinned_environment_is_accepted_and_the_locks_precision_applied(folder):
    jax.config.update("jax_enable_x64", False)
    report = check(folder, variant="cpu", base=True, installed=pinned("cpu"), devices=CPU, system=base_system())
    print(f"\n{report.text()}")
    assert jax.config.jax_enable_x64 is True and report.x64 is True
    assert report.base == BASE["image"] and report.devices == ("cpu: cpu",)
    assert "precision: float64, as the experiment's lock records" in report.text()


def test_a_gpu_run_on_the_type_it_requested_gets_the_highest_matmul_precision(folder):
    jax.config.update("jax_default_matmul_precision", None)
    report = check(folder, variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"),
                   devices=A100_40GB, system=base_system())
    assert jax.config.jax_default_matmul_precision == "highest" == report.matmul_precision


def test_a_planted_pin_is_refused_naming_it():
    environment = pinned("cpu")
    environment["numpy"] = ("1.0.0",)
    (refusal,) = refused(variant="cpu", base=True, installed=environment, devices=CPU, system=base_system())
    assert refusal[0] == "env.pin" and refusal[1].startswith("numpy is installed at 1.0.0 and pinned at ")


def test_a_missing_distribution_is_refused_naming_it():
    environment = pinned("cuda12")
    del environment["nvidia-cudnn-cu12"]
    (refusal,) = refused(variant="cuda12", gpu="A100-40GB", base=True, installed=environment, devices=A100_40GB,
                         system=base_system())
    assert refusal[0] == "env.missing" and refusal[1].startswith("nvidia-cudnn-cu12 9.")


def test_an_unpinned_distribution_is_refused_and_an_installers_own_only_off_the_base():
    environment = {**pinned("cpu", base=False), "jax-cuda12-plugin": ("0.12.0.1",), "pip": ("1.0",)}
    codes = refused(variant="cpu", installed=environment, devices=CPU, system=base_system())
    assert codes == [("env.unpinned", "jax-cuda12-plugin 0.12.0.1 is installed and nothing pins it")]
    codes = refused(variant="cpu", base=True, installed=environment, devices=CPU, system=base_system())
    assert {code for code, _ in codes} == {"env.unpinned", "env.pin"}


def test_another_python_is_refused():
    (refusal,) = refused(variant="cpu", base=True, installed=pinned("cpu"), devices=CPU,
                         system=base_system(python=f"{SERIES}.0"))
    assert refusal == ("env.python", f"Python is {SERIES}.0; the base image {BASE['image']} holds {BASE['python']}")
    (refusal,) = refused(variant="cpu", base=True, installed=pinned("cpu", base=False), devices=CPU,
                         system=base_system(python="3.11.9"))
    assert refusal == ("env.python", "MIME ships no base image for Python 3.11")


def test_another_os_is_refused():
    codes = refused(variant="cpu", base=True, installed=pinned("cpu"), devices=CPU,
                    system=base_system(debian="bookworm", glibc="2.36"))
    assert [code for code, _ in codes] == ["env.os", "env.os"]
    assert codes[0][1].startswith("Debian's release is bookworm") and codes[1][1].startswith("glibc is 2.36")


def test_a_requested_gpu_jax_does_not_see_is_refused_as_a_cpu_fallback():
    (refusal,) = refused(variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"), devices=CPU,
                         system=base_system())
    assert refusal == ("env.device", "the run requested a A100-40GB and JAX sees cpu: cpu (JAX fell back to the CPU: "
                                     "its CUDA plugin found no GPU)")


def test_a_gpu_of_another_type_is_refused_and_a_gpu_where_none_was_requested():
    (refusal,) = refused(variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"),
                         devices=[Device("gpu", "NVIDIA A100-SXM4-80GB")], system=base_system())
    assert refusal[0] == "env.device" and "NVIDIA A100-SXM4-80GB" in refusal[1]
    (refusal,) = refused(variant="cuda12", base=True, installed=pinned("cuda12"), devices=A100_40GB,
                         system=base_system())
    assert refusal == ("env.device", "the run requested the CPU and JAX computes on gpu: NVIDIA A100-SXM4-40GB")


def test_every_difference_is_refused_at_once():
    environment = pinned("cpu")
    environment["scipy"] = ("0.1",)
    codes = refused(variant="cpu", gpu="H100!", base=True, installed=environment, devices=CPU,
                    system=base_system(glibc="2.36"))
    assert sorted(code for code, _ in codes) == ["env.device", "env.os", "env.pin"]


def test_markers_are_evaluated_for_the_system_checked():
    linux, _ = expected_distributions("cuda12", base_system(markers={**default_environment(), "sys_platform": "linux"}),
                                      base=None)
    darwin, _ = expected_distributions("cuda12",
                                       base_system(markers={**default_environment(), "sys_platform": "darwin"}),
                                       base=None)
    nvidia = {name for name in linux if name.startswith("nvidia-")}
    assert len(nvidia) > 5 and not nvidia & set(darwin) and set(darwin) | nvidia == set(linux)


def test_a_loose_or_unknown_gpu_type_is_refused_naming_the_exact_ones():
    for name in LOOSE_GPUS:
        with pytest.raises(BuildRefused) as caught:
            gpu_type(name)
        assert caught.value.codes() == ["env.gpu-type"] and LOOSE_GPUS[name] in caught.value.refusals[0].action
    for name in ("A100-40GB:2", "a100", "V100"):
        with pytest.raises(BuildRefused):
            gpu_type(name)
    assert [gpu_type(name) for name in EXACT_GPUS] == list(EXACT_GPUS)


@pytest.mark.parametrize(("kind", "exact"), [
    ("NVIDIA A100-SXM4-40GB", "A100-40GB"), ("NVIDIA A100-PCIE-40GB", "A100-40GB"),
    ("NVIDIA A100-SXM4-80GB", "A100-80GB"), ("NVIDIA A100 80GB PCIe", "A100-80GB"),
    ("NVIDIA H100 80GB HBM3", "H100!"), ("NVIDIA H200", "H200"), ("NVIDIA L40S", "L40S"), ("NVIDIA L4", "L4"),
    ("NVIDIA A10", "A10"), ("Tesla T4", "T4"), ("NVIDIA B200", "B200"),
])
def test_a_device_kind_is_one_exact_type_only(kind, exact):
    assert [name for name in EXACT_GPUS if is_kind_of(kind, name)] == [exact]


def test_the_check_reads_this_environments_distributions():
    import numpy

    found = installed_distributions()
    assert found["numpy"] == (numpy.__version__,) and found["mime-engine"] == (MIME_VERSION,)


def test_mime_env_check_exits_two_and_records_what_it_refused(tmp_path, capsys):
    record = tmp_path / "out" / "check.json"
    assert mime_main(["env", "check", "--variant", "cpu", "--gpu", "A100", "--record", str(record)]) == 2
    assert "env.gpu-type" in capsys.readouterr().err
    assert json.loads(record.read_text())["refused"][0].startswith("[user] env.gpu-type")


def test_a_folder_whose_lock_records_no_precision_and_a_variant_without_pins_are_refused(tmp_path):
    for lock in (None, "[]", json.dumps({"environment": {"x64": "yes"}})):
        if lock is not None:
            (tmp_path / "mime.lock").write_text(lock)
        with pytest.raises(BuildRefused) as caught:
            check(tmp_path, variant="cpu", installed=pinned("cpu", base=False), devices=CPU, system=base_system())
        assert caught.value.codes() == ["env.lock"]
    with pytest.raises(BuildRefused) as caught:
        check(variant="rocm", installed=pinned("cpu", base=False), devices=CPU, system=base_system())
    assert caught.value.codes() == ["env.variant"]
