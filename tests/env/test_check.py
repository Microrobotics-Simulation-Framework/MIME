"""``mime env check``: an environment that differs from the pins, the base image or the GPU requested is refused,
each difference named, and one that matches has the lock's precision applied.

The distributions, the devices and the system are given to the check, as a runner's would be read; the controls plant
one difference each into an environment the check accepts, so each refusal is shown to come from its plant alone."""

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
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
    environment_modules,
    gpu_type,
    installed_distributions,
    is_kind_of,
    pins,
    shadow_refusals,
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


def lock_in(path, *, matmul_precision="highest"):
    """A folder whose lock records float64 and ``matmul_precision``."""
    (path / "mime.lock").write_text(json.dumps({"environment": {"x64": True, "matmul_precision": matmul_precision}}))
    return path


@pytest.fixture
def folder(tmp_path):
    return lock_in(tmp_path)


def refused(**arguments) -> list[tuple[str, str]]:
    with pytest.raises(EnvRefused) as caught:
        check(**arguments)
    print(f"\n{caught.value}")
    return [(r.code, r.message) for r in caught.value.refusals]


def test_the_pinned_environment_is_accepted_and_the_locks_precision_applied(tmp_path):
    folder = lock_in(tmp_path, matmul_precision="float32")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_default_matmul_precision", None)
    report = check(folder, variant="cpu", base=True, installed=pinned("cpu"), devices=CPU, system=base_system(),
                   mime_version=MIME_VERSION)
    print(f"\n{report.text()}")
    assert jax.config.jax_enable_x64 is True and report.x64 is True
    assert jax.config.jax_default_matmul_precision == "float32" == report.matmul_precision
    assert report.base == BASE["image"] and report.devices == ("cpu: cpu",) and report.mime == MIME_VERSION
    assert "precision: float64, as the experiment's lock records" in report.text()
    assert "MIME at " + MIME_VERSION in report.text()


def test_a_gpu_run_on_the_type_it_requested_computes_matmuls_at_the_highest_precision_its_lock_records(folder):
    jax.config.update("jax_default_matmul_precision", None)
    report = check(folder, variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"),
                   devices=A100_40GB, system=base_system())
    assert jax.config.jax_default_matmul_precision == "highest" == report.matmul_precision
    jax.config.update("jax_default_matmul_precision", None)
    report = check(variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"), devices=A100_40GB,
                   system=base_system())
    assert report.matmul_precision == "highest"


@pytest.mark.parametrize("recorded", [None, "default", "float32", "bfloat16"])
def test_a_gpu_run_whose_lock_records_another_matmul_precision_is_refused(tmp_path, recorded):
    jax.config.update("jax_default_matmul_precision", None)
    (refusal,) = refused(folder=lock_in(tmp_path, matmul_precision=recorded), variant="cuda12", gpu="A100-40GB",
                         base=True, installed=pinned("cuda12"), devices=A100_40GB, system=base_system())
    assert refusal[0] == "env.precision" and repr(recorded) in refusal[1]
    assert jax.config.jax_default_matmul_precision is None


def test_a_cpu_run_applies_whatever_matmul_precision_its_lock_records(tmp_path):
    for recorded in ("float32", None, "highest"):
        report = check(lock_in(tmp_path, matmul_precision=recorded), variant="cpu", installed=pinned("cpu", base=False),
                       devices=CPU, system=base_system())
        assert jax.config.jax_default_matmul_precision == recorded == report.matmul_precision


def test_a_lock_without_a_matmul_precision_is_refused(tmp_path):
    (tmp_path / "mime.lock").write_text(json.dumps({"environment": {"x64": True}}))
    with pytest.raises(BuildRefused) as caught:
        check(tmp_path, variant="cpu", installed=pinned("cpu", base=False), devices=CPU, system=base_system())
    assert caught.value.codes() == ["env.lock"]


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


@pytest.mark.parametrize("served", ["NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"])
@pytest.mark.parametrize("requested", ["A100-40GB", "A100-80GB"])
def test_a_request_for_an_a100_is_met_by_any_a100_and_the_kind_served_is_reported(requested, served):
    report = check(variant="cuda12", gpu=requested, base=True, installed=pinned("cuda12"),
                   devices=[Device("gpu", served)], system=base_system())
    assert report.devices == (f"gpu: {served}",) and f"devices: gpu: {served}" in report.text()


def test_a_gpu_of_another_type_is_refused_and_a_gpu_where_none_was_requested():
    for served in ("NVIDIA H100 80GB HBM3", "NVIDIA A10", "NVIDIA L40S"):
        (refusal,) = refused(variant="cuda12", gpu="A100-40GB", base=True, installed=pinned("cuda12"),
                             devices=[Device("gpu", served)], system=base_system())
        assert refusal[0] == "env.device" and served in refusal[1]
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


A100S = ["A100-40GB", "A100-80GB"]


@pytest.mark.parametrize(("kind", "exact"), [
    ("NVIDIA A100-SXM4-40GB", A100S), ("NVIDIA A100-PCIE-40GB", A100S), ("NVIDIA A100-SXM4-80GB", A100S),
    ("NVIDIA A100 80GB PCIe", A100S), ("NVIDIA H100 80GB HBM3", ["H100!"]), ("NVIDIA H200", ["H200"]),
    ("NVIDIA L40S", ["L40S"]), ("NVIDIA L4", ["L4"]), ("NVIDIA A10", ["A10"]), ("Tesla T4", ["T4"]),
    ("NVIDIA B200", ["B200"]),
])
def test_a_device_kind_is_one_exact_type_only_and_the_a100s_are_one_family(kind, exact):
    assert [name for name in EXACT_GPUS if is_kind_of(kind, name)] == exact


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
    for lock in (None, "[]", json.dumps({"environment": {"x64": "yes", "matmul_precision": None}})):
        if lock is not None:
            (tmp_path / "mime.lock").write_text(lock)
        with pytest.raises(BuildRefused) as caught:
            check(tmp_path, variant="cpu", installed=pinned("cpu", base=False), devices=CPU, system=base_system())
        assert caught.value.codes() == ["env.lock"]
    with pytest.raises(BuildRefused) as caught:
        check(variant="rocm", installed=pinned("cpu", base=False), devices=CPU, system=base_system())
    assert caught.value.codes() == ["env.variant"]


# -- K1: MIME's version is held to the pinned wheel's, never to the installed copy's own ------------------------------


def test_mime_is_held_to_the_version_of_the_pinned_wheel():
    environment = pinned("cpu")
    (refusal,) = refused(variant="cpu", base=True, installed=environment, devices=CPU, system=base_system(),
                         mime_version="999.0.0")
    assert refusal == ("env.pin", f"mime-engine is installed at {MIME_VERSION} and pinned at 999.0.0 by the pinned "
                                  f"MIME wheel")
    without = {name: versions for name, versions in environment.items() if name != "mime-engine"}
    (refusal,) = refused(variant="cpu", base=True, installed=without, devices=CPU, system=base_system(),
                         mime_version=MIME_VERSION)
    assert refusal[0] == "env.missing" and refusal[1].startswith(f"mime-engine {MIME_VERSION}")


def test_without_a_pinned_wheel_mime_is_present_but_held_to_no_version():
    report = check(variant="cpu", base=True, installed={**pinned("cpu"), "mime-engine": ("0.0.1",)}, devices=CPU,
                   system=base_system())
    assert report.mime is None and "MIME not held to a version" in report.text()


# -- K5: a lab's file that is not exact pins is refused, not a traceback ----------------------------------------------


@pytest.mark.parametrize("text", ["foo>=1\n", "foo\n", "-e .\n", "foo==1 ; python_version >\n"])
def test_a_lab_file_that_is_not_exact_pins_is_refused_with_a_message(tmp_path, text):
    (tmp_path / "lab.txt").write_text(text)
    with pytest.raises(BuildRefused) as caught:
        check(variant="cpu", lab=tmp_path / "lab.txt", installed=pinned("cpu", base=False), devices=CPU,
              system=base_system())
    assert caught.value.codes() == ["env.lab"] and "lab.txt" in caught.value.refusals[0].message
    with pytest.raises(BuildRefused) as caught:
        check(variant="cpu", lab=tmp_path / "missing.txt", installed=pinned("cpu", base=False), devices=CPU,
              system=base_system())
    assert caught.value.codes() == ["env.lab"]


def test_mime_env_check_refuses_a_malformed_lab_file_with_exit_two_and_no_traceback(tmp_path, capsys):
    (tmp_path / "lab.txt").write_text("foo>=1\n")
    assert mime_main(["env", "check", "--variant", "cpu", "--lab", str(tmp_path / "lab.txt")]) == 2
    assert "env.lab" in capsys.readouterr().err


# -- W1: a module Python imports from outside the environment's site-packages is refused, naming it ------------------


def test_a_module_imported_from_outside_the_environments_site_packages_is_refused_naming_it(tmp_path):
    inside = environment_modules()
    assert "numpy" in inside and inside["numpy"] == "numpy"
    assert shadow_refusals(inside) == []
    where = tmp_path / "shadow"
    where.mkdir()
    refusals = shadow_refusals({"idna": "idna", "numpy": "numpy"},
                               locate=lambda module: [str(where / module)] if module == "idna" else
                               [str(importlib.util.find_spec(module).submodule_search_locations[0])])
    (refusal,) = refusals
    assert refusal.code == "env.shadow" and "idna (idna)" in refusal.message and str(where) in refusal.message
    assert "numpy" not in refusal.message.split(" from ")[0]


SHADOW = """
import sys
from mime.env import environment_modules, shadow_refusals
modules = environment_modules()
planted = sys.argv[1]
refusals = shadow_refusals(modules)
print("MODULES", planted in modules)
print("REFUSALS", [(r.code, planted in r.message) for r in refusals])
"""


@pytest.fixture
def unimported_module() -> str:
    """A module of this environment that nothing in a fresh process importing ``mime.env`` has loaded yet."""
    probe = subprocess.run([sys.executable, "-c", "import sys, mime.env as e; "
                            "print(' '.join(sorted(m for m in e.environment_modules() if m not in sys.modules "
                            "and not m.startswith('_') and '.' not in m)))"],
                           capture_output=True, text=True, check=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    names = probe.stdout.split()
    assert names, "every module of the environment is imported by mime.env, so the plant cannot be shown"
    return names[0]


def test_a_pythonpath_entry_that_shadows_a_pinned_package_is_refused_in_a_real_process(tmp_path, unimported_module):
    """The review's plant: the same package, another copy, found through PYTHONPATH ahead of site-packages."""
    shadow = tmp_path / "shadow"
    (shadow / unimported_module).mkdir(parents=True)
    (shadow / unimported_module / "__init__.py").write_text("")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    environment.pop("PYTHONPATH", None)
    clean = subprocess.run([sys.executable, "-c", SHADOW, unimported_module], capture_output=True, text=True,
                           check=True, env=environment)
    assert "REFUSALS []" in clean.stdout
    planted = subprocess.run([sys.executable, "-c", SHADOW, unimported_module], capture_output=True, text=True,
                             check=True, env={**environment, "PYTHONPATH": str(shadow)})
    assert "MODULES True" in planted.stdout and f"[('env.shadow', True)]" in planted.stdout, planted.stdout
