"""The pins MIME ships: each variant's requirements file is ``uv export``'s output from ``uv.lock``, byte for byte, and
a lock changed without the files regenerated fails; every pin has a wheel for every Python series MIME supports, on the
base image's glibc; and ``bases.json`` pins each series' image by digest.

The export runs the uv the dev group pins, which CI installs too, so the bytes are tied to one release of uv."""

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from uv import find_uv_bin

from mime.env import EXPORT_ARGUMENTS, VARIANTS, applicable, bases, parse_requirements, pins, requirements_bytes
from mime.env._modal_script import UV_VERSION
from mime.env._pins import uv_digest, uv_image

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "src" / "mime" / "data" / "env"
REGENERATE = "uv export, as mime.env.EXPORT_ARGUMENTS gives it, into src/mime/data/env/requirements-<variant>.txt"


def export(project: Path, variant: str) -> bytes:
    done = subprocess.run([find_uv_bin(), *EXPORT_ARGUMENTS[variant], "--project", str(project)],
                          capture_output=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr.decode()
    return done.stdout


def committed_project(tmp_path: Path) -> Path:
    """The checkout, or, in the CI lane that re-resolves usd-core on purpose and so rewrites ``uv.lock``, a copy of the
    committed ``pyproject.toml`` and ``uv.lock``: the files ship the committed lock's export."""
    if os.environ.get("MIME_CI_LANE") != "usd-core-newest":
        return ROOT
    for name in ("pyproject.toml", "uv.lock"):
        shown = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, check=True)
        (tmp_path / name).write_bytes(shown.stdout)
    return tmp_path


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_requirements_file_is_the_locks_export_byte_for_byte(variant, tmp_path):
    exported = export(committed_project(tmp_path), variant)
    assert requirements_bytes(variant) == (SHIPPED / f"requirements-{variant}.txt").read_bytes()
    assert exported == requirements_bytes(variant), f"the lock changed: regenerate with {REGENERATE}"


def test_a_lock_changed_without_the_files_regenerated_fails(tmp_path):
    """The control: one hash of the lock changed, as a re-lock that moved one wheel would, and the export no longer
    matches the shipped file."""
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    numpy = re.search(r'name = "numpy"\nversion = "[^"]+"\n[^\[]*?hash = "sha256:([0-9a-f]{64})"', lock)
    assert numpy is not None
    planted = numpy.group(1)[::-1]
    (tmp_path / "uv.lock").write_text(lock.replace(numpy.group(1), planted), encoding="utf-8")
    for variant in VARIANTS:
        exported = export(tmp_path, variant)
        assert exported != requirements_bytes(variant)
        assert f"--hash=sha256:{planted}".encode() in exported


def test_the_uv_that_exports_is_cis_and_the_scripts():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    (ci,) = re.findall(r'UV_VERSION: "([^"]+)"', workflow)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    (pinned,) = [entry.split("==", 1)[1] for entry in project["dependency-groups"]["dev"] if entry.startswith("uv==")]
    version = subprocess.run([find_uv_bin(), "--version"], capture_output=True, text=True, check=True)
    installed = version.stdout.split()[1]
    assert ci == pinned == installed == UV_VERSION


def test_uv_is_copied_by_the_digest_of_its_image_and_the_release_is_the_one_that_exports():
    held = uv_image()
    assert held["version"] == UV_VERSION
    assert re.fullmatch(rf"ghcr\.io/astral-sh/uv:{re.escape(UV_VERSION)}@sha256:[0-9a-f]{{64}}", held["image"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", held["linux_amd64"]) and uv_digest() == held["image"].split("@")[1]
    assert uv_digest() != held["linux_amd64"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_line_of_a_file_is_a_hashed_pin(variant):
    text = requirements_bytes(variant).decode("utf-8")
    held = pins(variant)
    named = [line.split("==", 1)[0] for line in text.splitlines() if re.match(r"^[A-Za-z0-9]", line)]
    assert [pin.name for pin in held] == [canonicalize_name(name) for name in named] and held
    assert all(pin.hashes and all(re.fullmatch(r"sha256:[0-9a-f]{64}", h) for h in pin.hashes) for pin in held)


def test_the_cuda12_pins_hold_the_cpu_pins_and_the_plugin():
    cpu = {pin.name: pin.version for pin in pins("cpu")}
    cuda = {pin.name: pin.version for pin in pins("cuda12")}
    assert cpu.items() <= cuda.items()
    assert {"jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cudnn-cu12", "nvidia-cublas-cu12"} <= set(cuda) - set(cpu)


def test_a_line_that_is_not_a_pin_is_refused():
    for text in ("numpy>=2\n", "-e .\n", "numpy==2.0 ; python_version >>> '3'\n", "git+https://x/y\n"):
        with pytest.raises(ValueError):
            parse_requirements(text)


def _platforms(glibc: str) -> list[str]:
    major, minor = (int(part) for part in glibc.split(".")[:2])
    return ([f"manylinux_{major}_{m}_x86_64" for m in range(minor, 4, -1)]
            + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"])


def missing_wheels(lock: dict, variant: str, series: str, glibc: str) -> list[str]:
    """The pins of ``variant`` whose markers hold on Linux x86_64 under Python ``series`` and for which ``lock`` holds
    no wheel that Python and glibc can install."""
    version = tuple(int(part) for part in series.split("."))
    tags: set[Tag] = (set(cpython_tags(version, platforms=_platforms(glibc)))
                      | set(compatible_tags(version, platforms=_platforms(glibc))))
    environment = {**default_environment(), "python_version": series, "python_full_version": f"{series}.0",
                   "sys_platform": "linux", "platform_system": "Linux", "platform_machine": "x86_64",
                   "os_name": "posix", "implementation_name": "cpython", "platform_python_implementation": "CPython"}
    wheels = {canonicalize_name(str(package["name"])): [str(w["url"]).rsplit("/", 1)[1]
                                                        for w in package.get("wheels", [])]
              for package in lock["package"]}
    return sorted(name for name in applicable(pins(variant), environment)
                  if not any(set(parse_wheel_filename(file)[3]) & tags for file in wheels.get(name, [])))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("series", sorted(bases()))
def test_every_pin_has_a_wheel_for_every_python_on_its_base(variant, series):
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert missing_wheels(lock, variant, series, bases()[series]["glibc"]) == []


def test_a_pin_without_a_wheel_for_a_python_is_found():
    """The control: the lock with the CUDA plugin's Python 3.14 wheels taken out, and the check names the plugin."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == "jax-cuda12-plugin":
            package["wheels"] = [w for w in package["wheels"] if "-cp314-" not in w["url"]]
    assert missing_wheels(lock, "cuda12", "3.14", bases()["3.14"]["glibc"]) == ["jax-cuda12-plugin"]
    assert missing_wheels(lock, "cuda12", "3.13", bases()["3.13"]["glibc"]) == []


def test_bases_pin_each_supported_python_by_digest():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    supported = sorted(c.rsplit(" :: ", 1)[1] for c in project["project"]["classifiers"]
                       if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c))
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert sorted(bases()) == supported and f"python-version: {json.dumps(supported)}" in workflow
    for series, base in bases().items():
        assert re.fullmatch(rf"python:{re.escape(series)}-slim-{base['debian']}@sha256:[0-9a-f]{{64}}", base["image"])
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", base["linux_amd64"])
        assert base["python"].startswith(series + ".") and re.fullmatch(r"\d+\.\d+", base["glibc"])
        assert set(base["distributions"]) <= {"pip", "setuptools", "wheel"}
