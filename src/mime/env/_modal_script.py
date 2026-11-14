"""``mime modal <folder>``: write the script that runs a saved experiment on Modal, beside the experiment.

The script is the only file that imports ``modal``; MIME never does. Its image has four layers, in the order that
changes least often first:

1. the base image of the Python series, pinned by digest (``bases.json``);
2. the variant's hashed requirements, MIME's file copied byte for byte, installed by a uv copied from its image
   by digest, with ``--require-hashes --no-deps --no-build``;
3. MIME's wheel, by version and its hash on PyPI; before a release, a wheel the script builds of the checkout, by its
   hash;
4. a thin layer of the lab's distributions the experiment's lock names besides MIME.

The experiment folder is mounted when a container starts, never baked into a layer; assets live on a Volume filled by
``mime assets fetch`` inside Modal, and XLA's cache on another."""

from __future__ import annotations

import importlib.metadata
import hashlib
import importlib.resources
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mime._lock_json import read_lock
from mime.assets import folder_assets, licences_of, read_acceptances
from mime.build import BuildRefused, Refusal, RefusalClass
from mime.env._check import (
    DISTRIBUTION,
    LOCK_FILE,
    gpu_precision_refusals,
    lab_pins_of,
    lock_matmul_precision,
    lock_x64,
)
from mime.env._gpus import gpu_type
from mime.env._host import with_deterministic
from mime.env._pins import bases, requirements_bytes, requirements_name, series_of, uv_digest, uv_image

#: The uv that installs the pins: CI's, and the one whose ``uv export`` wrote them, copied from its image by digest.
UV_VERSION = uv_image()["version"]
#: Modal's image builder the script pins, and the client release it was written against.
IMAGE_BUILDER_VERSION = "2026.10"
MODAL_CLIENT = "1.6.3"
SCRIPT_FILE = "modal_app.py"
TEMPLATE = "modal_app.txt"
STUDY_SPEC = "study.mime.yaml"
MIME_REQUIREMENTS = "mime-requirements.txt"
LAB_REQUIREMENTS = "lab-requirements.txt"
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"

type WheelLookup = Callable[[str], str]


@dataclass(frozen=True)
class ModalScript:
    """What ``mime modal`` wrote: the folder, the script, and every file beside it."""

    out: Path
    script: Path
    files: tuple[Path, ...]
    variant: str
    gpu: str | None
    python: str
    dev: bool
    study: Path | None = None
    shards: int = 1

    def text(self) -> str:
        where = _shown(self.script)
        lines = [f"wrote {where} ({self.variant}, Python {self.python}, "
                 f"{self.gpu or 'CPU'}{', MIME from the checkout' if self.dev else ''})"]
        lines += [f"  and {_shown(path)}" for path in self.files if path != self.script]
        lines += ["run it with:", f"  modal run {where}::main --duration <seconds>", f"  modal run {where}::verify"]
        if self.study is not None:
            lines += [f"  modal run {where}::study --name <name>   # the study in {self.study.name}: "
                      + (f"{self.shards} containers, one shard of its groups each, each a partial study folder"
                         if self.shards > 1 else "one container runs every group; the same name resumes it")]
        return "\n".join(lines)


def _shown(path: Path) -> str:
    """``path`` from here when it is below here, else as it is (on Windows, a path on another drive has no relative
    form)."""
    try:
        relative = os.path.relpath(path)
    except ValueError:
        return str(path)
    return str(path) if relative.startswith("..") else relative


def _refuse(code: str, category: RefusalClass, message: str, action: str) -> BuildRefused:
    return BuildRefused([Refusal(code, category, message, action)])


def pypi_wheel_sha256(version: str) -> str:
    """The sha256 PyPI records for MIME's universal wheel of ``version``."""
    url = PYPI_JSON.format(name=DISTRIBUTION, version=version)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _refuse("modal.wheel", RefusalClass.ENVIRONMENT,
                      f"PyPI gives no record of {DISTRIBUTION} {version} ({error})",
                      "write the script where PyPI can be reached, or, before a release, pass --checkout") from error
    wheels = [entry for entry in cast(list[dict[str, object]], data.get("urls", []))
              if entry.get("packagetype") == "bdist_wheel" and str(entry.get("filename", "")).endswith("-none-any.whl")]
    if len(wheels) != 1:
        raise _refuse("modal.wheel", RefusalClass.ENVIRONMENT,
                      f"PyPI lists {len(wheels)} universal wheels of {DISTRIBUTION} {version}, not one",
                      "pass --checkout to build the wheel from a checkout of MIME")
    return str(cast(dict[str, object], wheels[0]["digests"])["sha256"])


def installed_checkout() -> Path | None:
    """The checkout MIME is installed from in development (an editable install), or ``None`` for a release."""
    try:
        text = importlib.metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    held = cast(dict[str, object], json.loads(text))
    url = str(held.get("url", ""))
    editable = cast(dict[str, object], held.get("dir_info") or {}).get("editable")
    if not (url.startswith("file://") and editable):
        return None
    return Path(urllib.request.url2pathname(urllib.parse.urlsplit(url).path))


def _read_lock(folder: Path) -> dict[str, object]:
    try:
        held: object = read_lock(folder / LOCK_FILE)
    except (OSError, ValueError) as error:
        raise _refuse("env.lock", RefusalClass.USER, f"{folder / LOCK_FILE} cannot be read as JSON ({error})",
                      "name the folder Experiment.save wrote, as in `mime run <folder> ...`") from error
    if not isinstance(held, dict):
        raise _refuse("env.lock", RefusalClass.USER, f"{folder / LOCK_FILE} is not a JSON object",
                      "name the folder Experiment.save wrote, as in `mime run <folder> ...`")
    return cast(dict[str, object], held)


def lab_pins(lock: dict[str, object]) -> list[tuple[str, str]]:
    """The distributions besides MIME that the lock's entries come from, each with its version, sorted."""
    found: set[tuple[str, str]] = set()
    for entry in cast(list[dict[str, str]], lock.get("entries", [])):
        if entry.get("distribution") not in (DISTRIBUTION, None):
            found.add((entry["distribution"], entry["distribution_version"]))
    return sorted(found)


def _python_series(lock: dict[str, object], requested: str | None) -> str:
    environment = cast(dict[str, object], lock.get("environment", {}))
    series = requested or series_of(str(environment.get("python", "")))
    if series not in bases():
        raise _refuse("env.python", RefusalClass.USER, f"MIME ships no base image for Python {series}",
                      f"pass --python with one of {', '.join(bases())}")
    return series


def _study_spec(study: str | os.PathLike[str] | None, shards: int, max_containers: int,
                lock: dict[str, object]) -> Path | None:
    """The spec file ``--study`` names, if it reads and is of this experiment (its input hash is the lock's); the shards
    refused where there is no study or the containers are not as many."""
    if study is None:
        if shards != 1:
            raise _refuse("modal.study-shards", RefusalClass.USER, "--shards splits a study, and no --study is given",
                          "give the study: mime modal <folder> --study study.mime.yaml --shards 4 --max-containers 4")
        return None
    if shards < 1 or (shards > 1 and max_containers != shards):
        raise _refuse("modal.study-shards", RefusalClass.USER,
                      f"--shards {shards} runs {shards} containers at once, and --max-containers is {max_containers}",
                      f"state both: --shards {shards} --max-containers {shards}; the fan-out is explicit, and one "
                      f"container runs every group by default")
    from mime.study._spec import read_spec

    path = Path(study).resolve()
    spec = read_spec(path)
    wanted = cast(dict[str, object], spec["base"]).get("input_hash")
    found = cast(dict[str, object], lock.get("hashes", {})).get("input")
    if wanted is not None and wanted != found:
        raise _refuse("modal.study-base", RefusalClass.USER,
                      f"the spec {path.name} is of the experiment with input hash {wanted}, and this folder holds {found}",
                      "write the script for the experiment the spec was written for: `mime modal <that folder> --study <spec>`")
    return path


def _accepted_licences(folder: Path) -> list[str]:
    """The ids of the licences with conditions that the experiment's assets are under and that were accepted on this
    machine (by id and title, as the gate asks): what the script carries as ``accept_licences`` for its run. A
    licence nobody accepted here is not carried, and the run's first read of that file is refused."""
    held = {(each.licence.id, each.licence.title) for each in read_acceptances()}
    found: dict[str, None] = {}
    for ref in folder_assets(folder):
        for licence in licences_of(ref):
            if (licence.id, licence.title) in held:
                found.setdefault(licence.id)
    return sorted(found)


def _constants(values: dict[str, object]) -> str:
    notes = {
        "EXPERIMENT": "the experiment's folder, from this script's",
        "BASE": "MIME's bases.json, for this Python series",
        "REQUIREMENTS": "MIME's file for the variant, byte for byte",
        "REQUIREMENTS_SHA256": "that file's hash, which the run's lock records",
        "UV_DIGEST": "the digest of uv's image (MIME's uv.json): the script copies uv from ghcr.io/astral-sh/uv@<it>",
        "MIME_WHEEL_SHA256": "MIME's wheel on PyPI; None: the script builds the checkout below",
        "LAB": "the lab's distributions the lock names, beside this script",
        "GPU": "an exact type; None for a CPU",
        "XLA_FLAGS": "XLA's deterministic GPU mode is on only when --deterministic asked for it",
        "X64": "the precision the experiment's lock records",
        "MAX_CONTAINERS": "containers a function may run in at once; --max-containers",
        "STUDY": "the study's spec file beside this script (--study); None: the script runs no study",
        "SHARDS": "containers a study is split over (--shards), one shard of its groups each; 1: one container",
        "ASSETS": "whether the lock references assets, fetched onto the assets Volume first",
        "ACCEPT_LICENCES": "the licences accepted where this was written, for this run only: a build reads a gated file "
                           "only once its licence is accepted, and nobody can be asked in a container",
        "IMAGE_BUILDER_VERSION": f"Modal's image builder; the script was written against client {MODAL_CLIENT}",
    }
    lines = ["# Written by `mime modal` from the experiment's lock and MIME's pins: write it again, never edit it."]
    for name, value in values.items():
        note = notes.get(name)
        lines.append(f"{name} = {value!r}" + (f"  # {note}" if note else ""))
    return "\n".join(lines)


def modal_script(folder: str | os.PathLike[str], *, gpu: str | None = None, deterministic: bool = False,
                 python: str | None = None, out: str | os.PathLike[str] | None = None,
                 checkout: str | os.PathLike[str] | None = None, lab_requirements: str | os.PathLike[str] | None = None,
                 xla_cache: bool = True, timeout: int = 7200, max_containers: int = 1, cpu: float = 4.0,
                 memory: int = 8192, study: str | os.PathLike[str] | None = None, shards: int = 1,
                 wheel_sha256: WheelLookup = pypi_wheel_sha256) -> ModalScript:
    """Write the Modal script for the saved experiment in ``folder`` into ``out`` (``<folder>-modal/`` beside it), with
    MIME's requirements file for the variant and, as needed, MIME's and the lab's. ``gpu`` is an exact type (a loose
    one is refused); ``deterministic`` turns XLA's deterministic GPU mode on. MIME's wheel is a release's on PyPI,
    unless MIME is installed from a checkout or ``checkout`` names one: then the script builds that checkout's wheel.
    ``study`` is a study's spec file (``study.mime.yaml``), copied beside the script, which then has a ``study``
    entrypoint that runs ``mime study run`` on it in one container (a pilot first), into a folder on the runs Volume a
    second run of the same ``--name`` resumes; ``shards`` (more than one, with ``max_containers`` the same) splits the
    study's groups over that many containers, explicit and recorded in the script, each writing a partial study folder.
    Raises ``BuildRefused``."""
    where = Path(folder).resolve()
    lock = _read_lock(where)
    spec_file = _study_spec(study, shards, max_containers, lock)
    requested = gpu_type(gpu) if gpu is not None else None
    if deterministic and requested is None:
        raise _refuse("env.gpu-type", RefusalClass.USER,
                      "--deterministic sets XLA's deterministic mode for a GPU, and the run requests none",
                      "request a GPU, `mime modal <folder> --gpu <type> --deterministic`, or drop --deterministic for "
                      "the CPU run")
    variant = "cuda12" if requested is not None else "cpu"
    series = _python_series(lock, python)
    x64 = lock_x64(where)
    if requested is not None and (refusals := gpu_precision_refusals(lock_matmul_precision(where), requested)):
        raise BuildRefused(refusals)
    if lab_requirements is not None:
        lab_pins_of(lab_requirements)
    has_assets = bool(folder_assets(where))
    target = Path(out).resolve() if out is not None else where.parent / f"{where.name}-modal"
    target.mkdir(parents=True, exist_ok=True)
    from mime import __version__

    source = Path(checkout).resolve() if checkout is not None else installed_checkout()
    files: list[Path] = []
    requirements = target / requirements_name(variant)
    requirements.write_bytes(requirements_bytes(variant))
    files.append(requirements)
    digest: str | None = None
    if source is None:
        digest = wheel_sha256(__version__)
        mime_requirements = target / MIME_REQUIREMENTS
        mime_requirements.write_text(f"{DISTRIBUTION}=={__version__} \\\n    --hash=sha256:{digest}\n",
                                     encoding="utf-8", newline="\n")
        files.append(mime_requirements)
    lab_file: str | None = None
    lab_hashed = False
    if lab_requirements is not None:
        given = Path(lab_requirements).read_bytes()
        (target / LAB_REQUIREMENTS).write_bytes(given)
        lab_file, lab_hashed = LAB_REQUIREMENTS, True
    elif pins := lab_pins(lock):
        (target / LAB_REQUIREMENTS).write_text("".join(f"{name}=={version}\n" for name, version in pins),
                                               encoding="utf-8", newline="\n")
        lab_file = LAB_REQUIREMENTS
    if lab_file is not None:
        files.append(target / lab_file)
    values: dict[str, object] = {
        "MIME_VERSION": __version__,
        "NAME": where.name,
        "EXPERIMENT": os.path.relpath(where, target),
        "VARIANT": variant,
        "PYTHON": series,
        "BASE": bases()[series]["image"],
        "REQUIREMENTS": requirements.name,
        "REQUIREMENTS_SHA256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "UV_VERSION": UV_VERSION,
        "UV_DIGEST": uv_digest(),
        "IMAGE_BUILDER_VERSION": IMAGE_BUILDER_VERSION,
        "MIME_WHEEL_SHA256": digest,
        "CHECKOUT": str(source) if source is not None else None,
        "LAB": lab_file,
        "LAB_HASHED": lab_hashed,
        "GPU": requested,
        "XLA_FLAGS": with_deterministic("", True) if deterministic else "",
        "X64": x64,
        "ASSETS": has_assets,
        "ACCEPT_LICENCES": _accepted_licences(where),
        "XLA_CACHE": xla_cache,
        "TIMEOUT": timeout,
        "MAX_CONTAINERS": max_containers,
        "CPU": cpu,
        "MEMORY": memory,
    }
    if spec_file is not None:
        (target / STUDY_SPEC).write_bytes(spec_file.read_bytes())
        files.append(target / STUDY_SPEC)
    values["STUDY"] = STUDY_SPEC if spec_file is not None else None
    values["SHARDS"] = shards
    template = (importlib.resources.files("mime.env") / TEMPLATE).read_text(encoding="utf-8")
    script = target / SCRIPT_FILE
    script.write_text(template.replace("@@CONSTANTS@@", _constants(values)), encoding="utf-8", newline="\n")
    return ModalScript(target, script, (script, *files), variant, requested, series, source is not None,
                       target / STUDY_SPEC if spec_file is not None else None, shards)


__all__ = ["IMAGE_BUILDER_VERSION", "MODAL_CLIENT", "SCRIPT_FILE", "UV_VERSION", "ModalScript", "installed_checkout",
           "lab_pins", "modal_script", "pypi_wheel_sha256"]
