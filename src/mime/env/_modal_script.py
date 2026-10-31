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

from mime.assets import folder_assets
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

    def text(self) -> str:
        where = _shown(self.script)
        lines = [f"wrote {where} ({self.variant}, Python {self.python}, "
                 f"{self.gpu or 'CPU'}{', MIME from the checkout' if self.dev else ''})"]
        lines += [f"  and {_shown(path)}" for path in self.files if path != self.script]
        lines += ["run it with:", f"  modal run {where}::main --duration <seconds>", f"  modal run {where}::verify"]
        return "\n".join(lines)


def _shown(path: Path) -> str:
    """``path`` from here when it is below here, else as it is."""
    relative = os.path.relpath(path)
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
        held: object = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise _refuse("env.lock", RefusalClass.USER, f"{folder / LOCK_FILE} cannot be read as JSON ({error})",
                      "give the folder a saved experiment was written to") from error
    if not isinstance(held, dict):
        raise _refuse("env.lock", RefusalClass.USER, f"{folder / LOCK_FILE} is not a JSON object",
                      "give the folder a saved experiment was written to")
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
        "ASSETS": "whether the lock references assets, fetched onto the assets Volume first",
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
                 memory: int = 8192,
                 wheel_sha256: WheelLookup = pypi_wheel_sha256) -> ModalScript:
    """Write the Modal script for the saved experiment in ``folder`` into ``out`` (``<folder>-modal/`` beside it), with
    MIME's requirements file for the variant and, as needed, MIME's and the lab's. ``gpu`` is an exact type (a loose
    one is refused); ``deterministic`` turns XLA's deterministic GPU mode on. MIME's wheel is a release's on PyPI,
    unless MIME is installed from a checkout or ``checkout`` names one: then the script builds that checkout's wheel.
    Raises ``BuildRefused``."""
    where = Path(folder).resolve()
    lock = _read_lock(where)
    requested = gpu_type(gpu) if gpu is not None else None
    if deterministic and requested is None:
        raise _refuse("env.gpu-type", RefusalClass.USER, "--deterministic sets XLA's deterministic mode for a GPU, "
                      "and the run requests none", "request a GPU, or leave the CPU run as it is")
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
                                     encoding="utf-8")
        files.append(mime_requirements)
    lab_file: str | None = None
    lab_hashed = False
    if lab_requirements is not None:
        given = Path(lab_requirements).read_bytes()
        (target / LAB_REQUIREMENTS).write_bytes(given)
        lab_file, lab_hashed = LAB_REQUIREMENTS, True
    elif pins := lab_pins(lock):
        (target / LAB_REQUIREMENTS).write_text("".join(f"{name}=={version}\n" for name, version in pins),
                                               encoding="utf-8")
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
        "XLA_CACHE": xla_cache,
        "TIMEOUT": timeout,
        "MAX_CONTAINERS": max_containers,
        "CPU": cpu,
        "MEMORY": memory,
    }
    template = (importlib.resources.files("mime.env") / TEMPLATE).read_text(encoding="utf-8")
    script = target / SCRIPT_FILE
    script.write_text(template.replace("@@CONSTANTS@@", _constants(values)), encoding="utf-8")
    return ModalScript(target, script, (script, *files), variant, requested, series, source is not None)


__all__ = ["IMAGE_BUILDER_VERSION", "MODAL_CLIENT", "SCRIPT_FILE", "UV_VERSION", "ModalScript", "installed_checkout",
           "lab_pins", "modal_script", "pypi_wheel_sha256"]
