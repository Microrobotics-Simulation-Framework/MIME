"""``mime env check``: what a runner calls before a run, so that a run in an environment other than the one it asked
for is refused before it starts, never discovered in its results.

It compares the installed distributions with the variant's pins, evaluating their markers here, and with MIME's own
version, the base image's distributions and the lab's pins; it compares Python and the OS with the base image MIME
ships for this Python series; it compares JAX's devices with the GPU requested, refusing a CPU fallback; and then it
applies the precision the experiment's lock records, with the highest matmul precision on a GPU, and points XLA's
compilation cache at its directory. Every difference is refused at once, each named."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import jax
from packaging.markers import default_environment
from packaging.utils import canonicalize_name

from mime.build import BuildRefused, Refusal, RefusalClass
from mime.build._maddening import devices as devices_of_jax
from mime.env._gpus import gpu_type, is_kind_of
from mime.env._host import runner_record
from mime.env._pins import Pin, applicable, bases, parse_requirements, pins, same_version, series_of
from mime.env._xla_cache import use_xla_cache

LOCK_FILE = "mime.lock"
DISTRIBUTION = "mime-engine"
#: An installer's own distributions, which a run never imports: allowed beside the pins outside a base image, whose
#: own are pinned by its digest and held exactly.
INSTALLERS = frozenset({"pip", "setuptools", "wheel"})
HIGHEST = "highest"


class DeviceLike(Protocol):
    @property
    def platform(self) -> str: ...

    @property
    def device_kind(self) -> str: ...


@dataclass(frozen=True)
class Device:
    platform: str
    device_kind: str


def jax_devices() -> list[Device]:
    return [Device(platform, kind) for platform, kind in devices_of_jax()]


@dataclass(frozen=True)
class System:
    """What the check reads of the interpreter and the OS: Python's version, Debian's release and point version
    (``None`` off Debian), glibc's version (``None`` without glibc) and the values markers are evaluated against."""

    python: str
    debian: str | None
    debian_version: str | None
    glibc: str | None
    markers: Mapping[str, str]
    os_name: str = ""


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except OSError:
        return None


def this_system() -> System:
    release = _read("/etc/os-release") or ""
    fields = dict(line.split("=", 1) for line in release.splitlines() if "=" in line)
    libc, version = platform.libc_ver()
    debian_version = _read("/etc/debian_version")
    debian = fields.get("ID", "").strip('"') == "debian"
    return System(platform.python_version(), fields.get("VERSION_CODENAME", "").strip('"') if debian else None,
                  debian_version.strip() if debian and debian_version else None,
                  version if libc == "glibc" else None, cast(Mapping[str, str], default_environment()),
                  fields.get("PRETTY_NAME", "").strip('"') or platform.platform())


def _environment_paths() -> list[str]:
    """This interpreter's own site-packages: what is installed in its environment, not whatever a runner appends to
    ``sys.path`` (Modal's own client, for one)."""
    paths = sysconfig.get_paths()
    return sorted({paths["purelib"], paths["platlib"]})


def installed_distributions(paths: Sequence[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Every distribution installed under ``paths`` (this environment's site-packages), by canonical name, with each
    version found: more than one is an environment that holds two copies."""
    found: dict[str, set[str]] = {}
    for dist in importlib.metadata.distributions(path=list(paths or _environment_paths())):
        name = dist.metadata["Name"]
        if name:
            found.setdefault(canonicalize_name(name), set()).add(dist.version)
    return {name: tuple(sorted(versions)) for name, versions in sorted(found.items())}


def fingerprint(paths: Sequence[str] | None = None) -> str:
    """A hash of every installed distribution's name, version and ``RECORD`` (its files and their hashes), bytecode
    aside: two environments with the same fingerprint hold the same installed files."""
    digest = hashlib.sha256()
    held = sorted(importlib.metadata.distributions(path=list(paths or _environment_paths())),
                  key=lambda d: (canonicalize_name(d.metadata["Name"] or ""), d.version))
    for dist in held:
        record = sorted(line for line in (dist.read_text("RECORD") or "").splitlines() if "__pycache__" not in line)
        digest.update(f"{canonicalize_name(dist.metadata['Name'] or '')}=={dist.version}\n".encode())
        digest.update("\n".join(record).encode() + b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class Expected:
    version: str
    source: str

    def matches(self, version: str) -> bool:
        return same_version(version, self.version)


class EnvRefused(BuildRefused):
    """Every way an environment differs from the one a run asked for, at once."""

    def __init__(self, refusals: Sequence[Refusal]) -> None:
        super().__init__(refusals)
        count = len(self.refusals)
        self.args = (f"the environment is refused for {count} reason{'s' if count != 1 else ''}:\n"
                     + "\n".join(str(r) for r in self.refusals),)


REBUILD = "rebuild the environment from the pinned requirements (`mime modal` writes a script that does)"


def expected_distributions(variant: str, system: System, *, base: Mapping[str, str] | None,
                           lab: Sequence[Pin] = ()) -> tuple[dict[str, Expected], list[Refusal]]:
    """What the environment must hold: the base image's distributions, the variant's pins whose markers hold, MIME at
    its own version, and the lab's pins; and a refusal for each name two of these pin at different versions."""
    found: dict[str, Expected] = {}
    refusals: list[Refusal] = []
    sources: list[tuple[str, Mapping[str, str]]] = [
        ("the base image", dict(base or {})),
        (f"the {variant} pins", {name: pin.version for name, pin in applicable(pins(variant), system.markers).items()}),
        ("MIME itself", {DISTRIBUTION: importlib.metadata.version(DISTRIBUTION)}),
        ("the lab's pins", {name: pin.version for name, pin in applicable(lab, system.markers).items()}),
    ]
    for source, held in sources:
        for name, version in held.items():
            earlier = found.get(name)
            if earlier is None:
                found[name] = Expected(version, source)
            elif not earlier.matches(version):
                refusals.append(Refusal(
                    "env.pin", RefusalClass.ENVIRONMENT,
                    f"{name} is pinned at {earlier.version} by {earlier.source} and at {version} by {source}",
                    f"pin {name} once: take it out of {source}, or match the versions"))
    return found, refusals


def distribution_refusals(installed: Mapping[str, Sequence[str]], expected: Mapping[str, Expected], *,
                          tolerated: frozenset[str] = frozenset()) -> list[Refusal]:
    """A refusal for each pinned distribution missing or at another version, and each installed one nothing pins."""
    refusals: list[Refusal] = []
    for name, want in sorted(expected.items()):
        held = installed.get(name, ())
        if not held:
            refusals.append(Refusal("env.missing", RefusalClass.ENVIRONMENT,
                                    f"{name} {want.version}, pinned by {want.source}, is not installed", REBUILD))
        elif len(held) > 1 or not want.matches(held[0]):
            refusals.append(Refusal("env.pin", RefusalClass.ENVIRONMENT,
                                    f"{name} is installed at {', '.join(held)} and pinned at {want.version} by "
                                    f"{want.source}", REBUILD))
    for name, held in sorted(installed.items()):
        if name not in expected and name not in tolerated:
            refusals.append(Refusal("env.unpinned", RefusalClass.ENVIRONMENT,
                                    f"{name} {', '.join(held)} is installed and nothing pins it",
                                    f"uninstall it, or pin it in the lab's requirements; {REBUILD}"))
    return refusals


def base_refusals(system: System) -> tuple[Mapping[str, str] | None, list[Refusal]]:
    """The base image MIME ships for this Python series, held to what this system is: its distributions, and a
    refusal for each fact that differs."""
    series = series_of(system.python)
    held = bases().get(series)
    if held is None:
        return None, [Refusal("env.python", RefusalClass.ENVIRONMENT, f"MIME ships no base image for Python {series}",
                              f"run on Python {' or '.join(bases())}")]
    refusals: list[Refusal] = []
    if system.python != held["python"]:
        refusals.append(Refusal("env.python", RefusalClass.ENVIRONMENT,
                                f"Python is {system.python}; the base image {held['image']} holds {held['python']}",
                                f"run in {held['image']}"))
    for name, here, there in (("Debian's release", system.debian, held["debian"]),
                              ("Debian's version", system.debian_version, held["debian_version"]),
                              ("glibc", system.glibc, held["glibc"])):
        if here != there:
            refusals.append(Refusal("env.os", RefusalClass.ENVIRONMENT,
                                    f"{name} is {here or 'unknown'}; the base image {held['image']} holds {there}",
                                    f"run in {held['image']}"))
    return held["distributions"], refusals


def device_refusals(devices: Sequence[DeviceLike], gpu: str | None) -> list[Refusal]:
    """A refusal when JAX's devices are not what the run requested: a GPU of the exact type, or, with none requested,
    the CPU. A GPU requested and a CPU seen is the fallback JAX makes without a word."""
    seen = ", ".join(sorted({f"{d.platform}: {d.device_kind}" for d in devices})) or "no device"
    if gpu is None:
        if all(d.platform == "cpu" for d in devices) and devices:
            return []
        return [Refusal("env.device", RefusalClass.ENVIRONMENT, f"the run requested the CPU and JAX computes on {seen}",
                        "request the GPU the run uses, or hide it from JAX (JAX_PLATFORMS=cpu)")]
    if devices and all(d.platform == "gpu" and is_kind_of(d.device_kind, gpu) for d in devices):
        return []
    fallback = (" (JAX fell back to the CPU: its CUDA plugin found no GPU)"
                if all(d.platform == "cpu" for d in devices) else "")
    return [Refusal("env.device", RefusalClass.ENVIRONMENT, f"the run requested a {gpu} and JAX sees {seen}{fallback}",
                    f"run where JAX sees a {gpu}: attach the GPU, and install the cuda12 pins")]


def lock_x64(folder: str | Path) -> bool:
    """The precision the experiment's lock records: whether it was built in float64."""
    path = Path(folder) / LOCK_FILE
    try:
        lock: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        lock = error
    environment = cast(dict[str, object], lock).get("environment") if isinstance(lock, dict) else None
    x64 = cast(dict[str, object], environment).get("x64") if isinstance(environment, dict) else None
    if not isinstance(x64, bool):
        raise BuildRefused([Refusal("env.lock", RefusalClass.USER,
                                    f"{path} records no precision a run can apply (its environment's x64)",
                                    "give the folder a saved experiment was written to")])
    return x64


def apply_lock_precision(folder: str | Path) -> bool:
    """Set JAX's float64 mode to what ``folder``'s lock records, and return it: a run of a saved experiment computes in
    the precision it was saved in."""
    x64 = lock_x64(folder)
    cast(Any, jax.config).update("jax_enable_x64", x64)
    return x64


@dataclass(frozen=True)
class EnvReport:
    """What the check found and what it applied."""

    variant: str
    distributions: int
    python: str
    system: str
    devices: tuple[str, ...]
    x64: bool | None
    matmul_precision: str | None
    xla_cache: str | None
    fingerprint: str
    runner: Mapping[str, str] = field(default_factory=dict[str, str])
    base: str | None = None
    lab: str | None = None

    def text(self) -> str:
        lines = [f"environment: every one of the {self.distributions} distributions installed as pinned "
                 f"({self.variant} pins, MIME{', the base image' if self.base else ''}"
                 f"{', the lab' if self.lab else ''}); fingerprint {self.fingerprint[:16]}",
                 f"python {self.python}, {self.system}" + (f": the base image {self.base}" if self.base else ""),
                 f"devices: {', '.join(self.devices)}"]
        if self.x64 is not None:
            lines.append(f"precision: {'float64' if self.x64 else 'float32'}, as the experiment's lock records")
        if self.matmul_precision is not None:
            lines.append(f"matmul precision: {self.matmul_precision}")
        lines.append(f"xla cache: {self.xla_cache or 'off'}")
        return "\n".join(lines)

    def record(self) -> dict[str, object]:
        return {"variant": self.variant, "distributions": self.distributions, "python": self.python,
                "system": self.system, "devices": list(self.devices), "x64": self.x64,
                "matmul_precision": self.matmul_precision, "xla_cache": self.xla_cache,
                "fingerprint": self.fingerprint, "runner": dict(self.runner), "base": self.base, "lab": self.lab}


def check(folder: str | Path | None = None, *, variant: str, gpu: str | None = None, base: bool = False,
          lab: str | Path | None = None, xla_cache: str | Path | None = None,
          installed: Mapping[str, Sequence[str]] | None = None, devices: Sequence[DeviceLike] | None = None,
          system: System | None = None) -> EnvReport:
    """Check this environment against the pins of ``variant`` (with ``base``, against the base image of this Python
    series too, and with ``lab``, a requirements file, against the lab's pins), and JAX's devices against ``gpu`` (an
    exact type, or ``None`` for the CPU). Then apply the precision ``folder``'s lock records, the highest matmul
    precision on a GPU, and XLA's cache under ``xla_cache``. Raises ``BuildRefused`` with every difference;
    ``installed``, ``devices`` and ``system`` stand in for what the check reads, in tests."""
    requested = gpu_type(gpu) if gpu is not None else None
    runner = runner_record()
    x64 = lock_x64(folder) if folder is not None else None
    held_system = system or this_system()
    lab_pins = parse_requirements(Path(lab).read_text(encoding="utf-8")) if lab is not None else []
    base_distributions, refusals = base_refusals(held_system) if base else (None, [])
    expected, conflicts = expected_distributions(variant, held_system, base=base_distributions, lab=lab_pins)
    found = installed if installed is not None else installed_distributions()
    refusals += conflicts
    refusals += distribution_refusals(found, expected, tolerated=frozenset() if base else INSTALLERS)
    seen: list[DeviceLike] = list(devices) if devices is not None else list(jax_devices())
    refusals += device_refusals(seen, requested)
    if refusals:
        raise EnvRefused(refusals)
    config = cast(Any, jax.config)
    if x64 is not None:
        config.update("jax_enable_x64", x64)
    if requested is not None:
        config.update("jax_default_matmul_precision", HIGHEST)
    cache = str(use_xla_cache(xla_cache)) if xla_cache is not None else None
    held_base = bases()[series_of(held_system.python)]["image"] if base else None
    os_text = (held_system.os_name or platform.platform()) + (f", glibc {held_system.glibc}" if held_system.glibc
                                                              else "")
    return EnvReport(variant, len(expected), held_system.python, os_text,
                     tuple(sorted({f"{d.platform}: {d.device_kind}" for d in seen})), x64,
                     config.jax_default_matmul_precision, cache,
                     fingerprint() if installed is None else "not read (the distributions were given)", runner,
                     held_base, str(lab) if lab is not None else None)


__all__ = ["DISTRIBUTION", "INSTALLERS", "EnvRefused", "EnvReport", "System", "apply_lock_precision", "base_refusals",
           "check", "device_refusals",
           "distribution_refusals", "expected_distributions", "fingerprint", "installed_distributions", "lock_x64",
           "this_system"]
