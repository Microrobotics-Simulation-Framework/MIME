"""The pinned requirements of each variant and the base image of each Python series, as MIME ships them in
``mime.data``'s ``env/`` folder.

A requirements file is ``uv export``'s output from MIME's ``uv.lock``, byte for byte: every distribution a variant
installs besides MIME, with its version, its environment marker and its hashes. ``bases.json`` gives, for each Python
series, the base image's reference pinned by digest, with what the image holds; ``uv.json`` the uv image that installs
the pins, pinned by digest too."""

from __future__ import annotations

import importlib.resources
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing_extensions import TypedDict

from packaging.markers import InvalidMarker, Marker
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from mime.build import BuildRefused, Refusal, RefusalClass

ENV_FOLDER = "env"
BASES_FILE = "bases.json"
UV_FILE = "uv.json"
#: The variants MIME ships pins for: JAX on the CPU alone, and JAX with its CUDA 12 plugin and libraries.
VARIANTS = ("cpu", "cuda12")
#: ``uv export``'s arguments for each variant's file, run in MIME's checkout. ``--no-header`` keeps the bytes free of
#: the command and the output path, so the file is the same wherever it is written.
EXPORT_ARGUMENTS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    variant: ("export", "--frozen", "--no-dev", "--no-emit-project", "--no-header", "--format", "requirements.txt",
              *(("--extra", variant) if variant != "cpu" else ()))
    for variant in VARIANTS})


class Base(TypedDict, closed=True):
    """A base image: its reference pinned by the digest of its multi-platform index, the digest of its linux/amd64
    manifest, and what it holds: Python's version, Debian's release and point version, glibc's version and the Python
    distributions it ships, by canonical name."""

    image: str
    linux_amd64: str
    python: str
    debian: str
    debian_version: str
    glibc: str
    distributions: dict[str, str]


class UvImage(TypedDict, closed=True):
    """The uv that installs the pins: its release, its image on GitHub's registry pinned by the digest of its index
    (``image``, the reference a build pulls) and its linux/amd64 manifest's digest."""

    image: str
    linux_amd64: str
    version: str


@dataclass(frozen=True)
class Pin:
    """One line of a requirements file: a distribution by canonical name, its exact version, its environment marker
    (``None`` when it applies everywhere) and its hashes."""

    name: str
    version: str
    marker: str | None
    hashes: tuple[str, ...]

    def applies(self, environment: Mapping[str, str] | None = None) -> bool:
        """Whether the marker holds in ``environment`` (this interpreter's, by default)."""
        return self.marker is None or Marker(self.marker).evaluate(dict(environment) if environment else None)

    def matches(self, version: str) -> bool:
        return same_version(version, self.version)


def same_version(a: str, b: str) -> bool:
    """Whether two versions are one release (``1.0`` is ``1.0.0``); one that does not parse only as the same text."""
    try:
        return Version(a) == Version(b)
    except InvalidVersion:
        return a == b


def requirements_name(variant: str) -> str:
    return f"requirements-{variant}.txt"


def _refuse_variant(variant: str) -> BuildRefused:
    return BuildRefused([Refusal("env.variant", RefusalClass.USER, f"MIME ships no pins for the variant {variant!r}",
                                 f"name one of {', '.join(VARIANTS)}")])


def requirements_bytes(variant: str) -> bytes:
    """The variant's requirements file as MIME ships it."""
    if variant not in VARIANTS:
        raise _refuse_variant(variant)
    return (importlib.resources.files("mime.data") / ENV_FOLDER / requirements_name(variant)).read_bytes()


_LINE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)\s*(?:;\s*(?P<marker>.+?))?\s*$")


def parse_requirements(text: str) -> list[Pin]:
    """Every pin of a hashed requirements file as ``uv export`` writes it: ``name==version [; marker]``, its
    ``--hash=`` options on continuation lines, and ``#`` comments. A line of any other form raises ``ValueError``
    naming it, since a pin the check cannot read is one it cannot hold."""
    pins: list[Pin] = []
    logical: list[str] = []
    current = ""
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            if current:
                logical.append(current)
                current = ""
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        logical.append(current + line)
        current = ""
    if current:
        logical.append(current)
    for entry in logical:
        words = entry.split()
        hashes = tuple(word.removeprefix("--hash=") for word in words if word.startswith("--hash="))
        requirement = " ".join(word for word in words if not word.startswith("--hash="))
        match = _LINE.match(requirement)
        if match is None:
            raise ValueError(f"a requirement MIME cannot read as name==version [; marker]: {requirement!r}")
        marker = match.group("marker")
        if marker is not None:
            try:
                Marker(marker)
            except InvalidMarker as error:
                raise ValueError(f"{requirement!r}: its marker does not parse: {error}") from error
        pins.append(Pin(canonicalize_name(match.group("name")), match.group("version"), marker, hashes))
    return pins


@cache
def pins(variant: str) -> tuple[Pin, ...]:
    """The variant's pins, as MIME ships them."""
    return tuple(parse_requirements(requirements_bytes(variant).decode("utf-8")))


@cache
def bases() -> Mapping[str, Base]:
    """Each Python series' base image, by series (``"3.14"``)."""
    text = (importlib.resources.files("mime.data") / ENV_FOLDER / BASES_FILE).read_text(encoding="utf-8")
    return MappingProxyType(json.loads(text))


@cache
def uv_image() -> UvImage:
    """The uv image the pins are installed with, pinned by digest."""
    text = (importlib.resources.files("mime.data") / ENV_FOLDER / UV_FILE).read_text(encoding="utf-8")
    return json.loads(text)


def uv_digest() -> str:
    """The digest of the uv image's index: the ``sha256:...`` after the ``@`` of its reference."""
    return uv_image()["image"].rsplit("@", 1)[1]


def series_of(version: str) -> str:
    """The series of a Python version: ``3.14`` for ``3.14.10``."""
    return ".".join(version.split(".")[:2])


def applicable(held: Sequence[Pin], environment: Mapping[str, str] | None = None) -> dict[str, Pin]:
    """The pins whose markers hold in ``environment``, by name. Two pins of one name that both hold raise
    ``ValueError``: a file that pins one distribution twice for one environment pins nothing."""
    found: dict[str, Pin] = {}
    for pin in held:
        if pin.applies(environment):
            if pin.name in found:
                raise ValueError(f"{pin.name} is pinned twice for this environment: {found[pin.name].version} and "
                                 f"{pin.version}")
            found[pin.name] = pin
    return found


__all__ = ["BASES_FILE", "ENV_FOLDER", "EXPORT_ARGUMENTS", "UV_FILE", "VARIANTS", "Base", "Pin", "UvImage",
           "applicable", "bases", "parse_requirements", "pins", "requirements_bytes", "requirements_name",
           "same_version", "series_of", "uv_digest", "uv_image"]
